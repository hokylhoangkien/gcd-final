"""RNG protocol and recovery checks; synthetic CPU fixtures, no benchmark scores."""
import ast
import copy
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gcd_runtime.config import ROOT, load_config
from selex_runtime.reference import SOURCE_ROOT, TRAIN
from selex_runtime.rng import (capture_rng, restore_rng, rng_digest, seed_training,
    seed_ss_setup, create_request, begin_plain_worker, finish_plain_worker,
    commit_plain_result, atomic_bundle)
from selex_runtime.trainer import SelExRuntime


def evaluation_draws():
    # Native loader base seed, global NumPy initialization and the CPU Torch
    # fallback used for empty clusters. No target values enter these operations.
    list(DataLoader(TensorDataset(torch.arange(12)), batch_size=4, num_workers=0))
    np.random.choice(12, 3, replace=False)
    torch.randint(12, (1,))


class SelExRngTests(unittest.TestCase):
    def setUp(self):
        self.state = capture_rng()
        self.flags = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic

    def tearDown(self):
        restore_rng(self.state)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = self.flags

    def test_training_seed_matches_source_and_leaves_python_flags_alone(self):
        source = TRAIN.read_text()
        self.assertIn('np.random.seed(args.seed)', source)
        self.assertIn('torch.manual_seed(args.seed)', source)
        py = random.getstate()
        flags = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
        seed_training(7)
        actual = capture_rng()
        np.random.seed(7)
        torch.manual_seed(7)
        self.assertEqual(rng_digest(actual), rng_digest())
        self.assertEqual(random.getstate(), py)
        self.assertEqual(flags, (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))

    def test_ss_setup_uses_source_helper_before_dataset_draws(self):
        seed_ss_setup()
        source_state = capture_rng()
        node = next(n for n in ast.parse((SOURCE_ROOT/'project_utils/general_utils.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'seed_torch')
        import os
        scope = dict(random=random, np=np, torch=torch, os=os)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<source seed>', 'exec'), scope)
        scope['seed_torch'](0)
        self.assertEqual(rng_digest(source_state), rng_digest())
        # Dataset subsampling advances NumPy; do not reset that before fitting.
        np.random.choice(100, 50, replace=False)
        after_data = rng_digest()
        self.assertNotEqual(after_data, rng_digest(source_state))
        from selex_runtime.trainer import evaluate_ss, run
        import inspect
        run_source = inspect.getsource(run)
        self.assertLess(run_source.index('seed_ss_setup()'), run_source.index('merged, test'))
        tree = ast.parse(inspect.getsource(evaluate_ss))
        resets = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'not upstream_policy(runtime.config)']
        self.assertEqual(len(resets), 1)
        self.assertIn('np.random.seed(0)', ast.unparse(resets[0]))

    def test_worker_handoff_matches_inline_and_rejects_stale_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = SimpleNamespace(run_dir=Path(tmp), config={'evaluation_rng':'upstream'},
                                      evaluation_epoch=2, evaluated_checkpoint=Path(tmp)/'model.pt')
            runtime.evaluated_checkpoint.write_bytes(b'fixture')
            from gcd_runtime.readiness import sha256
            request = create_request(runtime, 2, sha256(runtime.evaluated_checkpoint))
            evaluation_draws()
            expected = rng_digest()
            next_step = torch.rand(6)
            # Worker setup may consume arbitrary draws; handoff discards them.
            code = '''
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from selex_runtime.rng import seed_training, begin_plain_worker, finish_plain_worker
runtime=SimpleNamespace(run_dir=Path(sys.argv[1]),config={'evaluation_rng':'upstream'},
 evaluation_epoch=2,evaluated_checkpoint=Path(sys.argv[1])/'model.pt')
seed_training(999)
request=begin_plain_worker(runtime)
list(DataLoader(TensorDataset(torch.arange(12)),batch_size=4,num_workers=0))
np.random.choice(12,3,replace=False)
torch.randint(12,(1,))
finish_plain_worker(runtime,request)
'''
            child = subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT,
                                   capture_output=True, text=True, timeout=15)
            self.assertEqual(child.returncode, 0, child.stdout+child.stderr)
            seed_training(123)
            commit_plain_result(runtime, request)
            self.assertEqual(rng_digest(), expected)
            self.assertTrue(torch.equal(torch.rand(6), next_step))
            result_path = Path(tmp)/'plain_rng_output.pt'
            result = torch.load(result_path, weights_only=False)
            for changed in (dict(request_id='stale'), dict(all_acc=.99)):
                poisoned = dict(result, **changed)
                atomic_bundle(result_path, poisoned)
                before = rng_digest()
                with self.assertRaisesRegex(ValueError, 'Stale'):
                    commit_plain_result(runtime, request)
                self.assertEqual(before, rng_digest())
            poisoned = copy.deepcopy(result)
            poisoned['rng']['targets'] = [0,1]
            atomic_bundle(result_path, poisoned)
            with self.assertRaisesRegex(ValueError, 'Only RNG'):
                commit_plain_result(runtime, request)

    def test_pending_eval_resume_recovers_once_and_complete_resume_exports_backbone(self):
        cfg = load_config(ROOT/'configs/selex_baseline_smoke.json')
        cfg.update(device='cpu', epochs=1)
        model, head = torch.nn.Linear(2,2), torch.nn.Linear(2,2)
        optimizer = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
            runtime = SelExRuntime(a, cfg)
            runtime.plain_rng_pending = True
            runtime.save(0, model, head, optimizer, scheduler)
            evaluation_draws()
            expected = rng_digest()
            recovered = SelExRuntime(b, cfg, str(Path(a)/'resume.pt'))
            def recover(runtime, epoch, unused_model):
                self.assertEqual(epoch, 0)
                evaluation_draws()
                runtime.plain_rng_pending = False
            with patch('selex_runtime.monitoring.monitor_plain', side_effect=recover) as monitor:
                self.assertEqual(recovered.restore(model, head, optimizer, scheduler), 1)
                monitor.assert_called_once()
            self.assertEqual(rng_digest(), expected)
            self.assertTrue((Path(b)/'backbone_final.pt').is_file())
            completed = SelExRuntime(c, cfg, str(Path(b)/'resume.pt'))
            with patch('selex_runtime.monitoring.monitor_plain') as monitor:
                self.assertEqual(completed.restore(model, head, optimizer, scheduler), 1)
                monitor.assert_not_called()
            self.assertTrue((Path(c)/'backbone_final.pt').is_file())

    def test_plain_evaluator_matches_source_draws_and_ignores_target_values(self):
        code = '''
import ast, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from gcd_runtime.config import ROOT, load_config
from selex_runtime.reference import configure_environment, TRAIN
cfg=load_config(ROOT/'configs/selex_baseline.json')
configure_environment(Path(sys.argv[1]),cfg)
from project_utils.cluster_and_log_utils import split_cluster_acc_v2
from selex_runtime.trainer import evaluate_plain
from selex_runtime.rng import capture_rng, restore_rng, rng_digest, seed_training
args=SimpleNamespace(batch_size=4,num_workers=0,num_labeled_classes=1,
 num_unlabeled_classes=1,train_classes=[0],eval_funcs=['v2'],writer=None)
model=torch.nn.Linear(4,4).eval().requires_grad_(False)
images=torch.arange(32).reshape(8,4).float()
targets=torch.arange(8)%2
dataset=TensorDataset(images,targets,torch.arange(8))
seen=[]
def source_kmeans(X,num_clusters,**kwargs):
 assert kwargs['iter_limit']==1000 and kwargs['tqdm_flag'] is False
 np.random.choice(len(X),num_clusters,replace=False)
 torch.randint(len(X),(1,))
 seen.append(X.clone())
 return torch.arange(len(X))%2,torch.eye(2,4)
def adapted_kmeans(features,clusters,device,*,iteration_limit):
 pred,_=source_kmeans(features,clusters,iter_limit=iteration_limit,tqdm_flag=False)
 return pred.numpy(),{'backend':'synthetic_fixture'}
def log_accs_from_preds(y_true,y_pred,mask,**kwargs):
 return split_cluster_acc_v2(y_true,y_pred,mask)
scope=dict(torch=torch,np=np,device=torch.device('cpu'),F=F,tqdm=lambda x:x,
 kmeans=source_kmeans,log_accs_from_preds=log_accs_from_preds)
node=next(n for n in ast.parse(TRAIN.read_text()).body
 if isinstance(n,ast.FunctionDef) and n.name=='test_kmeans')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(TRAIN),'exec'),scope)
seed_training(5)
start=capture_rng()
runtime=SimpleNamespace(config=cfg,evaluation=lambda metrics:None)
with patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self), \
 patch('gcd_runtime.disjoint_evaluation.plain_kmeans',side_effect=adapted_kmeans):
 scope['test_kmeans'](model,DataLoader(dataset,batch_size=4,shuffle=False),0,'fixture',args)
 expected=rng_digest()
 restore_rng(start)
 evaluate_plain(model,dataset,args,runtime)
 assert rng_digest()==expected
 assert torch.equal(seen[0],seen[1])
 restore_rng(start)
 poisoned=TensorDataset(images,1-targets,torch.arange(8))
 evaluate_plain(model,poisoned,args,runtime)
 assert rng_digest()==expected
print('CPU source evaluator RNG parity; target-value invariance')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT,
                                    capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
