"""Execute pinned final-SS evaluation against the adapter on CPU fixtures.

Only the CUDA clustering kernel is replaced. These tests check features,
anchor-label visibility, fitter arguments, RNG and metrics, not CUDA accuracy.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gcd_runtime.config import ROOT


class SelExFinalProtocolTests(unittest.TestCase):
    def test_source_ss_feature_anchor_rng_and_matching_parity(self):
        code = r'''
import ast, math, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from gcd_runtime.config import ROOT,load_config
from selex_runtime.reference import configure_environment,SOURCE_ROOT
from selex_runtime.rng import capture_rng,restore_rng,rng_digest,seed_ss_setup
from selex_runtime.trainer import evaluate_ss,make_args
cfg=load_config(ROOT/'configs/selex_cifar100_baseline_200.json')
configure_environment(Path(sys.argv[1]),cfg)
from project_utils.cluster_and_log_utils import split_cluster_acc_v2
from methods.clustering import faster_mix_k_means_pytorch as fitter
args=make_args(cfg)
args.num_labeled_classes=args.num_unlabeled_classes=2
args.train_classes=[0,1]
args.max_kmeans_iter=args.ss_max_iter
args.k_means_init=args.ss_n_init
args.eval_funcs=['v2']
images=torch.eye(8)*torch.arange(1,9)[:,None]
truth=torch.tensor([0,1,0,1,2,2,3,3])
mask=torch.tensor([True,True,False,False,False,False,False,False])
ids=torch.arange(8)
calls=[]
prediction=torch.tensor([0,1,0,0,2,3,3,3])
class Cluster:
 def __init__(self,**kwargs):
  self.kwargs=dict(kwargs)
  self.kwargs.setdefault('is_code',False)
 def fit_mix(self,u_feats,l_feats,l_targets):
  assert len(l_targets)==2 and set(l_targets.tolist())=={0,1}
  calls.append((self.kwargs,u_feats.clone(),l_feats.clone(),l_targets.clone()))
  np.random.choice(len(u_feats),2,replace=False)
  torch.randint(8,(1,))
  self.labels_=torch.cat([l_targets.long(),prediction[u_feats.argmax(1)]])
def log_accs_from_preds(y_true,y_pred,mask,**kwargs):
 return split_cluster_acc_v2(y_true,y_pred,mask)
node=next(n for n in ast.parse((SOURCE_ROOT/'methods/clustering/k_means.py').read_text()).body
 if isinstance(n,ast.FunctionDef) and n.name=='test_kmeans_semi_sup')
scope=dict(np=np,torch=torch,math=math,device=torch.device('cpu'),
 tqdm=lambda x,**kwargs:x,SemiSupKMeans=Cluster,log_accs_from_preds=log_accs_from_preds)
exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned SS evaluator>','exec'),scope)
records=[]
runtime=SimpleNamespace(config=cfg,evaluation=records.append)
seed_ss_setup()
start=capture_rng()
cached=F.normalize(images,dim=-1)
expected=scope['test_kmeans_semi_sup'](
 DataLoader(TensorDataset(cached,truth,ids,mask),batch_size=args.batch_size),args,
 unbalanced=getattr(args,'unbalanced',False))
expected_rng=rng_digest()
restore_rng(start)
with patch.object(torch.Tensor,'cuda',lambda self,*unused,**kwargs:self), \
 patch.object(fitter,'K_Means',Cluster):
 evaluate_ss(torch.nn.Identity(),TensorDataset(images,truth,ids,mask[:,None]),args,runtime)
assert rng_digest()==expected_rng
assert len(calls)==2 and calls[0][0]==calls[1][0]
for a,b in zip(calls[0][1:],calls[1][1:]): torch.testing.assert_close(a,b,check_dtype=False)
np.testing.assert_allclose([records[-1][k] for k in ('all_acc','old_acc','new_acc')],expected[:3])
assert records[-1]['n_unlabeled']==6 and records[-1]['n_old']==2 and records[-1]['n_new']==4
assert abs(records[-1]['all_acc']-(2*records[-1]['old_acc']+4*records[-1]['new_acc'])/6)<1e-12
# Unlabeled evaluation labels may change metrics, never the fitter inputs/RNG.
poisoned=truth.clone(); poisoned[~mask]=torch.tensor([0,0,2,3,2,3])
restore_rng(start)
with patch.object(torch.Tensor,'cuda',lambda self,*unused,**kwargs:self), \
 patch.object(fitter,'K_Means',Cluster):
 evaluate_ss(torch.nn.Identity(),TensorDataset(images,poisoned,ids,mask[:,None]),args,runtime)
assert rng_digest()==expected_rng
for a,b in zip(calls[1][1:],calls[2][1:]): torch.testing.assert_close(a,b)
# Row restoration works even when an evaluation fixture interleaves anchors.
order=torch.tensor([6,0,3,1,4,7,2,5])
restore_rng(start)
with patch.object(torch.Tensor,'cuda',lambda self,*unused,**kwargs:self), \
 patch.object(fitter,'K_Means',Cluster):
 evaluate_ss(torch.nn.Identity(),TensorDataset(images[order],truth[order],ids[order],mask[order,None]),args,runtime)
np.testing.assert_allclose([records[-1][k] for k in ('all_acc','old_acc','new_acc')],expected[:3])
print('Synthetic CPU final SS: pinned evaluator parity, legal anchors, RNG, row mapping and weighted global-v2 metrics passed')
'''
        for dataset in ('cifar100', 'herbarium_19'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                program = code.replace('selex_cifar100_baseline_200.json', f'selex_{dataset}_baseline_200.json')
                result = subprocess.run([sys.executable, '-c', program, tmp], cwd=ROOT,
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
