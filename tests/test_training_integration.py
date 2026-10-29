"""Actual adapted train loop on synthetic batches; no dataset/weights/benchmark."""
import copy
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.worker import configure_environment
from gcd_runtime.checkpoint import EpochRuntime


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass


class TrainingIntegrationTests(unittest.TestCase):
    def test_two_epoch_loop_baseline_proposed_and_disabled_parity(self):
        upstream=configure_environment(ROOT/'runs/probe-placeholder')
        from gcd_runtime import trainer
        import importlib
        results={}
        for variant in ('original','baseline','neighborhood','crossview','proposed','proposed_relabelled','all_unlabeled','disabled'):
            torch.manual_seed(110)
            np.random.seed(110)
            profile='baseline' if variant=='original' else ('proposed' if variant in ('disabled','proposed_relabelled','all_unlabeled') else variant)
            cfg=load_config(ROOT/f'configs/cars_{profile}_dino_v1.json')
            cfg.update(epochs=2,synthetic_fixture=True)
            if variant not in ('baseline','original'):
                cfg['reliability'].update(k=1,warmup_epochs=0,ramp_epochs=0,diagnostics_every=1)
            if variant=='disabled':
                cfg['reliability'].update(reciprocal=False,cross_view=False)
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                args=SimpleNamespace(lr=.01,momentum=.9,weight_decay=0.,epochs=2,c=.1,
                     strategy='zero_one',cluster_momentum=1,unsupervised_smoothing=.5,
                     train_report_interval=200,prototype_extraction_interval=1,distance='euclidean',
                     num_labeled_classes=2,num_unlabeled_classes=2,contrast_unlabel_only=False,
                     sup_con_weight=.35,n_views=2,temperature=1.,hyper_start_epoch=0,
                     hyper_end_epoch=200,hyper_max_weight=1.,writer=NullWriter(),model_path=str(Path(tmp)/'model.pt'))
                runtime=EpochRuntime(tmp,cfg)
                trainer=importlib.reload(trainer)
                scope=vars(trainer)
                scope.update(_runtime=runtime,args=args,device=torch.device('cpu'))
                if variant=='original':
                    tree=ast.parse((upstream/'train/train_HypSelEx.py').read_text())
                    original_train=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='train')
                    exec(compile(ast.Module(body=[original_train],type_ignores=[]),'original_train','exec'),scope)
                from hyptorch.nn import ToPoincare
                model=torch.nn.Linear(4,8)
                head=torch.nn.Sequential(torch.nn.Linear(8,16),ToPoincare(c=.1,clip_r=2.,riemannian=False))
                batch=([torch.randn(4,4),torch.randn(4,4)],
                       torch.tensor([0,0,-333,88888] if variant=='proposed_relabelled' else [0,0,123456,-9876]),torch.arange(4),
                       torch.zeros(4,1,dtype=torch.long) if variant=='all_unlabeled' else torch.tensor([[1],[1],[0],[0]]))
                def extract(*a,**kw):
                    # Fixed synthetic pseudo labels, not the poison unlabeled targets.
                    pseudo=torch.zeros(4,dtype=torch.long)
                    return (np.arange(4),pseudo,[torch.zeros(4,8),torch.zeros(2,8)],[pseudo,pseudo.clone()],
                            dict(all_acc=0.,old_acc=0.,new_acc=0.))
                scope['extract_labeled_protos']=extract
                scope['test_kmeans']=lambda *a,**kw:(.2,.1,.3)
                scope['train'](head,model,[batch],[],[],[],[],args)
                if variant=='original':
                    results[variant]=copy.deepcopy(model.state_dict())
                    continue
                self.assertTrue((Path(tmp)/'resume.pt').is_file())
                self.assertTrue((Path(tmp)/'backbone_final.pt').is_file())
                self.assertEqual(len((Path(tmp)/'metrics.jsonl').read_text().splitlines()),2)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                if variant in ('neighborhood','crossview','proposed','proposed_relabelled','all_unlabeled'):
                    records=[json.loads(line) for line in (Path(tmp)/'reliability.jsonl').read_text().splitlines()]
                    self.assertEqual(len(records),4)
                    self.assertEqual({r['level'] for r in records},{0,1})
                    self.assertEqual(records[0]['level'],0)
                results[variant]=copy.deepcopy(model.state_dict())
        for key in results['baseline']:
            self.assertTrue(torch.equal(results['baseline'][key],results['original'][key]))
            self.assertTrue(torch.equal(results['baseline'][key],results['disabled'][key]))
            self.assertTrue(torch.equal(results['proposed'][key],results['proposed_relabelled'][key]))

class ResumeTrainingTests(unittest.TestCase):
    def test_proposed_epoch_resume_matches_uninterrupted_training(self):
        configure_environment(ROOT/'runs/probe-placeholder')
        from gcd_runtime import trainer
        from hyptorch.nn import ToPoincare
        import importlib
        cfg=load_config(ROOT/'configs/cars_proposed_dino_v1.json')
        cfg.update(epochs=3,synthetic_fixture=True)
        cfg['reliability'].update(k=1,warmup_epochs=1,ramp_epochs=2,diagnostics_every=1)

        class StopAfterEpochZero(EpochRuntime):
            def save(self,epoch,*args,**kwargs):
                super().save(epoch,*args,**kwargs)
                if epoch==0:
                    raise InterruptedError('Synthetic epoch-boundary interruption')

        def execute(runtime,seed):
            module=importlib.reload(trainer)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model=torch.nn.Linear(4,8)
            head=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.Dropout(.1),
                                     ToPoincare(c=.1,clip_r=2.,riemannian=False))
            generator=torch.Generator().manual_seed(87)
            batch=([torch.randn(4,4,generator=generator),torch.randn(4,4,generator=generator)],
                   torch.tensor([0,0,333,-777]),torch.arange(4),torch.tensor([[1],[1],[0],[0]]))
            args=SimpleNamespace(lr=.01,momentum=.9,weight_decay=0.,epochs=3,c=.1,
                 strategy='zero_one',cluster_momentum=1,unsupervised_smoothing=.5,
                 train_report_interval=200,prototype_extraction_interval=1,distance='euclidean',
                 num_labeled_classes=2,num_unlabeled_classes=2,contrast_unlabel_only=False,
                 sup_con_weight=.35,n_views=2,temperature=1.,hyper_start_epoch=0,
                 hyper_end_epoch=200,hyper_max_weight=1.,writer=NullWriter(),
                 model_path=str(runtime.run_dir/'model.pt'))
            module.args,module.device,module._runtime=args,torch.device('cpu'),runtime
            def extract(*a,**kw):
                pseudo=torch.zeros(4,dtype=torch.long)
                return np.arange(4),pseudo,[torch.zeros(4,8),torch.zeros(2,8)],[pseudo,pseudo.clone()],dict(all_acc=0.,old_acc=0.,new_acc=0.)
            module.extract_labeled_protos=extract
            module.test_kmeans=lambda *a,**kw:(.2,.1,.3)
            module.train(head,model,[batch],[],[],[],[],args)
            return copy.deepcopy(model.state_dict()),copy.deepcopy(head.state_dict())

        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as interrupted, tempfile.TemporaryDirectory() as resumed:
            expected=execute(EpochRuntime(full,cfg),110)
            with self.assertRaises(InterruptedError):
                execute(StopAfterEpochZero(interrupted,cfg),110)
            actual=execute(EpochRuntime(resumed,cfg,str(Path(interrupted)/'resume.pt')),999)
            for a,b in zip(expected,actual):
                for key in a:
                    self.assertTrue(torch.equal(a[key],b[key]),key)


if __name__=='__main__':
    unittest.main()
