"""SelEx source parity, active-level repair and unchanged Hyp-SelEx controls."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crish.config import CrishSettings
from crish.engine import CrishEngine
from crish.smoke import FeatureFixture
from crish.state import SupervisionState
from crish.rebuild import prototype_radii, rebuild, target_drift
from crish.candidates import Action
from crish.training import train_strict
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.checkpoint import EpochRuntime
from selex_runtime.hierarchy import coarse_levels
from selex_runtime.objective import selex_loss
from selex_runtime.reference import TRAIN, SOURCE_ROOT, verify_reference, definitions
from selex_runtime.trainer import make_args


def fixture(epoch=0):
    torch.manual_seed(311)
    x = torch.randn(40, 4)
    labels = torch.arange(40) % 4
    mask = torch.zeros(40, dtype=torch.bool)
    mask[[0, 1, 4, 5, 8, 9, 12, 13]] = True
    dataset = FeatureFixture(x, mask, labels[mask])
    features = F.normalize(x, dim=-1)
    centers = torch.stack([features[labels == i].mean(0) for i in range(4)])
    levels, protos = coarse_levels(features, labels, centers, 2, 2)
    active = min(len(levels), 1 + epoch//10)
    levels, protos = levels[:active], protos[:active]
    state = SupervisionState(torch.tensor(dataset.sample_ids), features, mask, labels[mask],
        levels, protos, tuple(prototype_radii(p) for p in protos),
        tuple(torch.zeros(40, dtype=torch.bool) for _ in levels))
    cfg = load_config(ROOT/'configs/selex_crish.json')
    args = make_args(cfg)
    args.num_labeled_classes, args.num_unlabeled_classes, args.hierarchy_levels = 2, 2, 2
    head = torch.nn.Linear(4, 16)
    return dataset, state, args, head


def original_objective(head, raw, state, args, epoch):
    """Execute the original factor, USE and SSE statements from the pinned file."""
    scope = definitions('cpu')
    n = len(state.ids)
    cfg = dict(scope, args=args, epoch=epoch, Distance='euclidean', strategy='zero_one',
        cluster_momentum=1., all_features=raw.detach(), features=F.normalize(head(raw),dim=-1),
        preds_ind_list=list(state.labels), cluster_protos_list=list(state.prototypes),
        cluster_radius_list=[p.clone() for p in state.radii], uq_index=np.arange(n),
        uq_idxs=np.arange(n), mask_lab=state.labeled,
        class_labels=torch.where(state.labeled, state.labels[0], torch.full_like(state.labels[0],-1)),
        unsupervised_smoothing=args.unsupervised_smoothing, sup_con_crit=scope['SupConLoss']())
    source = TRAIN.read_text()
    start = source.index('                confusion_factor=0')
    end = source.index('            # Choose which instances', start)
    exec(textwrap.dedent(source[start:end]), cfg)
    start = source.index('            confusion_factor = (confusion_factor - confusion_factor.min())', end)
    end = source.index('            # Train acc', start)
    block = textwrap.dedent(source[start:end]).replace('torch.cuda.empty_cache()', 'pass')
    cfg['con_feats'] = cfg['features']
    exec(block, cfg)
    return cfg['loss_in']


class SelExRuntimeTests(unittest.TestCase):
    def test_fresh_worker_uses_selex_models_without_hyperbolic_imports(self):
        code = """
import sys, torch
from gcd_runtime.config import ROOT, load_config
from selex_runtime.reference import configure_environment
cfg=load_config(ROOT/'configs/selex_crish.json')
configure_environment(ROOT/'runs/probe-placeholder',cfg)
from models.vision_transformer2 import DINOHead
from selex_runtime.trainer import make_args
head=DINOHead(4,16,hidden_dim=8,bottleneck_dim=4)
head(torch.randn(4,4)).square().mean().backward()
assert not any(name.startswith('hyptorch') for name in sys.modules)
print('source SelEx head backward; no hyperbolic import')
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,
                              capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_coarse_grouping_matches_executed_new_source(self):
        from sklearn.cluster import KMeans
        torch.manual_seed(77)
        features=F.normalize(torch.randn(80,5),dim=1)
        labels=torch.arange(80)%8
        centers=torch.stack([features[labels==g].mean(0) for g in range(8)])
        got_labels,got_centers=coarse_levels(features,labels,centers,4,4)
        source=TRAIN.read_text()
        start=source.index('    feats_t = torch.from_numpy(all_feats)')
        end=source.index('    return ids,all_preds',start)
        scope=dict(torch=torch,np=np,KMeans=KMeans,device=torch.device('cpu'),
            all_feats=features.numpy(),all_preds=labels,prototypes=centers,
            n_labeled=4,n_novel=4,preds_higher=[labels],prototype_higher=[centers])
        exec(textwrap.dedent(source[start:end]),scope)
        for got,want in zip(got_labels,scope['preds_higher']):
            self.assertTrue(torch.equal(got,want))
        for got,want in zip(got_centers,scope['prototype_higher']):
            self.assertTrue(torch.allclose(got,want,atol=1e-7))

    def test_pinned_source_config_and_matched_pairs(self):
        self.assertEqual(verify_reference()['sha'], 'd8107b5458f3337b2d5342f79345a976037a1265')
        import runpy
        official = runpy.run_path(str(SOURCE_ROOT/'config.py'))['dataset_configs']['scars']
        for suffix in ('', '_smoke'):
            b = load_config(ROOT/f'configs/selex_baseline{suffix}.json')
            c = load_config(ROOT/f'configs/selex_crish{suffix}.json')
            self.assertEqual({k for k in b if b[k] != c[k]}, {'method'})
            self.assertFalse({'c','cr','hyper_max_weight'} & b.keys())
            for key in ('temperature', 'grad_from_block', 'unsupervised_smoothing'):
                self.assertEqual(b[key], official[key])
            self.assertEqual(b['ss_n_init'], 10)

    def test_loss_and_gradients_match_executed_updated_source(self):
        for epoch in (0,9,10,19):
            with self.subTest(epoch=epoch):
                dataset, state, args, head = fixture(epoch)
                raw = torch.cat([dataset.features, dataset.features+.01]).requires_grad_()
                other = copy.deepcopy(head)
                result = selex_loss(head, raw, state.ids, state.labeled, state.known_targets, state, args, epoch)['loss']
                expected = original_objective(other, raw, state, args, epoch)
                self.assertTrue(torch.allclose(result,expected,atol=1e-6), (result,expected))
                got = torch.autograd.grad(result, tuple(head.parameters()))
                want = torch.autograd.grad(expected, tuple(other.parameters()))
                for a,b in zip(got,want):
                    self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=1e-5))

    def test_abstain_protection_and_finite_backward(self):
        dataset,state,args,head=fixture(10)
        sample=int(state.ids[2]); source=int(state.labels[0][2])
        updated,_=rebuild(state,Action('abstain',sample,0,source,version=state.version))
        raw=torch.cat([dataset.features,dataset.features+.01])
        drift=target_drift(state,updated,state.ids,raw,smoothing=.75,normalization='selex_legacy')
        self.assertEqual(drift['sse_levels'][0]['protected_coefficient_shift_max'],0.)
        result=selex_loss(head,raw,state.ids,state.labeled,state.known_targets,updated,args,10)
        result['loss'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_zero_use_support_matches_clamped_source(self):
        dataset,state,args,head=fixture()
        raw=torch.zeros(2*len(dataset),4)
        result=selex_loss(head,raw,state.ids,state.labeled,state.known_targets,state,args,0)['loss']
        expected=original_objective(head,raw,state,args,0)
        self.assertTrue(torch.isfinite(result))
        self.assertTrue(torch.allclose(result,expected,atol=1e-6))

    def test_inactive_levels_are_not_silently_scored(self):
        dataset,state,args,head=fixture(10)
        with self.assertRaisesRegex(ValueError,'active levels'):
            selex_loss(head,torch.cat([dataset.features]*2),state.ids,state.labeled,
                       state.known_targets,state,args,0)

    def test_real_crish_engine_uses_selex_objective_and_no_hidden_targets(self):
        outputs=[]
        for poison in (-1,98765):
            dataset,state,args,head=fixture()
            dataset.targets[~dataset.labeled]=poison
            cfg=load_config(ROOT/'configs/selex_crish.json')
            cfg['crish']=asdict(CrishSettings(warmup_epochs=0,refresh_every=1,evidence_samples=16,
                evidence_trials=1,k=2,boundaries=1,max_targets=1,min_support=1,min_edges=1,
                train_batch=8,probe_known=2,probe_unlabeled=2,max_protected_shift=1.,
                max_refresh_seconds=20,save_partitions=False))
            with tempfile.TemporaryDirectory() as tmp:
                runtime=EpochRuntime(tmp,cfg)
                optimizer=torch.optim.SGD(head.parameters(),lr=.002,momentum=.9)
                out=CrishEngine(runtime,objective=selex_loss).prepare(state,torch.nn.Identity(),
                    head,optimizer,dataset,args,0)
                records=[json.loads(x) for x in (Path(tmp)/'crish.jsonl').read_text().splitlines()]
                self.assertEqual(records[0]['status'],'completed',records)
                self.assertGreater(len(records[0]['gains']),0)
                outputs.append((out,records))
        self.assertEqual(outputs[0][1][0]['decisions'],outputs[1][1][0]['decisions'])
        for a,b in zip(outputs[0][0].labels,outputs[1][0].labels):
            self.assertTrue(torch.equal(a,b))

    def test_bounded_training_disabled_crish_parity_and_resume(self):
        def execute(tmp, method, *, enabled, resume=None, stop=False):
            dataset,state,args,head=fixture()
            args.epochs=2
            cfg=load_config(ROOT/'configs/selex_crish_smoke.json')
            cfg.update(device='cpu',method=method,epochs=2,mlp_out_dim=16,synthetic_fixture=True,
                       max_steps_per_epoch=1,final_ss_evaluation=False)
            cfg['crish']['enabled']=enabled
            class Synthetic(EpochRuntime):
                def monitor_epoch(self,*unused): pass
                def save(self,epoch,*values):
                    super().save(epoch,*values)
                    if stop and epoch==0: raise InterruptedError('fixture stop')
            runtime=Synthetic(tmp,cfg,resume)
            model=torch.nn.Linear(4,4)
            loader=DataLoader(dataset,batch_size=16,shuffle=False)
            def extract(model,loader,args):
                runtime.hierarchy_state=state
            train_strict(head,model,loader,loader,args,runtime,objective=selex_loss,extractor=extract)
            return model.state_dict(),head.state_dict()
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,\
             tempfile.TemporaryDirectory() as c,tempfile.TemporaryDirectory() as d:
            expected=execute(a,'baseline',enabled=False)
            disabled=execute(b,'crish',enabled=False)
            for x,y in zip(expected,disabled):
                self.assertTrue(all(torch.equal(x[k],y[k]) for k in x))
            with self.assertRaises(InterruptedError): execute(c,'baseline',enabled=False,stop=True)
            resumed=execute(d,'baseline',enabled=False,resume=str(Path(c)/'resume.pt'))
            for x,y in zip(expected,resumed):
                self.assertTrue(all(torch.equal(x[k],y[k]) for k in x))

    def test_selex_cli_dry_run_does_not_invoke_hypcd_worker(self):
        result=subprocess.run(['bash','scripts/train_selex.sh','--baseline','--dry-run'],
                              cwd=ROOT,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('"baseline_family": "selex"',result.stdout)
        self.assertNotIn('hyper_max_weight',result.stdout)


if __name__=='__main__':
    unittest.main()
