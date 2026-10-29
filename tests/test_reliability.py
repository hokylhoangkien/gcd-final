import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.worker import configure_environment
from gcd_runtime.method_config import ReliabilitySettings
from gcd_runtime.reliability import neighborhood_scores, pair_weights, weighted_supcon
from gcd_runtime.checkpoint import EpochRuntime


def scope_and_features():
    upstream = configure_environment(ROOT / 'runs/probe-placeholder')
    from gcd_runtime import trainer
    scope = vars(trainer)
    scope['args'] = SimpleNamespace(c=0.1)
    from hyptorch.nn import ToPoincare
    torch.manual_seed(31)
    features = ToPoincare(c=0.1, clip_r=2.0, riemannian=False)(torch.randn(6, 2, 8) * 0.2).detach()
    return scope, features


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope, cls.features = scope_and_features()

    def test_schedule(self):
        settings = ReliabilitySettings(warmup_epochs=2, ramp_epochs=4)
        self.assertEqual([settings.strength(e) for e in range(7)], [0, 0, .25, .5, .75, 1, 1])
        self.assertEqual(replace(settings, ramp_epochs=0).strength(2), 1)
        for kwargs in [dict(k=-1), dict(floor=1.1), dict(floor=float('nan')),
                       dict(reciprocal='False'), dict(pair_scope='novel_only')]:
            with self.assertRaises(ValueError):
                ReliabilitySettings(**kwargs)

    def test_score_bounds_symmetry_detachment_and_consistent_views(self):
        f = self.features.clone().requires_grad_()
        score, details = neighborhood_scores(f, torch.arange(6), .1, ReliabilitySettings(k=2))
        self.assertFalse(score.requires_grad)
        self.assertTrue(torch.equal(score, score.T))
        self.assertTrue(((score >= 0) & (score <= 1)).all())
        identical = self.features[:, :1].repeat(1, 2, 1)
        _, details = neighborhood_scores(identical, torch.arange(6), .1, ReliabilitySettings(k=2))
        self.assertTrue(torch.equal(details['agreement'], torch.ones(6)))

    def test_permutation_and_duplicate_id_exclusion(self):
        settings = ReliabilitySettings(k=2)
        ids = torch.arange(6)
        score, _ = neighborhood_scores(self.features, ids, .1, settings)
        order = torch.tensor([3, 1, 5, 0, 4, 2])
        permuted, _ = neighborhood_scores(self.features[order], ids[order], .1, settings)
        self.assertTrue(torch.equal(permuted, score[order][:, order]))
        duplicate = torch.cat([self.features, self.features[:1]])
        duplicated, details = neighborhood_scores(duplicate, torch.tensor([0,1,2,3,4,5,0]), .1, settings)
        self.assertTrue(torch.equal(duplicated[:6,:6], score))
        self.assertEqual(float(duplicated[0,6]), 1)
        self.assertEqual(int(details['unique_samples']), 6)
        self.assertEqual(float(details['reciprocal'][0,6]), 0)

    def test_invalid_geometry_and_nonfinite(self):
        for value in (float('nan'), float('inf'), 10.0):
            f = self.features.clone()
            f[0,0,0] = value
            with self.assertRaises(ValueError):
                neighborhood_scores(f, torch.arange(6), .1, ReliabilitySettings())

    def test_membership_mask_and_protected_pairs(self):
        score = torch.tensor([[1.,1.,.2,0.],[1.,1.,.5,.1],[.2,.5,1.,1.],[0.,.1,1.,1.]])
        pseudo = torch.zeros(4, dtype=torch.long)
        labeled = torch.tensor([True, True, False, False])
        ids = torch.arange(4)
        settings = ReliabilitySettings(floor=.1, warmup_epochs=0, ramp_epochs=0)
        weights, eligible, _ = pair_weights(score, pseudo, labeled, ids, settings, 0)
        self.assertTrue(torch.equal(weights[:2,:2], torch.ones(2,2)))
        self.assertTrue(torch.equal(weights.diag(), torch.ones(4)))
        self.assertTrue(torch.equal(weights, weights.T))
        self.assertLess(float(weights[0,2]), 1)
        only_unlabeled, eligible, _ = pair_weights(score, pseudo, labeled, ids,
                                                   replace(settings,pair_scope='both_unlabeled'),0)
        self.assertTrue(torch.equal(only_unlabeled[:2], torch.ones(2,4)))
        self.assertFalse(eligible[:2].any())
        # Different pseudo labels must never acquire weighted positive membership.
        _, eligible, _ = pair_weights(score, torch.tensor([0,1,0,1]), labeled, ids, settings, 0)
        self.assertFalse(bool(eligible[0,3]))

    def test_zero_neighbors_and_single_unique_sample_fallback(self):
        for k, ids in [(0,torch.arange(6)), (3,torch.zeros(6,dtype=torch.long))]:
            settings = ReliabilitySettings(k=k,floor=0,warmup_epochs=0,ramp_epochs=0)
            score, _ = neighborhood_scores(self.features, ids, .1, settings)
            weights, _, _ = pair_weights(score, torch.zeros(6,dtype=torch.long),
                                         torch.zeros(6,dtype=torch.bool), ids, settings, 0)
            self.assertTrue(torch.equal(weights,torch.ones(6,6)))

    def test_exact_baseline_value_and_gradient_parity(self):
        labels = torch.tensor([0,0,1,1,2,2])
        for curvature in (0, .1):
            criterion = self.scope['SupConLoss'](hyper_c=curvature)
            a = self.features.clone().requires_grad_()
            b = self.features.clone().requires_grad_()
            original = criterion(a, labels=labels)
            weighted = weighted_supcon(b, labels, torch.ones(6,6), criterion, .1)
            original.backward()
            weighted.backward()
            self.assertTrue(torch.equal(original,weighted))
            self.assertTrue(torch.equal(a.grad,b.grad))

    def test_weighted_loss_independent_oracle_and_gradients(self):
        f = self.features[:3].clone().requires_grad_()
        labels = torch.zeros(3,dtype=torch.long)
        weights = torch.tensor([[1.,.2,.7],[.2,1.,.1],[.7,.1,1.]],requires_grad=True)
        criterion = self.scope['SupConLoss'](hyper_c=0)
        loss = weighted_supcon(f,labels,weights,criterion,.1)
        flat = torch.cat([f[:,0],f[:,1]])
        logits = -torch.cdist(flat,flat) / criterion.temperature
        eye = torch.eye(6,dtype=torch.bool)
        oracle_log = torch.log_softmax(logits.masked_fill(eye,-torch.inf),dim=1)
        mass = weights.detach().repeat(2,2).masked_fill(eye,0)
        oracle = -(mass * oracle_log.masked_fill(eye,0)).sum(1).div(mass.sum(1)).mean()
        self.assertTrue(torch.allclose(loss,oracle,atol=1e-6))
        loss.backward()
        self.assertTrue(torch.isfinite(f.grad).all())
        self.assertGreater(float(f.grad.abs().sum()),0)
        self.assertIsNone(weights.grad)

    def test_finite_near_boundary(self):
        f = torch.randn(6,2,8)
        f = (f / f.norm(dim=-1,keepdim=True) * (0.1**-.5 * .99)).requires_grad_()
        loss = weighted_supcon(f,torch.zeros(6,dtype=torch.long),
                              torch.eye(6)*.5+.5,self.scope['SupConLoss'](hyper_c=.1),.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(f.grad).all())

    def test_runtime_switches_warmup_and_supervision_preserved(self):
        cfg = load_config(ROOT/'configs/cars_proposed_dino_v1.json')
        labels = torch.tensor([0,0,1,1,2,2])
        distance = self.scope['SupConLoss'](hyper_c=.1)
        angle = self.scope['SupConLoss'](hyper_c=0)
        baseline = (distance(self.features, labels=labels),angle(self.features,labels=labels))
        with tempfile.TemporaryDirectory() as tmp:
            runtime=EpochRuntime(tmp,cfg)
            output=runtime.hierarchy_losses(distance,angle,self.features,labels,
                                            torch.tensor([True,False]*3),torch.arange(6),0,0)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(output,baseline)))
            runtime.config=copy.deepcopy(cfg)
            runtime.config['reliability'].update(reciprocal=False,cross_view=False,warmup_epochs=0)
            output=runtime.hierarchy_losses(distance,angle,self.features,labels,
                                            torch.tensor([True,False]*3),torch.arange(6),100,0)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(output,baseline)))
            supervised=runtime.supervised_losses(distance,angle,self.features,labels)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(supervised,baseline)))
            zero=runtime.supervised_losses(distance,angle,self.features[:0],labels[:0])
            self.assertEqual(float(zero[0]),0)
            regular=torch.tensor([[0.,1.,2.],[3.,0.,4.],[5.,6.,0.]])
            self.assertTrue(torch.equal(runtime.normalize_confusion(regular),regular/regular.sum(1)))
            confusion=runtime.normalize_confusion(torch.zeros(4,4))
            self.assertTrue(torch.isfinite(confusion).all())
            self.assertTrue(torch.allclose(confusion.sum(1),torch.ones(4)))

    def test_active_method_preflight_on_cpu(self):
        from gcd_runtime.probe import method_probe
        cfg=load_config(ROOT/'configs/cars_proposed_dino_v1.json')
        report=method_probe(cfg,'cpu')
        self.assertEqual(report['trained_epochs'],0)
        self.assertEqual(report['status'],'synthetic_method_passed')
        self.assertGreater(report['diagnostics']['downweighted_fraction'],0)

    def test_all_profiles_load(self):
        for name in ('baseline','neighborhood','crossview','proposed'):
            cfg=load_config(ROOT/f'configs/cars_{name}_dino_v1.json')
            self.assertEqual(cfg['seed'],1)


if __name__=='__main__':
    unittest.main()
