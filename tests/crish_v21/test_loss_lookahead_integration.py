"""A bounded full loss -> train gradients -> legal cloned-step probe fixture."""
import copy
import math
import unittest

import torch

from audit.baseline_trace import baseline_oracle
from crish.candidates import Action
from crish.lookahead import exact_sgd_comparison, gradient_gain
from crish.rebuild import confusion_factor, pseudo_sse_loss, rebuild, target_drift

from .fixtures import backbone_views, probe_fixture, state_fixture


class LossLookaheadIntegrationTests(unittest.TestCase):
    def test_original_use_sse_rebuild_gradients_feed_an_isolated_exact_step(self):
        source = baseline_oracle()
        from hyptorch.nn import ToPoincare
        with torch.random.fork_rng():
            torch.manual_seed(12)
            model = torch.nn.Sequential(torch.nn.Linear(3, 16), ToPoincare(c=.1, clip_r=2., riemannian=False))
        probe = probe_fixture(model)
        state = state_fixture()
        trial, _ = rebuild(state, Action('move', 22, 0, 2, 3))
        inputs = backbone_views(state)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=.9, weight_decay=5e-5)
        before = copy.deepcopy(model.state_dict())

        def loss(snapshot):
            flat = model(inputs)
            paired = torch.stack(flat.chunk(2), dim=1)
            cf = confusion_factor(snapshot, state.ids, inputs)
            branches, protected = [], []
            for curvature in (0., .1):
                criterion = source['SupConLoss'](hyper_c=curvature)
                logits, labels, similarity = source['info_nce_logits'](flat, cf, source['args'], hyper_c=curvature)
                use = source['LabelSmoothingLoss']()(logits, labels, similarity, smoothing=.5)
                supervised = criterion(paired[state.labeled], labels=state.known_targets)
                protected.append(supervised.detach())
                pseudo = sum(pseudo_sse_loss(criterion, paired[:, :, :16 // 2**(ell+1)], snapshot, state.ids, ell) / 2**(ell+1)
                             for ell in range(len(state.labels)))
                branches.append(.65 * use + .35 * (supervised + pseudo) / 2)
            return .5 * (branches[0] + branches[1]), protected

        keep_loss, protected_keep = loss(state)
        move_loss, protected_move = loss(trial)
        parameters = tuple(model.parameters())
        g0 = torch.autograd.grad(keep_loss, parameters)
        ga = torch.autograd.grad(move_loss, parameters)
        q = torch.autograd.grad(probe.loss(model), parameters)
        mass = target_drift(state, trial, state.ids, inputs, smoothing=.5)['sse_levels'][0]['coefficient_half_l1_mean']
        self.assertGreater(mass, 0.)
        proxy = gradient_gain(q, ga, g0, changed_mass=mass)
        exact = exact_sgd_comparison(model, optimizer, g0, ga, probe)
        self.assertTrue(math.isfinite(proxy['raw']) and math.isfinite(exact['gain']))
        for a, b in zip(protected_keep, protected_move):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(all(p.grad is None for p in parameters))
        self.assertEqual(len(optimizer.state), 0)
