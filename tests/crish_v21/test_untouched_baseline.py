import unittest

import torch

from audit.baseline_trace import baseline_oracle
from crish.candidates import Action
from crish.policy import sequential_repair
from crish.rebuild import confusion_factor, pseudo_sse_loss, rebuild

from .fixtures import state_fixture, backbone_views


class BaselineParityTests(unittest.TestCase):
    def test_keep_targets_logits_loss_and_gradients_match_actual_source(self):
        state = state_fixture()
        kept, _ = rebuild(state, Action('keep'))
        self.assertIs(kept, state)
        original = baseline_oracle()
        generator = torch.Generator().manual_seed(8)
        features = (torch.randn(8, 2, 16, generator=generator) * .03).requires_grad_()
        for curvature in (0., .1):
            criterion = original['SupConLoss'](hyper_c=curvature)
            for ell in range(2):
                actual = pseudo_sse_loss(criterion, features, kept, kept.ids, ell)
                expected = criterion(features, labels=state.labels[ell])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(torch.autograd.grad(actual, features)[0],
                                           torch.autograd.grad(expected, features)[0], rtol=0, atol=0)
            cf = confusion_factor(kept, kept.ids, backbone_views(kept))
            cf_source = original['confusion_oracle'](backbone_views(state), state.labels, state.prototypes, state.radii)
            torch.testing.assert_close(cf, cf_source, rtol=0, atol=0)
            flat = torch.cat(torch.unbind(features, dim=1))
            logits, labels, similarity = original['info_nce_logits'](flat, cf, original['args'], hyper_c=curvature)
            source_logits, source_labels, source_similarity = original['info_nce_logits'](flat, cf_source, original['args'], hyper_c=curvature)
            for a, b in zip((logits, labels, similarity), (source_logits, source_labels, source_similarity)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            loss = original['LabelSmoothingLoss']()(logits, labels, similarity, smoothing=.5)
            source_loss = original['LabelSmoothingLoss']()(source_logits, source_labels, source_similarity, smoothing=.5)
            torch.testing.assert_close(loss, source_loss, rtol=0, atol=0)
            torch.testing.assert_close(torch.autograd.grad(loss, features)[0], torch.autograd.grad(source_loss, features)[0], rtol=0, atol=0)

    def test_disabled_policy_does_not_call_scorer_or_consume_rng(self):
        state = state_fixture()
        before = torch.get_rng_state().clone()
        def forbidden(*args):
            raise AssertionError('Disabled CRiSH invoked scorer')
        kept, trace = sequential_repair(state, [Action('move', 22, 0, 2, 3)], forbidden,
                                        max_accepted=3, threshold=0., agreement=1.,
                                        known_tolerance=0., enabled=False)
        self.assertIs(kept, state)
        self.assertEqual(trace, [])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
