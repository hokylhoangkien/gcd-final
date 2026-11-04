import unittest

import torch

from audit.baseline_trace import baseline_oracle
from crish.candidates import Action
from crish.rebuild import confusion_factor, pseudo_masks, pseudo_sse_loss, rebuild

from .fixtures import state_fixture, backbone_views


class AbstainTests(unittest.TestCase):
    def test_abstain_preserves_instance_labeled_pairs_and_use(self):
        state = state_fixture()
        changed, _ = rebuild(state, Action('abstain', 22, 1, 0))
        original, kept = pseudo_masks(changed, changed.ids, 1)
        self.assertTrue(kept.diagonal().all())
        self.assertTrue(torch.equal(kept[:2, :2], original[:2, :2]))
        self.assertFalse(torch.equal(original, kept))
        for a, b in zip(state.labels + state.prototypes + state.radii, changed.labels + changed.prototypes + changed.radii):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(confusion_factor(state, state.ids, backbone_views(state)),
                                    confusion_factor(changed, state.ids, backbone_views(state))))

    def test_loss_uses_original_positive_denominator_with_finite_backward(self):
        state, _ = rebuild(state_fixture(), Action('abstain', 22, 1, 0))
        criterion = baseline_oracle()['SupConLoss'](hyper_c=0)
        x = (torch.randn(8, 2, 5, generator=torch.Generator().manual_seed(3)) * .01).requires_grad_()
        original, kept = pseudo_masks(state, state.ids, 1)
        flat = torch.cat(torch.unbind(x, dim=1))
        logits = -torch.cdist(flat, flat) / criterion.temperature
        logits = logits - logits.max(1, keepdim=True)[0].detach()
        offdiag = ~torch.eye(16, dtype=torch.bool)
        lp = logits - (logits.exp() * offdiag).sum(1, keepdim=True).log()
        denom = (original.repeat(2, 2) & offdiag).sum(1)
        retained = kept.repeat(2, 2) & offdiag
        expected = -(criterion.temperature / criterion.base_temperature) * ((retained * lp).sum(1) / denom).mean()
        actual = pseudo_sse_loss(criterion, x, state, state.ids, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # A surviving-mask SupCon would renormalize and change protected weights.
        self.assertFalse(torch.isclose(actual, criterion(x, mask=kept)))
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        torch.testing.assert_close(retained.float().diagonal(offset=8) / denom[:8], 1. / denom[:8])

    def test_repeated_id_instances_keep_all_identity_positives(self):
        state, _ = rebuild(state_fixture(), Action('abstain', 22, 1, 0))
        original, kept = pseudo_masks(state, torch.tensor([22, 22, 40]), 1)
        self.assertTrue(kept[:2, :2].all())
        self.assertFalse(kept[0, 2])
        with self.assertRaisesRegex(ValueError, 'Already abstained'):
            rebuild(state, Action('abstain', 22, 1, 0, version=state.version))
