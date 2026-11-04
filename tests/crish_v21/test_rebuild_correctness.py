import unittest

import torch

from audit.baseline_trace import baseline_oracle
from crish.candidates import Action
from crish.ids import rows_for_ids
from crish.rebuild import confusion_factor, prototype_radii, rebuild, pseudo_masks, target_drift

from .fixtures import state_fixture, backbone_views


class RebuildTests(unittest.TestCase):
    def test_move_updates_assignment_centers_all_affected_radii_and_source_targets(self):
        state = state_fixture()
        moved, record = rebuild(state, Action('move', 22, 0, 2, 3))
        row = int(rows_for_ids(state.ids, [22])[0])
        self.assertEqual((moved.labels[0] != state.labels[0]).nonzero().flatten().tolist(), [row])
        self.assertEqual(moved.labels[0][row], 3)
        self.assertTrue(torch.equal(state.labels[0][state.labeled], moved.known_targets))
        self.assertTrue(torch.equal(moved.labels[1], state.labels[1]))
        self.assertTrue(torch.equal(moved.prototypes[1], state.prototypes[1]))
        self.assertTrue(torch.equal(moved.prototypes[0][:2], state.prototypes[0][:2]))
        for group in (2, 3):
            torch.testing.assert_close(moved.prototypes[0][group], state.features[moved.labels[0] == group].mean(0), rtol=0, atol=0)
        torch.testing.assert_close(moved.radii[0], prototype_radii(moved.prototypes[0]), rtol=0, atol=0)
        self.assertEqual(record['source_support_after'], 1)
        self.assertEqual(record['target_support_after'], 3)
        self.assertIn('origin_prototype', record)
        self.assertFalse(torch.equal(pseudo_masks(moved, moved.ids, 0)[0], pseudo_masks(state, state.ids, 0)[0]))
        oracle = baseline_oracle()['confusion_oracle']
        for strategy in ('zero_one', 'pair_dist', 'pair_cluster'):
            actual = confusion_factor(moved, moved.ids, backbone_views(moved), strategy)
            expected = oracle(backbone_views(moved), moved.labels, moved.prototypes, moved.radii, strategy=strategy)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(confusion_factor(state, state.ids, backbone_views(state), 'pair_cluster'),
                                     confusion_factor(moved, moved.ids, backbone_views(moved), 'pair_cluster')))

    def test_empty_source_stale_and_unverified_momentum_fail(self):
        state, _ = rebuild(state_fixture(), Action('move', 22, 0, 2, 3))
        with self.assertRaises(ValueError):
            rebuild(state, Action('move', 66, 0, 2, 3, state.version))
        with self.assertRaises(ValueError):
            rebuild(state, Action('move', 80, 0, 3, 2, 0))
        with self.assertRaises(ValueError):
            rebuild(state, Action('move', 80, 0, 3, 2, state.version), cluster_momentum=.5)

    def test_broadcast_defect_is_preserved_and_distinct_from_row_normalization(self):
        state = state_fixture()
        x = backbone_views(state)
        pair = torch.cdist(x, x)
        raw = sum((pair > 2 * radius[labels.repeat(2)]).float() / 2**ell
                  for ell, (labels, radius) in enumerate(zip(state.labels, state.radii)))
        scaled = (raw - raw.min()) / (raw.max() - raw.min() + 1e-7)
        actual = confusion_factor(state, state.ids, x)
        torch.testing.assert_close(actual, scaled / scaled.sum(1), rtol=0, atol=0)
        self.assertFalse(torch.allclose(actual, scaled / scaled.sum(1, keepdim=True)))

    def test_actual_target_drift_keep_abstain_move_and_protected_mass(self):
        state = state_fixture()
        zero = target_drift(state, state, state.ids, backbone_views(state), smoothing=.5)
        self.assertEqual(zero['use_target_half_l1_mean'], 0.)
        self.assertTrue(all(x['coefficient_half_l1_mean'] == 0. for x in zero['sse_levels']))
        abstained, _ = rebuild(state, Action('abstain', 22, 1, 0))
        drift = target_drift(state, abstained, state.ids, backbone_views(state), smoothing=.5)
        self.assertEqual(drift['use_target_half_l1_mean'], 0.)
        self.assertGreater(drift['sse_levels'][1]['coefficient_half_l1_mean'], 0.)
        self.assertEqual(drift['sse_levels'][1]['protected_coefficient_shift_max'], 0.)
        moved, _ = rebuild(state, Action('move', 22, 0, 2, 3))
        moved_drift = target_drift(state, moved, state.ids, backbone_views(state), smoothing=.5, strategy='pair_cluster')
        self.assertGreater(moved_drift['use_target_half_l1_mean'], 0.)
        self.assertGreater(moved_drift['sse_levels'][0]['coefficient_half_l1_mean'], 0.)
        # A Move changes the baseline cluster-positive denominator. This is
        # measured explicitly, unlike protected-coefficient preserving Abstain.
        self.assertGreater(moved_drift['sse_levels'][0]['protected_coefficient_shift_max'], 0.)
