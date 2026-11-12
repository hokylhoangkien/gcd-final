"""Real target-drift regressions for source-peer scoring coverage."""
import unittest

import torch

from crish.candidates import Action
from crish.ids import rows_for_ids
from crish.rebuild import rebuild, target_drift
from crish.scoring_batch import scoring_ids
from crish.state import SupervisionState


def fixture():
    features = torch.tensor([[1.+i*.01, 0.] for i in range(4)] +
                            [[0., 1.+i*.01] for i in range(4)])
    labels = torch.tensor([0]*4+[1]*4)
    return SupervisionState(torch.arange(1, 9)*10, features,
        torch.tensor([True, False, False, False, True, False, False, False]),
        torch.tensor([0, 1]), (labels,),
        (torch.stack([features[:4].mean(0), features[4:].mean(0)]),),
        (torch.ones(2),), (torch.zeros(8, dtype=torch.bool),))


def drift(state, proposed, ids):
    features = state.features[rows_for_ids(state.ids, ids)].repeat(2, 1)
    return target_drift(state, proposed, ids, features, smoothing=.75,
                        normalization='selex_legacy')['sse_levels'][0]


class ScoringBatchTests(unittest.TestCase):
    def test_abstain_gets_nonprobe_source_peer_and_real_target_change(self):
        state = fixture()
        action = Action('abstain', 20, source=0)
        old_ids = torch.tensor([20, 60])
        proposed, _ = rebuild(state, action)
        self.assertEqual(drift(state, proposed, old_ids)['coefficient_half_l1_mean'], 0.)
        ids, audit = scoring_ids(state, [action], {20: [(60, 1.)]},
            torch.tensor([1, 5]), torch.tensor([0, 4]), 6)
        self.assertFalse(set(ids.tolist()) & {10, 50})
        self.assertGreater(audit['action_support'][0]['source_peers'], 0)
        self.assertTrue(audit['added_support_ids'])
        measured = drift(state, proposed, ids)
        self.assertGreater(measured['coefficient_half_l1_mean'], 0.)
        self.assertEqual(measured['protected_coefficient_shift_max'], 0.)

    def test_move_has_source_destination_support_without_relaxing_cap(self):
        state = fixture()
        action = Action('move', 20, source=0, target=1)
        proposed, _ = rebuild(state, action, min_support=2)
        ids, audit = scoring_ids(state, [action], {20: [(60, 1.)]},
            torch.tensor([1, 5]), torch.tensor([0, 4]), 6)
        self.assertEqual(audit['action_support'][0]['source_peers'], 2)
        self.assertGreaterEqual(audit['action_support'][0]['target_peers'], 2)
        measured = drift(state, proposed, ids)
        self.assertGreater(measured['coefficient_half_l1_mean'], 0.)
        self.assertLessEqual(measured['protected_coefficient_shift_max'], .25)
        self.assertLessEqual(len(ids), 6)
        self.assertEqual(len(ids), len(ids.unique()))

    def test_missing_nonprobe_peer_stays_unscorable_without_using_probe(self):
        state = fixture()
        action = Action('abstain', 20, source=0)
        ids, audit = scoring_ids(state, [action], {}, torch.tensor([1, 5]),
            torch.tensor([0, 2, 3, 4]), 4)
        self.assertEqual(audit['action_support'][0]['source_peers'], 0)
        proposed, _ = rebuild(state, action)
        self.assertEqual(drift(state, proposed, ids)['coefficient_half_l1_mean'], 0.)

    def test_budget_determinism_and_probe_guard(self):
        state = fixture()
        action = Action('abstain', 20, source=0)
        args = (state, [action], {}, torch.tensor([1, 5]), torch.tensor([0, 4]), 2)
        first, _ = scoring_ids(*args)
        second, _ = scoring_ids(*args)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(first), 2)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            scoring_ids(state, [action], {}, torch.tensor([1, 5]), torch.tensor([1]), 2)
