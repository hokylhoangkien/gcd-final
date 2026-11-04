from dataclasses import replace
import unittest

import torch

from crish.candidates import Action, propose_actions
from crish.rebuild import rebuild
from crish.reliability import EdgeEvidence

from .fixtures import state_fixture


class ProtectionTests(unittest.TestCase):
    def test_coarse_known_merge_valid_and_labeled_edits_rejected(self):
        state = state_fixture()
        self.assertEqual(state.labels[1][:2].tolist(), [0, 0])
        for level, source in [(0, 0), (1, 0)]:
            for kind in ('abstain', 'move'):
                with self.assertRaises(ValueError):
                    rebuild(state, Action(kind, 40, level, source, 1))
        with self.assertRaises(ValueError):
            replace(state, known_targets=torch.tensor([1, 0]))
        with self.assertRaises(ValueError):
            replace(state, known_targets=torch.arange(8))

    def test_moves_need_positive_measured_support_and_bounded_candidates(self):
        state = state_fixture()
        evidence = {(22, 80): EdgeEvidence(1., 1., 1., 2), (12, 22): EdgeEvidence(1., 1., 1., 2)}
        actions = propose_actions(state, 22, 0, evidence, max_targets=1,
                                  min_support=1, min_edges=2, min_reliability=.5)
        self.assertEqual([a.kind for a in actions], ['keep', 'abstain', 'move'])
        self.assertEqual(actions[-1].target, 3)
        no_support = propose_actions(state, 22, 0, {}, max_targets=3,
                                     min_support=1, min_edges=1, min_reliability=.5)
        self.assertEqual([a.kind for a in no_support], ['keep', 'abstain'])
        anchors = propose_actions(state, 40, 0, evidence, max_targets=3,
                                  min_support=1, min_edges=1, min_reliability=.5)
        self.assertEqual([a.kind for a in anchors], ['keep'])
