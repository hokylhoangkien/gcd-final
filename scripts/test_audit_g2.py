"""Regression tests for G2's sealed shortlist AUROC."""

import unittest

import numpy as np

from scripts import audit_g1 as g1
from scripts.audit_g2 import SHORT, short_pack, shortlist_eligible


class ShortlistDetectionTest(unittest.TestCase):
    def test_current_slot_only_is_excluded_from_detection_auc(self):
        q, bank = np.arange(3), np.arange(3, 5)
        ids = np.arange(5)
        slots = np.array([0, 0, 0, 0, 1])
        truth = np.array([1, 0, 1, 0, 1])
        peers = np.tile([[3, 3], [3, 4], [3, 4]], (1, 32))
        score = np.tile([[0.8, 0.8], [0.9, 0.1], [0.2, 0.8]], (1, 32))
        packed = short_pack(score, q, peers, slots)
        self.assertTrue(np.isneginf(packed["wrong"][0]))
        np.testing.assert_array_equal(packed["dest"][0], -np.ones(5))
        frozen = {"pairs": np.tile(peers, (1, 2))}
        for method in SHORT:
            frozen.update({f"{method}_{k}": v.copy() for k, v in packed.items()})
        eligible = shortlist_eligible(frozen, q, slots)
        np.testing.assert_array_equal(eligible, [False, True, True])
        result, _ = g1.evaluate(
            frozen,
            ids,
            q,
            bank,
            slots,
            truth,
            methods=SHORT,
            baseline=SHORT[0],
            uniform_count=64,
            detection_eligible=eligible,
        )
        self.assertEqual(result["CLS@64"]["All"]["detection_auc"], 1.0)
        self.assertEqual(result["CLS@64"]["All"]["destination_recall"][1], 0.5)
        frozen["Local@64_wrong"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "slot availability"):
            shortlist_eligible(frozen, q, slots)
