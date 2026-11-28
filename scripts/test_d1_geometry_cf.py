import unittest

from scripts.audit_d1_geometry_cf import pick


class GeometryDecisionChecks(unittest.TestCase):
    def test_discovery_winner_must_pass_both_independent_guards(self):
        runner = {"gain": [0.1, 0.2], "ce_delta": [0.0, 0.0]}
        for second, guard in ((1e-7, 0.0), (0.2, 1e-10), (float("nan"), 0.0)):
            self.assertIsNone(
                pick([runner, {"gain": [0.3, second], "ce_delta": [0.0, guard]}])
            )
        self.assertEqual(pick([runner]), 0)
        self.assertIsNone(pick([]))
