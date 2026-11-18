"""Protect paper plots against protocol mixing, label selection and fake gaps."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.plot_observed_run import extract_refreshes, gap_series, ss_history, validate_ss


def ss(epoch=6):
    return dict(status="observed", unit="fraction", protocol="balanced_ss_kmeans_train_unlabeled",
                matching="hungarian_v2_global", epoch=epoch, seed=1, purpose="monitor_only",
                n_unlabeled=10, n_old=2, n_new=8, old_acc=1., new_acc=.5, all_acc=.6)


class ObservedFigureTests(unittest.TestCase):
    def test_all_must_be_sample_weighted_not_mean_of_old_new(self):
        row = ss()
        validate_ss(row)
        row["all_acc"] = .75
        with self.assertRaisesRegex(ValueError, "sample-weighted"):
            validate_ss(row)

    def test_missing_epochs_remain_nan_and_use_zero_based_indices(self):
        values = gap_series([ss(6), ss(8)], "all_acc", 10, 100)
        self.assertTrue(all(math.isnan(v) for v in values[:6]))
        self.assertEqual(values[6], 60)
        self.assertTrue(math.isnan(values[7]))
        self.assertEqual(values[8], 60)

    def test_plain_or_selected_metrics_cannot_enter_ss_curve(self):
        for changes in ({"protocol": "plain_kmeans_disjoint_test"},
                        {"purpose": "best_ss_all"}, {"status": "synthetic_fixture"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ss_history([ss(), dict(ss(7), **changes)])

    def test_conflicting_or_mismatched_observations_are_rejected(self):
        row = dict(ss(), old_acc=.5, new_acc=.5, all_acc=.5)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            ss_history([ss(), row])
        with self.assertRaisesRegex(ValueError, "conditions changed"):
            ss_history([ss(), dict(ss(7), seed=2)])

    def test_action_matching_includes_target_and_dose_is_net_not_summed(self):
        decision = dict(version=1, sample_id=12, level=0, kind="move", source=2,
                        target=3, status="accepted", changed_global_pseudo_pairs=900)
        gain = {k: v for k, v in decision.items() if k != "status"}
        gain.update(exact_discovery={"gain": .01}, confirmation={"gain": .02})
        wrong_target = dict(gain, target=4, exact_discovery={"gain": .99})
        dose = dict(changed_ids=1, changed_labeled_ids=0,
                    levels=[dict(level=0, changed_directed_image_positive_pairs=80,
                                 unweighted_target_mass_before=1., unweighted_target_mass_after=1.)])
        record = dict(epoch=10, status="completed", accepted=1, applied_actions=1,
                      decisions=[decision], gains=[gain, wrong_target], dose=dose, seconds=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crish.jsonl"
            path.write_text(json.dumps(record) + "\n")
            refreshes, actions = extract_refreshes(path, {10: copy.deepcopy(dose)})
            self.assertEqual(refreshes[0]["level0_changed_directed_pairs"], 80)
            self.assertEqual(actions[0]["discovery_gain"], .01)
            self.assertEqual(actions[0]["epoch_number"], 11)
            record["applied_actions"] = 0
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "action mismatch"):
                extract_refreshes(path, {10: dose})


if __name__ == "__main__":
    unittest.main()
