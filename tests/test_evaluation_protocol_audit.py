"""Independent metric, upstream-ordering and report-provenance regressions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gcd_runtime.checkpoint import EpochRuntime
from tests.test_crish_training import configure_trainer, training_config


class EvaluationProtocolAuditTests(unittest.TestCase):
    def test_global_matching_cannot_use_two_independent_perfect_assignments(self):
        configure_trainer()
        from project_utils.cluster_and_log_utils import split_cluster_acc_v1, split_cluster_acc_v2
        truth = np.array([0, 0, 1, 1, 1, 1])
        predictions = np.zeros(6, dtype=int)
        old = truth == 0
        self.assertEqual(split_cluster_acc_v1(truth, predictions, old), (1., 1., 1.))
        actual = split_cluster_acc_v2(truth, predictions, old)
        np.testing.assert_allclose(actual, (4/6, 0., 1.))
        self.assertAlmostEqual(actual[0], (2*actual[1]+4*actual[2])/6)
        order = np.array([5, 0, 3, 1, 4, 2])
        np.testing.assert_allclose(split_cluster_acc_v2(truth[order], predictions[order], old[order]), actual)

    def test_coarse_hssk_targets_follow_ids_when_predicted_membership_interleaves(self):
        # Isolate row mapping, not full k-means permutation invariance. The
        # upstream issue #11 arises when unlabeled rows mix predicted old/new.
        trainer = configure_trainer()
        images = torch.eye(8)
        base_predictions = torch.tensor([0, 1, 0, 1, 2, 0, 3, 1])
        ids = torch.tensor([110, 7, 99, 15, 42, 66, 80, 12])
        membership = torch.tensor([True]*4+[False]*4)
        legal = torch.tensor([0, 1, 0, 1, -1, -1, -1, -1])
        expected_coarse = (base_predictions >= 2).long()

        class Cluster:
            def __init__(self, k, **unused):
                self.k = k
            def fit_mix(self, unlabeled, labeled, targets):
                # Same grouping contract as the real upstream fit_mix.
                features = torch.cat([labeled, unlabeled])
                rows = features.argmax(1)
                self.labels_ = (base_predictions if self.k == 4 else expected_coarse)[rows]
                self.cluster_centers_ = torch.stack([features[self.labels_ == g].mean(0)
                                                     for g in range(self.k)])

        class Coarse:
            def __init__(self, n_clusters, **unused):
                pass
            def fit(self, features):
                self.labels_ = np.zeros(len(features), dtype=int)
                return self

        args = SimpleNamespace(train_classes=[0, 1], num_labeled_classes=2,
                               num_unlabeled_classes=2, unbalanced=False, eval_funcs=['v2'])
        with tempfile.TemporaryDirectory() as tmp:
            runtime = EpochRuntime(tmp, training_config('baseline'))
            with patch.object(trainer, '_runtime', runtime, create=True), \
                 patch.object(trainer, 'device', torch.device('cpu'), create=True), \
                 patch.object(trainer, 'SemiSupKMeans', Cluster), \
                 patch.object(trainer, 'KMeans', Coarse), \
                 patch.object(trainer, 'log_accs_from_preds', side_effect=AssertionError('Oracle in training')):
                for order in (torch.arange(8), torch.tensor([7, 3, 5, 1, 6, 0, 4, 2])):
                    trainer.extract_labeled_protos(torch.nn.Identity(),
                        [([images[order]], legal[order], ids[order], membership[order, None])], args)
                    state = runtime.hierarchy_state
                    self.assertEqual(state.ids.tolist(), ids[order].tolist())
                    torch.testing.assert_close(state.labels[0], base_predictions[order])
                    torch.testing.assert_close(state.labels[1], expected_coarse[order])

    def test_last_checkpoint_provenance_is_distinct_from_run_selector(self):
        for selection, role, selected, policy in (
            ('best_old_test_v2', 'final', False, True),
            ('best_old_test_v2', 'best_old_test', True, True),
            ('best_old_test_v2', 'monitor_only', False, True),
            ('final_epoch', 'final', False, False),
        ):
            with self.subTest(selection=selection, role=role), tempfile.TemporaryDirectory() as tmp:
                cfg = training_config('baseline')
                cfg.update(checkpoint_selection=selection, test_every_epochs=int(policy))
                runtime = EpochRuntime(tmp, cfg, evaluation_epoch=0 if role == 'monitor_only' else None)
                runtime.evaluation_role = role
                runtime.evaluation(dict(all_acc=.75, old_acc=.5, new_acc=1.))
                path = (Path(tmp)/'evaluations/epoch_0001.json' if role == 'monitor_only'
                        else Path(tmp)/('evaluation_best_old_test.json' if selected else 'evaluation_last.json'))
                report = json.loads(path.read_text())
                self.assertEqual(report['selected_using_evaluation_labels'], selected)
                self.assertEqual(report['checkpoint_selection_uses_evaluation_labels'], policy)
                self.assertTrue(report['matching_uses_evaluation_labels'])
