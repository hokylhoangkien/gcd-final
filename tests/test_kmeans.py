"""Real HSSK edge cases from the supplied epoch-3 prototype failure."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from gcd_runtime.config import ROOT
from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.worker import configure_environment

configure_environment(ROOT / 'runs/probe-placeholder')
from gcd_runtime.kmeans import SemiSupKMeans, SourceKMeans
from crish.rebuild import prototype_radii


def cluster(k=4, **kwargs):
    return SemiSupKMeans(k=k, max_iterations=4, n_init=1, random_state=7,
                        pairwise_batch_size=32, **kwargs)


class KMeansTests(unittest.TestCase):
    def test_original_empty_cluster_produces_the_reported_radius_error(self):
        # Both novel centers start on the same point. The source computes a
        # mean over the empty cluster, creating NaNs before radii are checked.
        known = torch.tensor([[0., 0.], [1., 0.]])
        other = torch.tensor([[2., 0.], [2., 0.], [3., 0.], [3., 0.]])
        initial = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [2., 0.]])
        targets = torch.tensor([0, 1])
        source = SourceKMeans(k=4, max_iterations=2, n_init=1)
        real_device = torch.device
        with patch.object(source, 'kpp', return_value=initial.clone()), \
             patch('torch.device', side_effect=lambda value: real_device('cpu' if value == 'cuda' else value)):
            source.fit_mix(other, known, targets)
        self.assertFalse(torch.isfinite(source.cluster_centers_).all())
        with self.assertRaisesRegex(ValueError, 'Invalid prototypes for radii'):
            prototype_radii(source.cluster_centers_)
        fixed = cluster()
        with patch.object(fixed, 'kpp', return_value=initial.clone()):
            fixed.fit_mix(other, known, targets)
        self.assertGreater(fixed.empty_cluster_repairs_, 0)
        self.assertTrue(torch.isfinite(prototype_radii(fixed.cluster_centers_)).all())
        self.assertTrue((torch.bincount(fixed.labels_, minlength=4) > 0).all())
        self.assertTrue(torch.equal(fixed.labels_[:2], targets))

    def test_nondegenerate_results_match_unmodified_source_exactly(self):
        generator = torch.Generator().manual_seed(47)
        centers = torch.eye(4) * 5
        known = torch.cat([centers[i] + .01 * torch.randn(3, 4, generator=generator) for i in range(2)])
        other = torch.cat([centers[i] + .01 * torch.randn(8, 4, generator=generator) for i in range(4)])
        targets = torch.tensor([0] * 3 + [1] * 3)
        real_device = torch.device
        for capacity in (None, 10):
            with self.subTest(capacity=capacity):
                settings = dict(k=4, max_iterations=4, n_init=1, random_state=7,
                                pairwise_batch_size=32, cluster_size=capacity)
                source, fixed = SourceKMeans(**settings), SemiSupKMeans(**settings)
                # Only adapt the source's hardcoded device for this CPU oracle.
                # Exercise all balancing iterations; source's unused iteration
                # counter is unbound if its balanced phase exits at iteration 0.
                source.tolerance = fixed.tolerance = 0
                with patch('torch.device', side_effect=lambda value: real_device('cpu' if value == 'cuda' else value)):
                    source.fit_mix(other, known, targets)
                fixed.fit_mix(other, known, targets)
                self.assertEqual(fixed.empty_cluster_repairs_, 0)
                self.assertEqual(fixed.degenerate_initializations_, 0)
                self.assertTrue(torch.equal(source.labels_, fixed.labels_))
                torch.testing.assert_close(source.cluster_centers_, fixed.cluster_centers_, rtol=0, atol=0)
                torch.testing.assert_close(source.inertia_, fixed.inertia_, rtol=0, atol=0)

    def test_collapsed_identical_features_terminate_with_all_clusters_occupied(self):
        for capacity in (None, 2):
            with self.subTest(capacity=capacity):
                model = cluster(cluster_size=capacity)
                model.fit_mix(torch.ones(6, 3), torch.ones(2, 3), torch.tensor([0, 1]))
                self.assertGreater(model.empty_cluster_repairs_, 0)
                self.assertGreater(model.degenerate_initializations_, 0)
                self.assertTrue((torch.bincount(model.labels_, minlength=4) > 0).all())
                self.assertTrue(torch.isfinite(model.cluster_centers_).all())
                self.assertLessEqual(model.n_iter_, 8)

    def test_repairs_preserve_anchors_and_singleton_donors(self):
        model = cluster()
        model.empty_cluster_repairs_ = 0
        features = torch.tensor([[0.], [1.], [2.], [10.], [20.]])
        labels = torch.tensor([0, 1, 2, 0, 0])
        centers = torch.tensor([[0.], [1.], [2.], [30.]])
        model._repair_empty(features, labels, 2, centers)
        self.assertEqual(labels.tolist(), [0, 1, 2, 0, 3])
        self.assertEqual(model.empty_cluster_repairs_, 1)

    def test_invalid_inputs_fail_instead_of_sanitizing_nans_or_moving_anchors(self):
        known, other, targets = torch.eye(2), torch.eye(2), torch.tensor([0, 1])
        for k, u, l, y in (
            (5, other, known, targets),
            (4, other * torch.nan, known, targets),
            (4, other, known, torch.tensor([0., .5])),
            (4, other, known, torch.tensor([0, 2])),
            (4, other[:0], known, targets),
        ):
            with self.subTest(k=k, y=y), self.assertRaises(ValueError):
                cluster(k=k).fit_mix(u, l, y)
        model = cluster()
        model.empty_cluster_repairs_ = 0
        with self.assertRaisesRegex(ValueError, 'without moving anchors'):
            model._repair_empty(torch.eye(4), torch.tensor([0, 0, 1, 2]), 4, torch.eye(4))

    def test_radius_errors_include_hierarchy_level_and_cluster_id(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = EpochRuntime(directory, {})
            with self.assertRaisesRegex(ValueError, r'level=0: .*nonfinite cluster IDs=\[1\]'):
                runtime.capture_hierarchy([0, 1], torch.eye(2), [True, False], [0],
                                          [torch.tensor([0, 1])], [torch.tensor([[1., 0.], [float('nan'), 1.]])])

    def test_real_extraction_and_backward_across_four_refreshes(self):
        from gcd_runtime import trainer
        from types import SimpleNamespace
        # No mocked clustering: reproduce repeated hierarchy rebuilding using
        # identical embeddings and sparse IDs, then run the actual shared loss.
        from crish.loss import hypselex_loss
        from hyptorch.nn import ToPoincare
        args = SimpleNamespace(num_labeled_classes=2, num_unlabeled_classes=2,
            unbalanced=False, lr=.001, c=.1, n_views=2, temperature=1.,
            contrast_unlabel_only=False, strategy='zero_one', unsupervised_smoothing=.5,
            sup_con_weight=.35, hyper_start_epoch=0, hyper_end_epoch=200, hyper_max_weight=1.)
        features = F.normalize(torch.tensor([[1., .1], [1., -.1]] + [[1., 0.]] * 10), dim=-1)
        ids = torch.tensor([110, 7, 99, 15, 42, 66, 80, 12, 50, 60, 70, 90])
        membership = torch.tensor([True, True] + [False] * 10)
        head = torch.nn.Sequential(torch.nn.Linear(2, 4), ToPoincare(c=.1, clip_r=2.))
        with tempfile.TemporaryDirectory() as directory:
            runtime = EpochRuntime(directory, {})
            with patch.object(trainer, '_runtime', runtime, create=True), \
                 patch.object(trainer, 'args', args, create=True), \
                 patch.object(trainer, 'device', torch.device('cpu'), create=True):
                for epoch in range(4):
                    state = trainer.extract_labeled_protos(torch.nn.Identity(),
                        [([features], torch.tensor([0, 1] + [-1] * 10), ids, membership[:, None])], args)
                    state.validate()
                    self.assertTrue(torch.equal(state.labels[0][:2], torch.tensor([0, 1])))
                    torch.manual_seed(epoch)
                    views = torch.cat([features + .02 * torch.randn_like(features) for _ in range(2)])
                    head.zero_grad(set_to_none=True)
                    loss = hypselex_loss(head, views, ids, membership, torch.tensor([0, 1]), state, args, epoch)['loss']
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
