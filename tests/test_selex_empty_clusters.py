"""Source-preserving empty-slot support at the SelEx/CRiSH boundary."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crish.candidates import Action, propose_actions, validate_action
from crish.config import CrishSettings
from crish.engine import CrishEngine
from crish.rebuild import prototype_radii, rebuild, target_drift
from crish.state import SupervisionState
from crish.training import train_strict
from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.cluster_alignment import restore_input_order
from gcd_runtime.config import ROOT, load_config
from selex_runtime.hierarchy import coarse_levels, extract_state, finite_base_centers, log_hierarchy
from selex_runtime.objective import selex_loss
from selex_runtime.reference import TRAIN
from selex_runtime.trainer import make_args
from tests.test_selex_runtime import fixture, original_objective


def source_coarse(features, base, centers, known, novel):
    """Execute the pinned hierarchy, including its empty-support fallback."""
    import textwrap
    source = TRAIN.read_text()
    start = source.index('    feats_t = torch.from_numpy(all_feats)')
    end = source.index('    return ids,all_preds', start)
    scope = dict(torch=torch, np=np, KMeans=KMeans, device=torch.device('cpu'),
                 all_feats=features.numpy(), all_preds=base, prototypes=centers,
                 n_labeled=known, n_novel=novel, preds_higher=[base],
                 prototype_higher=[centers])
    exec(textwrap.dedent(source[start:end]), scope)
    return scope['preds_higher'], scope['prototype_higher'], scope['prev_protos']


def empty_state():
    x = torch.tensor([[0., 1.], [1., 0.], [.7, .7], [.8, .6], [.6, .8], [.9, .1]])
    labels = torch.tensor([0, 1, 2, 2, 2, 1])
    mask = torch.tensor([True, True, False, False, False, False])
    centers = torch.stack([x[labels == i].mean(0) for i in range(3)] + [x[2]])
    return SupervisionState(torch.tensor([91, 7, 52, 11, 80, 16]), x, mask,
                            torch.tensor([0, 1]), (labels,), (centers,),
                            (prototype_radii(centers),), (torch.zeros(6, dtype=torch.bool),),
                            allow_empty_clusters=True)


def training_fixture():
    dataset, state, args, head = fixture()
    labels = state.labels[0].clone()
    labels[labels == 3] = 2  # a source-legal unsupported novel slot
    centers = state.prototypes[0].clone()
    centers[2] = state.features[labels == 2].mean(0)
    state = replace(state, labels=(labels,), prototypes=(centers,),
                    radii=(prototype_radii(centers),), allow_empty_clusters=True)
    return dataset, state, args, head


class SelExEmptyClusterTests(unittest.TestCase):
    def test_finite_slots_are_exactly_preserved_with_or_without_empty_ids(self):
        centers = torch.eye(4) + .1
        for labels in (torch.arange(4), torch.tensor([0, 1, 2, 2])):
            rng = torch.get_rng_state().clone()
            got, fallback = finite_base_centers(labels, centers, 2)
            torch.testing.assert_close(got, centers, rtol=0, atol=0)
            self.assertEqual(fallback, [])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_empty_center_fallback_matches_source_side_means_without_relabeling(self):
        labels = torch.tensor([0, 1, 2, 2])
        centers = torch.tensor([[0., 1.], [1., 0.], [.5, .5], [float('nan'), float('nan')]])
        original = labels.clone()
        expected_labels, raw_source_centers, expected_fallback = source_coarse(
            torch.eye(4, 2), labels, centers, 2, 2)
        got, fallback = finite_base_centers(labels, centers, 2)
        torch.testing.assert_close(got, expected_fallback, rtol=0, atol=0)
        # Upstream sanitizes a copy for coarse grouping but returns the raw
        # NaN base center. The runtime must use that finite copy at level 0 too.
        self.assertTrue(torch.isnan(raw_source_centers[0][3]).all())
        self.assertEqual(fallback, [3])
        self.assertTrue(torch.equal(labels, original))
        self.assertTrue(torch.isnan(centers[3]).all())
        self.assertTrue(torch.equal(expected_labels[0], labels))
        # Exactly the source zero fallback when the entire novel side is empty.
        both_bad = centers.clone(); both_bad[2:] = torch.nan
        got, fallback = finite_base_centers(torch.tensor([0, 1]), both_bad, 2)
        self.assertEqual(fallback, [2, 3])
        self.assertTrue(torch.equal(got[2:], torch.zeros(2, 2)))

    def test_nonfinite_occupied_centers_and_invalid_ids_are_rejected(self):
        labels = torch.tensor([0, 1, 2, 2])
        centers = torch.eye(4, 2)
        for bad in (float('nan'), float('inf')):
            damaged = centers.clone(); damaged[2, 0] = bad
            with self.assertRaisesRegex(ValueError, r'Nonfinite occupied.*level=0.*\[2\]'):
                finite_base_centers(labels, damaged, 2)
        for invalid in (torch.tensor([0, -1]), torch.tensor([0, 4]), torch.tensor([0., 1.])):
            with self.assertRaisesRegex(ValueError, 'Invalid assignment.*level=0'):
                finite_base_centers(invalid, centers, 2)

    def test_coarse_degeneracy_preserves_executed_source_labels_and_slot_budget(self):
        features = torch.ones(16, 3)
        # Include a base slot with no image, and identical prototype rows.
        base = torch.arange(16) % 7
        centers = torch.ones(8, 3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            labels, prototypes = coarse_levels(features, base, centers, 4, 4)
            expected_labels, expected_centers, _ = source_coarse(features, base, centers, 4, 4)
        self.assertEqual([len(p) for p in prototypes], [8, 4, 2])
        self.assertLess(labels[1].unique().numel(), 4)
        for got, expected in zip(labels, expected_labels):
            self.assertTrue(torch.equal(got, expected))
        for got, expected in zip(prototypes, expected_centers):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(prototype_radii(got)).all())

    def test_empty_slot_policy_is_explicit_preserved_by_copy_and_still_validates(self):
        state = empty_state()
        state.copy().validate()
        self.assertTrue(state.copy().allow_empty_clusters)
        with self.assertRaisesRegex(ValueError, 'empty cluster.*level=0'):
            replace(state, allow_empty_clusters=False)
        for bad in (1, 'yes', None):
            with self.assertRaisesRegex(ValueError, 'policy must be boolean'):
                replace(state, allow_empty_clusters=bad)
        invalid = state.labels[0].clone(); invalid[-1] = 4
        with self.assertRaisesRegex(ValueError, 'Invalid assignment.*level=0'):
            replace(state, labels=(invalid,))
        with self.assertRaisesRegex(ValueError, 'Base known anchors'):
            replace(state, known_targets=torch.tensor([1, 0]))
        for field, value, message in (
            ('prototypes', torch.full_like(state.prototypes[0], torch.nan), 'Nonfinite prototypes'),
            ('radii', torch.full_like(state.radii[0], torch.nan), 'Invalid prototype radii'),
            ('features', torch.full_like(state.features, torch.nan), 'Invalid frozen feature'),
        ):
            with self.assertRaisesRegex(ValueError, message):
                replace(state, **{field: value if field == 'features' else (value,)})

    def test_crish_cannot_fill_empty_slots_and_rebuild_preserves_other_empty_ids(self):
        state = empty_state()
        keep, _ = rebuild(state, Action('keep'))
        self.assertIs(keep, state)
        move = Action('move', int(state.ids[2]), 0, 2, 3)
        with self.assertRaisesRegex(ValueError, 'under-support'):
            validate_action(state, move)
        proposals = propose_actions(state, int(state.ids[2]), 0, {},
                                    max_targets=4, min_support=1, min_edges=1, min_reliability=.5)
        self.assertFalse(any(a.kind == 'move' and a.target == 3 for a in proposals))
        for action in (Action('abstain', int(state.ids[2]), 0, 2),
                       Action('move', int(state.ids[2]), 0, 2, 1)):
            changed, _ = rebuild(state, action)
            changed.validate()
            self.assertTrue(changed.allow_empty_clusters)
            self.assertFalse((changed.labels[0] == 3).any())
            torch.testing.assert_close(changed.prototypes[0][3], state.prototypes[0][3])
            self.assertTrue(torch.equal(changed.labels[0][state.labeled], state.known_targets))
            drift = target_drift(state, changed, state.ids, torch.cat([state.features]*2),
                                 smoothing=.75, normalization='selex_legacy')
            self.assertTrue(np.isfinite(drift['use_target_half_l1_mean']))

    def test_empty_slot_loss_and_gradients_match_source_even_when_all_centers_coincide(self):
        for collapsed in (False, True):
            state = empty_state()
            if collapsed:
                centers = torch.ones_like(state.prototypes[0])
                state = replace(state, features=torch.ones_like(state.features),
                                prototypes=(centers,), radii=(prototype_radii(centers),))
            args = make_args(load_config(ROOT/'configs/selex_crish.json'))
            args.hierarchy_levels = 1
            args.num_labeled_classes, args.num_unlabeled_classes = 2, 2
            head = torch.nn.Linear(2, 16)
            other = copy.deepcopy(head)
            raw = torch.cat([state.features, state.features+.01])
            loss = selex_loss(head, raw, state.ids, state.labeled, state.known_targets, state, args, 0)['loss']
            expected = original_objective(other, raw, state, args, 0)
            torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
            loss.backward(); expected.backward()
            for got, want in zip(head.parameters(), other.parameters()):
                self.assertTrue(torch.isfinite(got.grad).all())
                torch.testing.assert_close(got.grad, want.grad, rtol=1e-5, atol=2e-6)

    def test_diagnostics_record_real_support_without_changing_state_or_rng(self):
        state = empty_state()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = SimpleNamespace(epoch=5, run_dir=Path(tmp))
            rng = torch.get_rng_state().clone()
            log_hierarchy(runtime, state.features, state.labeled, state.labels, state.prototypes, 1, [3])
            record = json.loads((Path(tmp)/'hierarchy.jsonl').read_text())
        self.assertEqual(record, runtime.last_hierarchy_diagnostics)
        self.assertEqual(record['policy'], 'preserve_source_assignments')
        self.assertEqual(record['levels'][0]['empty_ids'], [3])
        self.assertEqual(record['levels'][0]['support'], [1, 2, 3, 0])
        self.assertEqual(sum(record['levels'][0]['labeled_support']), 2)
        self.assertEqual(sum(record['levels'][0]['unlabeled_support']), 4)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse((state.labels[0] == 3).any())

    def test_real_crish_engine_completes_with_an_unsupported_slot(self):
        from dataclasses import asdict
        dataset, state, args, head = training_fixture()
        cfg = load_config(ROOT/'configs/selex_crish.json')
        cfg['crish'] = asdict(CrishSettings(warmup_epochs=0, refresh_every=1,
            evidence_samples=16, evidence_trials=1, k=2, boundaries=1,
            max_targets=1, min_support=1, min_edges=1, train_batch=8,
            probe_known=2, probe_unlabeled=2, max_protected_shift=1.,
            max_refresh_seconds=20, save_partitions=False))
        with tempfile.TemporaryDirectory() as tmp:
            runtime = EpochRuntime(tmp, cfg)
            optimizer = torch.optim.SGD(head.parameters(), lr=.002, momentum=.9)
            out = CrishEngine(runtime, objective=selex_loss).prepare(
                state, torch.nn.Identity(), head, optimizer, dataset, args, 0)
            record = json.loads((Path(tmp)/'crish.jsonl').read_text())
        self.assertEqual(record['status'], 'completed', record)
        self.assertGreater(len(record['gains']), 0)
        out.validate()
        self.assertFalse((out.labels[0] == 3).any())
        self.assertTrue(torch.equal(out.labels[0][out.labeled], out.known_targets))

    def test_empty_slot_training_baseline_disabled_parity_and_checkpoint_resume(self):
        def execute(directory, method, *, resume=None, stop=False):
            dataset, state, args, head = training_fixture()
            args.epochs = 2
            cfg = load_config(ROOT/'configs/selex_crish_smoke.json')
            cfg.update(device='cpu', method=method, epochs=2, mlp_out_dim=16,
                       synthetic_fixture=True, max_steps_per_epoch=1, final_ss_evaluation=False)
            cfg['crish']['enabled'] = False
            class Synthetic(EpochRuntime):
                def monitor_epoch(self, *unused): pass
                def save(self, epoch, *values):
                    super().save(epoch, *values)
                    if stop and epoch == 0:
                        raise InterruptedError('fixture stop')
            runtime = Synthetic(directory, cfg, resume)
            model = torch.nn.Linear(4, 4)
            loader = DataLoader(dataset, batch_size=16, shuffle=False)
            def extract(model, loader, args):
                runtime.hierarchy_state = state.copy()
            train_strict(head, model, loader, loader, args, runtime,
                         objective=selex_loss, extractor=extract)
            return model.state_dict(), head.state_dict()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
             tempfile.TemporaryDirectory() as c, tempfile.TemporaryDirectory() as d:
            expected = execute(a, 'baseline')
            disabled = execute(b, 'crish')
            with self.assertRaises(InterruptedError):
                execute(c, 'baseline', stop=True)
            resumed = execute(d, 'baseline', resume=str(Path(c)/'resume.pt'))
            for result in (disabled, resumed):
                for got, want in zip(result, expected):
                    self.assertTrue(all(torch.equal(got[k], want[k]) for k in got))

    def test_real_pinned_empty_fit_keeps_exact_assignments_and_hidden_label_isolation(self):
        path = ROOT/'upstream/SelEx/methods/clustering/faster_mix_k_means_pytorch.py'
        spec = importlib.util.spec_from_file_location('selex_empty_fixture', path)
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(ROOT/'upstream/SelEx')] + sys.path):
            spec.loader.exec_module(module)
        points = torch.tensor([[1., 1.]] * 8)
        points[1], points[4] = torch.tensor([1., 0.]), torch.tensor([0., 1.])
        points = F.normalize(points, dim=1)
        ids = torch.tensor([91, 7, 52, 11, 80, 16, 33, 2])
        mask = torch.tensor([False, True, False, False, True, False, False, False])
        initial = torch.stack([points[1], points[4], points[2], points[2]])
        args = make_args(load_config(ROOT/'configs/selex_crish.json'))
        args.num_labeled_classes, args.num_unlabeled_classes = 2, 2
        model = torch.nn.Linear(2, 2, bias=False)
        model.weight.data.copy_(torch.eye(2))
        real_device = torch.device
        fitted = []
        def bounded_fit(**kwargs):
            kwargs['max_iterations'] = 1
            fitted.append(module.K_Means(**kwargs))
            return fitted[-1]
        states = []
        for poison in (-1, 98765):
            targets = torch.full((8,), poison, dtype=torch.long)
            targets[mask] = torch.tensor([0, 1])
            runtime = SimpleNamespace(epoch=args.level_interval)
            loader = [([points], targets, ids, mask[:, None])]
            with patch.dict('sys.modules', {'methods.clustering.faster_mix_k_means_pytorch':
                                          SimpleNamespace(K_Means=bounded_fit)}), \
                 patch.object(module.K_Means, 'kpp', return_value=initial.clone()), \
                 patch('torch.device', side_effect=lambda v: real_device('cpu' if v == 'cuda' else v)):
                state = extract_state(model, loader, args, runtime)
            before = restore_input_order(fitted[-1].labels_, mask)
            self.assertTrue((torch.bincount(before, minlength=4) == 0).any())
            self.assertTrue(torch.equal(state.labels[0], before))
            state.validate()
            self.assertIs(runtime.hierarchy_state, state)
            self.assertTrue(torch.equal(state.ids, ids))
            self.assertTrue(torch.equal(state.labels[0][mask], torch.tensor([0, 1])))
            fallback, _ = finite_base_centers(before, fitted[-1].cluster_centers_, 2)
            expected_labels, expected_centers, _ = source_coarse(state.features, before, fallback, 2, 2)
            for got, expected in zip(state.labels, expected_labels):
                self.assertTrue(torch.equal(got, expected))
            for got, expected in zip(state.prototypes, expected_centers):
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
            head = torch.nn.Linear(2, 8)
            loss = selex_loss(head, torch.cat([points, points+.01]), ids, mask,
                              state.known_targets, state, args, runtime.epoch)['loss']
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
            states.append(state)
        for a, b in zip(states[0].labels+states[0].prototypes, states[1].labels+states[1].prototypes):
            self.assertTrue(torch.equal(a, b))

    def test_nonfinite_features_fail_before_fitter_and_missing_anchors_are_rejected(self):
        # No fitter is needed: preconditions must fail before any fit starts.
        class NeverFit:
            def __init__(self, **kwargs):
                raise AssertionError('Invalid input reached fitter')
        args = make_args(load_config(ROOT/'configs/selex_crish.json'))
        args.num_labeled_classes, args.num_unlabeled_classes = 2, 2
        model = torch.nn.Linear(2, 2)
        runtime = SimpleNamespace(epoch=0)
        for points, targets, message in (
            (torch.full((4, 2), torch.nan), torch.tensor([0, 1, -1, -1]), 'Nonfinite extracted'),
            (torch.ones(4, 2), torch.tensor([0, 0, -1, -1]), 'legal anchors'),
        ):
            loader = [([points], targets, torch.arange(4), torch.tensor([[True], [True], [False], [False]]))]
            with patch.dict('sys.modules', {'methods.clustering.faster_mix_k_means_pytorch':
                                          SimpleNamespace(K_Means=NeverFit)}), \
                 self.assertRaisesRegex(ValueError, message):
                extract_state(model, loader, args, runtime)
