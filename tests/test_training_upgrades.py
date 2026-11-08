"""Corrected-target, neighbor-support and legal-probe controls; CPU only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import torch
import torch.nn.functional as F

from audit.baseline_trace import baseline_oracle
from audit.protocol import compare_worker
from crish.candidates import Action
from crish.config import CrishSettings
from crish.engine import _draw_context
from crish.loss import hypselex_loss
from crish.smoke import crish_method_probe
from crish.rebuild import confusion_factor, rebuild, target_drift
from crish.reliability import blockwise_query_knn
from crish.sampling import balanced_known_folds, global_neighbor_pool
from gcd_runtime.config import ROOT, load_config
from tests.crish_v21.fixtures import state_fixture, backbone_views
from tests.test_crish_training import args_fixture, configure_trainer, run_fixture, training_config


class CorrectedTargetTests(unittest.TestCase):
    def test_correction_preserves_supervised_terms_and_backbone_head_gradients_are_finite(self):
        trainer = configure_trainer()
        from hyptorch.nn import ToPoincare
        args = args_fixture(2)
        trainer.args, trainer.device = args, torch.device('cpu')
        state = state_fixture()
        head = torch.nn.Sequential(torch.nn.Linear(3, 16), ToPoincare(c=.1, clip_r=2.))
        inputs = backbone_views(state).requires_grad_()
        original = hypselex_loss(head, inputs, state.ids, state.labeled, state.known_targets, state, args, 1)
        args.use_target_normalization = 'row'
        corrected = hypselex_loss(head, inputs, state.ids, state.labeled, state.known_targets, state, args, 1)
        for left, right in zip(original['supervised'], corrected['supervised']):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        corrected['loss'].backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_nonself_targets_and_original_loss_have_unit_mass_and_finite_gradients(self):
        source = baseline_oracle()
        state = state_fixture()
        for strategy in ('zero_one', 'pair_dist', 'pair_cluster'):
            cf = confusion_factor(state, state.ids, backbone_views(state), strategy, normalization='row')
            self.assertFalse(cf.requires_grad)
            self.assertTrue(torch.equal(cf.diag(), torch.zeros(len(cf))))
            torch.testing.assert_close(cf.sum(1), torch.ones(len(cf)))
            # Reference helper removes self and permutes support. Its original
            # LabelSmoothingLoss now receives a probability target per anchor.
            flat = (torch.randn(16, 8)*.01).requires_grad_()
            logits, labels, similarity = source['info_nce_logits'](flat, cf, source['args'])
            targets = .5*F.one_hot(labels, logits.shape[1])+.5*similarity
            torch.testing.assert_close(targets.sum(1), torch.ones(len(flat)))
            loss = source['LabelSmoothingLoss']()(logits, labels, similarity, .5)
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(flat.grad).all())

    def test_zero_support_fallback_is_the_positive_view_and_legacy_still_rejects(self):
        state = state_fixture()
        views = torch.zeros_like(backbone_views(state))
        expected = torch.zeros(16, 16)
        expected[torch.arange(16), (torch.arange(16)+8) % 16] = 1
        actual = confusion_factor(state, state.ids, views, normalization='row')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            confusion_factor(state, state.ids, views)
        with self.assertRaises(ValueError):
            confusion_factor(state, state.ids, views, normalization='invalid')

    def test_legacy_stays_exact_and_row_drift_matches_actual_smoothing_targets(self):
        state = state_fixture()
        trial, _ = rebuild(state, Action('move', 22, 0, 2, 3))
        views = backbone_views(state)
        expected = baseline_oracle()['confusion_oracle'](views, state.labels, state.prototypes, state.radii)
        torch.testing.assert_close(confusion_factor(state, state.ids, views), expected, rtol=0, atol=0)
        cf0 = confusion_factor(state, state.ids, views, 'pair_cluster', normalization='row')
        cf1 = confusion_factor(trial, state.ids, views, 'pair_cluster', normalization='row')
        drift = target_drift(state, trial, state.ids, views, smoothing=.5,
                             strategy='pair_cluster', normalization='row')
        self.assertAlmostEqual(drift['use_target_half_l1_mean'], float(.5*(cf1-cf0).abs().sum(1).mean()/2))
        self.assertEqual(target_drift(state, state, state.ids, views, smoothing=.5,
                                     normalization='row')['use_target_half_l1_mean'], 0)


class EvidenceSamplingTests(unittest.TestCase):
    def test_query_knn_matches_dense_neighbors_for_multiple_blocks_and_excludes_self(self):
        features = torch.randn(47, 6, generator=torch.Generator().manual_seed(10))
        queries = torch.tensor([1, 21, 46, 0])
        distances = torch.cdist(features[queries], features)
        distances[torch.arange(len(queries)), queries] = torch.inf
        expected = distances.topk(5, largest=False).indices
        for block in (1, 7, 64):
            actual = blockwise_query_knn(features, queries, 5, query_block=2, reference_block=block)
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse((actual == queries[:, None]).any())
        with self.assertRaises(RuntimeError):
            blockwise_query_knn(features, queries, 5, deadline=time.monotonic()-1)
        for invalid in (torch.tensor([-1]), torch.tensor([47]), torch.tensor([1, 1])):
            with self.assertRaises(ValueError):
                blockwise_query_knn(features, invalid, 5)

    def test_global_pool_brings_nearest_references_outside_random_anchors_within_budget(self):
        state = state_fixture()
        available = torch.arange(len(state.ids))
        generator = torch.Generator().manual_seed(19)
        # Reconstruct only the declared anchor draw; no hidden labels exist.
        unlabeled = (~state.labeled).nonzero().flatten()
        anchor = unlabeled[torch.randperm(len(unlabeled), generator=generator)[:1]]
        normalized = F.normalize(state.features, dim=-1)
        distances = torch.cdist(normalized[anchor], normalized)
        distances[0, anchor] = torch.inf
        nearest = int(distances.argmin())
        pool = global_neighbor_pool(state, available, 2, 1,
                                    torch.Generator().manual_seed(19), reference_block=3)
        self.assertEqual(set(pool.tolist()), {int(anchor), nearest})
        self.assertEqual(len(pool.unique()), 2)
        self.assertTrue(torch.equal(pool, global_neighbor_pool(state, available, 2, 1,
                                   torch.Generator().manual_seed(19), reference_block=3)))

    def test_balanced_folds_cover_rare_legal_classes_without_duplicate_images(self):
        rows = torch.arange(20)
        targets = torch.tensor([0]*16+[1]*2+[2]*2)
        folds = balanced_known_folds(rows, targets, 4, torch.Generator().manual_seed(23))
        self.assertEqual(len(torch.cat(folds).unique()), 8)
        for fold in folds:
            self.assertEqual(set(targets[fold].tolist()), {0, 1, 2})
        with self.assertRaises(ValueError):
            balanced_known_folds(rows, targets, 11, torch.Generator())

    def test_global_context_has_disjoint_probes_evidence_and_uses_only_legal_labels(self):
        # Fixture has two legally labeled rows, so use one per independent fold.
        state = state_fixture()
        settings = CrishSettings(evidence_samples=4, k=1, probe_known=1, probe_unlabeled=2,
                                  evidence_sampling='global_neighbors', probe_sampling='class_balanced')
        # Six unlabeled rows minus four probes leave two eligible references.
        folds, pool = _draw_context(state, settings, torch.Generator().manual_seed(4))
        reserved = torch.cat([torch.cat(pair) for pair in folds]).tolist()
        self.assertEqual(len(set(reserved)), len(reserved))
        self.assertFalse(set(pool.tolist()) & set(reserved))
        self.assertLessEqual(len(pool), settings.evidence_samples)
        self.assertEqual(len(pool.unique()), len(pool))
        self.assertTrue(all(state.labeled[k].all() and not state.labeled[u].any() for k, u in folds))


def upgraded_fixture_config(method='crish'):
    cfg = training_config(method)
    cfg['use_target_normalization'] = 'row'
    cfg['crish'].update(evidence_sampling='global_neighbors', probe_sampling='class_balanced')
    return cfg


class UpgradeIntegrationTests(unittest.TestCase):
    def test_actual_dino_head_preflight_exercises_all_three_upgrade_modes(self):
        for normalization in ('legacy', 'row'):
            with self.subTest(normalization=normalization):
                cfg = upgraded_fixture_config()
                cfg['use_target_normalization'] = normalization
                result = crish_method_probe(cfg, 'cpu', backbone_dim=8, hidden_dim=32)
                self.assertGreater(result['scoring_attempts'], 0)
                self.assertEqual(result['use_target_normalization'], normalization)
                self.assertEqual(result['evidence_sampling'], 'global_neighbors')
                self.assertEqual(result['probe_sampling'], 'class_balanced')
                self.assertTrue(result['scaled_synthetic_probe'])
                self.assertEqual(result['trained_epochs'], 0)

    def test_upgraded_training_is_label_invariant_logs_context_and_changes_finite_weights(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_fixture(first, upgraded_fixture_config(), poison=123456)
            b = run_fixture(second, upgraded_fixture_config(), poison=-98765)
            for left, right in zip(a, b):
                for key in left:
                    self.assertTrue(torch.equal(left[key], right[key]), key)
                    self.assertTrue(torch.isfinite(left[key]).all())
            events = [json.loads(line) for line in (Path(first)/'crish.jsonl').read_text().splitlines()]
            self.assertTrue(any(event.get('gains') for event in events))
            self.assertTrue(all(event['evidence_sampling'] == 'global_neighbors' for event in events))
            self.assertTrue(all(event['probe_sampling'] == 'class_balanced' for event in events))
            self.assertTrue(all(event['use_target_normalization'] == 'row' for event in events))
            self.assertTrue(all(event['evidence_reference_rows'] > event['evidence_cache_rows'] for event in events))
            self.assertTrue(all(event['view_neighbor_scope'] == 'local_cached_two_views' for event in events))

    def test_corrected_baseline_matches_disabled_repair_and_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as baseline, tempfile.TemporaryDirectory() as disabled:
            cfg = upgraded_fixture_config()
            cfg['crish']['enabled'] = False
            a = run_fixture(baseline, upgraded_fixture_config('baseline'))
            b = run_fixture(disabled, cfg)
            for left, right in zip(a, b):
                for key in left:
                    self.assertTrue(torch.equal(left[key], right[key]))
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as interrupted, tempfile.TemporaryDirectory() as resumed:
            cfg = upgraded_fixture_config()
            expected = run_fixture(full, cfg)
            with self.assertRaises(InterruptedError):
                run_fixture(interrupted, cfg, stop=True)
            actual = run_fixture(resumed, cfg, seed=991, resume=str(Path(interrupted)/'resume.pt'))
            for left, right in zip(expected, actual):
                for key in left:
                    self.assertTrue(torch.equal(left[key], right[key]))

    def test_profiles_cli_and_protocol_distinguish_changed_use_from_reference(self):
        for prefix in ('', 'cub_'):
            for smoke in ('', '_smoke'):
                configs = [load_config(ROOT/f'configs/{prefix}{method}_upgraded{smoke}.json')
                           for method in ('baseline', 'crish')]
                self.assertEqual({k for k in configs[0] if configs[0][k] != configs[1][k]}, {'method'})
                for cfg in configs:
                    report = compare_worker(cfg, smoke=bool(smoke))
                    self.assertTrue(report['cli_hyperparameters_match'])
                    self.assertFalse(report['reference_use_arithmetic_preserved'])
                    self.assertFalse(report['exact_paper_reproduction_verified'])
                    self.assertEqual(cfg['crish']['probe_known'], 16)
                    self.assertEqual(cfg['crish']['probe_unlabeled'], 32)
        result = subprocess.run([sys.executable, '-m', 'gcd_runtime.train_cli', 'check',
                                 '--profile', 'upgraded', '--dataset', 'cub', '--baseline', '--dry-run'],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('configs/cub_baseline_upgraded.json', result.stdout)
        self.assertIn('"selected_profile": "upgraded"', result.stdout)

    def test_final_cli_and_prepared_pair_preserve_reference_loss_for_both_datasets(self):
        from contextlib import redirect_stdout
        import io
        from scripts.prepare_retraining import prepare
        for dataset, prefix in (('scars', ''), ('cub', 'cub_')):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                with redirect_stdout(io.StringIO()):
                    directory = prepare(dataset, 'final', [1], output=Path(tmp)/'plan')
                pair = [load_config(directory/f'{method}-seed-1.json') for method in ('baseline', 'crish')]
                self.assertEqual({k for k in pair[0] if pair[0][k] != pair[1][k]}, {'method'})
                for cfg in pair:
                    reference = load_config(ROOT/f'configs/{prefix}{cfg["method"]}.json')
                    self.assertEqual({k for k in reference if reference[k] != cfg[k]}, {'crish'})
                    self.assertEqual({k for k in reference['crish'] if reference['crish'][k] != cfg['crish'][k]},
                                     {'evidence_sampling', 'probe_sampling', 'probe_known', 'probe_unlabeled'})
                    self.assertEqual((cfg['crish']['probe_known'], cfg['crish']['probe_unlabeled']), (16, 32))
                    report = compare_worker(cfg)
                    self.assertTrue(report['cli_hyperparameters_match'])
                    self.assertTrue(report['reference_use_arithmetic_preserved'])
                for profile in ('final', 'reference'):
                    result = subprocess.run([sys.executable, '-m', 'gcd_runtime.train_cli', 'check', '--dataset', dataset,
                                             '--baseline', '--profile', profile, '--dry-run'],
                                            cwd=ROOT, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    entrypoint = json.JSONDecoder().raw_decode(result.stdout)[0]
                    suffix = '_final' if profile == 'final' else ''
                    self.assertEqual(entrypoint['stages'][0]['config'], f'configs/{prefix}baseline{suffix}.json')
                manifest = json.loads((directory/'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'prepared_not_executed')
                self.assertEqual(manifest['profile'], 'final')

    def test_invalid_modes_and_block_sizes_fail_before_training(self):
        for values in (dict(evidence_sampling='bad'), dict(probe_sampling='bad'),
                       dict(neighbor_reference_block=0), dict(neighbor_reference_block=True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CrishSettings(**values)
        cfg = json.loads((ROOT/'configs/crish.json').read_text())
        cfg['use_target_normalization'] = 'bad'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'invalid.json'
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                load_config(path)
