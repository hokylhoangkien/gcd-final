"""CPU regression tests for released-code selection and paired peak reporting."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.disjoint_evaluation import evaluate_disjoint, plain_kmeans
from gcd_runtime.launch import final_evaluations
from gcd_runtime.metrics import atomic_json, peaks, read_rows, summarize_run
from gcd_runtime.selection import BEST_NAME, TEST_PROTOCOL, load_evaluation_checkpoint, read_best, restore_artifacts
from tests.test_crish_training import configure_trainer, run_fixture, training_config


def selected_config(method='baseline'):
    cfg = training_config(method)
    cfg.update(checkpoint_selection='best_old_test_v2', test_every_epochs=1,
               final_ss_evaluation=True, ss_eval_every_epochs=0)
    return cfg


def score(directory, cfg, epoch, values, *, model=None):
    model = model if model is not None else torch.nn.Linear(2, 2)
    checkpoint = Path(directory)/'backbone_monitor.pt'
    torch.save(model.state_dict(), checkpoint)
    runtime = EpochRuntime(directory, cfg, evaluation_epoch=epoch,
                           evaluated_checkpoint=checkpoint, evaluation_kind='test')
    runtime.evaluation(dict(zip(('all_acc', 'old_acc', 'new_acc'), values)))


def fixture_selector(command, **kwargs):
    cfg_path, directory, mode, checkpoint, epoch = command[3:]
    if mode != 'select':
        raise AssertionError(f'Unexpected evaluator mode: {mode}')
    cfg = json.loads(Path(cfg_path).read_text())
    # Controlled fixture score; production evaluator is exercised separately.
    runtime = EpochRuntime(directory, cfg, evaluation_epoch=int(epoch),
                           evaluated_checkpoint=checkpoint, evaluation_kind='test')
    runtime.evaluation(dict(all_acc=.8, old_acc=.9-int(epoch)*.1, new_acc=.7))


class SelectionPolicyTests(unittest.TestCase):
    def test_strict_old_v2_selection_and_peak_triplets_are_independent(self):
        cfg = selected_config()
        cfg['epochs'] = 4
        observations = [(.8, .9, .7), (.9, .85, 1.), (.75, .95, .55), (.99, .95, .98)]
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            for epoch, values in enumerate(observations):
                with torch.no_grad():
                    model.weight.fill_(epoch+1)
                score(tmp, cfg, epoch, values, model=model)
            selected = read_best(Path(tmp)/BEST_NAME, cfg)
            self.assertEqual(selected['selection']['epoch'], 2)
            self.assertTrue(torch.equal(selected['model']['weight'], torch.full_like(model.weight, 3.)))
            rows = read_rows(Path(tmp)/'test_metrics.jsonl')
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['protocol'] == TEST_PROTOCOL for r in rows))
            summary = summarize_run(tmp)['histories'][TEST_PROTOCOL]['peaks']
            self.assertEqual(summary['all_acc']['epoch'], 3)
            self.assertEqual(summary['old_acc']['epoch'], 2)
            self.assertEqual(summary['new_acc']['epoch'], 1)
            for metric, epoch in [('all_acc', 3), ('old_acc', 2), ('new_acc', 1)]:
                row = summary[metric]
                self.assertEqual(tuple(row[k] for k in ('all_acc', 'old_acc', 'new_acc')), observations[epoch])
            self.assertFalse((Path(tmp)/'evaluation.json').exists())
            # Replaying the same exact snapshot/epoch does not duplicate history.
            runtime = EpochRuntime(tmp, cfg, evaluation_epoch=3, evaluation_kind='test',
                                   evaluated_checkpoint=Path(tmp)/'backbone_monitor.pt')
            runtime.evaluation(dict(zip(('all_acc', 'old_acc', 'new_acc'), observations[3])))
            self.assertEqual(len(read_rows(Path(tmp)/'test_metrics.jsonl')), 4)

    def test_first_zero_score_still_yields_a_usable_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = selected_config()
            score(tmp, cfg, 0, (0., 0., 0.))
            self.assertEqual(read_best(Path(tmp)/BEST_NAME, cfg)['selection']['old_acc'], 0.)

    def test_selector_keeps_training_updates_rng_and_label_masking_unchanged(self):
        for method in ('baseline', 'crish'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as off, tempfile.TemporaryDirectory() as on:
                expected = run_fixture(off, training_config(method), poison=-1)
                with patch('gcd_runtime.monitoring.subprocess.run', side_effect=fixture_selector):
                    actual = run_fixture(on, selected_config(method), poison=987654)
                for a, b in zip(expected, actual):
                    for key in a:
                        self.assertTrue(torch.equal(a[key], b[key]), key)
                before = torch.load(Path(off)/'resume.pt', weights_only=False)
                after = torch.load(Path(on)/'resume.pt', weights_only=False)
                self.assertTrue(torch.equal(before['torch_rng'], after['torch_rng']))
                self.assertTrue(np.array_equal(before['numpy_rng'][1], after['numpy_rng'][1]))
                self.assertEqual(before['python_rng'], after['python_rng'])
                self.assertEqual([r['epoch'] for r in read_rows(Path(on)/'test_metrics.jsonl')], [0, 1])
                self.assertEqual(read_best(Path(on)/BEST_NAME, selected_config(method))['selection']['epoch'], 0)

    def test_resume_recovers_pending_selector_without_changing_next_update(self):
        cfg = selected_config('crish')
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as stopped, tempfile.TemporaryDirectory() as resumed:
            with patch('gcd_runtime.monitoring.subprocess.run', side_effect=fixture_selector):
                expected = run_fixture(full, cfg)
                with self.assertRaises(InterruptedError):
                    run_fixture(stopped, cfg, stop=True)
                actual = run_fixture(resumed, cfg, seed=991, resume=str(Path(stopped)/'resume.pt'))
            for a, b in zip(expected, actual):
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertEqual(read_best(Path(resumed)/BEST_NAME, cfg)['selection']['epoch'], 0)
            self.assertEqual(len(read_rows(Path(resumed)/'test_metrics.jsonl')), 2)
            self.assertEqual(len(read_rows(Path(resumed)/'metrics.jsonl')), 2)

    def test_resume_after_last_selector_failure_runs_no_more_sgd(self):
        cfg = selected_config()
        def failing(command, **kwargs):
            if command[-1] == '1':
                raise subprocess.TimeoutExpired(command, 1)
            return fixture_selector(command, **kwargs)
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as stopped, tempfile.TemporaryDirectory() as resumed:
            with patch('gcd_runtime.monitoring.subprocess.run', side_effect=fixture_selector):
                expected = run_fixture(full, cfg)
            with patch('gcd_runtime.monitoring.subprocess.run', side_effect=failing), \
                 self.assertRaises(subprocess.TimeoutExpired):
                run_fixture(stopped, cfg)
            with patch('gcd_runtime.monitoring.subprocess.run', side_effect=fixture_selector) as evaluator:
                actual = run_fixture(resumed, cfg, resume=str(Path(stopped)/'resume.pt'))
            self.assertEqual(evaluator.call_count, 1)
            for a, b in zip(expected, actual):
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertTrue((Path(resumed)/'backbone_final.pt').is_file())
            self.assertEqual(len(read_rows(Path(resumed)/'test_metrics.jsonl')), 2)

    def test_resume_refuses_missing_or_future_selection_artifacts(self):
        cfg = selected_config()
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as destination:
            with self.assertRaisesRegex(ValueError, 'original run directory'):
                restore_artifacts(source, destination, cfg, 1)
            score(source, cfg, 1, (.8, .9, .7))
            with self.assertRaisesRegex(ValueError, 'ahead'):
                restore_artifacts(source, destination, cfg, 0)


class DisjointEvaluatorTests(unittest.TestCase):
    def test_worker_select_mode_routes_to_annotation_bearing_evaluator_only(self):
        import sys
        from gcd_runtime import worker
        trainer = configure_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['worker', str(ROOT/'configs/crish.json'), tmp, 'select', 'fixture.pt', '4']
            with patch.object(sys, 'argv', argv), patch.object(trainer, 'run') as run:
                worker.main()
            flags, runtime = run.call_args.args
            self.assertIn('--eval_only', flags)
            self.assertEqual(runtime.evaluation_kind, 'test')
            self.assertEqual(runtime.evaluation_epoch, 4)
            self.assertIsNone(runtime.resume)

    def test_cuda_rejects_legacy_backend_before_allocating_gpu_tensors(self):
        package = SimpleNamespace(read_text=lambda _: None, version='legacy')
        with patch('gcd_runtime.disjoint_evaluation.distribution', return_value=package), \
             self.assertRaisesRegex(RuntimeError, 'pinned Git'):
            plain_kmeans(torch.eye(2), 2, torch.device('cuda'))

    def test_actual_plain_kmeans_cpu_backend_and_feature_guards(self):
        predictions, backend = plain_kmeans(torch.eye(2), 2, torch.device('cpu'))
        self.assertEqual(len(set(predictions.tolist())), 2)
        self.assertEqual(backend['backend'], 'kmeans_pytorch')
        self.assertEqual(len(backend['backend_function_sha256']), 64)
        with self.assertRaises(ValueError):
            plain_kmeans(torch.full((2, 2), float('nan')), 2, torch.device('cpu'))

    def test_old_metric_uses_global_v2_not_independent_v1_assignment(self):
        configure_trainer()
        from project_utils.cluster_and_log_utils import split_cluster_acc_v1
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2])
        predictions = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        self.assertEqual(split_cluster_acc_v1(labels.numpy(), predictions, labels.numpy() < 2)[1], .75)
        cfg = selected_config()
        args = SimpleNamespace(train_classes=[0, 1], num_labeled_classes=2, num_unlabeled_classes=1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'backbone_monitor.pt'
            torch.save({}, checkpoint)
            runtime = EpochRuntime(tmp, cfg, evaluation_epoch=0, evaluated_checkpoint=checkpoint,
                                   evaluation_kind='test')
            with patch('gcd_runtime.disjoint_evaluation.plain_kmeans', return_value=(predictions, {})) as cluster:
                actual = evaluate_disjoint(torch.nn.Identity(),
                    [(torch.eye(8), labels, torch.arange(8))], args, runtime, torch.device('cpu'))
            self.assertEqual((actual['all_acc'], actual['old_acc'], actual['new_acc']), (.75, .5, 1.))
            self.assertEqual((actual['n_old'], actual['n_new']), (4, 4))
            # Clusterer receives features, cluster count and device, never labels.
            features, count, device = cluster.call_args.args
            self.assertEqual(features.shape, (8, 8))
            self.assertEqual(count, 3)
            row = read_rows(Path(tmp)/'test_metrics.jsonl')[0]
            self.assertEqual(row['matching'], 'hungarian_v2_global')


class FinalReportTests(unittest.TestCase):
    def test_best_bundle_loading_uses_its_actual_epoch_and_rejects_config_drift(self):
        cfg = selected_config()
        with tempfile.TemporaryDirectory() as tmp:
            score(tmp, cfg, 0, (.8, .9, .7))
            runtime = EpochRuntime(tmp, cfg)
            state = load_evaluation_checkpoint(Path(tmp)/BEST_NAME, cfg, runtime)
            self.assertEqual(runtime.evaluation_epoch, 0)
            self.assertEqual(runtime.evaluation_role, 'best_old_test')
            self.assertEqual(set(state), {'weight', 'bias'})
            with self.assertRaisesRegex(ValueError, 'exact source and config'):
                load_evaluation_checkpoint(Path(tmp)/BEST_NAME, dict(cfg, seed=99), runtime)

    def test_best_and_last_ss_reports_and_peak_protocols_remain_separate(self):
        cfg = selected_config()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            score(directory, cfg, 0, (.8, .9, .7))
            torch.save({}, directory/'backbone_final.pt')
            def evaluate(command, **kwargs):
                checkpoint = Path(command[-1])
                best = checkpoint.name == BEST_NAME
                runtime = EpochRuntime(directory, cfg, evaluation_epoch=0 if best else None)
                if best:
                    runtime.evaluation_role = 'best_old_test'
                runtime.evaluation(dict(all_acc=.6 if best else .7, old_acc=.8, new_acc=.5))
            with patch('gcd_runtime.launch.subprocess.run', side_effect=evaluate) as worker, \
                 contextlib.redirect_stdout(io.StringIO()):
                commands = final_evaluations(directory/'config.json', directory, cfg, {}, None)
            self.assertEqual([r['role'] for r in commands], ['last', 'best_old_test'])
            self.assertEqual(worker.call_count, 2)
            self.assertTrue(all(c.kwargs['timeout'] == cfg['final_eval_timeout_seconds'] for c in worker.call_args_list))
            primary = json.loads((directory/'evaluation.json').read_text())
            self.assertEqual(primary['purpose'], 'best_old_test')
            self.assertEqual(primary['epoch'], 0)
            summary = json.loads((directory/'results_summary.json').read_text())
            test = summary['histories'][TEST_PROTOCOL]['peaks']['all_acc']
            ss = summary['histories']['balanced_ss_kmeans_train_unlabeled']['peaks']['all_acc']
            self.assertEqual((test['all_acc'], test['epoch']), (.8, 0))
            self.assertEqual((ss['all_acc'], ss['epoch']), (.7, 1))
            self.assertTrue((directory/'evaluation_last.json').is_file())
            self.assertTrue((directory/'evaluation_best_old_test.json').is_file())

    def test_missing_best_is_not_silently_replaced_by_last(self):
        cfg = selected_config()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            torch.save({}, directory/'backbone_final.pt')
            def last(command, **kwargs):
                runtime = EpochRuntime(directory, cfg)
                runtime.evaluation(dict(all_acc=.7, old_acc=.8, new_acc=.6))
            with patch('gcd_runtime.launch.subprocess.run', side_effect=last), \
                 self.assertRaisesRegex(RuntimeError, 'Missing best_old_test'):
                final_evaluations(directory/'config.json', directory, cfg, {}, None)
            self.assertFalse((directory/'evaluation.json').exists())

    def test_config_rejects_mixed_selection_and_validation_policies(self):
        cfg = load_config(ROOT/'configs/crish.json')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            for change in (dict(test_every_epochs=0), dict(test_every_epochs=5),
                           dict(checkpoint_selection='final_epoch'), dict(final_ss_evaluation=False),
                           dict(early_stopping=True)):
                path.write_text(json.dumps(dict(cfg, **change)))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_config(path)
            path.write_text(json.dumps(dict(cfg, checkpoint_selection='final_epoch', test_every_epochs=0)))
            self.assertEqual(load_config(path)['checkpoint_selection'], 'final_epoch')
