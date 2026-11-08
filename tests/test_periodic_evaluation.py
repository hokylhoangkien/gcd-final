"""Bounded CPU checks for monitoring, label boundaries and checkpoint policy."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.monitoring import monitor_due
from tests.test_crish_training import configure_trainer, run_fixture, training_config


class PeriodicConfigTests(unittest.TestCase):
    def test_defaults_overrides_and_smoke_stays_disabled(self):
        for method in ('baseline', 'crish'):
            full = load_config(ROOT/f'configs/{method}.json')
            self.assertEqual(full['ss_eval_every_epochs'], 0)
            self.assertEqual(full['checkpoint_selection'], 'best_old_test_v2')
            self.assertEqual(full['test_every_epochs'], 1)
            self.assertEqual(load_config(ROOT/f'configs/{method}_smoke.json')['ss_eval_every_epochs'], 0)
            for interval in (0, 1, 7):
                self.assertEqual(load_config(ROOT/f'configs/{method}.json',
                                            eval_every_epochs=interval)['ss_eval_every_epochs'], interval)
            for value in (-1, True, 1.5):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_config(ROOT/f'configs/{method}.json', eval_every_epochs=value)
        with tempfile.TemporaryDirectory() as tmp:
            for value in (0, -1, True):
                path = Path(tmp)/'config.json'
                path.write_text(json.dumps(dict(full, ss_eval_timeout_seconds=value)))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_schedule_and_final_worker_are_not_duplicated(self):
        cfg = dict(epochs=12, ss_eval_every_epochs=5, final_ss_evaluation=True)
        self.assertEqual([e+1 for e in range(12) if monitor_due(cfg, e)], [5, 10])
        cfg['final_ss_evaluation'] = False
        self.assertEqual([e+1 for e in range(12) if monitor_due(cfg, e)], [5, 10, 12])
        cfg.update(ss_eval_every_epochs=1, final_ss_evaluation=True)
        self.assertEqual([e+1 for e in range(12) if monitor_due(cfg, e)], list(range(1, 12)))
        cfg['ss_eval_every_epochs'] = 0
        self.assertFalse(any(monitor_due(cfg, e) for e in range(12)))

    def test_cli_override_applies_to_full_run_not_smoke(self):
        for extra in ([], ['--baseline']):
            result = subprocess.run([sys.executable, '-m', 'gcd_runtime.train_cli',
                                     '--dry-run', '--eval-every', '1']+extra,
                                    cwd=ROOT, capture_output=True, text=True, timeout=10, check=True)
            decoder, rows, remaining = json.JSONDecoder(), [], result.stdout.lstrip()
            while remaining:
                item, end = decoder.raw_decode(remaining)
                rows.append(item)
                remaining = remaining[end:].lstrip()
            self.assertEqual([r['config']['ss_eval_every_epochs'] for r in rows[1:]], [0, 1])
        result = subprocess.run([sys.executable, '-m', 'gcd_runtime.train_cli',
                                 '--dry-run', '--eval-every', '-1'], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)


class PeriodicTrainingTests(unittest.TestCase):
    def test_monitor_snapshots_and_real_child_do_not_change_training(self):
        # The child exercises real process/RNG isolation and report writing on a
        # synthetic metric fixture; real SS metric computation is tested below.
        original_run = subprocess.run
        calls = []
        child_source = '''
import json, random, sys
from pathlib import Path
import numpy as np
import torch
from gcd_runtime.checkpoint import EpochRuntime
config_path, directory, mode, checkpoint, epoch = sys.argv[1:]
config = json.loads(Path(config_path).read_text())
assert mode == 'monitor'
assert all(t.device.type == 'cpu' for t in torch.load(checkpoint, weights_only=True).values())
random.seed(999); np.random.seed(999); torch.manual_seed(999)
random.random(); np.random.rand(50); torch.rand(50)
EpochRuntime(directory, config, evaluation_epoch=int(epoch), evaluated_checkpoint=checkpoint).evaluation(
    dict(all_acc=.75, old_acc=1., new_acc=.5))
'''
        def child(command, **kwargs):
            calls.append((command, kwargs))
            return original_run([sys.executable, '-c', child_source]+command[3:], **kwargs)
        for method in ('baseline', 'crish'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as off, tempfile.TemporaryDirectory() as on:
                cfg = training_config(method)
                before = run_fixture(off, cfg)
                cfg.update(ss_eval_every_epochs=1, ss_eval_timeout_seconds=20)
                with patch('gcd_runtime.monitoring.subprocess.run', side_effect=child):
                    after = run_fixture(on, cfg)
                for a, b in zip(before, after):
                    for key in a:
                        self.assertTrue(torch.equal(a[key], b[key]), key)
                saved_off = torch.load(Path(off)/'resume.pt', weights_only=False)
                saved_on = torch.load(Path(on)/'resume.pt', weights_only=False)
                self.assertTrue(torch.equal(saved_off['torch_rng'], saved_on['torch_rng']))
                self.assertTrue(np.array_equal(saved_off['numpy_rng'][1], saved_on['numpy_rng'][1]))
                snapshot = torch.load(Path(on)/'backbone_monitor.pt', weights_only=True)
                for key in after[0]:
                    self.assertTrue(torch.equal(snapshot[key], after[0][key]))
                reports = [json.loads(line) for line in (Path(on)/'epoch_metrics.jsonl').read_text().splitlines()]
                self.assertEqual([r['epoch_number'] for r in reports], [1, 2])
                self.assertTrue(all(r['purpose'] == 'monitor_only' for r in reports))
                self.assertTrue(all(len(r['checkpoint_sha256']) == 64 for r in reports))
                self.assertFalse((Path(on)/'evaluation.json').exists())
                self.assertFalse(any(Path(on).glob('*best*')))
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(kwargs['timeout'] == 20 and kwargs['check'] for _, kwargs in calls))

    def test_monitor_timeout_or_failure_preserves_completed_epoch(self):
        for error in (subprocess.TimeoutExpired('fixture', 1), subprocess.CalledProcessError(2, 'fixture')):
            with self.subTest(error=type(error)), tempfile.TemporaryDirectory() as tmp:
                cfg = training_config('baseline')
                cfg['ss_eval_every_epochs'] = 1
                with patch('gcd_runtime.monitoring.subprocess.run', side_effect=error), \
                     self.assertRaises(type(error)):
                    run_fixture(tmp, cfg)
                self.assertEqual(torch.load(Path(tmp)/'resume.pt', weights_only=False)['epoch'], 0)
                self.assertTrue((Path(tmp)/'backbone_monitor.pt').is_file())
                self.assertFalse((Path(tmp)/'epoch_metrics.jsonl').exists())
                self.assertFalse((Path(tmp)/'backbone_final.pt').exists())

    def test_enabled_monitor_resume_matches_uninterrupted_training(self):
        def report(command, **kwargs):
            directory, epoch = Path(command[4]), int(command[7])
            path = directory/'evaluations'/f'epoch_{epoch+1:04d}.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text('{"status": "synthetic_fixture"}')
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as stopped, tempfile.TemporaryDirectory() as resumed:
            cfg = training_config()
            cfg['ss_eval_every_epochs'] = 1
            with patch('gcd_runtime.monitoring.subprocess.run', side_effect=report):
                expected = run_fixture(full, cfg)
                with self.assertRaises(InterruptedError):
                    run_fixture(stopped, cfg, stop=True)
                actual = run_fixture(resumed, cfg, seed=991, resume=str(Path(stopped)/'resume.pt'))
            for a, b in zip(expected, actual):
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_successful_child_without_report_is_a_visible_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = training_config('baseline')
            cfg['ss_eval_every_epochs'] = 1
            with patch('gcd_runtime.monitoring.subprocess.run'), \
                 self.assertRaisesRegex(RuntimeError, 'without an epoch report'):
                run_fixture(tmp, cfg)
            self.assertEqual(torch.load(Path(tmp)/'resume.pt', weights_only=False)['epoch'], 0)


class PeriodicMetricTests(unittest.TestCase):
    def test_actual_balanced_ss_metric_and_separate_final_report(self):
        trainer = configure_trainer()
        labels = torch.tensor([2, 0, 3, 1, 0, 1, 2, 3])
        images = torch.eye(4)[labels]
        membership = torch.tensor([[0], [1], [0], [1], [0], [0], [0], [0]])
        args = SimpleNamespace(train_classes=[0, 1], num_labeled_classes=2,
                               num_unlabeled_classes=2, unbalanced=False, eval_funcs=['v2'])
        with tempfile.TemporaryDirectory() as tmp:
            cfg = training_config('baseline')
            cfg.update(ss_eval_every_epochs=1, epochs=3)
            output = io.StringIO()
            for epoch in (0, None):
                runtime = EpochRuntime(tmp, cfg, evaluation_epoch=epoch)
                model = torch.nn.Identity()
                with patch.object(trainer, '_runtime', runtime, create=True), \
                     patch.object(trainer, 'device', torch.device('cpu'), create=True), \
                     runtime.evaluation_context(0 if epoch == 0 else 2, 'final_ss', model), \
                     contextlib.redirect_stdout(output):
                    trainer.evaluate(model, [(images, labels, torch.arange(8), membership)], args)
            monitor = json.loads((Path(tmp)/'evaluations/epoch_0001.json').read_text())
            final = json.loads((Path(tmp)/'evaluation.json').read_text())
            for row in (monitor, final):
                self.assertEqual((row['all_acc'], row['old_acc'], row['new_acc']), (1., 1., 1.))
                self.assertEqual((row['n_unlabeled'], row['n_old'], row['n_new']), (6, 2, 4))
                self.assertAlmostEqual(row['all_acc'], (row['n_old']*row['old_acc']+row['n_new']*row['new_acc'])/6)
                self.assertEqual(row['unit'], 'fraction')
                self.assertEqual(row['checkpoint_selection'], 'final_epoch')
            self.assertEqual(monitor['purpose'], 'monitor_only')
            self.assertEqual(final['purpose'], 'final')
            self.assertEqual(final['epoch_number'], 3)
            self.assertEqual(len((Path(tmp)/'epoch_metrics.jsonl').read_text().splitlines()), 2)
            self.assertIn('All=100.00% Old=100.00% New=100.00%', output.getvalue())

    def test_invalid_metrics_are_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = EpochRuntime(tmp, training_config(), evaluation_epoch=0)
            for invalid in (float('nan'), float('inf'), -1., 1.01):
                with self.assertRaises(ValueError):
                    runtime.evaluation(dict(all_acc=invalid, old_acc=.5, new_acc=.5))
            self.assertFalse((Path(tmp)/'epoch_metrics.jsonl').exists())

    def test_monitor_worker_routes_to_eval_only_with_epoch_metadata(self):
        from gcd_runtime import worker
        trainer = configure_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            args = ['worker', str(ROOT/'configs/crish.json'), tmp, 'monitor', 'fixture.pt', '4']
            with patch.object(sys, 'argv', args), patch.object(trainer, 'run') as run:
                worker.main()
            flags, runtime = run.call_args.args
            self.assertIn('--eval_only', flags)
            self.assertEqual(runtime.evaluation_epoch, 4)
            self.assertEqual(runtime.evaluated_checkpoint, 'fixture.pt')
            self.assertIsNone(runtime.resume)

    def test_saved_table_formats_fractions_and_keeps_missing_epochs_unknown(self):
        from scripts.show_metrics import show
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/'metrics.jsonl').write_text('{"epoch": 0, "loss": 2.0}\n{"epoch": 1, "loss": 1.0}\n')
            (directory/'epoch_metrics.jsonl').write_text(json.dumps(dict(epoch=1, all_acc=.75,
                old_acc=.9, new_acc=.6, status='synthetic_fixture', unit='fraction',
                protocol='balanced_ss_kmeans_train_unlabeled'))+'\n')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                show(directory)
            self.assertIn('75.00', output.getvalue())
            self.assertIn('90.00', output.getvalue())
            self.assertIn('60.00', output.getvalue())
            self.assertIn('not_evaluated', output.getvalue())
            self.assertIn('synthetic_fixture', output.getvalue())
