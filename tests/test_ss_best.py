"""SS checkpoint selection uses paired scores and exact evaluated weights."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.readiness import sha256
from gcd_runtime.ss_best import PROTOCOL, restore_best, update_best
from tests.test_crish_training import training_config


class SsBestTests(unittest.TestCase):
    def test_selex_resume_replays_missing_ss_before_next_sgd(self):
        from selex_runtime.trainer import SelExRuntime
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(training_config('baseline'), ss_eval_every_epochs=1, epochs=3)
            runtime = SelExRuntime(tmp, cfg, resume=str(Path(tmp)/'resume.pt'))
            runtime.restored_epoch = 0
            runtime.restored_plain_rng_pending = False
            model = torch.nn.Linear(1, 1)
            with patch.object(EpochRuntime, 'restore', return_value=1), \
                 patch('gcd_runtime.monitoring.monitor_epoch') as monitor:
                self.assertEqual(runtime.restore(model, None, None, None), 1)
                monitor.assert_called_once_with(runtime, 0, model)
            report = Path(tmp)/'evaluations/epoch_0001.json'
            report.parent.mkdir()
            report.write_text('{}')
            with patch.object(EpochRuntime, 'restore', return_value=1), \
                 patch('gcd_runtime.monitoring.monitor_epoch') as monitor:
                runtime.restore(model, None, None, None)
                monitor.assert_not_called()

    def emit(self, directory, cfg, epoch, scores, role='monitor_only'):
        checkpoint = Path(directory)/'backbone_monitor.pt'
        torch.save({'weight': torch.tensor([epoch])}, checkpoint)
        runtime = EpochRuntime(directory, cfg, evaluation_epoch=epoch, evaluated_checkpoint=checkpoint)
        runtime.evaluation_role = role
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runtime.evaluation(dict(zip(('all_acc', 'old_acc', 'new_acc'), scores)))
        return output.getvalue()

    def test_independent_maxima_paired_metrics_weights_ties_and_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(training_config('baseline'), ss_eval_every_epochs=1, epochs=4)
            observations = [(.6, .9, .4), (.7, .8, .6), (.65, .7, .95)]
            for epoch, scores in enumerate(observations):
                text = self.emit(tmp, cfg, epoch, scores)
                self.assertIn('SS_BEST metric=all', text)
            index = json.loads((Path(tmp)/'ss_best.json').read_text())
            for metric, epoch in dict(all=1, old=0, new=2).items():
                best = index['best'][metric]
                self.assertEqual(best['epoch_number'], epoch+1)
                self.assertEqual(tuple(best[k] for k in ('all_acc', 'old_acc', 'new_acc')), observations[epoch])
                self.assertTrue(best['selected_using_evaluation_labels'])
                state = torch.load(best['checkpoint'], weights_only=True)
                self.assertEqual(state['weight'].item(), epoch)
                self.assertEqual(sha256(Path(best['checkpoint'])), best['checkpoint_sha256'])
            self.emit(tmp, cfg, 3, (.7, .8, .95), role='final')
            tied = json.loads((Path(tmp)/'ss_best.json').read_text())
            self.assertEqual(tied, index)
            self.emit(tmp, cfg, 3, (.8, .8, .95), role='final')
            final = json.loads((Path(tmp)/'ss_best.json').read_text())
            self.assertEqual(final['best']['all']['epoch'], 3)
            summary = json.loads((Path(tmp)/'results_summary.json').read_text())
            self.assertEqual(summary['ss_best_checkpoint_selection'], final)

    def test_best_old_test_is_not_an_extra_ss_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(training_config('baseline'), ss_eval_every_epochs=1)
            self.emit(tmp, cfg, 0, (.5, .6, .4))
            self.emit(tmp, cfg, 1, (.99, .99, .99), role='best_old_test')
            best = json.loads((Path(tmp)/'ss_best.json').read_text())
            self.assertTrue(all(r['epoch'] == 0 for r in best['best'].values()))

    def test_disabled_monitoring_does_not_select_ss_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.emit(tmp, training_config('baseline'), 0, (.5, .6, .4))
            self.assertFalse((Path(tmp)/'ss_best.json').exists())

    def test_hash_or_metric_failure_preserves_previous_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(training_config('baseline'), ss_eval_every_epochs=1)
            self.emit(tmp, cfg, 0, (.5, .6, .4))
            index = Path(tmp)/'ss_best.json'
            before = index.read_bytes()
            record = json.loads((Path(tmp)/'evaluations/epoch_0001.json').read_text())
            record.update(all_acc=.99, checkpoint_sha256='bad')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                update_best(tmp, cfg, record)
            record.update(all_acc=float('nan'))
            with self.assertRaisesRegex(ValueError, 'finite'):
                update_best(tmp, cfg, record)
            self.assertEqual(before, index.read_bytes())

    def test_resume_restores_weights_history_and_maxima(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            cfg = dict(training_config('baseline'), ss_eval_every_epochs=1)
            self.emit(src, cfg, 0, (.7, .8, .6))
            restore_best(src, dst, cfg, 0)
            self.emit(dst, cfg, 1, (.6, .7, .5))
            best = json.loads((Path(dst)/'ss_best.json').read_text())
            for entry in best['best'].values():
                self.assertEqual(entry['epoch'], 0)
                self.assertEqual(Path(entry['checkpoint']).parent, Path(dst))
                self.assertEqual(sha256(Path(entry['checkpoint'])), entry['checkpoint_sha256'])
            rows = (Path(dst)/'epoch_metrics.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertTrue((Path(dst)/'evaluations/epoch_0001.json').is_file())
            with self.assertRaisesRegex(ValueError, 'ahead'):
                restore_best(src, dst, cfg, -1)

    def test_resume_refuses_missing_best_weights(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            cfg = dict(training_config('baseline'), ss_eval_every_epochs=1)
            self.emit(src, cfg, 0, (.7, .8, .6))
            (Path(src)/'backbone_best_ss_all.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                restore_best(src, dst, cfg, 0)
