"""Secondary source Old-v1 selection with a fixed-final primary endpoint."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
from torch.utils.data import DataLoader

from gcd_runtime.config import ROOT, load_config
from gcd_runtime.launch import final_evaluations
from gcd_runtime.metrics import read_rows
from gcd_runtime.selection import load_evaluation_checkpoint
from selex_runtime.config import validate_config
from selex_runtime.selection import BEST_NAME, read_best, restore_artifacts
from selex_runtime.trainer import SelExRuntime
from selex_runtime.rng import capture_rng, restore_rng, rng_digest


def config():
    cfg = load_config(ROOT/'configs/selex_scars_baseline_200.json')
    cfg.update(device='cpu', synthetic_fixture=True, epochs=3)
    return cfg


def report(directory, cfg, epoch, old_v1, old_v2, model=None, *, pool='test'):
    model = torch.nn.Linear(2, 2) if model is None else model
    path = Path(directory)/'backbone_monitor.pt'
    torch.save(model.state_dict(), path)
    runtime = SelExRuntime(directory, cfg, evaluation_epoch=epoch,
                          evaluated_checkpoint=path, evaluation_kind='test')
    runtime.evaluation(dict(all_acc=.5, old_acc=old_v2, new_acc=.5,
        metrics_v1=dict(all_acc=.5, old_acc=old_v1, new_acc=.5), evaluation_pool=pool))


class SelExSelectionTests(unittest.TestCase):
    def test_selection_does_not_change_sgd_and_pending_resume_matches_full_run(self):
        from tests.test_selex_runtime import fixture
        from crish.training import train_strict
        from selex_runtime.objective import selex_loss
        def execute(directory, *, reporting=True, resume=None, fail=False):
            np.random.seed(77)
            dataset, state, args, head = fixture()
            args.epochs = 2
            cfg = config()
            cfg.update(epochs=2, report_best_old_test=reporting, mlp_out_dim=16,
                       max_steps_per_epoch=1, final_ss_evaluation=False)
            runtime = SelExRuntime(directory, cfg, resume)
            model = torch.nn.Linear(4, 4)
            loader = DataLoader(dataset, batch_size=16, shuffle=False)
            def extract(model, loader, args): runtime.hierarchy_state = state
            def monitor(runtime, epoch, model):
                report(runtime.run_dir, cfg, epoch, .8-epoch*.1, .2+epoch*.3, model)
                if fail and epoch == 0:
                    raise TimeoutError('fixture failure after selection before RNG commit')
                # Simulate only the source evaluation draws, identically on/off.
                torch.rand(2); np.random.rand(2)
                runtime.plain_rng_pending = False
            with patch('selex_runtime.monitoring.monitor_plain', side_effect=monitor):
                train_strict(head, model, loader, loader, args, runtime,
                             objective=selex_loss, extractor=extract)
            return model.state_dict(), head.state_dict(), rng_digest()
        state = capture_rng()
        try:
            with tempfile.TemporaryDirectory() as off, tempfile.TemporaryDirectory() as on, \
                 tempfile.TemporaryDirectory() as interrupted, tempfile.TemporaryDirectory() as resumed:
                expected = execute(off, reporting=False)
                actual = execute(on)
                with self.assertRaises(TimeoutError): execute(interrupted, fail=True)
                recovered = execute(resumed, resume=str(Path(interrupted)/'resume.pt'))
                for candidate in (actual, recovered):
                    for a, b in zip(expected[:2], candidate[:2]):
                        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
                    self.assertEqual(expected[2], candidate[2])
                resume_config = torch.load(Path(resumed)/'resume.pt', weights_only=False)['config']
                self.assertEqual(read_best(Path(resumed)/BEST_NAME, resume_config)['selection']['epoch'], 0)
        finally:
            restore_rng(state)

    def test_new_profiles_enable_both_reports_and_preserve_fixed_last(self):
        for dataset in ('scars', 'cub', 'aircraft', 'cifar100'):
            for arm in ('baseline', 'crish'):
                cfg = load_config(ROOT/f'configs/selex_{dataset}_{arm}_200.json')
                self.assertTrue(cfg['report_best_old_test'])
                self.assertEqual(cfg['checkpoint_selection'], 'final_epoch')
        cfg = config()
        for value in (1, 'true'):
            with self.assertRaises(ValueError):
                validate_config(dict(cfg, report_best_old_test=value))
        with self.assertRaisesRegex(ValueError, 'upstream'):
            validate_config(dict(cfg, evaluation_rng='isolated'))

    def test_old_v1_not_old_v2_selects_earliest_tie_and_zero_fallback(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            for epoch, (v1, v2) in enumerate(((.8, .2), (.7, .99), (.8, .9))):
                with torch.no_grad(): model.weight.fill_(epoch)
                report(tmp, cfg, epoch, v1, v2, model)
            selected = read_best(Path(tmp)/BEST_NAME, cfg)
            self.assertEqual(selected['selection']['epoch'], 0)
            self.assertEqual(selected['selection']['metrics_v1']['old_acc'], .8)
            self.assertTrue(torch.equal(selected['model']['weight'], torch.zeros_like(model.weight)))
            rows = read_rows(Path(tmp)/'test_metrics.jsonl')
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['status']=='synthetic_fixture' for r in rows))
            self.assertTrue(all(r['used_for_checkpoint_selection'] for r in rows))
            self.assertFalse((Path(tmp)/'evaluation.json').exists())
        with tempfile.TemporaryDirectory() as tmp:
            report(tmp, cfg, 0, 0., 0.)
            self.assertEqual(read_best(Path(tmp)/BEST_NAME, cfg)['selection']['epoch'], 0)

    def test_train_plain_cannot_select_and_replay_preserves_rng_and_history(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            state = capture_rng()
            try:
                before = rng_digest()
                report(tmp, cfg, 0, .99, .99, model, pool='train_unlabeled')
                self.assertFalse((Path(tmp)/BEST_NAME).exists())
                report(tmp, cfg, 0, .6, .1, model)
                report(tmp, cfg, 0, .6, .1, model)
                self.assertEqual(rng_digest(), before)
                self.assertEqual(len(read_rows(Path(tmp)/'test_metrics.jsonl')), 1)
                with torch.no_grad(): model.weight.add_(1)
                with self.assertRaisesRegex(ValueError, 'Different weights'):
                    report(tmp, cfg, 0, .6, .1, model)
            finally:
                restore_rng(state)

    def test_best_and_last_evaluations_keep_primary_and_provenance_separate(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report(tmp, cfg, 0, .9, .2)
            torch.save({}, directory/'backbone_final.pt')
            def evaluate(command, **kwargs):
                checkpoint = Path(command[-1])
                runtime = SelExRuntime(tmp, cfg, evaluated_checkpoint=checkpoint)
                load_evaluation_checkpoint(checkpoint, cfg, runtime)
                best = checkpoint.name == BEST_NAME
                runtime.evaluation(dict(all_acc=.99 if best else .6, old_acc=.8, new_acc=.5))
            with patch('gcd_runtime.launch.subprocess.run', side_effect=evaluate) as worker:
                commands = final_evaluations(directory/'config.json', directory, cfg, {}, None)
            self.assertEqual([item['role'] for item in commands], ['last', 'best_old_test_v1'])
            self.assertEqual(worker.call_count, 2)
            primary = json.loads((directory/'evaluation.json').read_text())
            best = json.loads((directory/'evaluation_best_old_test.json').read_text())
            self.assertEqual((primary['purpose'], primary['epoch'], primary['all_acc']), ('final', 2, .6))
            self.assertFalse(primary['selected_using_evaluation_labels'])
            self.assertFalse(primary['checkpoint_selection_uses_evaluation_labels'])
            self.assertEqual((best['purpose'], best['epoch']), ('best_old_test', 0))
            self.assertTrue(best['selected_using_evaluation_labels'])
            self.assertEqual(best['matching'], 'hungarian_v2_global')
            self.assertEqual(best['selection_matching'], 'hungarian_v1_separate')
            self.assertEqual(len(read_rows(directory/'epoch_metrics.jsonl')), 1)
            self.assertEqual(len(read_rows(directory/'best_old_ss_metrics.jsonl')), 1)

    def test_resume_restores_best_history_and_rejects_missing_or_ahead_artifacts(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            report(source, cfg, 0, .8, .2)
            report(source, cfg, 1, .7, .99)
            restore_artifacts(source, target, cfg, 1, pending=False)
            self.assertEqual(read_best(Path(target)/BEST_NAME, cfg)['selection']['epoch'], 0)
            self.assertEqual(len(read_rows(Path(target)/'test_metrics.jsonl')), 2)
            with self.assertRaisesRegex(ValueError, 'exact source/config'):
                read_best(Path(target)/BEST_NAME, dict(cfg, seed=17))
            with self.assertRaisesRegex(ValueError, 'original test_evaluations'):
                restore_artifacts(source, target, cfg, 2, pending=False)
            report(source, cfg, 2, .95, .1)
            with self.assertRaisesRegex(ValueError, 'ahead'):
                restore_artifacts(source, target, cfg, 1, pending=False)
            (Path(source)/BEST_NAME).unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                restore_artifacts(source, target, cfg, 2, pending=False)

    def test_pending_final_resume_replays_selection_then_exports_without_sgd(self):
        cfg = config()
        cfg['epochs'] = 1
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            model, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
            runtime = SelExRuntime(source, cfg)
            # Worker already selected but RNG handoff failed; replay is idempotent.
            report(source, cfg, 0, .8, .1, model)
            runtime.plain_rng_pending = True
            runtime.save(0, model, head, optimizer, scheduler)
            resumed = SelExRuntime(target, cfg, str(Path(source)/'resume.pt'))
            def recover(runtime, epoch, model):
                report(runtime.run_dir, cfg, epoch, .8, .1, model)
                runtime.plain_rng_pending = False
            with patch('selex_runtime.monitoring.monitor_plain', side_effect=recover) as monitor:
                self.assertEqual(resumed.restore(model, head, optimizer, scheduler), 1)
                monitor.assert_called_once()
            self.assertTrue((Path(target)/'backbone_final.pt').is_file())
            self.assertTrue((Path(target)/BEST_NAME).is_file())
            self.assertFalse(torch.load(Path(target)/'resume.pt', weights_only=False)['plain_rng_pending'])

    def test_missing_secondary_best_fails_instead_of_substituting_last(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            torch.save({}, directory/'backbone_final.pt')
            def evaluate(command, **kwargs):
                SelExRuntime(tmp, cfg).evaluation(dict(all_acc=.5, old_acc=.5, new_acc=.5))
            with patch('gcd_runtime.launch.subprocess.run', side_effect=evaluate), \
                 self.assertRaisesRegex(RuntimeError, 'Missing best_old_test_v1'):
                final_evaluations(directory/'config.json', directory, cfg, {}, None)
            self.assertFalse((directory/'evaluation_best_old_test.json').exists())


if __name__ == '__main__':
    unittest.main()
