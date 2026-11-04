import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from gcd_runtime.config import ROOT, load_config


class RuntimeTests(unittest.TestCase):
    def test_explicit_trainer_preserves_reference_losses_without_exec(self):
        source = (ROOT / 'upstream/HypCD/train/train_HypSelEx.py').read_text()
        old, new = ast.parse(source), ast.parse((ROOT/'gcd_runtime/trainer.py').read_text())
        for name in ('LabelSmoothingLoss', 'SupConLoss', 'info_nce_logits'):
            before = next(n for n in old.body if getattr(n, 'name', None) == name)
            after = next(n for n in new.body if getattr(n, 'name', None) == name)
            self.assertEqual(ast.dump(before), ast.dump(after))
        production = ['worker.py','probe.py','launch.py','trainer.py']
        for name in production:
            tree=ast.parse((ROOT/'gcd_runtime'/name).read_text())
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('exec','compile') for n in ast.walk(tree)))

    def test_config_rejects_unsupported_method(self):
        cfg = load_config(ROOT / 'configs/baseline.json')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            for key, value in [('method', 'unknown'), ('device', 'bad'), ('epochs', 0),
                               ('seed', -1), ('dino', 'invalid'), ('checkpoint_selection', 'test_best'),
                               ('training_diagnostics_every', -1), ('training_diagnostics_every', True),
                               ('test_every_epochs', -1), ('plain_train_eval_every', True),
                               ('evaluation_rng', 'bad'), ('final_ss_evaluation', 'yes'),
                               ('test_every_epochs', 1), ('early_stopping', True)]:
                changed = dict(cfg, **{key: value})
                path.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_config(path)

    def test_launcher_dry_run_without_data(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-m', 'gcd_runtime.launch', 'train', '--dry-run'],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'dry_run')

    def test_atomic_epoch_resume_rng_and_next_update(self):
        try:
            import torch
            import numpy as np
        except ImportError:
            self.skipTest('Install local torch and numpy to exercise checkpoint')
        from gcd_runtime.checkpoint import EpochRuntime
        import random
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.2))
        head = torch.nn.Linear(3, 1)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)

        def step(m, h, opt, sched):
            opt.zero_grad()
            h(m(torch.randn(4, 3))).square().mean().backward()
            opt.step()
            sched.step()

        step(model, head, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = EpochRuntime(tmp, {'epochs': 3, 'synthetic_fixture': True})
            runtime.save(0, model, head, optimizer, scheduler)
            expected_random = (random.random(), np.random.random())
            step(model, head, optimizer, scheduler)
            expected = copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict())
            model2 = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.2))
            head2 = torch.nn.Linear(3, 1)
            opt2 = torch.optim.SGD(list(model2.parameters()) + list(head2.parameters()), lr=0.1, momentum=0.9)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 3)
            resumed = EpochRuntime(tmp, {'epochs': 3, 'synthetic_fixture': True}, str(Path(tmp) / 'resume.pt'))
            self.assertEqual(resumed.restore(model2, head2, opt2, sched2), 1)
            self.assertEqual((random.random(), np.random.random()), expected_random)
            step(model2, head2, opt2, sched2)
            for expected_state, actual in zip(expected, [model2.state_dict(), head2.state_dict()]):
                for key, tensor in expected_state.items():
                    self.assertTrue(torch.equal(tensor, actual[key]))
            with self.assertRaises(ValueError):
                EpochRuntime(tmp, {'epochs': 4, 'synthetic_fixture': True}, str(Path(tmp) / 'resume.pt')).restore(model2, head2, opt2, sched2)

    def test_baseline_loss_and_hungarian_cpu(self):
        try:
            import torchvision
            import sklearn
            import tensorboard
        except ImportError:
            self.skipTest('Install common dependencies for upstream import probe')
        from gcd_runtime.probe import tensor_probe
        report = tensor_probe('cpu', baseline=True)
        self.assertTrue(report['baseline_probe'])
        self.assertFalse(report['cuda_verified'])

    def test_synthetic_geometry_cpu(self):
        try:
            import torch
            import scipy
        except ImportError:
            self.skipTest('Install local torch/scipy for geometry smoke')
        from gcd_runtime.probe import tensor_probe
        report = tensor_probe('cpu')
        self.assertEqual(report['device'], 'cpu')
        self.assertFalse(report['cuda_verified'])


if __name__ == '__main__':
    unittest.main()
