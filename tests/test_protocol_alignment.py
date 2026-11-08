"""Check actual launch arguments against the independent pinned upstream command."""
import unittest

from audit.protocol import compare_worker, upstream_reference
from gcd_runtime.config import ROOT, load_config


class CarsV2ProtocolTests(unittest.TestCase):
    def test_full_and_smoke_profiles_follow_source_command_not_python_default(self):
        expected, source = upstream_reference()
        self.assertEqual(expected['hyper_max_weight'], .5)
        self.assertEqual(expected['unsupervised_smoothing'], .5)
        self.assertEqual(expected['c'], .1)
        self.assertEqual(expected['cr'], 2.)
        for method in ('baseline', 'crish'):
            for suffix in ('', '_smoke'):
                cfg = load_config(ROOT/f'configs/{method}{suffix}.json')
                report = compare_worker(cfg, smoke=bool(suffix))
                with self.subTest(method=method, suffix=suffix):
                    self.assertEqual(report['differences'], {})
                    self.assertTrue(report['cli_hyperparameters_match'])
                    self.assertFalse(report['exact_paper_reproduction_verified'])
                    self.assertEqual(report['effective_reference_settings']['hyper_max_weight'], .5)
                    self.assertEqual(report['script_sha256'], source['script_sha256'])
                    self.assertEqual(set(report['smoke_overrides']), {'epochs'} if suffix else set())

    def test_checker_catches_material_hyperparameter_drift(self):
        cfg = load_config(ROOT/'configs/crish.json')
        for key, value in [('hyper_max_weight', 1.), ('grad_from_block', 9), ('c', .05),
                           ('unsupervised_smoothing', 1.), ('dino', 'v1')]:
            with self.subTest(key=key):
                report = compare_worker(dict(cfg, **{key: value}))
                self.assertFalse(report['cli_hyperparameters_match'])
                self.assertEqual(set(report['differences']), {key})
                self.assertEqual(report['differences'][key]['actual'], value)

    def test_smoke_exemption_does_not_hide_loss_changes(self):
        cfg = load_config(ROOT/'configs/crish_smoke.json')
        self.assertIn('epochs', compare_worker(cfg)['differences'])
        report = compare_worker(dict(cfg, hyper_max_weight=1.), smoke=True)
        self.assertEqual(set(report['differences']), {'hyper_max_weight'})
        self.assertEqual(report['smoke_overrides']['epochs'], dict(expected=200, actual=1))
