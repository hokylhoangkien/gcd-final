"""Configuration and checkpoint routing; CUDA/real weights require remote gate."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.readiness import inspect_inputs

class DinoV2ConfigTests(unittest.TestCase):
    def test_profiles_keep_optimization_matched(self):
        for method in ('baseline', 'proposed'):
            v1 = load_config(ROOT / f'configs/cars_{method}_dino_v1.json')
            v2 = load_config(ROOT / f'configs/cars_{method}_dino_v2.json')
            self.assertEqual(v2.pop('dino'), 'v2')
            v1.pop('dino')
            self.assertEqual(v1, v2)

    def test_gate_requires_v2_weights_even_when_v1_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'cars/devkit').mkdir(parents=True)
            for name in ('cars_train_annos.mat', 'cars_test_annos_withlabels.mat'):
                (root / 'cars/devkit' / name).touch()
            (root / 'dino_vitb16_pretrain.pth').touch()
            with patch('gcd_runtime.readiness.verify_upstream', return_value={'sha':'fixture', 'kind':'fixture'}), patch('gcd_runtime.readiness.check_split', return_value={}), patch('gcd_runtime.readiness.paths', return_value=(root,root,root)):
                with self.assertRaisesRegex(RuntimeError, 'dinov2_vitb14_reg4_pretrain.pth'):
                    inspect_inputs({'dino':'v2'})
