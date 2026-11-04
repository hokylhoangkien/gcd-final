"""Require the actual DINOv2 register checkpoint; never use a v1 substitute."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gcd_runtime.readiness import inspect_inputs


class DinoV2ReadinessTests(unittest.TestCase):
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
