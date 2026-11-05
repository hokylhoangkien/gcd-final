"""Regression guard for the one-pipeline handoff and its readable runbook."""
import ast
import json
from pathlib import Path
import re
import unittest

from gcd_runtime.config import ROOT, load_config


class LayoutTests(unittest.TestCase):
    def test_main_source_has_no_retired_pipeline_or_fragmented_docs(self):
        for path in ('docs', 'legacy', 'gcd_runtime/reliability.py',
                     'gcd_runtime/method_config.py', 'gcd_runtime/compat.py',
                     'scripts/train_proposed_linux.sh', 'scripts/start_dinov2_linux.sh',
                     'configs/revision2', 'configs/final_only', 'configs/train'):
            self.assertFalse((ROOT/path).exists(), path)
        self.assertEqual({p.name for p in ROOT.glob('*.md')}, {'README.md', 'AGENTS.md'})
        self.assertEqual({p.name for p in (ROOT/'presentation').glob('*.md')}, {'CRISH_GCD.md'})
        profiles = set()
        for path in (ROOT/'configs').glob('*.json'):
            cfg = json.loads(path.read_text())
            if 'method' in cfg:
                profiles.add(load_config(path)['method'])
        self.assertEqual(profiles, {'crish', 'baseline'})

    def test_readme_local_links_and_single_train_command_exist(self):
        text = (ROOT/'README.md').read_text()
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in target:
                self.assertTrue((ROOT/target.split('#')[0]).is_file(), target)
        self.assertIn('bash scripts/train_crish.sh', text)
        self.assertIn('GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda', text)
        self.assertIn('devkit/', text)

    def test_public_trainer_routes_only_to_the_shared_crish_loop(self):
        tree = ast.parse((ROOT/'gcd_runtime/trainer.py').read_text())
        train = next(node for node in tree.body if getattr(node, 'name', None) == 'train')
        self.assertFalse(any(isinstance(node, (ast.If, ast.For, ast.While)) for node in ast.walk(train)))
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == 'train_strict' for node in ast.walk(train)))
