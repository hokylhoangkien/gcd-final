"""Check the actual handoff archive, including execution outside the repository."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from gcd_runtime.config import ROOT


class ReleaseTests(unittest.TestCase):
    def test_clean_archive_is_complete_and_runs_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, 'scripts/build_release.py', '--output', tmp],
                                    cwd=ROOT, capture_output=True, text=True, timeout=20, check=True)
            archive_path = Path(json.loads(result.stdout)['archive'])
            with tarfile.open(archive_path) as archive:
                names = [member.name for member in archive]
                for member in archive:
                    target = Path(tmp)/member.name
                    self.assertTrue(target.resolve().is_relative_to(Path(tmp).resolve()))
                    self.assertTrue(member.isfile())
                    self.assertFalse(any(p in Path(member.name).parts for p in
                        ('.git', '.venv', '.archive', 'logs', 'weights', 'datasets', 'runs',
                         'legacy', 'docs', 'presentation', '__pycache__')))
                    self.assertNotIn(target.suffix, ('.pt', '.pth', '.ckpt'))
                    if target.suffix == '.pdf':
                        self.assertEqual(Path(member.name).parts[-2:], ('paper', 'manuscript.pdf'))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
            root = Path(tmp)/names[0].split('/')[0]
            manifest = json.loads((root/'release_manifest.json').read_text())
            self.assertEqual(manifest['methods'], ['crish', 'baseline'])
            for path, digest in manifest['files'].items():
                self.assertEqual(hashlib.sha256((root/path).read_bytes()).hexdigest(), digest, path)
            for path in ('README.md', 'AGENTS.md', 'NOTICE', 'reports/04_method_and_research_plan.md',
                         'reports/01_baseline_protocol.md', 'paper/manuscript.tex', 'paper/manuscript.pdf',
                         'paper/manuscript.build.json',
                         'crish/training.py', 'crish/rebuild.py',
                         'audit/baseline_trace.py', 'upstream/HypCD.SNAPSHOT.json'):
                self.assertTrue((root/path).is_file(), path)
            for path in ('gcd_runtime/reliability.py', 'gcd_runtime/method_config.py', 'configs/train'):
                self.assertFalse((root/path).exists(), path)
            result = subprocess.run([sys.executable, 'scripts/bootstrap_upstream.py'], cwd=root,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            # Bounded subset avoids recursively testing archive construction.
            config = json.loads((root/'configs/local_smoke.json').read_text())
            config['tests'] = 'tests/crish_v21'
            target = root/'configs/release_smoke.json'
            target.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, 'scripts/run_smoke.py', '--config', str(target)],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(json.loads((root/'results/local_validation.json').read_text())['status'], 'passed')
            for flag in ([], ['--baseline']):
                result = subprocess.run([sys.executable, '-m', 'gcd_runtime.train_cli', '--dry-run']+flag,
                                        cwd=root, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                method = 'baseline' if flag else 'crish'
                self.assertIn('"selected_method": "'+method+'"', result.stdout)
