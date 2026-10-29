import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from gcd_runtime.config import ROOT


class ReleaseTests(unittest.TestCase):
    def test_release_contains_source_without_private_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'scripts/build_release.py','--output',tmp],
                                  cwd=ROOT,capture_output=True,text=True,timeout=20,check=True)
            archive_path=Path(json.loads(result.stdout)['archive'])
            with tarfile.open(archive_path) as archive:
                names=[member.name for member in archive]
                self.assertTrue(any(name.endswith('/gcd_runtime/trainer.py') for name in names))
                self.assertTrue(any(name.endswith('/upstream/HypCD.SNAPSHOT.json') for name in names))
                for name in names:
                    parts=Path(name).parts
                    self.assertFalse(any(p in parts for p in ('.git','.venv','logs','weights','datasets','runs','legacy','__pycache__')))
                    self.assertNotIn(Path(name).suffix,('.pt','.pth','.ckpt'))
                # Extract only this tool's known regular-file release into a temp dir.
                for member in archive:
                    target=Path(tmp)/member.name
                    self.assertTrue(target.resolve().is_relative_to(Path(tmp).resolve()))
                    self.assertTrue(member.isfile())
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
                root=Path(tmp)/names[0].split('/')[0]
                result=subprocess.run([sys.executable,'scripts/bootstrap_upstream.py'],cwd=root,
                                      capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,0,result.stderr)
                result=subprocess.run([sys.executable,'-m','gcd_runtime.launch','train','--config',
                                       'configs/cars_proposed_dino_v1.json','--dry-run'],cwd=root,
                                      capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,0,result.stderr)
