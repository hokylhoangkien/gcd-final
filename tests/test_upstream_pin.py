"""Real Git fixtures: Python caches do not invalidate an unchanged source pin."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from gcd_runtime.readiness import verify_upstream


class UpstreamPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.upstream = self.root/'upstream/HypCD'
        self.upstream.mkdir(parents=True)
        self.git('init', '-q')
        (self.upstream/'source.py').write_text('value = 1\n')
        self.git('add', 'source.py')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit', '-qm', 'fixture')
        self.pin = self.git('rev-parse', 'HEAD').strip()
        for name, value in (('ROOT', self.root), ('UPSTREAM_SHA', self.pin)):
            patched = patch('gcd_runtime.readiness.'+name, value)
            patched.start()
            self.addCleanup(patched.stop)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.upstream), *args],
                                       text=True, timeout=10)

    def test_generated_untracked_bytecode_is_accepted_without_deletion(self):
        import py_compile
        folder = self.upstream/'data with spaces'
        folder.mkdir()
        # Compile an existing tracked source into an untracked cache folder.
        cache = folder/'__pycache__'
        cache.mkdir()
        output = cache/'source.cpython-39.pyc'
        py_compile.compile(str(self.upstream/'source.py'), cfile=str(output), doraise=True)
        self.assertEqual(verify_upstream(), {'sha':self.pin, 'kind':'git_submodule'})
        self.assertTrue(output.is_file())

    def test_modified_staged_and_deleted_tracked_source_are_rejected(self):
        source = self.upstream/'source.py'
        source.write_text('value = 2\n')
        with self.assertRaisesRegex(RuntimeError, 'unexpected changes'):
            verify_upstream()
        self.git('add', 'source.py')
        with self.assertRaises(RuntimeError):
            verify_upstream()
        source.unlink()
        with self.assertRaises(RuntimeError):
            verify_upstream()

    def test_untracked_source_even_inside_cache_and_symlinks_are_rejected(self):
        cache = self.upstream/'__pycache__'
        cache.mkdir()
        unexpected = cache/'injected.py'
        unexpected.write_text('value = 2\n')
        with self.assertRaises(RuntimeError):
            verify_upstream()
        unexpected.unlink()
        (cache/'linked.pyc').symlink_to(self.upstream/'source.py')
        with self.assertRaises(RuntimeError):
            verify_upstream()

    def test_wrong_commit_is_still_rejected(self):
        with patch('gcd_runtime.readiness.UPSTREAM_SHA', '0'*40):
            with self.assertRaisesRegex(RuntimeError, 'HEAD='):
                verify_upstream()


if __name__ == '__main__':
    unittest.main()
