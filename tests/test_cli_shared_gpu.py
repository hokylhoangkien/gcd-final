"""Shared GPU launch is allowed while preflight, smoke and error handling remain."""
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from gcd_runtime import train_cli
from gcd_runtime.config import ROOT
from selex_runtime import cli as selex_cli


class SharedGpuCliTests(unittest.TestCase):
    def invoke(self, module, arguments, *, system="Linux", fail=False):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[0] == "nvidia-smi":
                # Simulated occupied GPU: querying it must not prevent launch.
                return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")
            self.assertEqual(command[1:3], ["-m", "gcd_runtime.launch"])
            self.assertTrue(kwargs["check"])
            self.assertGreater(kwargs["timeout"], 0)
            if fail:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            if module is selex_cli:
                stack.enter_context(patch.object(module, "ROOT", Path(tmp)))
                arguments = [*arguments, "--config", str(ROOT / "configs/selex_crish.json")]
            stack.enter_context(patch("sys.argv", ["cli", *arguments]))
            stack.enter_context(patch.object(module.platform, "system", return_value=system))
            stack.enter_context(patch.object(module.subprocess, "run", side_effect=run))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                module.main()
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                return calls, exc
        return calls, None

    def test_fresh_train_reaches_smoke_then_full_without_idle_gpu_query(self):
        for module in (selex_cli, train_cli):
            with self.subTest(entrypoint=module.__name__):
                calls, error = self.invoke(module, ["train"])
                self.assertIsNone(error)
                self.assertEqual([c[3] for c in calls], ["train", "train"])
                self.assertIn("smoke", calls[0][calls[0].index("--config") + 1])
                self.assertNotIn("--dry-run", calls[0])
                self.assertNotIn("--dry-run", calls[1])

    def test_check_still_launches_real_preflight(self):
        for module in (selex_cli, train_cli):
            with self.subTest(entrypoint=module.__name__):
                calls, error = self.invoke(module, ["check"])
                self.assertIsNone(error)
                self.assertEqual([c[3] for c in calls], ["preflight"])
                self.assertNotIn("--dry-run", calls[0])

    def test_resume_keeps_checkpoint_and_skips_fresh_smoke(self):
        for module in (selex_cli, train_cli):
            with self.subTest(entrypoint=module.__name__):
                calls, error = self.invoke(module, ["train", "--resume", "trusted/resume.pt"])
                self.assertIsNone(error)
                self.assertEqual([c[3] for c in calls], ["train"])
                self.assertEqual(calls[0][-2:], ["--checkpoint", "trusted/resume.pt"])

    def test_failed_smoke_or_preflight_prevents_following_stage(self):
        for module in (selex_cli, train_cli):
            for mode in ("train", "check"):
                with self.subTest(entrypoint=module.__name__, mode=mode):
                    calls, error = self.invoke(module, [mode], fail=True)
                    self.assertIsInstance(error, subprocess.CalledProcessError)
                    self.assertEqual(len(calls), 1)

    def test_mac_full_runs_stay_blocked(self):
        for module in (selex_cli, train_cli):
            with self.subTest(entrypoint=module.__name__):
                calls, error = self.invoke(module, ["train"], system="Darwin")
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(calls, [])

    def test_dry_run_remains_available_on_mac(self):
        for module in (selex_cli, train_cli):
            with self.subTest(entrypoint=module.__name__):
                calls, error = self.invoke(module, ["train", "--dry-run"], system="Darwin")
                self.assertIsNone(error)
                self.assertEqual(len(calls), 2)
                self.assertTrue(all("--dry-run" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
