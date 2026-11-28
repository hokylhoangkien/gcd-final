"""Focused operations checks without launching a production epoch."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from crish.experiment import file_digest
from relational.export import verify
from relational.state import write_json

from .ablation import PRODUCTION, run
from .ablation_benchmark import PURPOSE, verified_snapshot
from .ablation_ops import archive_dir, checkpoint_receipt, drive, publish


def committed_fixture(root, epoch):
    folder = root / "run"
    folder.mkdir(exist_ok=True)
    identity = {"purpose": PRODUCTION}
    write_json(folder / "config.json", identity)
    journal = folder / "matched.jsonl"
    journal.write_text(
        "".join(
            json.dumps({"epoch": e, "partial": False, "accuracy": {}, "arms": {}})
            + "\n"
            for e in range(1, epoch + 1)
        ),
        encoding="utf8",
    )
    torch.save(
        {
            "identity": identity,
            "epoch": epoch,
            "journal_bytes": journal.stat().st_size,
            "checkpoint_tensor": torch.tensor([epoch]),
        },
        folder / "recovery.pt",
    )
    return folder


class OperationsChecks(unittest.TestCase):
    def test_benchmark_cannot_be_used_by_production_or_load_wrong_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "run"
            folder.mkdir()
            write_json(folder / "config.json", {"purpose": PURPOSE})
            clip = root / "clip"
            with patch("dual.ablation.build") as build:
                with self.assertRaisesRegex(ValueError, "reserved"):
                    run(folder, 50, clip)
                build.assert_not_called()
            with self.assertRaisesRegex(ValueError, "benchmark"):
                checkpoint_receipt(folder)
            with self.assertRaisesRegex(ValueError, "dedicated run root"):
                drive(folder, clip, 50, micro=False)
            proof, snapshot = root / "proof.json", root / "snapshot.pt"
            snapshot.write_bytes(b"untrusted snapshot")
            write_json(
                proof, {"provenance": {"r2_checkpoint_sha256": {"R2-100": "x" * 64}}}
            )
            with (
                patch("dual.ablation_benchmark.PROOF", proof),
                patch("dual.ablation_benchmark.torch.load") as load,
            ):
                with self.assertRaisesRegex(ValueError, "verified R2-100"):
                    verified_snapshot(snapshot, "fixture")
                load.assert_not_called()

    def test_verified_milestone_is_immutable_and_contains_only_committed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = committed_fixture(root, 50)
            epoch, offset = checkpoint_receipt(folder)
            journal = folder / "matched.jsonl"
            with journal.open("ab") as stream:
                stream.write(b'{"epoch":51')
            bundle = publish(root, folder, epoch, offset)
            verify(bundle)
            self.assertEqual(journal.stat().st_size, offset)
            sidecar = bundle.with_suffix(".tar.sha256")
            receipts = [
                (file_digest(p), p.stat().st_mtime_ns) for p in (bundle, sidecar)
            ]
            self.assertEqual(publish(root, folder, epoch, offset), bundle)
            self.assertEqual(
                [(file_digest(p), p.stat().st_mtime_ns) for p in (bundle, sidecar)],
                receipts,
            )
            raw = journal.read_bytes().replace(b"false", b"true ", 1)
            self.assertEqual(len(raw), offset)
            journal.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "immutable milestone"):
                publish(root, folder, epoch, offset)
            self.assertEqual(
                [(file_digest(p), p.stat().st_mtime_ns) for p in (bundle, sidecar)],
                receipts,
            )

    def test_epoch20_export_retry_preserves_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            committed_fixture(root, 20)
            drive(root, root / "clip", 20, micro=False)
            (archive,) = archive_dir(root, 20).glob("*.tar")
            first = (file_digest(archive), archive.stat().st_mtime_ns)
            drive(root, root / "clip", 20, micro=False)
            self.assertEqual(first, (file_digest(archive), archive.stat().st_mtime_ns))

    def test_verify_failure_never_publishes_partial_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = committed_fixture(root, 50)
            with (
                patch(
                    "dual.ablation_ops.verify",
                    side_effect=ValueError("verification failed"),
                ),
                self.assertRaisesRegex(ValueError, "verification failed"),
            ):
                publish(root, folder, *checkpoint_receipt(folder))
            self.assertFalse(archive_dir(root, 50).exists())
            self.assertEqual(list((root / "exports").iterdir()), [])

    def test_explicit_milestones_and_resume_after_export_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "clip"
            events = []

            def run_stub(folder, target, clip_path, *, micro):
                assert folder == root / "run" and clip_path == clip and micro
                start, _ = checkpoint_receipt(folder)
                if start:
                    assert archive_dir(root, start).is_dir()
                events.append((start, target))
                committed_fixture(root, target)

            with patch("dual.ablation_ops.run", side_effect=run_stub):
                with self.assertRaisesRegex(ValueError, "next milestone is 50"):
                    drive(root, clip, 100, micro=True)
                # Checkpoint 50 is already committed, but its publish was interrupted.
                committed_fixture(root, 50)
                drive(root, clip, 50, micro=True)
                self.assertEqual(events, [])
                for target in (100, 150, 200):
                    drive(root, clip, target, micro=True)
                    self.assertTrue(archive_dir(root, target).is_dir())
                drive(root, clip, 200, micro=True)
                self.assertEqual(events, [(50, 100), (100, 150), (150, 200)])
                status = json.loads((root / "status.json").read_text(encoding="utf8"))
                self.assertEqual(
                    (status["status"], status["epoch"]), ("STOPPED_AT_MILESTONE", 200)
                )


if __name__ == "__main__":
    unittest.main()
