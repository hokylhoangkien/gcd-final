"""Run one immutable milestone of the coordinated D1 ablation on Linux."""

import argparse
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

import torch

from crish.experiment import file_digest, recover_ledger
from relational.export import verify
from relational.state import write_json

from .ablation import PRODUCTION, run

MILESTONES = (50, 100, 150, 200)


def checkpoint_receipt(folder):
    config = folder / "config.json"
    identity = json.loads(config.read_text(encoding="utf8")) if config.exists() else {}
    if identity and identity.get("purpose") != PRODUCTION:
        raise ValueError(
            "Ablation production cannot use a benchmark or smoke directory"
        )
    path = folder / "recovery.pt"
    if not path.exists():
        return 0, 0
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not identity or saved.get("identity") != identity:
        raise ValueError(
            "Recovery is missing its matching official-initialization manifest"
        )
    epoch, offset = saved["epoch"], saved["journal_bytes"]
    if not 1 <= epoch <= MILESTONES[-1] or offset <= 0:
        raise ValueError("Invalid production recovery epoch or journal offset")
    return epoch, offset


def archive_dir(root, epoch):
    return root / "exports" / f"epoch-{epoch:03d}"


def publish(root, folder, epoch, offset):
    final = archive_dir(root, epoch)
    name = f"D1-ablation-epoch-{epoch:03d}.tar"
    sources = [folder / n for n in ("config.json", "recovery.pt", "matched.jsonl")]
    config, recovery, journal = sources
    recover_ledger(journal, offset)
    if final.exists():
        bundle = final / name
        verify(bundle)
        with tarfile.open(bundle) as tar:
            receipt = json.load(tar.extractfile("milestone.json"))
        if (
            receipt["epoch"] != epoch
            or receipt["journal_bytes"] != offset
            or receipt["checkpoint_sha256"] != file_digest(recovery)
            or receipt["config_sha256"] != file_digest(config)
            or receipt["journal_sha256"] != file_digest(journal)
        ):
            raise ValueError("An immutable milestone belongs to different run state")
        return bundle

    rows = [
        json.loads(line) for line in journal.read_text(encoding="utf8").splitlines()
    ]
    if [row["epoch"] for row in rows] != list(range(1, epoch + 1)) or any(
        row["partial"] for row in rows
    ):
        raise ValueError("Milestone requires exactly the committed complete epochs")
    final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".epoch-{epoch:03d}-", dir=final.parent
    ) as tmp:
        staging = Path(tmp)
        checks = {p.name: file_digest(p) for p in sources}
        status = staging / "milestone.json"
        write_json(
            status,
            {
                "epoch": epoch,
                "journal_bytes": offset,
                "purpose": PRODUCTION,
                "checkpoint_sha256": checks[recovery.name],
                "config_sha256": checks[config.name],
                "journal_sha256": checks[journal.name],
                "last_epoch_accuracy": rows[-1]["accuracy"],
                "applied_per_arm": {
                    a: record["applied"] for a, record in rows[-1]["arms"].items()
                },
            },
        )
        sources.append(status)
        checks[status.name] = file_digest(status)
        bundle = staging / name
        with tarfile.open(bundle, "w") as tar:
            for path in sources:
                tar.add(path, arcname=path.name, recursive=False)
            sums = "".join(f"{sha}  {item}\n" for item, sha in checks.items()).encode(
                "ascii"
            )
            info = tarfile.TarInfo("SHA256SUMS")
            info.size = len(sums)
            tar.addfile(info, io.BytesIO(sums))
        with bundle.with_suffix(".tar.sha256").open("x", encoding="ascii") as f:
            f.write(f"{file_digest(bundle)}  {bundle.name}\n")
        verify(bundle)
        # The run lock excludes another publisher. Move both verified files at once.
        status.unlink()
        staging.rename(final)
    return final / name


def drive(root, clip, until, *, micro):
    if (root / "config.json").exists():
        raise ValueError(
            "Production requires a dedicated run root, separate from a benchmark"
        )
    folder = root / "run"
    epoch, offset = checkpoint_receipt(folder)
    if epoch == until:
        bundle = publish(root, folder, epoch, offset)
    else:
        following = next((mark for mark in MILESTONES if mark > epoch), None)
        if until != following:
            raise ValueError(f"Run is at epoch {epoch}; next milestone is {following}")
        if epoch in MILESTONES:
            # Retry a timed-out export before allowing another SGD step.
            publish(root, folder, epoch, offset)
        for mark in MILESTONES:
            if mark < epoch and not archive_dir(root, mark).is_dir():
                raise ValueError(
                    f"Previously completed milestone {mark} is missing its archive"
                )
        write_json(
            root / "status.json",
            {"status": "RUNNING", "from_epoch": epoch, "until": until},
        )
        run(folder, until, clip, micro=micro)
        epoch, offset = checkpoint_receipt(folder)
        if epoch != until:
            raise ValueError(f"Worker did not commit requested milestone {until}")
        bundle = publish(root, folder, epoch, offset)
    write_json(
        root / "status.json",
        {"status": "STOPPED_AT_MILESTONE", "epoch": epoch, "archive": str(bundle)},
    )
    print(f"Stopped at epoch {epoch}; verified archive {bundle}", flush=True)


def main():
    import fcntl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--until", type=int, choices=MILESTONES, required=True)
    p.add_argument("--micro", action="store_true")
    args = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".ablation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        drive(args.root, args.clip, args.until, micro=args.micro)


if __name__ == "__main__":
    main()
