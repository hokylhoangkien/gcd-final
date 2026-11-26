"""Committed epoch boundaries and append-only diagnostic journal recovery."""

import json
import os
from pathlib import Path

from crish.experiment import (
    atomic_checkpoint,
    load_checkpoint,
    recover_ledger,
    restore_rng,
    rng_state,
)

JOURNALS = ("metrics.jsonl", "relationships.jsonl")
MILESTONES = (100, 150, 200)


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(tmp, path)


def append(path, data):
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")


def checkpoints(folder):
    return sorted(folder.glob("epoch-[0-9][0-9][0-9].pt"))


def save(folder, epoch, identity, model, optimizer, scheduler):
    state = {
        "epoch": epoch,
        "identity": identity,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_state(),
        "journal_bytes": {k: (folder / k).stat().st_size for k in JOURNALS},
    }
    atomic_checkpoint(state, folder / f"epoch-{epoch:03}.pt")
    for old in checkpoints(folder)[:-1]:
        if int(old.stem.split("-")[1]) not in MILESTONES:
            old.unlink()


def restore(folder, identity, model, optimizer, scheduler):
    files = checkpoints(folder)
    saved = load_checkpoint(files[-1]) if files else None
    if saved and (
        saved["identity"] != identity or files[-1].stem != f"epoch-{saved['epoch']:03}"
    ):
        raise ValueError("Checkpoint identity or epoch mismatch")
    for name in JOURNALS:
        path = folder / name
        path.touch(exist_ok=True)
        recover_ledger(path, saved["journal_bytes"][name] if saved else 0)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        return saved["epoch"]
    return 0
