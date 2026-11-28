"""Bounded Cars benchmark from the verified R2-100 training state."""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crish.experiment import file_digest, restore_rng
from model import DistillLoss
from relational.state import write_json
from relational.train import build

from . import cache
from .ablation import ARMS, matched_step, refresh
from .policy import Settings
from .train import optimizer_for, scheduler_for

EPOCH = 100
PROOF = Path(__file__).resolve().parents[1] / "reports/r2_e0_baseline_audit.json"
PURPOSE = "ablation-benchmark-only"


def verified_snapshot(path, split):
    provenance = json.loads(PROOF.read_text(encoding="utf8"))["provenance"]
    digest = file_digest(path)
    if digest != provenance["r2_checkpoint_sha256"]["R2-100"]:
        raise ValueError("Benchmark needs the verified R2-100 checkpoint")
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if (
        saved["epoch"] != EPOCH
        or saved["identity"]["protocol"]["split"] != split
        or saved["scheduler"]["T_max"] != 200
        or saved["scheduler"]["last_epoch"] != EPOCH
        or saved["scheduler"]["_last_lr"]
        != [g["lr"] for g in saved["optimizer"]["param_groups"]]
    ):
        raise ValueError(
            "R2-100 split, optimizer or scheduler does not match the benchmark"
        )
    return saved, digest


def run(folder, clip_path, snapshot_path, steps, *, micro=False, workers=None):
    # Reserve a fresh benchmark-only directory and never write any checkpoints.
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", {"purpose": PURPOSE})
    started = time.monotonic()
    model, loader, _, center, _, protocol = build()
    device = next(model.parameters()).device
    saved, digest = verified_snapshot(snapshot_path, protocol["split"])
    clip, receipt = cache.load(
        clip_path, loader.dataset.unlabelled_dataset, protocol["split"], device
    )
    models, opts, starting_weights = {}, {}, {}
    for arm in ARMS:
        m = model if arm == ARMS[0] else build()[0]
        m.load_state_dict(saved["model"], strict=True)
        opt = optimizer_for(m)
        opt.load_state_dict(saved["optimizer"])
        scheduler_for(opt).load_state_dict(saved["scheduler"])
        models[arm], opts[arm] = m, opt
        starting_weights[arm] = m[1].last_layer.weight_v.detach().cpu().clone()
    restore_rng(saved["rng"])
    del saved
    if workers is None:
        workers = protocol["workers"]
    elif workers != loader.num_workers:
        loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=loader.sampler,
            drop_last=loader.drop_last,
            num_workers=workers,
            pin_memory=loader.pin_memory,
        )
    print(f"Verified R2-100 {digest}; measuring three full-bank refreshes", flush=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = time.monotonic()
    plans = refresh(models, loader, center, clip, EPOCH, Settings(), device, workers)
    refresh_seconds = time.monotonic() - before
    refresh_peak = torch.cuda.max_memory_allocated(device)
    refresh_reserved = torch.cuda.max_memory_reserved(device)
    if any(p.details["status"] != "ok" for p in plans.values()):
        write_json(
            folder / "benchmark.json",
            {
                "purpose": PURPOSE,
                "status": "refresh_failure",
                "verified_R2_100_sha256": digest,
                "refresh_seconds": refresh_seconds,
                "peak_refresh_allocated_bytes": refresh_peak,
                "arms": {a: p.details for a, p in plans.items()},
            },
        )
        raise RuntimeError(
            "Full-bank benchmark refresh did not finish; see benchmark.json"
        )
    print(
        f"Refresh {refresh_seconds:.2f}s, proposed "
        + str({a: len(p.proposals) for a, p in plans.items()}),
        flush=True,
    )
    for m in models.values():
        m.train()
    counts, losses, mixed_rows = [], [], 0
    crit = DistillLoss(30, 200)
    torch.cuda.reset_peak_memory_stats(device)
    before = time.monotonic()
    for index, batch in enumerate(itertools.islice(loader, steps), 1):
        loss, matched = matched_step(
            models, opts, batch, crit, EPOCH, plans, index, micro=micro
        )
        counts.append(matched)
        losses.append(loss)
        mixed_rows += len(batch[1])
        print(
            f"Batch {index}/{steps} matched MOVE per arm: {matched['applied_per_arm']}",
            flush=True,
        )
    torch.cuda.synchronize(device)
    train_seconds = time.monotonic() - before
    train_peak = torch.cuda.max_memory_allocated(device)
    dose = sum(x["applied_per_arm"] for x in counts)
    if len(counts) != steps or any(len(p.applied) != dose for p in plans.values()):
        raise RuntimeError("Incorrect benchmark batch or matched counts")
    updates = {
        a: not torch.equal(m[1].last_layer.weight_v.detach().cpu(), starting_weights[a])
        for a, m in models.items()
    }
    records = {a: p.record(mixed_rows) for a, p in plans.items()}
    result = {
        "purpose": PURPOSE,
        "verified_R2_100_sha256": digest,
        "clip": receipt,
        "teacher_epoch_zero_based": EPOCH,
        "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "microbatch8": micro,
        "workers": workers,
        "bank_ids": len(clip),
        "measured_batches": len(counts),
        "shared_mixed_rows": mixed_rows,
        "matched_move_per_arm": dose,
        "realized_exposure": dose / mixed_rows,
        "moved_mass_per_arm": dose * Settings().mass,
        "classifier_updated": updates,
        "dino_positive_candidates": records["D1-DINO"]["eligible_candidates"],
        "dino_candidate_quota": records["D1-DINO"]["candidate_quota"],
        "dino_candidate_shortfall": records["D1-DINO"]["candidate_shortfall"],
        "dino_live_shortfall_total": sum(
            x["dino_shortfall_before_matching"] for x in counts
        ),
        "refresh_seconds": refresh_seconds,
        "bounded_training_seconds": train_seconds,
        "wall_seconds": time.monotonic() - started,
        "peak_refresh_allocated_bytes": refresh_peak,
        "peak_training_allocated_bytes": train_peak,
        "peak_reserved_bytes": max(
            refresh_reserved, torch.cuda.max_memory_reserved(device)
        ),
        "epoch_core_seconds_estimate_excluding_eval_and_export": refresh_seconds
        + train_seconds / steps * len(loader),
        "batch_matching": counts,
        "losses": losses,
        "arms": records,
        "status": "ok"
        if dose and all(updates.values())
        else "insufficient_exercised_support",
    }
    write_json(folder / "benchmark.json", result)
    if result["status"] != "ok":
        raise RuntimeError(
            "Benchmark did not exercise MOVE and update all arms; see benchmark.json"
        )
    print(f"Benchmark complete: {folder / 'benchmark.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("clip", type=Path)
    p.add_argument("snapshot", type=Path)
    p.add_argument(
        "--steps", type=int, choices=range(2, 25), default=12, metavar="2..24"
    )
    p.add_argument("--micro", action="store_true")
    p.add_argument("--workers", type=int, choices=(0, 8))
    args = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(
        args.folder,
        args.clip,
        args.snapshot,
        args.steps,
        micro=args.micro,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
