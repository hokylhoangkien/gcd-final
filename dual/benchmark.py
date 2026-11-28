"""Bounded full-bank D1 pilot, cache preflight and optional image-only cache build."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest, load_checkpoint, restore_rng
from model import DistillLoss
from relational.state import write_json
from relational.train import build, train_epoch
from scripts.audit_e3_proposals import json_native

from . import cache
from .policy import Settings, refresh
from .train import optimizer_for, step


def benchmark(model, loader, center, clip, receipt, snapshot, steps, micro, workers):
    tag = "R2-100"
    proof = json.loads(Path("reports/r2_e0_baseline_audit.json").read_text())[
        "provenance"
    ]
    digest = file_digest(snapshot)
    if digest != proof["r2_checkpoint_sha256"][tag]:
        raise ValueError(
            "Expected the verified R2-100 snapshot for the locked benchmark"
        )
    saved = load_checkpoint(snapshot)
    assert saved["epoch"] == 100
    model.load_state_dict(saved["model"])
    opt = optimizer_for(model)
    opt.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    plan, pair = refresh(
        model,
        loader,
        center,
        clip,
        100,
        Settings(),
        device,
        workers=workers,
        diagnostic=True,
    )
    if pair is None:
        raise ValueError("Full-bank D1 benchmark timed out")
    refresh_peak = torch.cuda.max_memory_allocated()
    # Production sampler, batch size and drop_last, bounded at the first N iterations.
    bench = DataLoader(
        loader.dataset,
        batch_size=128,
        sampler=loader.sampler,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
    )
    model.train()
    criterion = DistillLoss(30, 200)
    measured, base_times, rows = [], [], 0
    before = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    stream = iter(bench)
    for i in range(1, steps + 1):
        t = time.monotonic()
        batch = next(stream)
        _a, b, grad = step(
            model,
            opt,
            batch,
            criterion,
            100,
            plan,
            i,
            micro=micro,
            diagnose=len(measured) < 3,
        )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t
        rows += len(batch[1])
        if grad:
            measured.append({"batch": i, "gradient": grad})
        else:
            base_times.append(elapsed)
        print(
            "D1 pilot",
            i,
            "applied cumulative",
            len(plan.applied),
            "loss",
            b,
            flush=True,
        )
    runtime = time.monotonic() - before
    if not measured or not plan.applied:
        raise ValueError("Pilot had no applied edits")
    result = plan.record(rows)
    breakdown = result["runtime"]
    estimate = (
        float(np.mean(base_times)) * len(loader)
        if base_times
        else runtime / steps * len(loader)
    )
    # Center evaluation is charged as another complete center pass for a rough same-device forecast.
    core = breakdown["dino_snapshot"] + breakdown["clip_retrieval"]
    eval_cost = breakdown["dino_snapshot"]
    pair.pop("slots")
    pair.pop("partition")
    diag_dest = np.asarray(pair["dino_dest"])
    pair["same_candidates_destination_agreement"] = float(
        np.mean(diag_dest == np.asarray(pair["clip_dest"]))
    )
    pair["same_candidates_dino_valid"] = int((diag_dest >= 0).sum())
    return {
        "snapshot": tag,
        "checkpoint_sha256": digest,
        "clip": receipt,
        "gpu": torch.cuda.get_device_name(),
        "microbatch8": micro,
        "workers": workers,
        "full_bank": len(clip),
        "batches": steps,
        "total_steps_per_epoch": len(loader),
        "policy": result,
        "diagnostic": measured,
        "paired_evidence": pair,
        "bounded_train_seconds": runtime,
        "peak_refresh_bytes": refresh_peak,
        "peak_training_bytes": torch.cuda.max_memory_allocated(),
        "forecast_epoch_seconds": estimate + core + eval_cost,
        "forecast_200h": ((estimate + eval_cost) * 200 + core * 190) / 3600,
    }


def parity(model, loader, snapshot):
    proof = json.loads(Path("reports/r2_e0_baseline_audit.json").read_text())[
        "provenance"
    ]
    if file_digest(snapshot) != proof["r2_checkpoint_sha256"]["R2-100"]:
        raise ValueError("Expected the verified R2-100 snapshot for parity")
    saved = load_checkpoint(snapshot)
    n = len(loader.dataset.labelled_dataset)
    (batch,) = DataLoader(
        Subset(loader.dataset, list(range(4)) + list(range(n, n + 4))), batch_size=8
    )
    batch = (batch[0], batch[1].long(), *batch[2:])
    criterion = DistillLoss(30, 200)

    def initial():
        model.load_state_dict(saved["model"])
        opt = optimizer_for(model)
        opt.load_state_dict(saved["optimizer"])
        model.train()
        return opt

    opt = initial()
    train_epoch(model, [batch], opt, criterion, 100, None, torch.device("cuda"))
    expected = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    results = {}
    for label, micro, tol in (("direct", False, 0), ("micro", True, 2e-6)):
        opt = initial()
        step(model, opt, batch, criterion, 100, None, 1, micro=micro)
        observed = [p.detach() for p in model.parameters() if p.requires_grad]
        for a, b in zip(expected, observed, strict=True):
            torch.testing.assert_close(a, b, atol=tol, rtol=0)
        results[label] = {
            "max_parameter_error": max(
                float((a - b).abs().max())
                for a, b in zip(expected, observed, strict=True)
            )
        }
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("preflight", "encode", "benchmark", "parity"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--weights", type=Path)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--steps", type=int, choices=range(2, 17), default=12)
    p.add_argument("--workers", type=int, choices=(0, 8), default=8)
    p.add_argument("--micro", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    model, loader, _, center, _, protocol = build()
    blind = loader.dataset.unlabelled_dataset
    device = torch.device("cuda:0")
    if a.mode == "encode":
        if a.weights is None or a.cache.exists():
            p.error("Encode needs --weights and a new cache path")
        _, receipt = cache.encode(a.cache, a.weights, blind, protocol["split"], device)
        print(json.dumps(receipt, indent=2))
    else:
        clip, receipt = cache.load(a.cache, blind, protocol["split"], device)
        print("Verified D1 cache", receipt["sha256"], flush=True)
        if a.mode in ("benchmark", "parity"):
            if a.snapshot is None or a.output is None:
                p.error("Benchmark needs --snapshot and --output")
            value = (
                benchmark(
                    model,
                    loader,
                    center,
                    clip,
                    receipt,
                    a.snapshot,
                    a.steps,
                    a.micro,
                    a.workers,
                )
                if a.mode == "benchmark"
                else parity(model, loader, a.snapshot)
            )
            a.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(a.output, json_native(value))
            print("Benchmark complete", a.output, flush=True)


if __name__ == "__main__":
    main()
