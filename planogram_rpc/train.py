"""RPC-only SimGCD/CRISH trainer. Last checkpoint; no K-means inside training."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from crish.experiment import atomic_checkpoint, load_checkpoint, restore_rng, rng_state
from crish.global_repair import EpochRepairs
from crish.runtime import arguments, global_defaults
from crish.sampling import balanced_known_folds
from model import DINOHead, DistillLoss, SupConLoss, get_params_groups, info_nce_logits

from .assets import backbone
from .common import DINO_REVISION, DINO_SHA256, digest, write_json
from .data import BalancedBatches, load_manifest, training_data
from .prepare import verify


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    arguments(p)
    global_defaults(p)
    p.set_defaults(crish_mode="E3", crish_warmup=10)
    p.add_argument("--data", type=Path, default=Path("datasets/rpc40"))
    p.add_argument("--assets", type=Path, default=Path("assets_local"))
    p.add_argument("--output", type=Path, default=Path("runs/rpc40-e3-s1"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="Real two-view CUDA forward/backward; no checkpoint, no training run",
    )
    p.add_argument(
        "--stop-after-epochs",
        type=int,
        help="Pause after this many epochs this invocation, preserving the 200-epoch schedule",
    )
    p.add_argument(
        "--max-hours",
        type=float,
        default=24.0,
        help="Save and stop at the next epoch boundary",
    )
    p.add_argument("--fp32", action="store_true")
    return p


def configure(args):
    if (
        args.crish_mode not in ("E0", "E3")
        or args.crish_global_suite
        or args.crish_dose
    ):
        raise ValueError(
            "RPC runner supports standalone E0 or E3 only; no Cars dose suite"
        )
    if args.epochs < 1 or args.max_hours <= 0 or args.num_workers < 0 or args.lr <= 0:
        raise ValueError("Invalid epochs, runtime budget, workers or learning rate")
    if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
        raise ValueError("--stop-after-epochs must be positive")
    args.dataset_name, args.n_views = "rpc", 2
    args.run_seed, args.grad_from_block = args.seed, 11
    args.weight_decay, args.momentum = 5e-5, 0.9
    args.sup_weight, args.memax_weight = 0.35, 1.0
    args.warmup_teacher_temp_epochs = min(30, args.epochs)
    args.warmup_teacher_temp, args.teacher_temp = 0.07, 0.04
    args.fp16 = args.device == "cuda" and not args.fp32
    return args


def objective(student, batch, criterion, epoch, args, repairs=None):
    views, labels, ids, mask = batch
    device = next(student.parameters()).device
    images = torch.cat(views).to(device, non_blocking=True)
    labels = labels.to(device)
    mask = mask[:, 0].to(device).bool()
    if not mask.any() or not (~mask).any():
        raise ValueError(
            "A batch must contain both legally labeled and unlabeled images"
        )
    projections, logits = student(images)
    supervised_logits = torch.cat([v[mask] for v in (logits / 0.1).chunk(2)])
    supervised_labels = labels[mask].repeat(2)
    ce = F.cross_entropy(supervised_logits, supervised_labels)
    distill = (
        repairs.loss(logits, ids, mask, criterion, epoch)
        if repairs is not None
        else criterion(logits, logits.detach(), epoch)
    )
    probs = (logits / 0.1).softmax(-1).mean(0)
    memax = (probs * probs.clamp_min(1e-12).log()).sum() + math.log(len(probs))
    nce_logits, nce_targets = info_nce_logits(projections, device=device)
    nce = F.cross_entropy(nce_logits, nce_targets)
    known_views = torch.stack([v[mask] for v in projections.chunk(2)], dim=1)
    supcon = SupConLoss()(F.normalize(known_views, dim=-1), labels[mask])
    return (1 - args.sup_weight) * (
        distill + args.memax_weight * memax + nce
    ) + args.sup_weight * (ce + supcon)


def main(argv=None):
    args = configure(parser().parse_args(argv))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; choose an NVIDIA PyTorch image and check nvidia-smi"
        )
    if args.device == "cpu" and not args.preflight:
        raise RuntimeError(
            "Full RPC training is restricted to the rented CUDA host; CPU is only for preflight"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    manifest = load_manifest(args.data)
    verify(args.data, manifest)
    dataset, evaluation_transform = training_data(args.data, manifest)
    if args.crish_mode == "E3":
        legal_targets = torch.tensor([r[1] for r in dataset.labelled_dataset.rows])
        folds = 1 + min(args.crish_cap, args.crish_confirmations)
        try:
            balanced_known_folds(
                torch.arange(len(legal_targets)),
                legal_targets,
                args.crish_probe_known,
                torch.Generator().manual_seed(args.seed),
                per_class=2,
                folds=folds,
            )
        except ValueError as error:
            raise ValueError(
                "Insufficient legal known images/classes for CRISH probes. Use the default RPC40 dataset or explicitly configure smaller probe folds."
            ) from error
    sampler = BalancedBatches(
        len(dataset.labelled_dataset),
        len(dataset.unlabelled_dataset),
        args.batch_size,
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        timeout=120 if args.num_workers else 0,
    )
    model = backbone(args.assets)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("blocks.11.")
    student = nn.Sequential(model, DINOHead(768, len(manifest["classes"]))).to(
        args.device
    )
    optimizer = torch.optim.SGD(
        get_params_groups(student),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 1e-3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        2,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )
    if args.preflight:
        start = time.monotonic()
        student.train()
        batch = next(iter(loader))
        # A new AMP scaler can legitimately overflow initially. Bound retries,
        # allow its normal scale reduction and require one finite real update.
        for attempt in range(16):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=args.device, dtype=torch.float16, enabled=args.fp16
            ):
                loss = objective(student, batch, criterion, 0, args)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CUDA preflight loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = all(
                torch.isfinite(p.grad).all()
                for p in student.parameters()
                if p.grad is not None
            )
            if not finite and not args.fp16:
                raise FloatingPointError("Nonfinite FP32 preflight gradient")
            scaler.step(optimizer)
            scaler.update()
            if finite:
                break
        else:
            raise FloatingPointError(
                "AMP preflight could not produce finite gradients after 16 attempts"
            )
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "loss": float(loss.detach()),
                    "seconds": time.monotonic() - start,
                    "batch_size": args.batch_size,
                    "amp_scale_attempts": attempt + 1,
                    "views": 2,
                    "device": args.device,
                    "gpu": torch.cuda.get_device_name()
                    if args.device == "cuda"
                    else None,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30
                    if args.device == "cuda"
                    else None,
                }
            ),
            flush=True,
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "last.pt"
    args.model_path = str(checkpoint_path)
    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k
        not in (
            "resume",
            "preflight",
            "stop_after_epochs",
            "max_hours",
            "data",
            "assets",
            "output",
            "model_path",
        )
    }
    identity = {
        "settings": settings,
        "manifest_sha256": digest(args.data / "manifest.json"),
        "dino_revision": DINO_REVISION,
        "pretrained_sha256": DINO_SHA256,
        "code_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): digest(p)
            for p in sorted(
                list(Path(__file__).resolve().parent.glob("*.py"))
                + list((Path(__file__).resolve().parents[1] / "crish").glob("*.py"))
                + [
                    Path(__file__).resolve().parents[1] / "model.py",
                    Path(__file__).resolve().parents[1] / "data/data_utils.py",
                ]
            )
        },
        "pipeline_version": 1,
    }
    start_epoch = 0
    ledger = args.output / "crish-global.jsonl"
    if args.resume:
        saved = load_checkpoint(checkpoint_path)
        if saved["identity"] != identity:
            raise ValueError(
                "Resume settings/data differ; reuse the exact original command"
            )
        student.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start_epoch = saved["epoch"]
        args.resume_ledger_bytes = saved["ledger_bytes"]
        args.resume_carry = saved.get("carry", [])
        # Metrics are an epoch-committed prefix, just like the repair ledger.
        metrics_path = args.output / "metrics.jsonl"
        if metrics_path.stat().st_size < saved["metrics_bytes"]:
            raise ValueError(
                "Metrics file is shorter than the committed checkpoint prefix"
            )
        with metrics_path.open("r+b") as stream:
            stream.truncate(saved["metrics_bytes"])
        del saved
    elif any(args.output.iterdir()):
        raise ValueError(
            "Output directory is not empty; use --resume or choose a new --output"
        )
    else:
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            revision = "unavailable"
        write_json(
            args.output / "run.json",
            {
                "identity": identity,
                "git_sha": revision,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": os.getpid(),
                "eval_policy": "last checkpoint; SS-KMeans and retrieval after training only",
            },
        )
    repairs = EpochRepairs(args) if args.crish_mode == "E3" else None
    pause = {"requested": False}

    def request_pause(signum, frame):
        pause["requested"] = True
        print(
            "Pause requested; saving at the next completed epoch boundary", flush=True
        )

    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    started = time.monotonic()
    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        sampler.epoch = epoch
        student.train()
        active = repairs is not None and repairs.due(epoch)
        if active:
            record = repairs.prepare(
                student, optimizer, dataset, evaluation_transform, criterion, epoch
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "repair_stop": record.get("stop"),
                        "accepted": len(record["accepted"]),
                    }
                ),
                flush=True,
            )
        total, count, skipped = 0.0, 0, 0
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=args.device, dtype=torch.float16, enabled=args.fp16
            ):
                loss = objective(
                    student, batch, criterion, epoch, args, repairs if active else None
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite loss at epoch {epoch + 1}, step {step}"
                )
            before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            stepped = not args.fp16 or scaler.get_scale() >= before
            skipped += int(not stepped)
            if active:
                repairs.finish_step(stepped)
            total += float(loss.detach())
            count += 1
            if step % 10 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {step + 1}/{len(loader)} loss {float(loss.detach()):.5f}",
                    flush=True,
                )
        if active:
            repairs.close_epoch()
        if skipped == count:
            raise FloatingPointError(
                "AMP skipped every optimizer step in this epoch; no trained epoch will be committed. Check the preflight, or use a NEW --fp32 run."
            )
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        metric = {
            "epoch": epoch + 1,
            "loss": total / count,
            "lr": lr,
            "amp_skipped_steps": skipped,
            "seconds": time.monotonic() - epoch_start,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(metric, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        state = {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "identity": identity,
            "epoch": epoch + 1,
            "classes": manifest["classes"],
            "known_count": manifest["known_count"],
            "ledger_bytes": ledger.stat().st_size if ledger.exists() else 0,
            "metrics_bytes": metrics_path.stat().st_size,
            "carry": repairs.carry_state() if repairs and repairs.persistence else [],
        }
        atomic_checkpoint(state, checkpoint_path)
        print(json.dumps(dict(metric, checkpoint=str(checkpoint_path))), flush=True)
        if (
            pause["requested"]
            or time.monotonic() - started >= args.max_hours * 3600
            or (
                args.stop_after_epochs
                and epoch + 1 - start_epoch >= args.stop_after_epochs
            )
        ):
            break
    print(
        f"Saved last.pt through epoch {epoch + 1 if start_epoch < args.epochs else start_epoch}/{args.epochs}. Run evaluate only after epoch {args.epochs}.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "CUDA OOM. No automatic batch change. For a NEW run use --batch-size 64 --lr 0.05 (or 32 / 0.025) and a NEW --output, then repeat preflight. Resume requires the original batch/LR."
        )
