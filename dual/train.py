"""Single-GPU D1 worker on the original SimGCD objective and epoch clock."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from crish.repair import corrected_distill
from model import DistillLoss, get_params_groups
from relational.export import export, verify
from relational.sensitivity_metrics import block_gradient, cls, prefix
from relational.state import MILESTONES, append, restore, save, write_json
from relational.train import baseline, build
from train import test as official_test

from . import cache
from .policy import Settings, refresh


def optimizer_for(model):
    return torch.optim.SGD(
        get_params_groups(model), lr=0.1, momentum=0.9, weight_decay=5e-5
    )


def scheduler_for(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=0.0001
    )


def losses(proj, logits, legal, mask, criterion, epoch, actions, mass):
    keep = baseline(proj, logits, legal, mask, criterion, epoch)
    if not actions:
        return keep, keep, None
    edited = corrected_distill(logits, logits.detach(), criterion, epoch, actions, mass)
    delta = 0.65 * (edited - criterion(logits, logits.detach(), epoch))
    return keep, keep + delta, delta


def gradient_receipt(keep, delta, model, features, tokens):
    v = model[1].last_layer.weight_v
    params = tuple(model[0].blocks[-1].parameters())

    def score(x, y):
        base, move = float(x.norm()), float(y.norm())
        return {
            "keep_norm": base,
            "edit_norm": move,
            "ratio": move / base if base else None,
            "cosine": float(torch.dot(x, y) / (base * move)) if base and move else None,
        }

    def grads(loss):
        head = torch.autograd.grad(loss, v, retain_graph=True)[0].flatten()
        if tokens is None:
            block = torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(loss, params, retain_graph=True)
                ]
            )
        else:
            block = block_gradient(loss, features, model[0], tokens, params)
        return head, block

    a, b = grads(keep), grads(delta)
    return {key: score(a[i], b[i]) for i, key in enumerate(("classifier", "block11"))}


def step(
    model,
    optimizer,
    batch,
    criterion,
    epoch,
    plan,
    index,
    *,
    micro=False,
    diagnose=False,
):
    images, legal, ids, mask = batch
    device = next(model.parameters()).device
    mask = mask.flatten().bool().to(device)
    legal = legal.to(device, dtype=torch.long)
    if not torch.equal(mask, legal >= 0) or (legal[~mask] != -1).any():
        raise ValueError(
            "Unexpected hidden-label or mask payload in the training batch"
        )
    optimizer.zero_grad()
    images = torch.cat(images)
    tokens = None
    if micro:
        tokens = prefix(model[0], images)
        with torch.no_grad():
            h = torch.cat([cls(model[0], t) for t in tokens])
        h.requires_grad_()
    else:
        h = model[0](images.to(device, non_blocking=True))
    proj, logits = model[1](h)
    actions, pending = (
        plan.select(logits, legal.tolist(), ids.tolist(), epoch, criterion)
        if plan
        else ([], [])
    )
    keep, loss, delta = losses(
        proj,
        logits,
        legal,
        mask,
        criterion,
        epoch,
        actions,
        plan.settings.mass if plan else 0.0,
    )
    gradient = (
        gradient_receipt(keep, delta, model, h, tokens)
        if diagnose and pending
        else None
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite SimGCD/D1 loss")
    loss.backward()
    if tokens is not None:
        for t, incoming in zip(tokens, h.grad.split(8), strict=True):
            cls(model[0], t).backward(incoming)
    optimizer.step()
    if plan:
        plan.commit(index, pending)
    return float(keep.detach()), float(loss.detach()), gradient


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cudnn": torch.backends.cudnn.version(),
        "packages": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines(),
    }


def stage(folder, until, clip_path, micro=False):
    device = torch.device("cuda:0")
    model, loader, evaluation, center, args, protocol = build()
    clip, receipt = cache.load(
        clip_path, loader.dataset.unlabelled_dataset, protocol["split"], device
    )
    protocol.pop("relational")
    settings = Settings()
    identity = {
        "variant": "D1",
        "protocol": protocol,
        "settings": asdict(settings),
        "clip": receipt,
        "microbatch8": micro,
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "config.json"
    if path.is_file() and json.loads(path.read_text()) != identity:
        raise ValueError("D1 run settings or cache identity changed")
    write_json(path, identity)
    env = folder / "environment.json"
    current_env = environment()
    if env.exists() and json.loads(env.read_text()) != current_env:
        raise ValueError("D1 environment changed during exact resume")
    write_json(env, current_env)
    opt = optimizer_for(model)
    schedule = scheduler_for(opt)
    mark = restore(folder, identity, model, opt, schedule)
    if until != next((x for x in MILESTONES if x > mark), None):
        raise ValueError("D1 stage must end at the next milestone")
    criterion = DistillLoss(30, 200)
    for epoch in range(mark, until):
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        plan = (
            refresh(model, loader, center, clip, epoch, settings, device)[0]
            if epoch >= settings.warmup
            else None
        )
        model.train()
        totals = np.zeros(2)
        grad, rows = None, 0
        for i, batch in enumerate(loader, 1):
            base, total, g = step(
                model,
                opt,
                batch,
                criterion,
                epoch,
                plan,
                i,
                micro=micro,
                diagnose=grad is None,
            )
            grad = g if g is not None else grad
            rows += len(batch[1])
            totals += [base, total]
            if i % 10 == 0:
                print(
                    f"epoch={epoch + 1} step={i}/{len(loader)} applied={len(plan.applied) if plan else 0} loss={total:.6f}",
                    flush=True,
                )
        # Official transductive evaluation is isolated from selection and optimization.
        a, o, n = official_test(model, evaluation, epoch, "Train ACC Unlabelled", args)
        lr = opt.param_groups[0]["lr"]
        schedule.step()
        done = epoch + 1
        journal = (
            plan.record(rows)
            if plan
            else {
                "status": "warmup",
                "rows": rows,
                "proposed": 0,
                "applied": 0,
                "mass": 0.0,
                "exposure": 0.0,
                "unique_applied": 0,
            }
        )
        report = {
            "epoch": done,
            "all": float(a),
            "old": float(o),
            "new": float(n),
            "lr": lr,
            "seconds": time.monotonic() - start,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "baseline_loss": float(totals[0] / len(loader)),
            "loss": float(totals[1] / len(loader)),
            "gradient_first_edited_batch": grad,
        }
        append(folder / "metrics.jsonl", report)
        append(folder / "relationships.jsonl", {"epoch": done, **journal})
        print(report, flush=True)
        interrupt = (folder / "operator-request.json").exists()
        if done % 10 == 0 or interrupt:
            save(folder, done, identity, model, opt, schedule)
        if interrupt:
            return
        if done % 10 == 0 and done not in MILESTONES:
            bundle = export(folder, recovery=True)
            verify(bundle)
            print(f"Verified recovery archive {bundle}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("until", type=int, choices=MILESTONES)
    p.add_argument("cache", type=Path)
    p.add_argument("--micro", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    stage(a.folder, a.until, a.cache, a.micro)


if __name__ == "__main__":
    main()
