"""Locked Cars stage worker. Budget decisions are made by the external controller."""

import argparse
import hashlib
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from crish.data import ImageOnly
from crish.experiment import file_digest
from data.augmentations import get_transform
from data.get_datasets import get_class_splits, get_datasets
from model import (
    ContrastiveLearningViewGenerator,
    DINOHead,
    DistillLoss,
    SupConLoss,
    get_params_groups,
    info_nce_logits,
)
from relational.export import export
from relational.graph import Settings, refresh
from relational.state import MILESTONES, append, restore, save, write_json


def sampler_for(data):
    n, u = len(data.labelled_dataset), len(data.unlabelled_dataset)
    return WeightedRandomSampler(torch.DoubleTensor([1] * n + [n / u] * u), len(data))


def baseline(proj, logits, legal, mask, criterion, epoch):
    sup_logits = torch.cat([f[mask] for f in (logits / 0.1).chunk(2)])
    ce = F.cross_entropy(sup_logits, legal[mask].repeat(2))
    cluster = criterion(logits, logits.detach(), epoch)
    p = (logits / 0.1).softmax(1).mean(0)
    cluster += -torch.sum(torch.log(p ** (-p))) + math.log(len(p))
    q, target = info_nce_logits(proj, device=proj.device)
    contrast = F.cross_entropy(q, target)
    f = torch.cat([x[mask].unsqueeze(1) for x in proj.chunk(2)], dim=1)
    supcon = SupConLoss()(F.normalize(f, dim=-1), labels=legal[mask])
    return (0.65 * cluster + 0.35 * ce) + (0.65 * contrast + 0.35 * supcon)


def grad_stats(base, relation, difference, block):
    params = tuple(block.parameters())

    def vector(scalar):
        g = torch.autograd.grad(scalar, params, retain_graph=True)
        return torch.cat([v.detach().flatten() for v in g])

    ref = vector(base)

    def compare(term):
        g = vector(term)
        norm, ref_norm = g.norm().item(), ref.norm().item()
        return {
            "norm": norm,
            "ratio": norm / ref_norm if ref_norm else None,
            "cosine": float(torch.dot(g, ref) / (norm * ref_norm))
            if norm and ref_norm
            else None,
        }

    return {
        "baseline_norm": ref.norm().item(),
        "relational": compare(relation),
        "edit_delta": compare(difference),
    }


def train_epoch(model, loader, optimizer, criterion, epoch, graph, device):
    model.train()
    totals = np.zeros(3)
    diagnostic = None
    for step, (images, legal, ids, mask) in enumerate(loader):
        if graph is not None and any(
            (key in graph.lookup) == labeled
            for key, labeled in zip(ids.tolist(), mask.flatten().bool().tolist())
        ):
            raise ValueError("Graph membership disagrees with legal training split")
        mask = mask.flatten().bool().to(device)
        legal = legal.to(device)
        features = model[0](torch.cat(images).to(device))
        base = baseline(*model[1](features), legal, mask, criterion, epoch)
        loss = base
        rel = plain = base.new_zeros(())
        if graph is not None:
            rel, plain, pending = graph.loss(features, ids)
            coeff = graph.settings.coefficient
            loss = base + coeff * rel
            if step == 0:
                diagnostic = grad_stats(
                    base, coeff * rel, coeff * (rel - plain), model[0].blocks[-1]
                )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if graph is not None:
            graph.applied(pending)
        if step % 10 == 0:
            print(
                f"epoch={epoch + 1} batch={step}/{len(loader)} baseline={base.item():.8f} "
                f"relational={rel.item():.8f} total={loss.item():.8f}",
                flush=True,
            )
        totals += [base.detach().item(), rel.detach().item(), plain.detach().item()]
    means = totals / len(loader)
    return {
        "baseline_loss": means[0],
        "relational_loss": means[1],
        "unedited_relational_loss": means[2],
        "steps": len(loader),
        "gradient_first_batch": diagnostic,
    }


def build():
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    args = get_class_splits(
        SimpleNamespace(
            dataset_name="scars",
            use_ssb_splits=True,
            prop_train_labels=0.5,
            interpolation=3,
            crop_pct=0.875,
            eval_funcs=["v2"],
            logger=logging.getLogger(),
        )
    )
    source, weights = os.environ["SIMGCD_DINO_REPO"], os.environ["SIMGCD_DINO_WEIGHTS"]
    weight_digest = file_digest(weights)
    if (
        weight_digest
        != "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e"
    ):
        raise ValueError("Expected the locked official dino_vitbase16_pretrain.pth")
    backbone = torch.hub.load(source, "dino_vitb16", source="local", pretrained=False)
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    for name, param in backbone.named_parameters():
        param.requires_grad_(name.startswith("blocks.11."))
    train, center = get_transform("imagenet", 224, args)
    data, _, evaluation, _ = get_datasets(
        "scars", ContrastiveLearningViewGenerator(train, 2), center, args
    )
    data.unlabelled_dataset = ImageOnly(data.unlabelled_dataset, "scars")
    known, unknown = data.labelled_dataset, data.unlabelled_dataset
    if set(known.uq_idxs) & set(unknown.ids) or len(set(unknown.ids)) != len(unknown):
        raise ValueError("Overlapping or duplicate training IDs")
    h = hashlib.sha256()
    for array in (
        args.train_classes,
        args.unlabeled_classes,
        known.uq_idxs,
        known.target,
        unknown.ids,
    ):
        h.update(np.asarray(array, dtype="<i8").tobytes())
    workers = 8
    loader = DataLoader(
        data,
        batch_size=128,
        sampler=sampler_for(data),
        pin_memory=True,
        drop_last=True,
        num_workers=workers,
    )
    test = DataLoader(evaluation, batch_size=256, shuffle=False, num_workers=workers)
    model = torch.nn.Sequential(backbone, DINOHead(768, 196)).cuda()
    protocol = {
        "seed": 1,
        "epochs": 200,
        "batch": 128,
        "workers": workers,
        "sampler": "official weighted replacement",
        "backbone": "DINOv1 ViT-B/16",
        "grad_from_block": 11,
        "lr": 0.1,
        "schedule": "cosine-200",
        "eta_min": 0.0001,
        "momentum": 0.9,
        "decay": 0.00005,
        "fp16": False,
        "transform": "imagenet",
        "views": 2,
        "teacher": [0.07, 0.04, 30],
        "sup_weight": 0.35,
        "memax": 1.0,
        "split": h.hexdigest(),
        "evaluation": "v2",
        "dino_sha256": weight_digest,
        "dino_source": file_digest(Path(source) / "vision_transformer.py"),
        "relational": asdict(Settings()),
        "deterministic": True,
        "torch": str(torch.__version__),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    return model, loader, test, center, args, protocol


def stage(folder, variant, until):
    # Only the controller schedules stages. A stage cannot cross a research guard.
    from train import test as evaluate

    device = torch.device("cuda:0")
    model, loader, evaluation, center, args, protocol = build()
    identity = {"variant": variant, "protocol": protocol}
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "config.json"
    if manifest.exists():
        import json

        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Run folder belongs to a different experiment")
    write_json(manifest, identity)
    write_json(
        folder / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cudnn": torch.backends.cudnn.version(),
            "packages": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            ).splitlines(),
        },
    )
    optimizer = torch.optim.SGD(
        get_params_groups(model), lr=0.1, momentum=0.9, weight_decay=5e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=0.0001
    )
    criterion = DistillLoss(30, 200)
    start = restore(folder, identity, model, optimizer, scheduler)
    standalone = (folder / "standalone.json").is_file()
    if until != next((e for e in (100, 150, 200) if e > start), None):
        raise ValueError("Stage must end at the next milestone")
    for epoch in range(start, until):
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        graph = None
        if variant != "R0" and epoch >= Settings().warmup:
            graph = refresh(
                model[0],
                loader.dataset.unlabelled_dataset,
                center,
                variant,
                epoch,
                device,
                workers=8,
            )
        refresh_time = time.monotonic() - begin
        stats = train_epoch(model, loader, optimizer, criterion, epoch, graph, device)
        lr = optimizer.param_groups[0]["lr"]
        # Isolated official evaluator. Results are exported, never supplied to train_epoch or refresh.
        all_acc, old_acc, new_acc = evaluate(
            model, evaluation, epoch, "Train ACC Unlabelled", args
        )
        scheduler.step()
        completed = epoch + 1
        metrics = dict(
            epoch=completed,
            all=float(all_acc),
            old=float(old_acc),
            new=float(new_acc),
            lr=lr,
            refresh_seconds=refresh_time,
            epoch_seconds=time.monotonic() - begin,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            **stats,
        )
        append(folder / "metrics.jsonl", metrics)
        append(
            folder / "relationships.jsonl",
            {
                "epoch": completed,
                "graph": graph.record() if graph else None,
                "applied": graph.exposure_record(128) if graph else None,
            },
        )
        print(metrics, flush=True)
        del graph
        interrupt = standalone and (folder / "operator-request.json").is_file()
        if completed % 10 == 0 or interrupt:
            save(folder, completed, identity, model, optimizer, scheduler)
        if interrupt:
            return
        if standalone and completed % 10 == 0 and completed not in MILESTONES:
            print(f"Recovery bundle: {export(folder, recovery=True)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", choices=("R0", "R1", "R2"))
    p.add_argument("folder", type=Path)
    p.add_argument("until", type=int, choices=(100, 150, 200))
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    stage(a.folder, a.variant, a.until)
