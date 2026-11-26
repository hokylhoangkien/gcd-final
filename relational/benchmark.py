"""Bounded, disposable Cars benchmark or preflight. Never opens a run checkpoint."""

import argparse
import json
import os
import platform
import shutil
import time
from itertools import islice
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from model import DistillLoss, get_params_groups
from relational import graph as graph_module
from relational.state import write_json
from relational.train import build, train_epoch


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


class TimedBatches:
    """Include input wait, host transfers, forward/backward and synchronization per batch."""

    def __init__(self, loader, count):
        self.loader = loader
        self.count = count
        self.seconds = []

    def __len__(self):
        return self.count

    def __iter__(self):
        start = now()
        for batch in islice(self.loader, self.count):
            yield batch
            end = now()
            self.seconds.append(end - start)
            start = end


def measure_graph(model, dataset, center, device, workers, micro):
    encoding, searches = [], []
    original_knn = graph_module.blockwise_euclidean_knn

    def timed_loader(*args, **kwargs):
        start = now()
        yield from DataLoader(*args, **kwargs)
        encoding.append(now() - start)
        print(f"Finished image pass {len(encoding)} in {encoding[-1]:.2f}s", flush=True)

    def timed_knn(*args, **kwargs):
        start = now()
        result = original_knn(*args, **kwargs)
        searches.append(now() - start)
        return result

    start = now()
    with (
        patch.object(graph_module, "DataLoader", timed_loader),
        patch.object(graph_module, "blockwise_euclidean_knn", timed_knn),
    ):
        graph = graph_module.refresh(
            model[0], dataset, center, "R2", 10, device, workers=workers, micro=micro
        )
    total = now() - start
    assert len(encoding) == 2 and len(searches) == 3
    return graph, {
        "center_encoding_seconds": encoding[0],
        "two_crop_encoding_seconds": encoding[1],
        "exact_knn_seconds": searches,
        "refresh_seconds": total,
        "nodes": graph.health["nodes"],
        "edges": graph.health["edges"],
        "peak_graph_bytes": torch.cuda.max_memory_allocated(),
    }


def run(args):
    # Exercise imports used only by training/endpoint export too.
    from relational.offline import run as sealed_geometry
    from train import test as evaluate

    if not callable(evaluate) or not callable(sealed_geometry):
        raise TypeError("Missing evaluation entry points")
    model, train, evaluation, center, _, protocol = build()
    blind = train.dataset.unlabelled_dataset
    if len(blind) != 6144 or len(set(blind.ids.tolist())) != 6144:
        raise ValueError(
            f"Expected the official 6144-image unlabeled Cars split, found {len(blind)}"
        )
    missing = [p for p in blind.images if not Path(p).is_file()]
    missing += [p for p in train.dataset.labelled_dataset.data if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} Cars files missing, first: {missing[0]}"
        )
    result = {
        "device": torch.cuda.get_device_name(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "protocol": protocol,
        "unlabeled": len(blind),
        "training_batches": len(train),
        "eval_batches": len(evaluation),
        "mode": args.mode,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "tools": {k: shutil.which(k) for k in ("tmux", "timeout", "ssh", "scp")},
    }
    if args.mode == "preflight":
        return result
    result.update(
        probe_batch=args.batch,
        eval_batch=args.eval_batch,
        graph_micro=args.micro,
        workers=args.workers,
    )
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    graph, result["graph"] = measure_graph(
        model, blind, center, device, args.workers, args.micro
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, dict(result, status="graph_complete_probes_pending"))
    probe = DataLoader(
        train.dataset,
        batch_size=args.batch,
        sampler=train.sampler,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    optimizer = torch.optim.SGD(
        get_params_groups(model), lr=0.1, momentum=0.9, weight_decay=5e-5
    )
    criterion = DistillLoss(30, 200)
    estimates = []
    for key, active in (("warmup", None), ("active", graph)):
        batches = TimedBatches(probe, 3)
        train_epoch(
            model,
            batches,
            optimizer,
            criterion,
            0 if active is None else 10,
            active,
            device,
        )
        estimates.append(
            (batches.seconds[0] + (len(train) - 1) * mean(batches.seconds[1:]))
            * train.batch_size
            / args.batch
        )
        result[key + "_probe_seconds"] = batches.seconds
    batches = TimedBatches(
        DataLoader(
            evaluation.dataset, batch_size=args.eval_batch, num_workers=args.workers
        ),
        3,
    )
    model.eval()
    with torch.no_grad():
        for images, _, _ in batches:
            model(images.to(device))[1].argmax(1).cpu()
    evaluation_time = batches.seconds[0] + (
        len(evaluation.dataset) / args.eval_batch - 1
    ) * mean(batches.seconds[1:])
    warm = estimates[0] + evaluation_time
    active_time = estimates[1] + evaluation_time + result["graph"]["refresh_seconds"]
    result.update(
        status="complete",
        eval_probe_seconds=batches.seconds,
        peak_total_bytes=torch.cuda.max_memory_allocated(),
        warm_epoch_seconds_estimate=warm,
        active_epoch_seconds_estimate=active_time,
        uninterrupted_200h_estimate=(10 * warm + 190 * active_time) / 3600,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("preflight", "benchmark"))
    p.add_argument("--output", type=Path)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--batch", type=int, choices=(8, 16, 32, 64, 128), default=128)
    p.add_argument(
        "--eval-batch", type=int, choices=(8, 16, 32, 64, 128, 256), default=256
    )
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.micro < 1 or args.workers < 0:
        p.error("Invalid microbatch or worker count")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    measured = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, measured)
    print(json.dumps(measured, indent=2))
