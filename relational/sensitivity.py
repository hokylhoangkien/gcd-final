"""Paired, label-blind Cars diagnostic. No evaluation loader or trajectory checkpoints."""

import argparse
import copy
import hashlib
import os
import random
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import DINOHead, DistillLoss, get_params_groups
from relational.graph import refresh
from relational.sensitivity_metrics import measure_batch
from relational.state import write_json
from relational.train import baseline, build


def tensor_digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def legal_batch(batch):
    images, legal, ids, mask = batch
    known = mask.flatten().bool()
    if not (legal[~known] == -1).all():
        raise ValueError("Unlabeled rows must use the image-only sentinel")
    return torch.cat(images), legal.cuda(), ids, known.cuda()


def run(output):
    model, train, _, center, _, protocol = build()
    backbone, fresh = model
    # DINOv1 has zero drop path in this protocol. Fix model mode across all arms.
    backbone.eval()
    device = torch.device("cuda:0")
    start = time.monotonic()
    blind = train.dataset.unlabelled_dataset
    if len(blind) != 6144:
        raise ValueError("Expected the complete Cars training split")
    print("Building the full reciprocal graph", flush=True)
    graph = refresh(backbone, blind, center, "R2", 10, device, workers=0, micro=8)

    # Explicit seeds decouple sampling/crops from any diagnostic initialization.
    def draw(seed, count):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        loader = DataLoader(train.dataset, sampler=train.sampler, batch_size=128)
        yield from islice(loader, count)

    learned = DINOHead(768, 196).cuda()
    learned.load_state_dict(fresh.state_dict())
    opt = torch.optim.SGD(
        get_params_groups(learned), lr=0.1, momentum=0.9, weight_decay=5e-5
    )
    criterion = DistillLoss(30, 200)
    warmed_ids = set()
    print("Running eight disposable head-only SimGCD steps", flush=True)
    for batch in draw(1279, 8):
        images, legal, ids, mask = legal_batch(batch)
        warmed_ids.update(ids.tolist())
        with torch.no_grad():
            h = torch.cat([backbone(x.cuda()) for x in images.split(8)])
        loss = baseline(*learned(h), legal, mask, criterion, 0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    del opt, batch, images, h, loss
    head_shift = (
        sum(
            (a.detach() - b.detach()).square().sum().item()
            for a, b in zip(fresh.parameters(), learned.parameters())
        )
        ** 0.5
    )
    assert head_shift > 0
    result = {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(),
        "graph": graph.health,
        "edge_sha256": tensor_digest(graph.edge_ids),
        "votes_sha256": tensor_digest(graph.votes),
        "key_sha256": tensor_digest(graph.keys),
        "head_parameter_l2_shift": head_shift,
        "head_only_warm_steps": 8,
        "warm_draw_seed": 1279,
        "measurement_draw_seed": 4421,
        "probe_seed": 4001,
        "microbatch": 8,
        "batches": [],
    }
    panel = copy.copy(blind)
    panel.transform = center
    order = np.random.default_rng(4001).permutation(len(panel))
    for number, batch in enumerate(draw(4421, 4), 1):
        print(f"Measuring paired batch {number}/4", flush=True)
        images, legal, ids, mask = legal_batch(batch)
        excluded = warmed_ids.union(ids.tolist())
        indices = [i for i in order if int(panel.ids[i]) not in excluded][:48]
        probe, sentinel, probe_ids = next(
            iter(
                DataLoader(
                    torch.utils.data.Subset(panel, indices), batch_size=len(indices)
                )
            )
        )
        assert (sentinel == -1).all()
        record = measure_batch(
            backbone,
            (fresh, learned),
            graph,
            criterion,
            images,
            legal,
            ids,
            mask,
            probe,
            probe_ids,
        )
        result["batches"].append(
            dict(
                batch=number,
                ids=ids.tolist(),
                input_sha256=tensor_digest(images),
                probe_ids=probe_ids.tolist(),
                **record,
            )
        )
        result["completed_batches"] = number
        result["seconds"] = time.monotonic() - start
        write_json(output, result)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    path = parser.parse_args().output
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(path)
