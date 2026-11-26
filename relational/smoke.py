"""Bounded real-Cars diagnostic on a small unlabeled graph and one mixed batch."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DistillLoss
from relational.graph import Settings, refresh
from relational.state import write_json
from relational.train import baseline, build, sampler_for


def main(out):
    model, loader, _, center, _, protocol = build()
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    ds = copy.copy(loader.dataset)
    blind = copy.copy(ds.unlabelled_dataset)
    pick = np.random.default_rng(701).choice(len(blind), 128, replace=False)
    blind.images = [blind.images[i] for i in pick]
    blind.ids = blind.ids[pick]
    ds.unlabelled_dataset = blind
    device = torch.device("cuda:0")
    backbone = model[0]
    graph = refresh(backbone, blind, center, "R2", 10, device, micro=4)
    # Same graph/keys for all arms. Real mixed weighted draws, with replacement.
    batch = next(iter(DataLoader(ds, sampler=sampler_for(ds), batch_size=128)))
    images, legal, ids, mask = batch
    with torch.no_grad():
        tokens, h = [], []
        for images4 in torch.cat(images).split(4):
            x = backbone.prepare_tokens(images4.to(device))
            for block in list(backbone.blocks)[:-1]:
                x = block(x)
            tokens.append(x.cpu())
            h.append(backbone.norm(backbone.blocks[-1](x))[:, 0])
    h = torch.cat(h).requires_grad_()
    base = baseline(
        *model[1](h),
        legal.to(device),
        mask.flatten().bool().to(device),
        DistillLoss(30, 200),
        10,
    )
    edited, plain, pending = graph.loss(h, ids)
    coeff = Settings().coefficient
    block = backbone.blocks[-1]
    params = list(block.parameters())
    original = [p.detach().clone() for p in params]
    grads = []
    for objective in (base, coeff * plain, coeff * edited):
        (dh,) = torch.autograd.grad(objective, h, retain_graph=True)
        totals = [torch.zeros_like(p) for p in params]
        for x, incoming in zip(tokens, dh.split(4)):
            y = backbone.norm(block(x.to(device)))[:, 0]
            for acc, grad in zip(totals, torch.autograd.grad(y, params, incoming)):
                acc.add_(grad)
        grads.append(totals)
    vectors = [torch.cat([g.flatten() for g in group]) for group in grads]
    outcomes = []
    with torch.no_grad():
        for rel in (grads[1], grads[2]):
            for (name, p), orig, gb, gr in zip(
                block.named_parameters(), original, grads[0], rel
            ):
                decay = 0 if p.ndim == 1 or name.endswith(".bias") else 5e-5
                p.copy_(orig - 0.1 * (gb + gr + decay * orig))
            after = torch.cat(
                [backbone.norm(block(x.to(device)))[:, 0] for x in tokens]
            )
            outcomes.append(F.normalize(after, dim=-1))
        for param, orig in zip(params, original):
            param.copy_(orig)
    graph.applied(pending)
    exposure = graph.exposure_record(128)

    def compare(x):
        return {
            "norm": x.norm().item(),
            "ratio": (x.norm() / vectors[0].norm()).item(),
            "cosine": F.cosine_similarity(x, vectors[0], dim=0).item(),
        }

    diff = outcomes[1] - outcomes[0]
    pair_diff = outcomes[1] @ outcomes[1].T - outcomes[0] @ outcomes[0].T
    rows = [graph.lookup.get(key, len(graph.ids)) for key in ids.tolist()]
    keys = graph.keys[graph.peer_slots[rows].to(device)]
    cos_delta = torch.einsum("vbd,bkd->vbk", diff.reshape(2, len(ids), -1), keys)
    dose = (graph.weights - graph.base)[graph.edge_slots[rows].to(device)]
    signed = {
        kind: float(cos_delta[:, selected].mean())
        for kind, selected in (("strengthen", dose > 0), ("weaken", dose < 0))
        if selected.any()
    }
    torch.cuda.synchronize()
    result = {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(),
        "graph": graph.health,
        "proposed_strengthen": int((graph.delta > 0).sum()),
        "proposed_weaken": int((graph.delta < 0).sum()),
        "exposure": exposure,
        "population": 128,
        "batch": 128,
        "unlabeled_draws": int((~mask.bool()).sum()),
        "baseline_loss": base.item(),
        "plain_relational_loss": plain.item(),
        "edited_relational_loss": edited.item(),
        "baseline_block_norm": vectors[0].norm().item(),
        "weighted_R1": compare(vectors[1]),
        "weighted_R2": compare(vectors[2]),
        "edit": compare(vectors[2] - vectors[1]),
        "post_step_unit_cls_rms_l2": diff.norm(dim=1).square().mean().sqrt().item(),
        "post_step_pairwise_cosine_rms": pair_diff.square().mean().sqrt().item(),
        "post_step_peer_cosine_change": signed,
        "seconds": time.monotonic() - begin,
        "peak_vram": torch.cuda.max_memory_allocated(),
    }
    assert exposure["strengthen"] and exposure["weaken"] and result["edit"]["norm"] > 0
    assert result["post_step_pairwise_cosine_rms"] > 0
    write_json(out, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("exposure", "protocol")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main(parser.parse_args().output)
