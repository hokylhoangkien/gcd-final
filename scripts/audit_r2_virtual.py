"""Sealed A/B/C/D/E checkpoint-conditioned one-step interventions. Never writes model state."""

import argparse
import copy
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import load_checkpoint
from model import DistillLoss, get_params_groups
from relational.sensitivity import legal_batch
from relational.sensitivity_metrics import block_gradient, cls, prefix
from relational.state import write_json
from relational.train import baseline, build
from scripts.audit_r2_evidence import graph_at, sealed_labels


def cosine(x, y):
    return F.cosine_similarity(x, y, dim=0).item()


def rms(x):
    return x.square().mean().sqrt().item()


def encode(backbone, tokens):
    with torch.no_grad():
        return torch.cat([cls(backbone, x) for x in tokens])


def probe_scores(f, bank, idx, truth):
    sim = f @ bank.T
    sim[torch.arange(len(f)), idx] = -torch.inf
    same = truth[idx, None] == truth[None, :]
    knn = sim.topk(5).indices
    purity = (truth[knn] == truth[idx, None]).double().mean(1)
    margin = (
        sim.masked_fill(~same, -torch.inf).max(1).values
        - sim.masked_fill(same, -torch.inf).max(1).values
    )
    return {
        "purity5": purity.mean().item(),
        "margin": margin.mean().item(),
        "variance": f.var(0, unbiased=False).mean().item(),
    }, knn


def attraction_gradient(cost, denominator, h, backbone, tokens, params, weights):
    w = torch.as_tensor(weights, device=cost.device, dtype=cost.dtype)
    return block_gradient((cost * w).sum() / denominator, h, backbone, tokens, params)


def run(work, out):
    model, train, _, center, _, _ = build()
    backbone, head = model
    model.eval()
    blind = train.dataset.unlabelled_dataset
    panel = copy.copy(blind)
    panel.transform = center
    y = sealed_labels()
    write = {}
    for epoch, index in ((100, 1), (150, 2)):
        state = load_checkpoint(work / f"input/epoch-{epoch}.pt")
        model.load_state_dict(state["model"])
        graph = graph_at(work, index)
        bank = graph.keys.double()
        id_to_row = graph.lookup
        bank_y = torch.tensor(y[graph.ids.numpy()], device="cuda")
        if epoch == 100:
            with (work / "input/relationships.jsonl").open() as f:
                row = json.loads(next(line for i, line in enumerate(f, 1) if i == 101))
            edges = np.array(row["graph"]["edges"])[:, :3].astype(int)
            source = "committed_epoch101_support_and_votes"
        else:
            edges = np.column_stack([graph.edge_ids.numpy(), graph.votes.numpy()])
            source = "checkpoint150_reconstructed_policy_no_epoch151"
        incident = defaultdict(list)
        for edge, (u, v, vote) in enumerate(edges):
            incident[u].append((edge, id_to_row[v]))
            incident[v].append((edge, id_to_row[u]))
        base_parameters = [p for p in backbone.blocks[-1].parameters()]
        head_parameters = [p for p in head.parameters() if p.requires_grad]
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        truth = y[edges[:, 0]] == y[edges[:, 1]]
        evidence = edges[:, 2] - 1
        random.seed(4927)
        np.random.seed(4927)
        torch.manual_seed(4927)
        batches = iter(DataLoader(train.dataset, sampler=train.sampler, batch_size=128))
        results = []
        for step in range(3):
            with torch.no_grad():
                for name, p in named:
                    p.copy_(state["model"][name])
            images, legal, ids, mask = legal_batch(next(batches))
            used_ids = set(ids.tolist())
            used_ids.update(
                int(graph.ids[peer])
                for key in ids.tolist()
                for _, peer in incident[key]
            )
            candidates = np.array(
                [k for k in range(len(blind)) if int(blind.ids[k]) not in used_ids]
            )
            rows = np.random.default_rng(820 + step).choice(
                candidates, 384, replace=False
            )
            probe, sentinel, pids = next(
                iter(DataLoader(Subset(panel, rows), batch_size=len(rows)))
            )
            assert (sentinel == -1).all() and not set(ids.tolist()).intersection(
                pids.tolist()
            )
            tokens, pt = prefix(backbone, images), prefix(backbone, probe)
            h = encode(backbone, tokens).requires_grad_()
            pre = F.normalize(encode(backbone, pt).double(), dim=1)
            pi = torch.tensor(rows, device="cuda")
            before, near = probe_scores(pre, bank, pi, bank_y)
            factual = baseline(*head(h), legal, mask, DistillLoss(30, 200), epoch)
            gh = torch.autograd.grad(factual, head_parameters, retain_graph=True)
            gs = block_gradient(factual, h, backbone, tokens, base_parameters)
            occurrences = [
                (i, e, peer)
                for i, k in enumerate(ids.tolist())
                for e, peer in incident[k]
            ]
            anchor, ed, peer = np.array(occurrences).T
            hits = np.bincount(ed, minlength=len(edges))
            perm = evidence.copy()
            rng = np.random.default_rng(937 + step)
            for n in np.unique(hits):
                use = hits == n
                perm[use] = rng.permutation(evidence[use])
            oracle = np.where(truth, 1, -1)
            tg = torch.as_tensor(anchor, device="cuda")
            keys = bank[peer].float()
            cost = 1 - torch.einsum(
                "vnd,nd->vn",
                F.normalize(h, dim=1).reshape(2, len(ids), -1)[:, tg],
                keys,
            )
            den = 2 * len(ids) * 5

            gradient = partial(
                attraction_gradient, cost, den, h, backbone, tokens, base_parameters
            )

            keep = gradient(np.full(len(ed), 0.5))
            edits = {
                k: gradient(0.25 * s[ed])
                for k, s in (("C", evidence), ("D", perm), ("E", oracle))
            }
            if epoch == 150:
                actual, _, _ = graph.loss(h, ids)
                torch.testing.assert_close(
                    actual,
                    (cost * torch.as_tensor(0.5 + 0.25 * evidence[ed], device="cuda"))
                    .sum()
                    .float()
                    / den,
                )
            signs = {
                "A": np.zeros(len(ed)),
                "B": np.zeros(len(ed)),
                "C": evidence[ed],
                "D": perm[ed],
                "E": oracle[ed],
            }
            assert sorted(signs["C"]) == sorted(signs["D"])
            post = {}
            response = {}
            for arm in "ABCDE":
                with torch.no_grad():
                    for name, p in named:
                        p.copy_(state["model"][name])
                opt = torch.optim.SGD(
                    get_params_groups(model), lr=0.1, momentum=0.9, weight_decay=5e-5
                )
                # load_state_dict can alias CPU tensors on CPU; the target parameters live on CUDA.
                opt.load_state_dict(state["optimizer"])
                delta = edits.get(arm)
                rel = keep if arm != "A" else torch.zeros_like(keep)
                if delta is not None:
                    rel = rel + delta
                for p, g in zip(head_parameters, gh, strict=True):
                    p.grad = g.detach()
                for p, g in zip(
                    base_parameters,
                    (gs + rel).split([p.numel() for p in base_parameters]),
                    strict=True,
                ):
                    p.grad = g.view_as(p)
                opt.step()
                opt.zero_grad(set_to_none=True)
                feat = F.normalize(encode(backbone, pt).double(), dim=1)
                local = F.normalize(encode(backbone, tokens).double(), dim=1).reshape(
                    2, len(ids), -1
                )
                post[arm] = (feat, local)
                scores, nn = probe_scores(feat, bank, pi, bank_y)
                response[arm] = {
                    **scores,
                    "neighbor_turnover": 1
                    - (nn[:, :, None] == near[:, None, :])
                    .any(2)
                    .double()
                    .mean()
                    .item(),
                    "positive_occurrences": int((signs[arm] > 0).sum()),
                    "negative_occurrences": int((signs[arm] < 0).sum()),
                    "rel_ratio": float(rel.norm() / gs.norm()),
                    "rel_cos_base": cosine(rel, gs) if arm != "A" else None,
                    "edit_ratio": float(delta.norm() / gs.norm())
                    if delta is not None
                    else None,
                    "edit_cos_base": cosine(delta, gs) if delta is not None else None,
                    "edit_cos_oracle": cosine(delta, edits["E"])
                    if delta is not None
                    else None,
                }
                del opt
            kept, klocal = post["B"]
            for arm, (f, loc) in post.items():
                peer_shift = torch.einsum(
                    "vnd,nd->vn", (loc - klocal)[:, tg], keys.double()
                )
                response[arm].update(
                    probe_cls_rms_vs_B=rms((f - kept).norm(dim=1)),
                    probe_pair_rms_vs_B=rms(f @ f.T - kept @ kept.T),
                    shift_by_actual_evidence={
                        str(s): peer_shift[:, evidence[ed] == s].mean().item()
                        for s in (-1, 0, 1)
                    },
                )
                response[arm]["edge_cos_shift_vs_B"] = {}
                for sign in (-1, 0, 1):
                    for semantic in (False, True):
                        subset = (evidence[ed] == sign) & (truth[ed] == semantic)
                        if subset.any():
                            response[arm]["edge_cos_shift_vs_B"][
                                f"{sign}/{int(semantic)}"
                            ] = {
                                "occurrences": int(subset.sum()),
                                "mean": peer_shift[:, subset].mean().item(),
                            }
            results.append(
                {
                    "batch": step,
                    "batch_ids": ids.tolist(),
                    "probe_ids": pids.tolist(),
                    "base_grad_norm": gs.norm().item(),
                    "reference": before,
                    "arms": response,
                }
            )
            write[epoch] = {
                "graph_source": source,
                "lr": state["optimizer"]["param_groups"][0]["lr"],
                "batches": results,
            }
            write_json(out, write)
            print(epoch, step, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    run(args.work, args.out)
