"""Bounded two-stage verifier diagnostic using the completed R2 audit cache."""

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest, load_checkpoint
from crish.global_repair import frozen
from crish.repair import VirtualClassifier, corrected_distill, proposals
from model import DistillLoss, get_params_groups
from relational.sensitivity import legal_batch
from relational.sensitivity_metrics import block_gradient, prefix
from relational.state import write_json
from relational.train import build
from scripts.audit_r2_virtual import encode

ARGS = SimpleNamespace(crish_mass=0.2, sup_weight=0.35, memax_weight=1.0)
EPS = 1e-7


def checkpoint_optimizer(model, state):
    opt = torch.optim.SGD(
        get_params_groups(model), lr=0.1, momentum=0.9, weight_decay=5e-5
    )
    opt.load_state_dict(state["optimizer"])
    return opt


def margin_targets(vc, cached, proxy, reference):
    """Choose slots and competitors before any update, with no semantic labels."""
    center, crops = cached[0], cached[1:]
    q = (vc.logits(crops[:, reference], vc.v) / 0.04).softmax(-1)
    conf, slots = q.max(-1)
    reliable = (slots[0] == slots[1]) & (conf.min(0).values >= 0.5)
    ref_slots = slots[0, reliable]
    sums = torch.zeros((vc.v.shape[0], center.shape[-1]), device=center.device)
    # Matrix reduction avoids nondeterministic CUDA scatter-add.
    if reliable.any():
        sums = (
            F.one_hot(ref_slots, num_classes=len(sums)).float().T
            @ center[reference][reliable]
        )
    counts = ref_slots.bincount(minlength=len(sums))
    centroids = F.normalize(sums, dim=-1)
    ps = vc.logits(proxy, vc.v).argmax(-1)
    target = ps[0]
    keep = (ps[0] == ps[1]) & (counts[target] >= 3) & ((counts >= 3).sum() > 1)
    sim = proxy.mean(0)[keep] @ centroids.T
    sim[:, counts < 3] = -torch.inf
    sim.scatter_(1, target[keep, None], -torch.inf)
    competitor = sim.argmax(1)
    directions = centroids[target[keep]] - centroids[competitor]
    meta = {
        "reference_n": len(reference),
        "reliable_n": int(reliable.sum()),
        "supported_slots": int((counts >= 3).sum()),
        "proxy_covered": int(keep.sum()),
        "targets": target[keep].tolist(),
        "competitors": competitor.tolist(),
        "covered_rows": keep.nonzero().flatten().tolist(),
    }
    return keep, directions, meta


def grad_bases(backbone, tokens, h, vc, teacher, legal, criterion, actions):
    v = vc.v.detach().clone().requires_grad_()
    logits = vc.logits(F.normalize(h, dim=-1).reshape(2, len(legal), -1), v)
    flat = logits.flatten(0, 1)
    p = (flat / 0.1).softmax(-1).mean(0)
    reg = 0.65 * ((p * p.clamp_min(1e-30).log()).sum() + math.log(len(p)))
    mask = legal >= 0
    reg += 0.35 * F.cross_entropy(
        logits[:, mask].flatten(0, 1) / 0.1, legal[mask].repeat(2)
    )
    params = list(backbone.blocks[-1].parameters())
    return [
        (
            torch.autograd.grad(loss, v, retain_graph=True)[0],
            block_gradient(loss, h, backbone, tokens, params),
        )
        for plan in [[], *[[a] for a in actions]]
        for loss in [
            reg
            + 0.65
            * corrected_distill(flat, teacher.flatten(0, 1), criterion, 0, plan, 0.2)
        ]
    ]


def predict(work, prior, scratch):
    start = time.monotonic()
    model, train, _, _, _, protocol = build()
    assert (
        protocol["split"]
        == json.loads((work / "input/config.json").read_text())["protocol"]["split"]
    )
    backbone, head = model.eval()
    bank_ids = np.load(work / "ids.npy")
    lookup = {int(i): j for j, i in enumerate(bank_ids)}
    saved_features = np.load(work / "features.npy", mmap_mode="r")
    criterion = DistillLoss(0, 200)
    results = {"checkpoints": {}, "semantic_epsilon": EPS}
    known_ds, blind = train.dataset.labelled_dataset, train.dataset.unlabelled_dataset
    for index, epoch in ((1, 100), (2, 150)):
        ckpt = work / f"input/epoch-{epoch}.pt"
        state = load_checkpoint(ckpt)
        model.load_state_dict(state["model"])
        cache = F.normalize(
            torch.from_numpy(saved_features[index].copy()).cuda(), dim=-1
        )
        reference_ids = torch.as_tensor(bank_ids, device="cuda")
        random.seed(4927)
        np.random.seed(4927)
        torch.manual_seed(4927)
        batches = iter(DataLoader(train.dataset, sampler=train.sampler, batch_size=128))
        endpoint = {"checkpoint_sha256": file_digest(ckpt), "batches": []}
        results["checkpoints"][str(epoch)] = endpoint
        for step, old in enumerate(prior[str(epoch)]["batches"]):
            begin = time.monotonic()
            model.load_state_dict(state["model"])
            images, legal, ids, _ = legal_batch(next(batches))
            if ids.tolist() != old["batch_ids"]:
                raise ValueError("Archived batch ID order differs")
            torch.empty(
                (), dtype=torch.int64
            ).random_()  # prior audit center-probe iterator
            full_probe_ids = np.asarray(old["probe_ids"])
            proxy_ids, semantic_ids = full_probe_ids[:128], full_probe_ids[128:]
            train_ids = set(ids.tolist())
            known_rows = [
                j for j, k in enumerate(known_ds.uq_idxs) if int(k) not in train_ids
            ]
            known_rows = (
                np.random.default_rng(861 + step)
                .choice(known_rows, 32, replace=False)
                .tolist()
            )
            with frozen(backbone, 907 + step, torch.device("cuda:0")):
                pviews, plegal, pids, _ = next(
                    iter(
                        DataLoader(
                            Subset(
                                train.dataset,
                                known_rows
                                + [len(known_ds) + lookup[k] for k in proxy_ids],
                            ),
                            batch_size=160,
                        )
                    )
                )
                sviews, sentinel, sids = next(
                    iter(
                        DataLoader(
                            Subset(blind, [lookup[k] for k in semantic_ids]),
                            batch_size=256,
                        )
                    )
                )
            assert (sentinel == -1).all() and np.array_equal(sids.numpy(), semantic_ids)
            assert (
                np.array_equal(pids[32:].numpy(), proxy_ids)
                and (plegal[32:] == -1).all()
            )
            assert not train_ids.intersection(pids.tolist() + sids.tolist())
            ref = ~torch.isin(
                reference_ids,
                torch.tensor(list(train_ids) + full_probe_ids.tolist(), device="cuda"),
            )
            tokens, probe_tokens, sem_tokens = (
                prefix(backbone, x) for x in (images, torch.cat(pviews), sviews[0])
            )
            h = encode(backbone, tokens).requires_grad_()
            x = F.normalize(h.detach(), dim=-1).reshape(2, len(ids), -1)
            px = F.normalize(encode(backbone, probe_tokens), dim=-1).reshape(
                2, len(pids), -1
            )
            vc = VirtualClassifier(head.last_layer, checkpoint_optimizer(model, state))
            raw = vc.logits(x, vc.v)
            teacher = (raw / 0.04).softmax(-1)
            counts = Counter(ids.tolist())
            unique = [i for i, k in enumerate(ids.tolist()) if counts[k] == 1]
            eligible = [
                replace(a, row=unique[a.row])
                for a in proposals(
                    x[:, unique],
                    teacher[:, unique],
                    ids[unique].tolist(),
                    legal[unique],
                    5,
                    len(unique),
                    0.2,
                    None,
                    False,
                )
            ]
            # Reserve up to two places per action type, then fill from the shared ranking.
            actions = [
                a
                for kind in ("move", "abstain")
                for a in [
                    candidate for candidate in eligible if candidate.kind == kind
                ][:2]
            ]
            actions = (actions + [a for a in eligible if a not in actions])[:4]
            keep, dirs, coverage = margin_targets(vc, cache, px[:, 32:], ref)
            anchor = (vc.logits(px, vc.v) / 0.1).softmax(-1)
            plegal = plegal.cuda()
            gradients = grad_bases(
                backbone, tokens, h, vc, raw, legal, criterion, actions
            )
            params = list(backbone.blocks[-1].parameters())
            inputs = [("keep", None), *[(a.kind, a) for a in actions]]
            rows, semantic = [], []
            for i, (kind, action) in enumerate(inputs):
                plan = [] if action is None else [action]
                v0 = vc.step(x, raw, legal, criterion, 0, plan, ARGS)
                with torch.no_grad():
                    model.load_state_dict(state["model"])
                opt = checkpoint_optimizer(model, state)
                gv, gb = gradients[i]
                head.last_layer.weight_v.grad = gv
                for p, g in zip(
                    params, gb.split([p.numel() for p in params]), strict=True
                ):
                    p.grad = g.view_as(p)
                opt.step()
                opt.zero_grad()
                torch.testing.assert_close(head.last_layer.weight_v, v0)
                features = F.normalize(
                    encode(backbone, probe_tokens), dim=-1
                ).reshape_as(px)
                v0_obj, v0_guard = vc.probe(v0, px, plegal, anchor)
                v1_obj, v1_guard = vc.probe(v0, features, plegal, anchor)
                margin = (
                    float((features[:, 32:][:, keep] * dirs).sum(-1).mean())
                    if keep.any()
                    else None
                )
                rows.append(
                    {
                        "kind": kind,
                        "id": None if action is None else int(ids[action.row]),
                        "source": None if action is None else action.source,
                        "target": None if action is None else action.target,
                        "objective": [v0_obj, v1_obj, margin],
                        "known_ce": [v0_guard, v1_guard, v1_guard],
                    }
                )
                semantic.append(
                    F.normalize(encode(backbone, sem_tokens), dim=-1).cpu().numpy()
                )
            filename = scratch / f"post-{epoch}-{step}.npy"
            np.save(filename, np.stack(semantic))
            endpoint["batches"].append(
                {
                    "batch": step,
                    "lr": vc.group["lr"],
                    "train_ids": ids.tolist(),
                    "proxy_ids": proxy_ids.tolist(),
                    "known_ids": pids[:32].tolist(),
                    "semantic_ids": semantic_ids.tolist(),
                    "coverage": coverage,
                    "eligible": dict(Counter(a.kind for a in eligible)),
                    "missing": 4 - len(actions),
                    "arms": rows,
                    "seconds": time.monotonic() - begin,
                    "semantic_sha256": file_digest(filename),
                }
            )
            print(epoch, step, len(actions), flush=True)
    results["prediction_seconds"] = time.monotonic() - start
    write_json(scratch / "frozen_predictions.json", results)


def summarize(groups, segment, variant):
    from scipy.stats import spearmanr

    pairs = []
    selected, random_gains, ranked = [], [], []
    abstentions = 0
    for batch in groups:
        trials = batch["arms"][1:]
        current = [
            (t["proxy_gain"][variant], t["semantic_gain"][segment])
            for t in trials
            if t["proxy_gain"][variant] is not None
        ]
        pairs.extend(current)
        if len(current) > 1 and np.ptp(np.array(current), axis=0).min() > 0:
            ranked.append(float(spearmanr(*zip(*current)).statistic))
        valid = [
            t
            for t in trials
            if t["proxy_gain"][variant] is not None
            and t["proxy_gain"][variant] > EPS
            and t["guard_delta"][variant] <= 0
        ]
        winner = max(valid, key=lambda t: t["proxy_gain"][variant]) if valid else None
        abstentions += winner is None
        selected.append(winner["semantic_gain"][segment] if winner else 0.0)
        # Match each verifier's decision budget, including no-op batches.
        random_gains.append(
            float(np.mean([t["semantic_gain"][segment] for t in trials]))
            if winner
            else 0.0
        )
    direction = lambda x: 0 if abs(x) <= EPS else (1 if x > 0 else -1)
    determinate = [
        (direction(p), direction(s)) for p, s in pairs if direction(p) and direction(s)
    ]
    rho = (
        float(spearmanr(*zip(*pairs)).statistic)
        if len(pairs) > 1 and np.ptp(np.array(pairs), axis=0).min() > 0
        else None
    )
    return {
        "pairs": len(pairs),
        "spearman_pooled": rho,
        "mean_within_batch_spearman": float(np.mean(ranked)) if ranked else None,
        "rankable_batches": len(ranked),
        "sign_matches": sum(p == s for p, s in determinate),
        "sign_determinate": len(determinate),
        "proxy_indeterminate": sum(not direction(p) for p, s in pairs),
        "semantic_indeterminate": sum(not direction(s) for p, s in pairs),
        "abstentions": abstentions,
        "selected_mean": float(np.mean(selected)),
        "dose_matched_random_mean": float(np.mean(random_gains)),
    }


def seal(work, scratch, out):
    from scripts.audit_r2_evidence import sealed_labels

    frozen_path = scratch / "frozen_predictions.json"
    receipt = json.loads(frozen_path.read_text())
    receipt["frozen_predictions_sha256"] = file_digest(frozen_path)
    start = time.monotonic()
    y = sealed_labels()
    ids = np.load(work / "ids.npy")
    features = np.load(work / "features.npy", mmap_mode="r")
    groups = []
    for index, (epoch, segment) in enumerate(receipt["checkpoints"].items(), 1):
        for b in segment["batches"]:
            groups.append(b)
            f = scratch / f"post-{epoch}-{b['batch']}.npy"
            assert file_digest(f) == b["semantic_sha256"]
            query = torch.from_numpy(np.load(f)).cuda().double()
            reference = ~np.isin(
                ids, b["train_ids"] + b["proxy_ids"] + b["semantic_ids"]
            )
            bank = F.normalize(
                torch.from_numpy(features[index, 0, reference]).cuda().double(), dim=-1
            )
            same = torch.from_numpy(
                y[b["semantic_ids"], None] == y[ids[reference]][None, :]
            ).cuda()
            covered = same.any(-1) & (~same).any(-1)
            similarities = query @ bank.T
            margin = similarities.masked_fill(~same, -torch.inf).amax(
                -1
            ) - similarities.masked_fill(same, -torch.inf).amax(-1)
            delta = margin - margin[:1]
            b["semantic_coverage"] = {}
            for group, mask in (
                ("all", covered),
                ("old", covered & torch.from_numpy(y[b["semantic_ids"]] < 98).cuda()),
                ("new", covered & torch.from_numpy(y[b["semantic_ids"]] >= 98).cuda()),
            ):
                b["semantic_coverage"][group] = int(mask.sum())
                if not mask.any():
                    raise ValueError(f"No supported semantic probes for {group}")
                for i, arm in enumerate(b["arms"]):
                    arm.setdefault("semantic_gain", {})[group] = (
                        delta[i, mask].mean().item()
                    )
            base = b["arms"][0]
            for arm in b["arms"]:
                arm["proxy_gain"] = [
                    (b - a if v < 2 else a - b) if a is not None else None
                    for v, (a, b) in enumerate(
                        zip(arm["objective"], base["objective"], strict=True)
                    )
                ]
                arm["guard_delta"] = [
                    a - b
                    for a, b in zip(arm["known_ce"], base["known_ce"], strict=True)
                ]
    receipt["summary"] = {
        seg: {f"V{v}": summarize(groups, seg, v) for v in range(3)}
        for seg in ("all", "old", "new")
    }
    receipt["sealed_seconds"] = time.monotonic() - start
    write_json(out, receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "previous_virtual", "scratch", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    a.scratch.mkdir(exist_ok=True, parents=True)
    predict(a.work, json.loads(a.previous_virtual.read_text()), a.scratch)
    seal(a.work, a.scratch, a.output)
