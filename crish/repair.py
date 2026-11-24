"""Single-device CRiSH for the two-view SimGCD classifier."""

import math
import random
import time
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .reliability import blockwise_euclidean_knn, sparse_evidence


@dataclass(frozen=True)
class Action:
    row: int
    kind: str
    source: int
    target: int = -1


def corrected_distill(logits, teacher, criterion, epoch, actions, mass):
    if not actions:
        return criterion(logits, teacher, epoch)
    q = (teacher.detach().float() / criterion.teacher_temp_schedule[epoch]).softmax(-1)
    q = q.reshape(2, -1, q.size(-1))
    for a in actions:
        if a.kind == "abstain":
            q[:, a.row] = 0
        elif a.kind == "move":
            if (q[:, a.row, a.source] < mass).any():
                raise ValueError("Insufficient teacher mass")
            q[:, a.row, a.source] -= mass
            q[:, a.row, a.target] += mass
        else:
            raise ValueError("Unknown intervention")
    logp = (logits.float().reshape_as(q) / criterion.student_temp).log_softmax(-1)
    return -(q.flip(0) * logp).sum(-1).mean()


def proposals(x, teacher, ids, legal, k, limit, mass, generator, random_order):
    # x and teacher have shape [2, score rows, feature/classes].
    votes = teacher.mean(0).argmax(-1)
    neighbors = tuple(
        blockwise_euclidean_knn(F.normalize(v, dim=-1), min(k, len(ids) - 1)) for v in x
    )
    edges = sparse_evidence(ids, [neighbors])
    rows = {key: i for i, key in enumerate(ids)}
    adjacent = [[] for _ in ids]
    for (i, j), e in edges.items():
        if e.joint:
            adjacent[rows[i]].append((rows[j], e.joint))
            adjacent[rows[j]].append((rows[i], e.joint))
    ranked = []
    for i, peers in enumerate(adjacent):
        if legal[i] >= 0 or not peers:
            continue
        source = int(votes[i])
        support = Counter()
        for j, weight in peers:
            support[int(votes[j])] += weight
        priority = (sum(support.values()) - 2 * support[source]) / len(peers)
        choices = []
        if int((votes == source).sum()) >= 3 and (teacher[:, i, source] >= mass).all():
            targets = [
                t for t in support if t != source and int((votes == t).sum()) >= 2
            ]
            if targets:
                target = max(
                    targets,
                    key=lambda t: (support[t], float(teacher[:, i, t].mean()), -t),
                )
                choices.append(Action(i, "move", source, target))
        choices.append(Action(i, "abstain", source))
        ranked.append((-priority, ids[i], choices))
    ranked.sort()
    if random_order:
        generator.shuffle(ranked)
    return [a for _, _, choices in ranked[:limit] for a in choices]


class VirtualClassifier:
    """Functional WeightNorm classifier, fresh parameter and SGD state per call."""

    def __init__(self, layer, optimizer):
        if not isinstance(optimizer, torch.optim.SGD) or layer.weight_g.requires_grad:
            raise ValueError(
                "Expected official frozen-scale weight-normalized SGD classifier"
            )
        self.v = layer.weight_v.detach().float().clone()
        self.g = layer.weight_g.detach().float().clone()
        self.group = next(
            group.copy()
            for group in optimizer.param_groups
            if any(p is layer.weight_v for p in group["params"])
        )
        buf = optimizer.state.get(layer.weight_v, {}).get("momentum_buffer")
        self.buffer = None if buf is None else buf.detach().float().clone()

    def logits(self, x, v):
        return F.linear(x, self.g * F.normalize(v, dim=1))

    def step(self, x, teacher, legal, criterion, epoch, actions, args):
        v = self.v.clone().requires_grad_()
        logits = self.logits(x, v)
        flat = logits.flatten(0, 1)
        distill = corrected_distill(
            flat, teacher.flatten(0, 1), criterion, epoch, actions, args.crish_mass
        )
        p = (flat / 0.1).softmax(-1).mean(0)
        memax = (p * p.clamp_min(1e-30).log()).sum() + math.log(len(p))
        mask = legal >= 0
        cls = F.cross_entropy(
            logits[:, mask].flatten(0, 1) / 0.1, legal[mask].repeat(2)
        )
        loss = (1 - args.sup_weight) * (
            distill + args.memax_weight * memax
        ) + args.sup_weight * cls
        (grad,) = torch.autograd.grad(loss, v)
        group = self.group
        update = grad + group["weight_decay"] * v.detach()
        if group["momentum"]:
            buf = (
                update
                if self.buffer is None
                else group["momentum"] * self.buffer + (1 - group["dampening"]) * update
            )
            update = update + group["momentum"] * buf if group["nesterov"] else buf
        return self.v - group["lr"] * update

    @torch.no_grad()
    def probe(self, v, x, legal, anchor):
        logits = self.logits(x, v) / 0.1
        known = legal >= 0
        known_loss = F.cross_entropy(
            logits[:, known].flatten(0, 1), legal[known].repeat(2)
        )
        # Opposite-view frozen classifier distribution. Fixed for every alternative.
        cross = (
            -(anchor.flip(0)[:, ~known] * logits[:, ~known].log_softmax(-1))
            .sum(-1)
            .mean()
        )
        return float(known_loss + cross), float(known_loss)


def select(
    x, flat_teacher, ids, legal, layer, optimizer, criterion, epoch, args, quota=None
):
    start = time.perf_counter()
    rng = random.Random(args.crish_seed + epoch * 100003)
    record = {
        "epoch": epoch,
        "mode": args.crish_mode,
        "decisions": [],
        "accepted": [],
        "stop": "cap",
    }
    x = F.normalize(x.detach().float().reshape(2, len(ids), -1), dim=-1)
    teacher = flat_teacher.detach().float().reshape(2, len(ids), -1)
    ids = list(map(int, ids))
    counts = Counter(ids)
    # Drop all occurrences of repeated IDs from selection, so applied dose is exact.
    known, other = [], []
    for i, key in enumerate(ids):
        if counts[key] == 1:
            (known if legal[i] >= 0 else other).append(i)
    rng.shuffle(known)
    rng.shuffle(other)
    cap = args.crish_cap if args.crish_mode == "E4" else args.crish_fixed
    folds = 1 + max(args.crish_fixed, args.crish_cap)
    per = min(args.crish_probe, (len(known) - 1) // folds, (len(other) - 3) // folds)
    chosen = []
    if per < 1:
        record["stop"] = "insufficient_disjoint_ids"
    else:
        probes = [
            known[i * per : (i + 1) * per] + other[i * per : (i + 1) * per]
            for i in range(folds)
        ]
        score = known[per * folds :] + other[per * folds :]
        xs, ts, labels = x[:, score], teacher[:, score], legal[score]
        targets = (ts / criterion.teacher_temp_schedule[epoch]).softmax(-1)
        actions = proposals(
            xs,
            targets,
            [ids[i] for i in score],
            labels,
            args.crish_k,
            args.crish_boundaries,
            args.crish_mass,
            rng,
            args.crish_mode == "E1",
        )
        record.update(
            score_ids=[ids[i] for i in score],
            probe_ids=[[ids[i] for i in fold] for fold in probes],
            candidates=len(actions),
        )
        verified = args.crish_mode in ("E3", "E4")
        if verified:
            virtual = VirtualClassifier(layer, optimizer)
            anchors = [
                (virtual.logits(x[:, f], virtual.v) / 0.1).softmax(-1) for f in probes
            ]

            def evaluate(v, index):
                f = probes[index]
                return virtual.probe(v, x[:, f], legal[f], anchors[index])

            def step(plan):
                return virtual.step(xs, ts, labels, criterion, epoch, plan, args)

            original = step([])
            original_base, original_known = evaluate(original, 0)

        for n in range(cap):
            pool = [a for a in actions if all(a.row != b.row for b in chosen)]
            if any(a.kind == "move" for a in chosen):
                updated = targets.clone()
                for applied in chosen:
                    if applied.kind == "move":
                        updated[:, applied.row, applied.source] -= args.crish_mass
                        updated[:, applied.row, applied.target] += args.crish_mass
                legal_now = proposals(
                    xs,
                    updated,
                    [ids[i] for i in score],
                    labels,
                    args.crish_k,
                    len(score),
                    args.crish_mass,
                    rng,
                    False,
                )
                pool = [a for a in pool if a in legal_now]
            if quota is not None:
                if n >= len(quota):
                    record["stop"] = "matched_dose"
                    break
                pool = [a for a in pool if a.kind == quota[n]]
            if not pool:
                record["stop"] = "no_candidate"
                break
            if time.perf_counter() - start >= args.crish_seconds:
                record["stop"] = "deadline"
                break
            if verified:
                factual = step(chosen) if chosen else original
                base, known_base = (
                    evaluate(factual, 0) if chosen else (original_base, original_known)
                )
                winner = None
                best = args.crish_gain
                for a in pool:
                    if time.perf_counter() - start >= args.crish_seconds:
                        break
                    trial = step(chosen + [a])
                    value, known_value = evaluate(trial, 0)
                    gain = base - value
                    group_gain = original_base - value
                    row = {
                        "id": ids[score[a.row]],
                        "kind": a.kind,
                        "round": n,
                        "source": a.source,
                        "target": a.target,
                        "known_increase": known_value - known_base
                        if math.isfinite(known_value - known_base)
                        else None,
                        "discovery": gain if math.isfinite(gain) else None,
                        "group_discovery": group_gain
                        if math.isfinite(group_gain)
                        else None,
                        "accepted": False,
                    }
                    record["decisions"].append(row)
                    if (
                        math.isfinite(gain)
                        and gain > best
                        and known_value <= known_base
                        and math.isfinite(group_gain)
                        and group_gain > args.crish_gain
                        and known_value <= original_known
                    ):
                        best, winner = gain, (a, trial, row)
                if time.perf_counter() - start >= args.crish_seconds:
                    record["stop"] = "deadline"
                    break
                if winner is None:
                    record["stop"] = "discovery"
                    break
                a, trial, row = winner
                b, bk = evaluate(factual, n + 1)
                orig_b, orig_bk = evaluate(original, n + 1) if chosen else (b, bk)
                c, ck = evaluate(trial, n + 1)
                gain = b - c
                row["confirmation_known_increase"] = (
                    ck - bk if math.isfinite(ck - bk) else None
                )
                row["confirmation"] = gain if math.isfinite(gain) else None
                group_gain = orig_b - c
                row["group_confirmation"] = (
                    group_gain if math.isfinite(group_gain) else None
                )
                if (
                    not math.isfinite(gain)
                    or gain <= args.crish_gain
                    or ck > bk
                    or not math.isfinite(group_gain)
                    or group_gain <= args.crish_gain
                    or ck > orig_bk
                ):
                    record["stop"] = "confirmation"
                    break
            else:
                a = rng.choice(pool) if args.crish_mode == "E1" else pool[0]
            if time.perf_counter() - start >= args.crish_seconds:
                record["stop"] = "deadline"
                break
            if verified:
                row["accepted"] = True
            chosen.append(a)
            record["accepted"].append(
                {
                    "id": ids[score[a.row]],
                    "kind": a.kind,
                    "source": a.source,
                    "target": a.target,
                }
            )
        chosen = [Action(score[a.row], a.kind, a.source, a.target) for a in chosen]
    if quota is not None and [a.kind for a in chosen] != quota:
        raise RuntimeError("Requested E3 dose unavailable; comparison cannot proceed")
    record["seconds"] = time.perf_counter() - start
    record["counts"] = dict(Counter(a.kind for a in chosen))
    return chosen, record
