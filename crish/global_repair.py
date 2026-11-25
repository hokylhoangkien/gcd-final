"""Epoch-scoped CRiSH for the flat SimGCD classifier."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .confirmation import MAX_ATTEMPTS, confirm_ranked, sign_audit
from .experiment import paired_settings, recover_ledger, reference_dose
from .ids import rows_for_ids
from .reliability import blockwise_euclidean_knn, sparse_evidence
from .repair import Action, VirtualClassifier, corrected_distill
from .sampling import balanced_known_folds, global_neighbor_pool
from .scoring_batch import scoring_ids


class Deadline(Exception):
    pass


@dataclass(frozen=True)
class Proposal:
    sample_id: int
    kind: str
    source: int
    target: int | None = None
    level: int = 0


class Clock:
    def __init__(self, seconds, device):
        self.device = device
        self.start = time.monotonic()
        self.end = self.start + seconds
        self.times = Counter()
        self.encoded = Counter()
        self.current = None

    def check(self):
        if time.monotonic() >= self.end:
            raise Deadline

    @contextmanager
    def stage(self, name):
        self.check()
        self.current = name
        start = time.monotonic()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.times[name] += time.monotonic() - start


@contextmanager
def frozen(model, seed, device):
    modes = [(m, m.training) for m in model.modules()]
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        try:
            model.eval()
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            yield
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(py)
            np.random.set_state(np_state)


def eval_dataset(dataset, transform):
    result = copy.copy(dataset)
    for name in ("labelled_dataset", "unlabelled_dataset"):
        part = copy.copy(getattr(dataset, name))
        part.transform = transform
        setattr(result, name, part)
    return result


@torch.no_grad()
def encode(backbone, dataset, rows, device, args, clock, expected=None):
    subset = dataset if rows is None else Subset(dataset, rows.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.crish_encode_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    xs, keys, labels = [], [], []
    clock.check()
    for images, legal, ids, mask in loader:
        clock.check()
        views = images if isinstance(images, (list, tuple)) else [images]
        if len(views) != (1 if rows is None else 2):
            raise ValueError(
                "Expected deterministic snapshot or independent paired training crops"
            )
        # Limit activation memory to one encoder microbatch, including two-view reads.
        out = [F.normalize(backbone(v.to(device)).float(), dim=-1).cpu() for v in views]
        xs.append(torch.stack(out))
        keys.append(ids.long())
        labels.append(legal.long().masked_fill(~mask.flatten().bool(), -1))
        clock.encoded[clock.current] += len(ids) * len(views)
        clock.check()
    ids = torch.cat(keys)
    if expected is not None and not torch.equal(ids, expected):
        raise ValueError("Repair dataset order/ID mismatch")
    return torch.cat(xs, dim=1), ids, torch.cat(labels)


def candidates(state, pool, edges, counts, limit, random_order=None, carried=None):
    local = {int(state.ids[i]): int(state.labels[0][i]) for i in pool}
    adjacency = defaultdict(list)
    for (u, v), edge in edges.items():
        if edge.joint > 0:
            adjacency[u].append((v, edge.joint))
            adjacency[v].append((u, edge.joint))
    ranking = []
    for i in pool.tolist():
        key = int(state.ids[i])
        if state.labeled[i] or not adjacency[key]:
            continue
        source = local[key]
        support = Counter()
        for peer, weight in adjacency[key]:
            support[local[peer]] += weight
        score = (sum(support.values()) - 2 * support[source]) / len(adjacency[key])
        options = [Proposal(key, "abstain", source)]
        dest = [t for t in support if t != source and counts[t] >= 2]
        if counts[source] >= 3 and dest:
            target = min(dest, key=lambda t: (-support[t], t))
            options.append(Proposal(key, "move", source, target))
        ranking.append((-score, key, options))
    ranking.sort()
    if carried:
        # Reserve a boundary slot only when the exact old edit is legal again.
        retained, fresh = [], []
        for score, key, options in ranking:
            if key not in carried:
                fresh.append((score, key, options))
            elif carried[key][0] in options:
                retained.append((score, key, [carried[key][0]]))
        ranking = retained + fresh
    if random_order is not None:
        random_order.shuffle(ranking)
    return [a for _, _, options in ranking[:limit] for a in options], adjacency


def source_ok(q, row, action, mass):
    return int(q[:, row].mean(0).argmax()) == action.source and (
        action.kind != "move" or bool((q[:, row, action.source] >= mass).all())
    )


class EpochRepairs:
    def __init__(self, args):
        self.args = args
        self.fallback = getattr(args, "crish_confirmation_fallback", False)
        self.persistence = getattr(args, "crish_revalidated_persistence", False)
        if (self.fallback or self.persistence) and args.crish_mode not in ("E3", "E4"):
            raise ValueError(
                "Confirmation fallback and revalidated persistence require a verified global mode"
            )
        if (self.fallback or self.persistence) and args.crish_cap > 8:
            raise ValueError(
                "Verified extensions permit at most eight accepted actions"
            )
        self.carry = {}
        if self.persistence:
            for entry in getattr(args, "resume_carry", []):
                action = Proposal(**entry["action"])
                self.carry[action.sample_id] = (action, entry["origin_epoch"])
        self.path = Path(args.model_path).parent / "crish-global.jsonl"
        if (
            args.n_views != 2
            or not 0 < args.crish_mass <= 1
            or not math.isfinite(args.crish_gain)
            or args.crish_gain < 0
            or not math.isfinite(args.crish_seconds)
            or args.crish_seconds <= 0
            or min(
                args.crish_cap,
                args.crish_every,
                args.crish_k,
                args.crish_boundaries,
                args.crish_confirmations,
                args.crish_encode_batch,
                args.crish_trials,
                args.crish_probe_unknown,
            )
            < 1
            or args.crish_warmup < 0
            or args.crish_scoring <= args.crish_boundaries
            or args.crish_evidence < args.crish_scoring
            or args.crish_evidence <= args.crish_k
            or args.crish_probe_known < 4
            or args.crish_probe_known % 2
        ):
            raise ValueError("Invalid global CRiSH configuration")
        self.reference = None
        if getattr(args, "crish_global_suite", False) and args.crish_mode in (
            "E1",
            "E2",
        ):
            self.reference = reference_dose(args)
        if getattr(args, "resume_ledger_bytes", None) is None:
            self.path.touch(exist_ok=False)
        else:
            recover_ledger(self.path, args.resume_ledger_bytes)
        self.plan = {}
        self.epoch = None

    def due(self, epoch):
        return (
            epoch >= self.args.crish_warmup
            and (epoch - self.args.crish_warmup) % self.args.crish_every == 0
        )

    def log(self, record):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    def carry_state(self):
        return [
            {"action": asdict(a), "origin_epoch": born}
            for a, born in self.carry.values()
        ]

    def prepare(self, student, optimizer, dataset, transform, criterion, epoch):
        if self.epoch is not None:
            raise RuntimeError("Previous repair epoch was not closed")
        self.epoch = epoch
        self.plan = {}
        self.exposure = Counter()
        self.applied = Counter()
        self.applied_ids = set()
        args = self.args
        device = next(student.parameters()).device
        seed = (args.crish_seed * 1000003 + epoch * 103 + 7) % 2**32
        g = torch.Generator().manual_seed(seed)
        clock = Clock(args.crish_seconds, device)
        rec = {
            "event": "refresh",
            "epoch": epoch,
            "seed": seed,
            "accepted": [],
            "decisions": [],
            "confirmations": 0,
        }
        if getattr(args, "crish_global_suite", False):
            rec.update(mode=args.crish_mode, settings=paired_settings(args))
        if self.fallback:
            rec.update(
                confirmation_attempts=[],
                attempt_limit=MAX_ATTEMPTS,
                required_probe_folds=1 + MAX_ATTEMPTS,
            )
        if self.persistence:
            expired = [
                item for item in self.carry_state() if epoch - item["origin_epoch"] > 3
            ]
            for item in expired:
                del self.carry[item["action"]["sample_id"]]
            rec.update(expired=expired, carried=self.carry_state(), reaccepted=[])
        with frozen(student, seed, device):
            try:
                self._refresh(
                    student,
                    optimizer,
                    dataset,
                    transform,
                    criterion,
                    epoch,
                    device,
                    g,
                    clock,
                    rec,
                )
                clock.check()
            except Deadline:
                rec["stop"] = "timeout"
                if self.fallback and rec["confirmation_attempts"]:
                    last = rec["confirmation_attempts"][-1]
                    if last["result"] == "started":
                        last["result"] = "timeout"
        if self.persistence:
            rec["reaccepted"] = [
                {"action": asdict(a), "origin_epoch": self.carry[key][1]}
                for key, a in self.plan.items()
                if key in self.carry and a == self.carry[key][0]
            ]
        rec.update(
            runtime_seconds=time.monotonic() - clock.start,
            stages_seconds=dict(clock.times),
            encoded_images=dict(clock.encoded),
            accepted_counts=dict(Counter(a.kind for a in self.plan.values())),
        )
        self.log(rec)
        return rec

    def _refresh(
        self,
        student,
        optimizer,
        dataset,
        transform,
        criterion,
        epoch,
        device,
        g,
        clock,
        rec,
    ):
        args = self.args
        virtual = VirtualClassifier(student[1].last_layer, optimizer)
        with clock.stage("snapshot"):
            x, ids, legal = encode(
                student[0], eval_dataset(dataset, transform), None, device, args, clock
            )
            if len(ids.unique()) != len(ids):
                raise ValueError("Duplicate snapshot IDs")
            slots = virtual.logits(x[0].to(device), virtual.v).argmax(-1).cpu()
            state = SimpleNamespace(
                ids=ids,
                features=x[0],
                labels=[slots],
                labeled=legal >= 0,
                abstained=[torch.zeros_like(legal, dtype=torch.bool)],
            )
            rec["snapshot_rows"] = len(ids)
        with clock.stage("global_shortlist"):
            known = state.labeled.nonzero().flatten()
            other = (~state.labeled).nonzero().flatten()
            carry_rows = (
                [i for i in other.tolist() if int(ids[i]) in self.carry]
                if self.carry
                else []
            )
            if carry_rows:
                excluded = set(carry_rows)
                other = torch.tensor(
                    [i for i in other.tolist() if i not in excluded], dtype=torch.long
                )
            fold_count = 1 + (
                args.crish_confirmations
                if getattr(args, "crish_global_suite", False)
                else min(args.crish_cap, args.crish_confirmations)
            )
            try:
                if self.fallback:
                    fold_count = 9
                kfold = balanced_known_folds(
                    known,
                    legal[known],
                    args.crish_probe_known,
                    g,
                    per_class=2,
                    folds=fold_count,
                )
                if self.fallback:
                    remaining = ~torch.isin(known, torch.cat(kfold))
                    kfold += balanced_known_folds(
                        known[remaining],
                        legal[known[remaining]],
                        args.crish_probe_known,
                        g,
                        per_class=2,
                        folds=1 + MAX_ATTEMPTS - fold_count,
                    )
                    fold_count = len(kfold)
            except ValueError:
                rec["stop"] = "insufficient_known_probes"
                return
            if len(other) < fold_count * args.crish_probe_unknown:
                rec["stop"] = "insufficient_unlabeled_probes"
                return
            order = other[torch.randperm(len(other), generator=g)]
            folds = [
                torch.cat(
                    [
                        kf,
                        order[
                            i * args.crish_probe_unknown : (i + 1)
                            * args.crish_probe_unknown
                        ],
                    ]
                )
                for i, kf in enumerate(kfold)
            ]
            reserved = torch.cat(folds)
            if self.fallback:
                rec["probe_support"] = {
                    "folds": len(folds),
                    "known_per_fold": args.crish_probe_known,
                    "unlabeled_per_fold": args.crish_probe_unknown,
                    "distinct_ids": int(reserved.unique().numel()),
                }
            allowed = torch.ones(len(ids), dtype=torch.bool)
            allowed[reserved] = False
            available = allowed.nonzero().flatten()
            if (
                len(available) < args.crish_evidence
                or not state.labeled[available].any()
                or state.labeled[available].all()
                or len(carry_rows) > args.crish_evidence
            ):
                rec["stop"] = "insufficient_context"
                return
            try:
                pool = global_neighbor_pool(
                    state,
                    available,
                    args.crish_evidence,
                    args.crish_k,
                    g,
                    deadline=clock.end,
                    required_rows=carry_rows,
                )
            except RuntimeError:
                clock.check()
                raise
            rec.update(
                probe_ids=[ids[f].tolist() for f in folds],
                evidence_ids=ids[pool].tolist(),
            )
        with clock.stage("evidence"):
            trials = []
            for trial in range(args.crish_trials):
                views, _, _ = encode(
                    student[0], dataset, pool, device, args, clock, ids[pool]
                )
                if trial == 0:
                    first = views
                trials.append(
                    tuple(
                        blockwise_euclidean_knn(v.to(device), args.crish_k)
                        for v in views
                    )
                )
            edges = sparse_evidence(ids[pool], trials)
            counts = Counter(slots[available].tolist())
            peer_slots = dict(zip(ids[pool].tolist(), slots[pool].tolist()))
            random_order = (
                random.Random(rec["seed"])
                if self.reference is not None and args.crish_mode == "E1"
                else None
            )
            options, adj = candidates(
                state,
                pool,
                edges,
                counts,
                args.crish_boundaries,
                random_order,
                self.carry,
            )
            if self.persistence:
                rec["carry_geometry_ids"] = [
                    a.sample_id for a in options if a.sample_id in self.carry
                ]
            rec.update(
                joint_edges=sum(e.joint > 0 for e in edges.values()),
                proposed=len(options),
            )
            if not options:
                rec["stop"] = "no_boundary"
                return
        with clock.stage("scoring_batch"):
            score_ids, support = scoring_ids(
                state, options, adj, pool, reserved, args.crish_scoring
            )
            rows = rows_for_ids(ids, score_ids)
            if not state.labeled[rows].any():
                rows[-1] = available[state.labeled[available]][0]
                score_ids = ids[rows]
                support["known_anchor_replaced_filler"] = True
            cache = {int(row): first[:, i] for i, row in enumerate(pool)}
            missing = torch.tensor(
                [r for r in rows.tolist() if r not in cache], dtype=torch.long
            )
            if len(missing):
                extra, _, _ = encode(
                    student[0], dataset, missing, device, args, clock, ids[missing]
                )
                cache.update({r: extra[:, i] for i, r in enumerate(missing.tolist())})
            xs = torch.stack([cache[r] for r in rows.tolist()], dim=1).to(device)
            teacher = virtual.logits(xs, virtual.v)
            probs = (teacher / criterion.teacher_temp_schedule[epoch]).softmax(-1)
            labels = legal[rows].to(device)
            lookup = {k: i for i, k in enumerate(score_ids.tolist())}
            rec.update(score_ids=score_ids.tolist(), scoring=support)
            options = [
                a
                for a in options
                if source_ok(probs, lookup[a.sample_id], a, args.crish_mass)
            ]
            rec["score_eligible"] = len(options)
            if self.persistence:
                rec["carry_teacher_eligible_ids"] = [
                    a.sample_id for a in options if a.sample_id in self.carry
                ]
            if self.reference is not None:
                kinds, _ = self.reference[epoch]
                rec["planned_reference"] = kinds
                if not kinds:
                    rec["stop"] = "reference_keep"
                    return
                eligible = [a for a in options if a.kind == kinds[0]]
                if not eligible:
                    rec["stop"] = "reference_type_unavailable"
                    return
                clock.check()
                pick = (
                    random_order.choice(eligible)
                    if random_order is not None
                    else eligible[0]
                )
                self.plan[pick.sample_id] = pick
                rec["accepted"].append(asdict(pick))
                rec["stop"] = "matched_plan"
                return
            if not options:
                rec["stop"] = "no_current_candidate"
                return

        def get_probe(index):
            with clock.stage("probe_encoding"):
                f = folds[index]
                v, _, _ = encode(student[0], dataset, f, device, args, clock, ids[f])
                v = v.to(device)
                return (
                    v,
                    legal[f].to(device),
                    (virtual.logits(v, virtual.v) / 0.1).softmax(-1),
                )

        def step(plan):
            return virtual.step(xs, teacher, labels, criterion, epoch, plan, args)

        discovery = get_probe(0)
        with clock.stage("discovery"):
            original = step([])
            orig = virtual.probe(original, *discovery)
        accepted = []
        rejected = set()
        limit = MAX_ATTEMPTS if self.fallback else args.crish_confirmations
        current, baseline = original, orig
        for n in range(args.crish_cap):
            if rec["confirmations"] >= limit:
                rec["stop"] = "confirmation_budget"
                return
            best = args.crish_gain
            ranked = []
            with clock.stage("discovery"):
                for a in options:
                    clock.check()
                    if a.sample_id in self.plan or a in rejected:
                        continue
                    if a.kind == "move" and (
                        counts[a.source] < 3
                        or counts[a.target] < 2
                        or not any(
                            peer_slots[j] == a.target for j, _ in adj[a.sample_id]
                        )
                    ):
                        continue
                    edit = Action(lookup[a.sample_id], a.kind, a.source, a.target)
                    trial = step(accepted + [edit])
                    total, known = virtual.probe(trial, *discovery)
                    gain, group = baseline[0] - total, orig[0] - total
                    trace = {
                        "round": n,
                        "id": a.sample_id,
                        "kind": a.kind,
                        "accepted": False,
                    }
                    rec["decisions"].append(trace)
                    if not all(math.isfinite(x) for x in (gain, group, known)):
                        continue
                    trace.update(
                        gain=gain, group_gain=group, known_delta=known - baseline[1]
                    )
                    if (
                        gain > best
                        and group > args.crish_gain
                        and known <= min(baseline[1], orig[1])
                    ):
                        if self.fallback:
                            # Recreate the selected exact SGD step rather than retaining every weight matrix.
                            ranked.append((a, edit, None, (total, known), trace))
                        else:
                            ranked = [(a, edit, trial, (total, known), trace)]
                            best = gain
            if not ranked:
                rec["stop"] = "discovery"
                return
            if self.fallback:
                ranked.sort(key=lambda item: -item[-1]["gain"])
            winner = confirm_ranked(
                ranked,
                fallback=self.fallback,
                limit=limit,
                rejected=rejected,
                rec=rec,
                clock=clock,
                get_probe=get_probe,
                probe=virtual.probe,
                step=step,
                accepted=accepted,
                current=current,
                original=original,
                threshold=args.crish_gain,
                round_index=n,
                audit=lambda f, t, h, m: sign_audit(virtual, f, t, h, m),
            )
            if winner is None:
                return
            a, edit, trial, new_base, trace = winner
            accepted.append(edit)
            self.plan[a.sample_id] = a
            trace["accepted"] = True
            rec["accepted"].append(asdict(a))
            current, baseline = trial, new_base
            if a.kind == "move":
                edited = probs[:, edit.row].mean(0).clone()
                edited[a.source] -= args.crish_mass
                edited[a.target] += args.crish_mass
                new_slot = int(edited.argmax())
                counts[a.source] -= 1
                counts[new_slot] += 1
                peer_slots[a.sample_id] = new_slot
        rec["stop"] = "cap"

    def loss(self, logits, ids, labeled, criterion, epoch):
        if self.epoch != epoch:
            raise ValueError("Expired repair plan")
        self.pending = []
        self.exposure["training_rows"] += len(ids)
        hits = [(i, self.plan[k]) for i, k in enumerate(ids.tolist()) if k in self.plan]
        if hits:
            q = (
                logits.detach().float().reshape(2, len(ids), -1)
                / criterion.teacher_temp_schedule[epoch]
            ).softmax(-1)
            for i, a in hits:
                if bool(labeled[i]):
                    raise ValueError("Repair matched a legally labeled row")
                self.exposure["encounters"] += 1
                if int(q[:, i].mean(0).argmax()) != a.source:
                    self.exposure["stale_source"] += 1
                    continue
                if a.kind == "move" and not bool(
                    (q[:, i, a.source] >= self.args.crish_mass).all()
                ):
                    self.exposure["low_teacher_mass"] += 1
                    continue
                if self.reference is not None:
                    quota = self.reference[epoch][1].get(a.kind, 0)
                    pending = sum(edit.kind == a.kind for _, edit in self.pending)
                    if self.applied[a.kind] + pending >= quota:
                        self.exposure["dose_cap"] += 1
                        continue
                self.pending.append(
                    (a.sample_id, Action(i, a.kind, a.source, a.target))
                )
        return corrected_distill(
            logits,
            logits.detach(),
            criterion,
            epoch,
            [a for _, a in self.pending],
            self.args.crish_mass,
        )

    def finish_step(self, stepped):
        self.exposure["optimizer_steps" if stepped else "skipped_steps"] += 1
        for key, a in self.pending:
            if stepped:
                self.applied[a.kind] += 1
                self.applied_ids.add(key)
            else:
                self.exposure["amp_skipped_rows"] += 1
        self.pending = []

    def close_epoch(self):
        record = {
            "event": "exposure",
            "epoch": self.epoch,
            "accepted": len(self.plan),
            "applied": dict(self.applied),
            "applied_ids": sorted(self.applied_ids),
            "exposure": dict(self.exposure),
        }
        if self.reference is not None:
            kinds, target = self.reference[self.epoch]
            record.update(
                reference_applied=target,
                plan_matched=[a.kind for a in self.plan.values()] == kinds,
                exposure_matched=dict(self.applied) == target,
            )
        if self.persistence:
            for key, action in self.plan.items():
                if key in self.applied_ids:
                    self.carry.pop(key, None)
                else:
                    self.carry.setdefault(key, (action, self.epoch))
            record["carry_pending"] = self.carry_state()
        self.log(record)
        self.plan.clear()
        self.epoch = None
        return record
