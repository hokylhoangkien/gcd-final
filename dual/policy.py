"""Label-blind complementary-fold CLIP ranking and live MOVE eligibility."""

import copy
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from crish.data import ImageOnly
from crish.global_repair import Clock, Deadline, frozen
from crish.repair import Action


@dataclass(frozen=True)
class Settings:
    warmup: int = 10
    coverage: float = 0.1
    mixed_exposure_cap: float = 0.04
    batch_cap: int = 8
    mass: float = 0.2
    refresh_timeout: float = 300
    seed: int = 1


def folds(n, epoch, seed):
    order = np.random.default_rng(seed + (epoch + 1) * 100003).permutation(n)
    a, b = np.array_split(order, 2)
    return ((np.sort(a), np.sort(b)), (np.sort(b), np.sort(a)))


@torch.inference_mode()
def scores(x, slots, ids, partition, clock=None):
    s = torch.as_tensor(slots, dtype=torch.long, device=x.device)
    risk = np.full(len(ids), -np.inf)
    target = np.full(len(ids), -1, dtype=np.int64)
    for q, bank in partition:
        if np.intersect1d(ids[q], ids[bank]).size:
            raise ValueError("D1 query/reference IDs overlap")
        refs = s[bank]
        present = np.unique(slots[bank])
        for group in np.array_split(q, max(1, (len(q) + 255) // 256)):
            if clock:
                clock.check()
            similarity = x[group] @ x[bank].T
            support = torch.full((len(group), 196), -torch.inf, device=x.device)
            for slot in present:
                support[:, slot] = similarity[:, refs == int(slot)].max(1).values
            row = torch.arange(len(group), device=x.device)
            current = support[row, s[group]].clamp_min(-1)
            support[row, s[group]] = -torch.inf
            alternative, index = support.max(1)
            risk[group] = (alternative - current).cpu().numpy()
            target[group] = torch.where(alternative.isfinite(), index, -1).cpu().numpy()
    return risk, target


@torch.inference_mode()
def snapshot(model, blind, center, device, clock, *, workers=8, batch=32):
    if not isinstance(blind, ImageOnly):
        raise TypeError("Expected unlabeled ImageOnly snapshot")
    ds = copy.copy(blind)
    ds.transform = center
    logits, x, ids = [], [], []
    for image, legal, keys in DataLoader(
        ds, batch_size=batch, num_workers=workers, pin_memory=True
    ):
        clock.check()
        if (legal != -1).any():
            raise ValueError("Unexpected label in D1 snapshot")
        h = F.normalize(model[0](image.to(device, non_blocking=True)), dim=1)
        x.append(h)
        logits.append(model[1].last_layer(h))
        ids.extend(keys.tolist())
    np.testing.assert_array_equal(ids, blind.ids)
    return torch.cat(logits).argmax(1).cpu().numpy(), torch.cat(x)


class Plan:
    def __init__(self, learned, proposals, budget, settings, record):
        self.learned = frozenset(learned)
        self.proposals = {
            key: (src, dst, rank) for rank, (key, src, dst, _) in enumerate(proposals)
        }
        if set(self.proposals) & self.learned:
            raise ValueError("Known ID in D1 proposals")
        self.budget, self.settings, self.details = budget, settings, record
        self.applied, self.counts = [], Counter()

    @torch.no_grad()
    def select(self, logits, legal, keys, epoch, criterion):
        active = []
        for row, (key, label) in enumerate(zip(keys, legal, strict=True)):
            if (label >= 0) != (key in self.learned):
                raise ValueError("ID and legal-label mask disagree")
            if label < 0 and label != -1:
                raise ValueError("Unexpected unlabeled target")
            if key in self.proposals:
                active.append((row, key, *self.proposals[key]))
        self.counts["encounters"] += len(active)
        if not active:
            return [], []
        p = (logits.detach().float() / criterion.teacher_temp_schedule[epoch]).softmax(
            -1
        )
        p = p.reshape(2, len(keys), -1)
        rows = [r[0] for r in active]
        inferred = p[:, rows].mean(0).argmax(1).tolist()
        mass = p[:, rows, [r[2] for r in active]].min(0).values.tolist()
        eligible = []
        for (row, key, src, dst, rank), current, dose in zip(
            active, inferred, mass, strict=True
        ):
            if current != src:
                self.counts["stale_source"] += 1
            elif dose < self.settings.mass:
                self.counts["low_teacher_mass"] += 1
            else:
                eligible.append((rank, row, key, src, dst))
        chosen = sorted(eligible)
        available = max(0, self.budget - len(self.applied))
        cap = min(available, self.settings.batch_cap)
        self.counts["batch_cap_skipped"] += max(
            0, len(chosen) - self.settings.batch_cap
        )
        self.counts["epoch_budget_skipped"] += max(
            0, min(len(chosen), self.settings.batch_cap) - available
        )
        chosen = chosen[:cap]
        return [Action(row, "move", src, dst) for _, row, _, src, dst in chosen], [
            [row, key, src, dst] for _, row, key, src, dst in chosen
        ]

    def commit(self, step, pending):
        self.applied.extend([step, *v] for v in pending)

    def record(self, rows):
        n = len(self.applied)
        return {
            **self.details,
            "counts": dict(self.counts),
            "rows": rows,
            "budget": self.budget,
            "applied": n,
            "unique_applied": len({x[2] for x in self.applied}),
            "mass": n * self.settings.mass,
            "exposure": n / rows if rows else 0,
            "ledger": self.applied,
        }


def refresh(
    model, loader, center, clip, epoch, settings, device, *, workers=8, diagnostic=False
):
    ds = loader.dataset
    ids = np.asarray(ds.unlabelled_dataset.ids)
    known = ds.labelled_dataset.uq_idxs.tolist()
    if len(np.unique(np.r_[ids, known])) != len(ids) + len(known) or len(clip) != len(
        ids
    ):
        raise ValueError("Overlapping, duplicate or misaligned D1 ID bank")
    clock = Clock(settings.refresh_timeout, device)
    partition = folds(len(ids), epoch, settings.seed)
    budget = int(len(loader) * loader.batch_size * settings.mixed_exposure_cap)
    proposal, pair = [], None
    details = {
        "status": "ok",
        "fold_sizes": [[len(q), len(b)] for q, b in partition],
        "candidate_budget": int(len(ids) * settings.coverage),
    }
    try:
        with frozen(model, settings.seed + epoch, device):
            with clock.stage("dino_snapshot"):
                slots, dino = snapshot(
                    model, ds.unlabelled_dataset, center, device, clock, workers=workers
                )
            with clock.stage("clip_retrieval"):
                risk, target = scores(clip, slots, ids, partition, clock)
                good = np.flatnonzero(np.isfinite(risk) & (risk > 0) & (target >= 0))
                ranked = good[np.lexsort((ids[good], -risk[good]))][
                    : details["candidate_budget"]
                ]
                proposal = [
                    [int(ids[i]), int(slots[i]), int(target[i]), float(risk[i])]
                    for i in ranked
                ]
                details.update(
                    eligible_positive=len(good),
                    proposed=len(proposal),
                    failed_destination=int((target < 0).sum()),
                    candidate_ids=ids[ranked].tolist(),
                    slot_counts=np.bincount(slots, minlength=196).tolist(),
                    feature_variance=float(dino.var(0, unbiased=False).mean()),
                )
            if diagnostic:
                with clock.stage("paired_dino_retrieval"):
                    dr, dt = scores(dino, slots, ids, partition, clock)
                    other = np.flatnonzero(np.isfinite(dr) & (dr > 0) & (dt >= 0))
                    other = other[np.lexsort((ids[other], -dr[other]))][: len(ranked)]
                    pair = {
                        "common_ids": ids[ranked].tolist(),
                        "same_source_slots": slots[ranked].tolist(),
                        "clip_dest": target[ranked].tolist(),
                        "dino_dest": dt[ranked].tolist(),
                        "clip_risk": risk[ranked].tolist(),
                        "dino_risk": [
                            float(x) if np.isfinite(x) else None for x in dr[ranked]
                        ],
                        "equal_budget_rank_overlap": len(set(other) & set(ranked)),
                        "equal_budget_dino_n": len(other),
                        "slots": slots,
                        "partition": partition,
                    }
                    alt_risk, alt_target = scores(
                        clip,
                        slots,
                        ids,
                        folds(len(ids), epoch, settings.seed + 1),
                        clock,
                    )
                    alt = np.flatnonzero(
                        np.isfinite(alt_risk) & (alt_risk > 0) & (alt_target >= 0)
                    )
                    alt = alt[np.lexsort((ids[alt], -alt_risk[alt]))][
                        : details["candidate_budget"]
                    ]
                    pair["stability"] = {
                        "different_reference_split_n": len(alt),
                        "selected_overlap": len(set(alt) & set(ranked)),
                        "destination_agreement_on_same_candidates": float(
                            np.mean(alt_target[ranked] == target[ranked])
                        )
                        if len(ranked)
                        else None,
                    }
        clock.check()
    except Deadline:
        proposal, pair = [], None
        details = {"status": "timeout", "proposed": 0}
    details.update(
        seconds=time.monotonic() - clock.start,
        runtime=dict(clock.times),
        proposals=proposal,
    )
    return Plan(known, proposal, budget, settings, details), pair
