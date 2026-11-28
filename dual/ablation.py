"""Dose-matched D1 CLIP, random-destination and DINO evidence ablations."""

import argparse
import copy
import math
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from crish.experiment import atomic_checkpoint, recover_ledger, restore_rng, rng_state
from crish.global_repair import Clock, Deadline, frozen
from crish.repair import Action
from model import DistillLoss
from relational.sensitivity_metrics import cls, prefix
from relational.state import append
from relational.train import build
from train import test as official_test

from . import cache
from .policy import Plan, Settings, folds, scores, snapshot
from .train import optimizer_for, scheduler_for, step

ARMS = ("D1-matched", "D1-Random", "D1-DINO")


def ranked(scores_and_targets, slots, ids, limit):
    risk, dst = scores_and_targets
    good = np.flatnonzero(np.isfinite(risk) & (risk > 0) & (dst >= 0))
    order = good[np.lexsort((ids[good], -risk[good]))][:limit]
    rows = [(int(ids[i]), int(slots[i]), int(dst[i]), float(risk[i])) for i in order]
    return rows, len(good)


def random_destinations(clip_rows, random_slots, ids, partition, epoch, seed):
    """Use a per-ID RNG and alternative occupied slots in the Random arm's bank."""
    banks = {}
    for q, ref in partition:
        if np.intersect1d(ids[q], ids[ref]).size:
            raise ValueError("Random proposal query/reference overlap")
        present = np.unique(random_slots[ref])
        banks.update({int(ids[i]): present for i in q})
    output, unavailable = [], 0
    for key, source, clip_target, risk in clip_rows:
        bank = banks[key]
        eligible = bank[(bank != source) & (bank != clip_target)]
        if len(eligible) == 0:
            unavailable += 1
            continue
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, key]))
        output.append((key, source, int(generator.choice(eligible)), risk))
    return output, unavailable


def refresh(models, loader, center, clip, epoch, settings, device, workers):
    ds = loader.dataset
    known = ds.labelled_dataset.uq_idxs.tolist()
    ids = np.asarray(ds.unlabelled_dataset.ids)
    if len(np.unique(np.r_[ids, known])) != len(ids) + len(known) or len(clip) != len(
        ids
    ):
        raise ValueError("Overlapping, duplicate or misaligned ablation IDs")
    partition = folds(len(ids), epoch, settings.seed)
    quota = int(len(ids) * settings.coverage)
    budget = int(len(loader) * loader.batch_size * settings.mixed_exposure_cap)
    records, plans = {}, {}
    clip_rows = []
    for arm in ARMS:
        clock = Clock(settings.refresh_timeout, device)
        proposals, eligible, short = [], 0, 0
        status = "warmup" if epoch < settings.warmup else "ok"
        if status == "ok":
            try:
                with frozen(models[arm], settings.seed + epoch, device):
                    with clock.stage("snapshot"):
                        slots, dino = snapshot(
                            models[arm],
                            ds.unlabelled_dataset,
                            center,
                            device,
                            clock,
                            workers=workers,
                        )
                    with clock.stage("ranking"):
                        if arm == "D1-Random":
                            proposals, short = random_destinations(
                                clip_rows, slots, ids, partition, epoch, settings.seed
                            )
                            eligible = len(proposals)
                        else:
                            representation = clip if arm == "D1-matched" else dino
                            proposals, eligible = ranked(
                                scores(representation, slots, ids, partition, clock),
                                slots,
                                ids,
                                quota,
                            )
                clock.check()
            except Deadline:
                status, proposals, eligible, short = "timeout", [], 0, 0
        if arm == "D1-matched":
            clip_rows = proposals
        records[arm] = {
            "status": status,
            "candidate_quota": quota,
            "eligible_candidates": eligible,
            "candidate_shortfall": quota - len(proposals) if status != "warmup" else 0,
            "random_destination_shortfall": short,
            "runtime": dict(clock.times),
            "proposed": len(proposals),
            "proposals": proposals,
        }
        plans[arm] = Plan(known, proposals, budget, settings, records[arm])
    return plans


@torch.no_grad()
def preview(model, batch, micro):
    """Use the same train-mode logits as step(), restoring RNG after the preview."""
    device = next(model.parameters()).device
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        images = torch.cat(batch[0])
        if micro:
            h = torch.cat([cls(model[0], t) for t in prefix(model[0], images)])
        else:
            h = model[0](images.to(device, non_blocking=True))
        return model[1](h)[1]


def live_rows(plan, logits, batch, epoch, criterion):
    """Apply the original live guard before any cross-arm truncation."""
    shadow = copy.copy(plan)
    shadow.settings = replace(plan.settings, batch_cap=len(batch[1]))
    shadow.budget = len(batch[1])
    shadow.applied = []
    shadow.counts = Counter()
    _, eligible = shadow.select(
        logits, batch[1].tolist(), batch[2].tolist(), epoch, criterion
    )
    plan.counts.update(shadow.counts)
    return eligible


def matched_rows(plans, eligible):
    clip, random, dino = (eligible[a] for a in ARMS)
    random_by_row = {row[0]: row for row in random}
    common = []
    random_common = []
    for row, key, src, dst in clip:
        other = random_by_row.get(row)
        if other and other[:3] == [row, key, src]:
            if other[3] == dst:
                raise ValueError(
                    "Matched random control reproduced the CLIP destination"
                )
            common.append([row, key, src, dst])
            random_common.append(other)
    limit = min(p.settings.batch_cap for p in plans.values())
    left = min(p.budget - len(p.applied) for p in plans.values())
    count = max(0, min(limit, left, len(common), len(dino)))
    rows = dict(
        zip(ARMS, (common[:count], random_common[:count], dino[:count]), strict=True)
    )
    return rows, {
        "desired": max(0, min(limit, left, len(clip))),
        "clip_live": len(clip),
        "random_live": len(random),
        "shared_clip_random": len(common),
        "dino_live": len(dino),
        "dino_shortfall_before_matching": max(
            0, min(limit, left, len(clip)) - len(dino)
        ),
        "random_shortfall_before_matching": max(
            0, min(limit, left, len(clip)) - len(common)
        ),
        "applied_per_arm": count,
    }


class LockedBatch:
    """Consume paired edits once, after checking the real forward matches preview."""

    def __init__(self, plan, expected, rows):
        self.plan = plan
        self.expected = expected
        self.rows = rows
        self.settings = plan.settings

    def select(self, logits, legal, keys, epoch, criterion):
        if not torch.equal(logits.detach(), self.expected):
            raise RuntimeError(
                "Ablation selection preview differs from real training logits"
            )
        return [
            Action(row, "move", src, dst) for row, _, src, dst in self.rows
        ], self.rows

    def commit(self, index, pending):
        self.plan.commit(index, pending)


def matched_step(models, optimizers, batch, criterion, epoch, plans, index, *, micro):
    expected, eligible = {}, {}
    # No explicit forward randomness in the locked DINOv1 architecture. A parity
    # check also catches different kernels or a future stateful training layer.
    if any(p.proposals for p in plans.values()):
        for arm in ARMS:
            expected[arm] = preview(models[arm], batch, micro)
            eligible[arm] = live_rows(
                plans[arm], expected[arm], batch, epoch, criterion
            )
    else:
        eligible = {arm: [] for arm in ARMS}
    selected, counts = matched_rows(plans, eligible)
    losses = {}
    # Share forward RNG too, and advance the common stream exactly once.
    initial = rng_state()
    advanced = None
    for arm in ARMS:
        restore_rng(initial)
        locked = (
            LockedBatch(plans[arm], expected[arm], selected[arm])
            if selected[arm]
            else None
        )
        base, total, _ = step(
            models[arm],
            optimizers[arm],
            batch,
            criterion,
            epoch,
            locked,
            index,
            micro=micro,
        )
        losses[arm] = [base, total]
        if advanced is None:
            advanced = rng_state()
    restore_rng(advanced)
    return losses, counts


def run(folder, until, clip_path, *, micro=False, steps=None):
    models, optimizers, schedulers = {}, {}, {}
    model, loader, evaluation, center, args, protocol = build()
    device = next(model.parameters()).device
    if steps and folder.exists():
        raise ValueError("Use a new output directory for a partial-epoch smoke")
    protocol.pop("relational")
    settings = Settings()
    clip, receipt = cache.load(
        clip_path, loader.dataset.unlabelled_dataset, protocol["split"], device
    )
    for arm in ARMS:
        # build() reseeds both official model initialization and the sampler RNG.
        models[arm] = model if arm == ARMS[0] else build()[0]
        if arm != ARMS[0]:
            for k, v in model.state_dict().items():
                if not torch.equal(v, models[arm].state_dict()[k]):
                    raise ValueError(f"Different ablation initialization: {k}")
        optimizers[arm] = optimizer_for(models[arm])
        schedulers[arm] = scheduler_for(optimizers[arm])
    identity = {
        "protocol": protocol,
        "settings": vars(settings),
        "clip": receipt,
        "arms": ARMS,
        "micro": micro,
    }
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / "recovery.pt"
    journal = folder / "matched.jsonl"
    journal.touch(exist_ok=True)
    start = 0
    if state.exists():
        saved = torch.load(state, map_location="cpu", weights_only=False, mmap=True)
        if saved["identity"] != identity:
            raise ValueError("Ablation resume identity changed")
        start = saved["epoch"]
        for arm in ARMS:
            models[arm].load_state_dict(saved["models"][arm])
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
            schedulers[arm].load_state_dict(saved["schedulers"][arm])
            sched = schedulers[arm]
            if sched.last_epoch != start or any(
                not math.isclose(g["lr"], lr)
                for g, lr in zip(
                    optimizers[arm].param_groups, sched.get_last_lr(), strict=True
                )
            ):
                raise ValueError("Ablation optimizer/scheduler checkpoint mismatch")
        restore_rng(saved["rng"])
        recover_ledger(journal, saved["journal_bytes"])
        del saved
    elif journal.stat().st_size:
        raise ValueError("The journal is missing its committed recovery checkpoint")
    if not start < until <= 200:
        raise ValueError("Target must advance the original 200-epoch schedule")
    criterion = DistillLoss(30, 200)
    for epoch in range(start, until):
        plans = refresh(
            models, loader, center, clip, epoch, settings, device, protocol["workers"]
        )
        for model in models.values():
            model.train()
        counts, dose, totals, n_rows = [], 0, {a: np.zeros(2) for a in ARMS}, 0
        for i, batch in enumerate(loader, 1):
            loss, tally = matched_step(
                models, optimizers, batch, criterion, epoch, plans, i, micro=micro
            )
            for arm in ARMS:
                totals[arm] += loss[arm]
            counts.append(tally)
            dose += tally["applied_per_arm"]
            n_rows += len(batch[1])
            if steps is not None and i >= steps:
                break
        if any(len(p.applied) != dose for p in plans.values()):
            raise ValueError("Matched intervention counts diverged")
        acc = {}
        if steps is None:
            for arm in ARMS:
                acc[arm] = [
                    float(x)
                    for x in official_test(
                        models[arm], evaluation, epoch, f"{arm} ACC", args
                    )
                ]
                schedulers[arm].step()
        append(
            journal,
            {
                "epoch": epoch + 1,
                "steps": len(counts),
                "partial": steps is not None,
                "batch_matching": counts,
                "accuracy": acc,
                "loss": {a: (v / len(counts)).tolist() for a, v in totals.items()},
                "arms": {a: p.record(n_rows) for a, p in plans.items()},
            },
        )
        print(
            f"epoch={epoch + 1} applies_per_arm={dose} mixed_rows={n_rows}", flush=True
        )
        if steps is not None:
            break
        atomic_checkpoint(
            {
                "identity": identity,
                "epoch": epoch + 1,
                "journal_bytes": journal.stat().st_size,
                "rng": rng_state(),
                "models": {a: m.state_dict() for a, m in models.items()},
                "optimizers": {a: o.state_dict() for a, o in optimizers.items()},
                "schedulers": {a: s.state_dict() for a, s in schedulers.items()},
            },
            state,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--until", type=int, default=1)
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--steps", type=int, choices=range(1, 64))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(args.folder, args.until, args.clip, micro=args.micro, steps=args.steps)


if __name__ == "__main__":
    main()
