"""Four paired, panel-conditioned epochs following the one-epoch Oracle MOVE audit."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts import audit_oracle_move as audit

ARMS = ("O0", "O5", "O6")
WORK = Path("dev_outputs/r2-audit")
SOURCE = WORK / "oracle-move"
RUN = WORK / "oracle-five"
HISTORY = Path("reports/r2_oracle_move_trajectory.json")
OUTPUT = Path("reports/r2_oracle_move_five_epochs.json")
REPORT = Path("reports/CRISH_ORACLE_MOVE_FIVE_EPOCHS.md")


def digest_array(x):
    return hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()).hexdigest()


def prepare():
    history = json.loads(HISTORY.read_text())
    model, loader, _, center, _, _ = audit.build()
    origin = WORK / "input/epoch-100.pt"
    state = audit.load_checkpoint(origin)
    truth, slots, mapping, _, _, _, panel, _, lookup = audit.prep(
        model, loader, center, WORK, state
    )
    assert audit.file_digest(origin) == history["checkpoint_sha256"]
    assert panel == history["panel_ids"] and len(set(panel)) == 588
    assert mapping == {int(k): v for k, v in history["frozen_slot_to_class"].items()}
    cached = torch.load(SOURCE / "step0-panel-tokens.pt", weights_only=True)
    assert cached["checkpoint_sha256"] == history["checkpoint_sha256"]
    assert cached["panel_ids"] == panel
    ds = loader.dataset
    ids = np.r_[ds.labelled_dataset.uq_idxs, ds.unlabelled_dataset.ids]
    weights = loader.sampler.weights.clone()
    weights[np.isin(ids, panel)] = 0
    plans = {}
    for epoch in range(2, 6):
        generator = torch.Generator().manual_seed(716000 + epoch)
        draw = torch.multinomial(weights, len(ds), True, generator=generator)
        plan = draw[: len(loader) * 128].reshape(-1, 128).numpy()
        assert not np.isin(ids[plan], panel).any()
        plans[epoch] = plan
    assert len(loader) == history["epoch_budget"] == 63
    return (
        model,
        ds,
        ids,
        truth,
        slots,
        mapping,
        sorted({slots[k] for k in lookup}),
        panel,
        cached["tokens"],
        history,
        plans,
        state,
    )


def preflight(context):
    model, _, _, truth, _, mapping, _, panel, tokens, hist, plans, origin = context
    signature = {
        "history": audit.file_digest(HISTORY),
        "endpoints": {},
        "plans": {str(e): digest_array(p) for e, p in plans.items()},
    }
    for arm in ARMS:
        path = SOURCE / f"{arm}-recovery.pt"
        state = audit.load_checkpoint(path)
        progress = state["progress"]
        assert (progress["arm"], progress["step"]) == (arm, 63)
        original = hist["arms"][arm]
        assert progress["receipt"]["steps"] == original["steps"]
        assert state["scheduler"]["last_epoch"] == origin["scheduler"]["last_epoch"] + 1
        opt, sched = audit.restore(model, state)
        assert sched.get_last_lr() == [g["lr"] for g in opt.param_groups]
        assert opt.state and all(
            s["momentum_buffer"].shape == p.shape for p, s in opt.state.items()
        )
        for name, p in model.named_parameters():
            if name.startswith("0.") and not name.startswith("0.blocks.11."):
                assert torch.equal(state["model"][name], origin["model"][name]), name
        measured, _ = audit.assess(model, tokens, truth[panel], mapping)
        assert measured == original["milestones"]["63"]["panel"], arm
        signature["endpoints"][arm] = audit.file_digest(path)
        print(
            "Verified endpoint, SGD, schedule, prefix and panel metrics",
            arm,
            flush=True,
        )
        del state, opt, sched
    return signature


def summarize(point, applied, total, stats):
    return {
        "panel": point,
        "applied": applied,
        "rows": total,
        "exposure": applied / total,
        "matched_stale": stats.get("matched_stale_source", 0),
        "matched_low_mass": stats.get("matched_low_mass", 0),
    }


def publish(results, signature):
    audit.write_json(
        OUTPUT,
        {
            "verification": signature,
            "epochs": results,
            "move_mass": 0.2,
            "panel_size": 588,
            "sampler_seed_base": 716000,
        },
    )
    lines = [
        "# Five-epoch Oracle MOVE trajectory",
        "",
        (
            "Epoch 1 is the published endpoint. Epochs 2–5 set the original sampler weights "
            "of all 588 held-out panel IDs to zero. The remaining relative weights, replacement, "
            "draw length and drop-last behavior are preserved. This changes the expected labeled "
            "fraction. All arms share the exact draw and two crops. Exposure divides applied edits "
            "by all 8,064 mixed rows per epoch. O6 replays eligible O5 occurrences with random "
            "supported destinations and skips eligibility divergence. The 10% target is nominal."
        ),
        "",
        (
            "Fixed accuracy uses the original global mapping. Rematching uses the panel labels. "
            "Geometry uses the same held-out panel. All values are single-trajectory diagnostics."
        ),
        "",
        "| Epoch | Arm | Fixed All/Old/New (%) | Rematched All/Old/New (%) | kNN-5 | Margin | Applied | Exposure |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for epoch, arms in results.items():
        for arm, row in arms.items():
            panel = row["panel"]
            acc = [
                " / ".join(
                    f"{100 * panel[m][g]['accuracy']:.2f}"
                    for g in ("all", "old", "new")
                )
                for m in ("fixed", "rematched")
            ]
            geometry = panel["geometry"]["all"]
            lines.append(
                f"| {epoch} | {arm} | {acc[0]} | {acc[1]} | {geometry['purity5']:.4f} | "
                f"{geometry['centroid_margin']:+.4f} | {row['applied']} | {100 * row['exposure']:.2f}% |"
            )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def train(context, signature):
    model, ds, ids, truth, slots, mapping, choices, panel, tokens, hist, plans, _ = (
        context
    )
    protected = set(panel)
    inverse = {v: k for k, v in mapping.items()}
    criterion = audit.DistillLoss(30, 200)
    # The original step helper indexes epoch 100. Temperature is constant for this extension.
    assert np.all(criterion.teacher_temp_schedule[100:105] == 0.04)
    results = {1: {}}
    for arm in ARMS:
        p = hist["arms"][arm]
        stats = Counter()
        for step in p["steps"]:
            stats.update(step["stats"])
        results[1][arm] = summarize(
            p["milestones"]["63"]["panel"],
            p["realized_moves"],
            hist["denominator"],
            stats,
        )
    for epoch, plan in plans.items():
        results[epoch] = {}
        oracle = None
        for arm in ARMS:
            end = RUN / f"{arm}-epoch-{epoch}.pt"
            if end.exists():
                progress = audit.load_checkpoint(end)["progress"]
                assert (progress["signature"], progress["arm"], progress["epoch"]) == (
                    signature,
                    arm,
                    epoch,
                )
            else:
                start = (
                    SOURCE / f"{arm}-recovery.pt"
                    if epoch == 2
                    else RUN / f"{arm}-epoch-{epoch - 1}.pt"
                )
                state = audit.load_checkpoint(start)
                opt, sched = audit.restore(model, state)
                del state
                applied, ledger, stats = 0, [], Counter()
                for i, rows in enumerate(plan, 1):
                    sequence = (epoch - 1) * len(plan) + i
                    cache = RUN / f"batch-{i}.pt"
                    if arm == "O0" or not cache.exists():
                        batch = audit.load_batch(model, ds, rows.tolist(), sequence)
                        batch["sequence"] = sequence
                        audit.atomic_checkpoint(batch, cache)
                    else:
                        batch = torch.load(cache, weights_only=True)
                        if batch["sequence"] != sequence:
                            batch = audit.load_batch(model, ds, rows.tolist(), sequence)
                            batch["sequence"] = sequence
                            audit.atomic_checkpoint(batch, cache)
                    assert batch["ids"].tolist() == ids[rows].tolist()
                    quota = max(0, int(np.floor(0.1 * 128 * i + 0.5)) - applied)
                    matched = oracle[i - 1] if arm == "O6" else None

                    def choose(
                        q, keys, labels, quota=quota, sequence=sequence, matched=matched
                    ):
                        return audit.select_moves(
                            q,
                            keys,
                            labels,
                            truth,
                            mapping,
                            inverse,
                            slots,
                            protected,
                            quota,
                            3893 + sequence,
                            matched,
                            choices,
                        )

                    random.seed(69401 + sequence)
                    np.random.seed(69401 + sequence)
                    torch.manual_seed(69401 + sequence)
                    edits, counts = audit.training_step(
                        model,
                        opt,
                        batch,
                        criterion,
                        audit.no_edit if arm == "O0" else choose,
                    )
                    ledger.append(edits)
                    applied += len(edits)
                    stats.update(counts)
                sched.step()
                point, _ = audit.assess(model, tokens, truth[panel], mapping)
                progress = {
                    "epoch": epoch,
                    "arm": arm,
                    "signature": signature,
                    "edits": ledger,
                    "result": summarize(point, applied, plan.size, stats),
                }
                audit.save_epoch(end, model, opt, sched, progress)
            results[epoch][arm] = progress["result"]
            if arm == "O5":
                oracle = progress["edits"]
            print("Completed", epoch, arm, progress["result"]["exposure"], flush=True)
    publish(results, signature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(exist_ok=True)
    data = prepare()
    signature = preflight(data)
    if args.verify:
        audit.write_json(RUN / "verified.json", signature)
    else:
        assert signature == json.loads((RUN / "verified.json").read_text())
        train(data, signature)
