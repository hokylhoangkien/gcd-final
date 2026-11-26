"""Bounded action-strength response and frozen E3 verifier decisions."""

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest, load_checkpoint
from crish.global_repair import frozen
from crish.repair import Action, VirtualClassifier, corrected_distill, proposals
from model import DistillLoss
from relational.sensitivity import legal_batch
from relational.sensitivity_metrics import prefix
from relational.state import write_json
from relational.train import build
from scripts.audit_r2_v3 import gradient_components
from scripts.audit_r2_verifier import (
    ARGS,
    EPS,
    checkpoint_optimizer,
    seal,
)
from scripts.audit_r2_virtual import encode

MOVE = (0.05, 0.1, 0.2, 0.4)
ABSTAIN = (0.25, 0.5, 1.0)
ORIGINAL = {"move": 0.2, "abstain": 1.0}


def decide(arms, adaptive):
    """E3 singleton discovery ranks only positive, strict-guarded proposals."""
    keep = arms[0]["v0"]
    candidates = [
        (i, keep[0][0] - a["v0"][0][0])
        for i, a in enumerate(arms[1:], 1)
        if (adaptive or a["strength"] == ORIGINAL[a["kind"]])
        and a["v0"][0][0] < keep[0][0]
        and a["v0"][0][1] <= keep[0][1]
    ]
    if not candidates:
        return {"selected": None, "reason": "discovery", "winner": None}
    i = max(candidates, key=lambda p: p[1])[0]
    score, known = arms[i]["v0"][1]
    if score >= keep[1][0]:
        return {"selected": None, "reason": "confirmation_gain", "winner": i}
    if known > keep[1][1]:
        return {"selected": None, "reason": "confirmation_known_ce", "winner": i}
    return {"selected": i, "reason": "accepted", "winner": i}


def strength_support(actions, teacher):
    moves = [a for a in actions if a.kind == "move"]

    def legal(a, mass):
        return bool((teacher[:, a.row, a.source] >= mass).all())

    invalid = {str(m): [a.row for a in moves if not legal(a, m)] for m in MOVE}
    common = [a for a in moves if legal(a, max(MOVE))]
    return (
        [*common[:2], *[a for a in actions if a.kind == "abstain"][:2]],
        invalid,
        len(common),
    )


def predict(work, previous, scratch):
    model, train, _, _, _, _ = build()
    backbone, head = model.eval()
    known, blind = train.dataset.labelled_dataset, train.dataset.unlabelled_dataset
    bank_ids = np.load(work / "ids.npy")
    assert np.array_equal(bank_ids, blind.ids)
    out = {"checkpoints": {}, "semantic_epsilon": EPS, "v0_gain_threshold": 0.0}
    criterion = DistillLoss(0, 200)
    for epoch in (100, 150):
        state = load_checkpoint(work / f"input/epoch-{epoch}.pt")
        endpoint = {
            "batches": [],
            "lr": state["optimizer"]["param_groups"][0]["lr"],
            "checkpoint_sha256": file_digest(work / f"input/epoch-{epoch}.pt"),
        }
        out["checkpoints"][str(epoch)] = endpoint

        seed = 73109 + epoch
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        loader = iter(DataLoader(train.dataset, sampler=train.sampler, batch_size=128))
        old_ids = [
            b["train_ids"]
            for receipt in previous
            for b in receipt["checkpoints"][str(epoch)]["batches"]
        ]
        for step in range(6):
            begin = time.monotonic()
            model.load_state_dict(state["model"])
            images, legal, ids, _ = legal_batch(next(loader))
            assert ids.tolist() not in old_ids
            rng = np.random.default_rng(seed + step)
            exclusion = ids.tolist()
            available = np.flatnonzero(~np.isin(bank_ids, exclusion))
            drawn = rng.choice(available, 448, replace=False)
            available_known = np.flatnonzero(~np.isin(known.uq_idxs, exclusion))
            labeled = rng.choice(available_known, 64, replace=False)
            all_probe = []
            semantic_rows = drawn[192:]
            with frozen(backbone, seed + step, torch.device("cuda:0")):
                for f in range(2):
                    split = drawn[96 * f : 96 * (f + 1)]
                    indices = (
                        labeled[32 * f : 32 * (f + 1)].tolist()
                        + (len(known) + split).tolist()
                    )
                    views, y, keys, _ = next(
                        iter(DataLoader(Subset(train.dataset, indices), batch_size=128))
                    )
                    assert (y[32:] == -1).all() and (y[:32] >= 0).all()
                    all_probe.append(
                        (prefix(backbone, torch.cat(views)), y.cuda(), keys.tolist())
                    )
                views, sy, sids = next(
                    iter(DataLoader(Subset(blind, semantic_rows), batch_size=256))
                )
                assert (sy == -1).all()
                sem_tokens = prefix(backbone, views[0])
            independent = [set(keys) for _, _, keys in all_probe] + [
                set(sids.tolist()),
                set(exclusion),
            ]
            assert sum(map(len, independent)) == len(set.union(*independent))
            tokens = prefix(backbone, images)
            h = encode(backbone, tokens).requires_grad_()
            x = F.normalize(h.detach(), dim=-1).reshape(2, 128, -1)
            vc = VirtualClassifier(head.last_layer, checkpoint_optimizer(model, state))
            raw = vc.logits(x, vc.v)
            teacher = (raw / 0.04).softmax(-1)
            counts = Counter(exclusion)
            unique = [i for i, k in enumerate(exclusion) if counts[k] == 1]
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
            actions, invalid, common_count = strength_support(eligible, teacher)
            folds = []
            for tok, y, _ in all_probe:
                px = F.normalize(encode(backbone, tok), dim=-1).reshape(2, 128, -1)
                folds.append((px, y, (vc.logits(px, vc.v) / 0.1).softmax(-1)))
            first, *diffs = list(
                gradient_components(model, tokens, h, raw, legal, criterion, actions)
            )
            params = list(backbone.blocks[-1].parameters())
            sizes = [p.numel() for p in params]
            variants = [(None, 0.0, None)]
            for action, delta in zip(actions, diffs, strict=True):
                variants.extend(
                    (action, strength, delta)
                    for strength in (MOVE if action.kind == "move" else ABSTAIN)
                )
            arms, semantic = [], []
            for action, strength, delta in variants:
                model.load_state_dict(state["model"])
                opt = checkpoint_optimizer(model, state)
                gv, gb = first
                dose = {"classifier": 0.0, "block11": 0.0}
                if action is not None:
                    scale = strength / ORIGINAL[action.kind]
                    dv, db = delta
                    gv, gb = gv + scale * dv, gb + scale * db
                    dose = {
                        "classifier": scale * dv.norm().item(),
                        "block11": scale * db.norm().item(),
                    }
                head.last_layer.weight_v.grad = gv
                for p, g in zip(params, gb.split(sizes), strict=True):
                    p.grad = g.view_as(p)
                opt.step()
                opt.zero_grad()
                current = head.last_layer.weight_v.detach()
                if action is None or strength == ORIGINAL[action.kind]:
                    exact = vc.step(
                        x,
                        raw,
                        legal,
                        criterion,
                        0,
                        [] if action is None else [action],
                        ARGS,
                    )
                    torch.testing.assert_close(current, exact)
                scores = [
                    list(vc.probe(current, px, y, anchor)) for px, y, anchor in folds
                ]
                arms.append(
                    {
                        "kind": "keep" if action is None else action.kind,
                        "id": None if action is None else int(ids[action.row]),
                        "source": None if action is None else action.source,
                        "target": None if action is None else action.target,
                        "strength": strength,
                        "dose": dose,
                        "v0": scores,
                        "objective": [scores[0][0], scores[0][0], None],
                        "known_ce": [scores[0][1]] * 3,
                    }
                )
                semantic.append(
                    F.normalize(encode(backbone, sem_tokens), dim=-1).cpu().numpy()
                )
            file = scratch / f"post-{epoch}-{step}.npy"
            np.save(file, np.stack(semantic))
            endpoint["batches"].append(
                {
                    "batch": step,
                    "train_ids": exclusion,
                    "fold_ids": [keys for _, _, keys in all_probe],
                    "proxy_ids": [key for _, _, keys in all_probe for key in keys],
                    "semantic_ids": sids.tolist(),
                    "eligible": dict(Counter(a.kind for a in eligible)),
                    "common_move": common_count,
                    "invalid_move_ids": {
                        k: [int(ids[row]) for row in v] for k, v in invalid.items()
                    },
                    "semantic_sha256": file_digest(file),
                    "seconds": time.monotonic() - begin,
                    "arms": arms,
                    "decisions": {
                        k: decide(arms, k == "adaptive")
                        for k in ("original", "adaptive")
                    },
                }
            )
            print(epoch, step, len(actions), len(arms) - 1, flush=True)
    write_json(scratch / "frozen_predictions.json", out)


def aggregate(rows):
    return {
        group: float(np.mean([r[group] for r in rows])) if rows else None
        for group in ("all", "old", "new")
    }


def mixed_gain(arms):
    return aggregate([a["semantic_gain"] for a in arms])


def evaluate(work, scratch, out):
    sealed = scratch / "sealed.json"
    seal(work, scratch, sealed)
    result = json.loads(sealed.read_text())
    batches = [b for e in result["checkpoints"].values() for b in e["batches"]]
    assert len(batches) == 12
    summary = {}
    zero = dict.fromkeys(("all", "old", "new"), 0.0)
    for name in ("original", "adaptive"):
        gains, fixed, random_strength, random_id = [], [], [], []
        reasons = Counter()
        selected = 0
        for b in batches:
            decision = b["decisions"][name]
            reasons[decision["reason"]] += 1
            if decision["selected"] is None:
                gains.append(zero)
                fixed.append(zero)
                random_strength.append(zero)
                random_id.append(zero)
                continue
            selected += 1
            a = b["arms"][decision["selected"]]
            same = [
                r for r in b["arms"][1:] if (r["id"], r["kind"]) == (a["id"], a["kind"])
            ]
            same_kind = [
                r
                for r in b["arms"][1:]
                if r["kind"] == a["kind"] and r["strength"] == a["strength"]
            ]
            gains.append(a["semantic_gain"])
            fixed.append(
                next(
                    r["semantic_gain"]
                    for r in same
                    if r["strength"] == ORIGINAL[a["kind"]]
                )
            )
            random_strength.append(mixed_gain(same))
            random_id.append(mixed_gain(same_kind))
        summary[name] = {
            "accepted": selected,
            "reasons": dict(reasons),
            "harmful": sum(g["all"] < -EPS for g in gains),
            "gain": aggregate(gains),
            "fixed_same_candidate_and_count": aggregate(fixed),
            "random_strength_same_candidate_and_count": aggregate(random_strength),
            "random_candidate_same_kind_strength_count": aggregate(random_id),
        }
    curves = {}
    for kind, strengths in (("move", MOVE), ("abstain", ABSTAIN)):
        for strength in strengths:
            rows = [
                (a, b["arms"][0])
                for b in batches
                for a in b["arms"][1:]
                if a["kind"] == kind and a["strength"] == strength
            ]
            curves[f"{kind}/{strength}"] = {
                "n": len(rows),
                "gain": mixed_gain([a for a, _ in rows]),
                "harmful": sum(a["semantic_gain"]["all"] < -EPS for a, _ in rows),
                "classifier_dose": float(
                    np.mean([a["dose"]["classifier"] for a, _ in rows])
                )
                if rows
                else None,
                "block11_dose": float(np.mean([a["dose"]["block11"] for a, _ in rows]))
                if rows
                else None,
                "v0_discovery_gain": float(
                    np.mean([k["v0"][0][0] - a["v0"][0][0] for a, k in rows])
                )
                if rows
                else None,
                "v0_confirmation_gain": float(
                    np.mean([k["v0"][1][0] - a["v0"][1][0] for a, k in rows])
                )
                if rows
                else None,
            }
    result["summary"] = {
        "keep": zero,
        "curves": curves,
        "policies": summary,
        "invalid_move": {
            str(m): sum(len(b["invalid_move_ids"][str(m)]) for b in batches)
            for m in MOVE
        },
    }
    write_json(out, result)
    return result


def report(path, result):
    def g(values):
        return " / ".join(
            "—" if values[k] is None else f"{values[k] * 1e6:+.3f}"
            for k in ("all", "old", "new")
        )

    lines = [
        "# Bounded CRiSH action-strength audit",
        "",
        (
            "Twelve fresh batch-128 draws from the ordinary Cars sampler, six per R2 checkpoint "
            "at 100 and 150. Batch IDs differ from the prior V0/V2 and V3 audits. Training, "
            "the two probe folds, and the sealed queries are ID-disjoint within each batch. "
            "The sealed reference bank excludes all those IDs."
        ),
        "",
        (
            "We freeze the E3 eligibility and evidence ranking computed at MOVE .2. "
            "MOVE curves retain only proposals whose source probability is at least .4 in both views. "
            "Choose up to two common-support MOVE and two ABSTAIN candidates per batch. "
            "Invalid counts below cover all E3 MOVE proposals at each requested mass, before the cap. "
            "The same source/destination and action IDs are used across strengths."
        ),
        "",
        (
            "KEEP uses original SimGCD. The full joint step updates the weight-normalized classifier "
            "and block 11 through classification, InfoNCE and SupCon, with checkpoint SGD state, "
            "excluding the R2 loss. MOVE transfers the indicated mass. ABSTAIN scales the original "
            "teacher rows by 1−β with the same denominator and no renormalization. "
            "At fixed parameters and detached targets the edit gradient is affine in strength, "
            "so one cached edit differential per candidate scales exactly in real arithmetic. "
            "Each strength still gets its own SGD step and feature forward. The original doses and "
            "KEEP are checked against the functional E3 classifier step."
        ),
        "",
        (
            "V0 uses the original frozen classifier probe objective. Discovery ranks positive gains "
            "that satisfy the non-increasing known-CE guard. The discovery winner must pass both "
            "checks on the separate confirmation fold. No action follows rejection. All decisions "
            "are frozen before sealed semantic evaluation. These singleton decisions do not model "
            "E3 multi-action or fallback behavior."
        ),
        "",
        "| Arm | Candidate-batches | Semantic gain All / Old / New (×10⁶) | Harmful | Edit gradient norm classifier / block 11 | V0 gain discovery / confirmation (×10⁶) |",
        "|---|---:|---|---:|---|---|",
        "| KEEP | 12 | 0 / 0 / 0 | 0 | 0 / 0 | 0 / 0 |",
    ]
    for cell, s in result["summary"]["curves"].items():
        if s["n"]:
            detail = f"{s['classifier_dose']:.3g} / {s['block11_dose']:.3g} | {s['v0_discovery_gain'] * 1e6:+.2f} / {s['v0_confirmation_gain'] * 1e6:+.2f}"
        else:
            detail = "— | —"
        lines.append(
            f"| {cell} | {s['n']} | {g(s['gain'])} | {s['harmful']} | {detail} |"
        )
    lines += [
        "",
        "Invalid MOVE proposal counts by mass — "
        + ", ".join(f"{m}: {n}" for m, n in result["summary"]["invalid_move"].items()),
        "",
        "| V0 policy | Accepted | Harmful / accepted | Gain | Same-ID fixed E3 dose | Same-ID random strength | Same-kind/strength random ID |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for name, s in result["summary"]["policies"].items():
        values = [
            g(s[k])
            for k in (
                "gain",
                "fixed_same_candidate_and_count",
                "random_strength_same_candidate_and_count",
                "random_candidate_same_kind_strength_count",
            )
        ]
        lines.append(
            f"| {name} | {s['accepted']} | {s['harmful']}/{s['accepted']} | "
            + " | ".join(values)
            + " |"
        )
    for name, s in result["summary"]["policies"].items():
        lines += [
            "",
            f"- {name} — " + ", ".join(f"{k}: {v}" for k, v in s["reasons"].items()),
        ]
    lines += [
        "",
        (
            "Policy gains use the same All / Old / New units and average over all 12 batches with "
            "zero for KEEP/no action. Harmful means All gain < −1e−7. The fixed-dose and "
            "random-strength controls intervene on exactly the selected candidate and batches. "
            "They are forced comparisons and can choose a strength V0 would reject. The random-ID "
            "control retains the selected kind and dose and is uniform over the batch's eligible IDs. "
            "Unmatched original-versus-adaptive policy means can reflect different intervention counts. "
            "These checkpoint-conditioned one-step effects do not estimate train-time accuracy."
        ),
        "",
        "[Sealed results and frozen decision provenance](r2_strength_diagnostic.json)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def validate_policy():
    keep = {"v0": [[1.0, 1.0], [1.0, 1.0]]}
    original = {"kind": "move", "strength": 0.2, "v0": [[0.8, 0.9], [0.8, 0.9]]}
    strongest = {"kind": "move", "strength": 0.4, "v0": [[0.7, 0.9], [1.1, 0.9]]}
    assert decide([keep, original, strongest], False)["selected"] == 1
    assert decide([keep, original, strongest], True)["reason"] == "confirmation_gain"
    guarded = {**strongest, "v0": [[0.7, 1.01], [0.7, 0.9]]}
    assert decide([keep, original, guarded], True)["selected"] == 1
    moves = [Action(0, "move", 0, 1), Action(1, "move", 0, 1)]
    chosen, invalid, n = strength_support(
        moves, torch.tensor([[[0.35, 0.65], [0.6, 0.4]], [[0.5, 0.5], [0.4, 0.6]]])
    )
    assert chosen == moves[1:] and invalid["0.4"] == [0] and n == 1
    criterion = DistillLoss(0, 2)
    teacher = torch.tensor([[0.2, 0.0], [0.0, 0.2], [0.22, 0.0], [0.0, 0.22]])
    logits = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.1, 0.5], [0.5, 0.1]], requires_grad=True
    )
    plain = criterion(logits, teacher, 0)
    (keep_grad,) = torch.autograd.grad(plain, logits, retain_graph=True)
    for action, strengths in ((moves[0], MOVE), (Action(0, "abstain", 0), ABSTAIN)):
        exact = corrected_distill(logits, teacher, criterion, 0, [action], 0.2)
        (base,) = torch.autograd.grad(exact - plain, logits, retain_graph=True)
        for s in strengths:
            if action.kind == "move":
                direct = corrected_distill(logits, teacher, criterion, 0, [action], s)
            else:
                q = (teacher / 0.04).softmax(-1).reshape(2, 2, -1)
                q[:, action.row] *= 1 - s
                logp = (logits.reshape_as(q) / criterion.student_temp).log_softmax(-1)
                direct = -(q.flip(0) * logp).sum(-1).mean()
            (measured,) = torch.autograd.grad(direct, logits, retain_graph=True)
            torch.testing.assert_close(
                measured, keep_grad + (s / ORIGINAL[action.kind]) * base
            )


if __name__ == "__main__":
    validate_policy()
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "scratch", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    a.scratch.mkdir(exist_ok=True, parents=True)
    previous = [
        json.loads(a.output.with_name(name).read_text())
        for name in ("r2_verifier.json", "r2_v3_diagnostic.json")
    ]
    predict(a.work, previous, a.scratch)
    result = evaluate(a.work, a.scratch, a.output)
    report(a.output.with_name("CRISH_R2_STRENGTH_DIAGNOSTIC.md"), result)
