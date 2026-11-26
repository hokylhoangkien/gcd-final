"""Bounded full-SimGCD counterfactual decisions and sealed evaluation."""

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
from crish.repair import VirtualClassifier, corrected_distill, proposals
from model import DistillLoss
from relational.sensitivity import legal_batch
from relational.sensitivity_metrics import block_gradient, prefix
from relational.state import write_json
from relational.train import baseline, build
from scripts.audit_r2_verifier import (
    ARGS,
    EPS,
    checkpoint_optimizer,
    margin_targets,
    seal,
)
from scripts.audit_r2_virtual import encode

POLICIES = {
    "V0": (0, False, False),
    "V2": (2, False, False),
    "V3-A": (2, True, False),
    "V3-B": (2, False, True),
    "V3": (2, True, True),
}


def decisions(arms):
    keep = arms[0]
    policies = {}
    for name, (metric, confirm, relax) in POLICIES.items():
        guard_index = 0 if metric == 0 else 2
        folds = (0, 1) if confirm else (0,)

        def guard(i, f, guard_index=guard_index, relax=relax):
            base = keep["folds"][f]["known_ce"][guard_index]
            return arms[i]["folds"][f]["known_ce"][guard_index] - base <= (
                0.001 * base if relax else 0.0
            )

        def gain(i, f, metric=metric):
            a, b = (x["folds"][f]["objective"][metric] for x in (arms[i], keep))
            if a is None or b is None:
                return None
            return b - a if metric == 0 else a - b

        def positive(i, f, metric=metric):
            g = gain(i, f)
            if g is None or g <= EPS:
                return False
            if metric == 0:
                return True
            check = (
                arms[i]["folds"][f]["precision_margin"]
                - keep["folds"][f]["precision_margin"]
            )
            return check > EPS and abs(g - check) <= max(EPS, 0.1 * abs(g))

        candidates = list(range(1, len(arms)))
        ranked = sorted(
            (i for i in candidates if gain(i, 0) is not None), key=lambda i: -gain(i, 0)
        )
        top = ranked[0] if ranked else None
        reason = "accepted"
        if top is None:
            reason = "no_eligible_action" if not candidates else "no_discovery_coverage"
        else:
            for f in folds:
                label = ("discovery", "confirmation")[f]
                if gain(top, f) is None:
                    reason = f"{label}_coverage"
                elif gain(top, f) <= EPS:
                    reason = f"{label}_gain"
                elif not positive(top, f):
                    reason = f"{label}_precision"
                elif not guard(top, f):
                    reason = f"{label}_known_ce"
                if reason != "accepted":
                    break
        policies[name] = {
            "selected": top if reason == "accepted" else None,
            "top": top,
            "reason": reason,
            "guard_pool": [i for i in candidates if all(guard(i, f) for f in folds)],
        }
    return policies


def gradient_components(model, tokens, h, raw, legal, criterion, actions):
    proj, logits = model[1](h)
    teacher = raw.flatten(0, 1)
    plain = criterion(logits, teacher, 0)
    loss = baseline(proj, logits, legal, legal >= 0, criterion, 0)
    params = list(model[0].blocks[-1].parameters())
    classifier = model[1].last_layer.weight_v

    def both(obj):
        return (
            torch.autograd.grad(obj, classifier, retain_graph=True)[0],
            block_gradient(obj, h, model[0], tokens, params),
        )

    gv, gb = both(loss)
    yield gv, gb
    for a in actions:
        yield both(
            0.65 * (corrected_distill(logits, teacher, criterion, 0, [a], 0.2) - plain)
        )


def gradients(model, tokens, h, raw, legal, criterion, actions):
    parts = gradient_components(model, tokens, h, raw, legal, criterion, actions)
    gv, gb = next(parts)
    yield gv, gb
    for dv, db in parts:
        yield gv + dv, gb + db


def predict(work, previous, scratch):
    model, train, _, _, _, _ = build()
    backbone, head = model.eval()
    known, blind = train.dataset.labelled_dataset, train.dataset.unlabelled_dataset
    bank_ids = np.load(work / "ids.npy")
    assert np.array_equal(bank_ids, blind.ids)
    saved = np.load(work / "features.npy", mmap_mode="r")
    out = {"checkpoints": {}, "semantic_epsilon": EPS, "ce_relative_tolerance": 0.001}
    criterion = DistillLoss(0, 200)
    for index, epoch in ((1, 100), (2, 150)):
        state = load_checkpoint(work / f"input/epoch-{epoch}.pt")
        endpoint = {"batches": []}
        out["checkpoints"][str(epoch)] = endpoint
        cached = F.normalize(torch.from_numpy(saved[index].copy()).cuda(), dim=-1)
        seed = 57901 + epoch
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        loader = iter(DataLoader(train.dataset, sampler=train.sampler, batch_size=128))
        old_ids = [
            b["train_ids"] for b in previous["checkpoints"][str(epoch)]["batches"]
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
            ref = ~torch.from_numpy(
                np.isin(bank_ids, exclusion + bank_ids[drawn].tolist())
            ).cuda()
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
            actions = [
                a
                for kind in ("move", "abstain")
                for a in [v for v in eligible if v.kind == kind][:2]
            ]
            actions = (actions + [a for a in eligible if a not in actions])[:4]
            fold_states = []
            for tok, y, keys in all_probe:
                px = F.normalize(encode(backbone, tok), dim=-1).reshape(2, 128, -1)
                mask, directions, meta = margin_targets(vc, cached, px[:, 32:], ref)
                fold_states.append(
                    (
                        px,
                        (vc.logits(px, vc.v) / 0.1).softmax(-1),
                        mask,
                        directions,
                        meta,
                    )
                )
            bases = list(gradients(model, tokens, h, raw, legal, criterion, actions))
            params = list(backbone.blocks[-1].parameters())
            arms, semantic, confirmation_features = [], [], []
            for action, (gv, gb) in zip([None, *actions], bases, strict=True):
                model.load_state_dict(state["model"])
                opt = checkpoint_optimizer(model, state)
                head.last_layer.weight_v.grad = gv
                for p, g in zip(
                    params, gb.split([p.numel() for p in params]), strict=True
                ):
                    p.grad = g.view_as(p)
                opt.step()
                opt.zero_grad()
                v0 = vc.step(
                    x,
                    raw,
                    legal,
                    criterion,
                    0,
                    [] if action is None else [action],
                    ARGS,
                )
                torch.testing.assert_close(head.last_layer.weight_v, v0)
                folds = []
                for fold, ((tok, y, _), (px, anchor, mask, directions, _)) in enumerate(
                    zip(all_probe, fold_states, strict=True)
                ):
                    h_probe = encode(backbone, tok).reshape(2, 128, -1)
                    features = F.normalize(h_probe, dim=-1)
                    s0, k0 = vc.probe(v0, px, y, anchor)
                    s1, k1 = vc.probe(v0, features, y, anchor)
                    if fold:
                        confirmation_features.append(h_probe)
                    vals = [
                        float((f[:, 32:][:, mask] * d).sum(-1).mean())
                        if not fold and mask.any()
                        else None
                        for f, d in (
                            (
                                F.normalize(h_probe.double(), dim=-1),
                                directions.double(),
                            ),
                            (features, directions),
                        )
                    ]
                    folds.append(
                        {
                            "objective": [s0, s1, vals[0]],
                            "known_ce": [k0, k1, k1],
                            "precision_margin": vals[1],
                        }
                    )
                arms.append(
                    {
                        "action": None
                        if action is None
                        else [
                            int(ids[action.row]),
                            action.kind,
                            action.source,
                            action.target,
                        ],
                        "folds": folds,
                        **{k: folds[0][k] for k in ("objective", "known_ce")},
                    }
                )
                semantic.append(
                    F.normalize(encode(backbone, sem_tokens), dim=-1).cpu().numpy()
                )
            rankable = [
                i for i in range(1, len(arms)) if arms[i]["objective"][2] is not None
            ]
            if rankable and fold_states[1][2].any():
                top = max(rankable, key=lambda i: arms[i]["objective"][2])
                _, _, mask, directions, _ = fold_states[1]
                for i in (0, top):
                    f = confirmation_features[i]
                    score = arms[i]["folds"][1]
                    score["objective"][2] = float(
                        (
                            F.normalize(f.double(), dim=-1)[:, 32:][:, mask]
                            * directions.double()
                        )
                        .sum(-1)
                        .mean()
                    )
                    score["precision_margin"] = float(
                        (F.normalize(f, dim=-1)[:, 32:][:, mask] * directions)
                        .sum(-1)
                        .mean()
                    )
            path = scratch / f"post-{epoch}-{step}.npy"
            np.save(path, np.stack(semantic))
            endpoint["batches"].append(
                {
                    "batch": step,
                    "train_ids": exclusion,
                    "proxy_ids": [key for _, _, keys in all_probe for key in keys],
                    "fold_ids": [keys for _, _, keys in all_probe],
                    "semantic_ids": sids.tolist(),
                    "semantic_sha256": file_digest(path),
                    "eligible": dict(Counter(a.kind for a in eligible)),
                    "missing": 4 - len(actions),
                    "arms": arms,
                    "decisions": decisions(arms),
                    "coverage": [s[-1] for s in fold_states],
                    "seconds": time.monotonic() - begin,
                }
            )
            print(epoch, step, len(actions), flush=True)
    write_json(scratch / "frozen_predictions.json", out)


def evaluate(work, scratch, out):
    sealed = scratch / "sealed.json"
    seal(work, scratch, sealed)
    result = json.loads(sealed.read_text())
    batches = [b for e in result["checkpoints"].values() for b in e["batches"]]
    summary = {}
    for name in POLICIES:
        gains = {group: [] for group in ("all", "old", "new")}
        count_random = {k: [] for k in gains}
        guard_random = {k: [] for k in gains}
        why = Counter()
        for b in batches:
            d = b["decisions"][name]
            why[d["reason"]] += 1
            for group, gains_list in gains.items():
                values = [a["semantic_gain"][group] for a in b["arms"]]
                i = d["selected"]
                gains_list.append(values[i] if i is not None else 0.0)
                count_random[group].append(
                    float(np.mean(values[1:])) if i is not None else 0.0
                )
                guard_random[group].append(
                    float(np.mean([values[j] for j in d["guard_pool"]]))
                    if i is not None
                    else 0.0
                )
        summary[name] = {
            "interventions": why["accepted"],
            "harmful": sum(g < -EPS for g in gains["all"]),
            "reasons": dict(why),
            "gain": {k: float(np.mean(v)) for k, v in gains.items()},
            "matched_count_random": {
                k: float(np.mean(v)) for k, v in count_random.items()
            },
            "guard_matched_random": {
                k: float(np.mean(v)) for k, v in guard_random.items()
            },
        }
    result["summary"] = summary
    write_json(out, result)
    return result


def report(path, result):
    lines = [
        "# V3 bounded verifier decision diagnostic",
        "",
        (
            "Twelve fresh batch-128 weighted draws across R2 checkpoints 100/150. "
            "KEEP, MOVE (.2) and ABSTAIN use E3 targets. Joint virtual SGD uses full SimGCD, "
            "including InfoNCE and SupCon through block 11, with the saved optimizer state "
            "and without the R2 relational term. The classifier result is checked against the "
            "functional V0 update for each arm. V0 serves as a classifier-only predictor of the "
            "same sealed joint-update effect assessed by the other policies."
        ),
        "",
        (
            "Train, reference, discovery, confirmation and sealed probe IDs are disjoint "
            "within each batch. Each proxy fold has 32 known and 96 unlabeled images. "
            "Centroids, target slots and competitors are fixed before intervention. "
            "The discovery top candidate alone is eligible for acceptance, with no fallback. "
            "V0 ranks the original objective. V2 ranks relative margin. V3-A adds confirmation "
            "to strict V2, V3-B relaxes only the CE guard, and V3 combines both. Positive margin "
            "must exceed 1e-7 in both FP64 and FP32 scoring, with their gains agreeing within "
            "max(1e-7, 10% of gain). The backbone forward uses FP32 in both. "
            "Strict CE requires non-increase; relaxed CE permits +0.1% of fold KEEP CE. "
            "With confirmation the gate applies separately on both folds."
        ),
        "",
        (
            "Decisions and guard pools are frozen before the sealed label join. "
            "Both random controls match each policy's exact intervention batches and count. "
            "The guard-matched random pool further restricts eligible candidates by that policy's "
            "CE rule. Entries below average over all 12 batches, with zero for no action. "
            "Harmful means sealed All gain below −1e-7. The small, checkpoint-conditioned sample "
            "cannot establish trajectory-level effects."
        ),
        "",
        "| Policy | Actions | Harmful / accepted | Gain All / Old / New (×10⁶) | Count-matched random All / Old / New | Guard-matched random All / Old / New |",
        "|---|---:|---:|---|---|---|",
    ]
    for key, s in result["summary"].items():
        fmt = lambda k, s=s: " / ".join(
            f"{s[k][g] * 1e6:+.3f}" for g in ("all", "old", "new")
        )
        lines.append(
            f"| {key} | {s['interventions']} | {s['harmful']}/{s['interventions']} | {fmt('gain')} | {fmt('matched_count_random')} | {fmt('guard_matched_random')} |"
        )
    lines.extend(["", "Rejection and acceptance counts", ""])
    for key, s in result["summary"].items():
        lines.append(
            f"- {key} — "
            + ", ".join(f"{reason} {n}" for reason, n in sorted(s["reasons"].items()))
        )
    batches = [
        b for endpoint in result["checkpoints"].values() for b in endpoint["batches"]
    ]
    assert len(batches) == 12 and all(len(b["arms"]) <= 5 for b in batches)
    actions = Counter(a["action"][1] for b in batches for a in b["arms"][1:])
    lines.extend(
        [
            "",
            (
                f"Evaluated {sum(actions.values())} actions ({dict(actions)}); "
                f"{sum(b['missing'] for b in batches)} unfilled positions, "
                f"{sum('move' not in b['eligible'] for b in batches)} batches without eligible MOVE. "
                f"Fold coverage ranged from {min(c['proxy_covered'] for b in batches for c in b['coverage'])} "
                f"to {max(c['proxy_covered'] for b in batches for c in b['coverage'])} of 96. "
                f"Sealed coverage: {sum(b['semantic_coverage']['all'] for b in batches)} of 3072. "
                f"Batch compute: {sum(b['seconds'] for b in batches):.1f} s."
            ),
            "",
            "[Per-batch results, decisions and provenance](r2_v3_diagnostic.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def validate_decisions():
    def arm(discovery, confirmation, ce, numerical=None):
        return {
            "folds": [
                {
                    "objective": [0.0, 0.0, gain],
                    "known_ce": [ce] * 3,
                    "precision_margin": gain if numerical is None else numerical,
                }
                for gain in (discovery, confirmation)
            ]
        }

    keep = arm(0.0, 0.0, 1.0)
    second = arm(0.02, 0.02, 0.99)
    rejected = decisions([keep, arm(0.03, -0.01, 1.0005), second])
    assert (
        rejected["V3"]["reason"] == "confirmation_gain"
        and rejected["V3"]["selected"] is None
    )
    assert rejected["V2"]["reason"] == "discovery_known_ce"
    assert rejected["V3-B"]["selected"] == 1
    accepted = decisions([keep, arm(0.03, 0.01, 1.0005), second])
    assert accepted["V3"]["selected"] == 1 and accepted["V3-A"]["selected"] is None
    assert (
        decisions([keep, arm(0.03, 0.03, 0.99, 0.0)])["V3"]["reason"]
        == "discovery_precision"
    )
    assert decisions([keep])["V3"]["reason"] == "no_eligible_action"


if __name__ == "__main__":
    validate_decisions()
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "previous", "scratch", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    a.scratch.mkdir(exist_ok=True, parents=True)
    predict(a.work, json.loads(a.previous.read_text()), a.scratch)
    result = evaluate(a.work, a.scratch, a.output)
    report(a.output.with_name("CRISH_R2_V3_DIAGNOSTIC.md"), result)
