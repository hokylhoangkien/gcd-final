"""Bounded expanded-slot counterfactuals on frozen E3 MOVE IDs."""

import argparse
import copy
import functools
import json
import operator
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest, load_checkpoint
from crish.global_repair import source_ok
from crish.repair import Action, VirtualClassifier, corrected_distill
from model import DistillLoss
from relational.sensitivity import legal_batch
from relational.sensitivity_metrics import cls, prefix
from relational.state import write_json
from relational.train import build
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_r2_v3 import gradient_components
from scripts.audit_r2_verifier import ARGS, EPS, checkpoint_optimizer, seal
from scripts.audit_r2_virtual import encode

DEPTHS = {"D0": 0, "D1": 1, "D2": 3, "D3": 5}


def decide(arms, indices):
    keep = arms[0]["v0"]
    valid = [
        i
        for i in indices
        if arms[i]["v0"][0][0] < keep[0][0] and arms[i]["v0"][0][1] <= keep[0][1]
    ]
    if not valid:
        return {
            "selected": None,
            "discovery_winner": None,
            "reason": "discovery" if indices else "no_eligible_arm",
        }
    top = min(valid, key=lambda i: arms[i]["v0"][0][0])
    score, known = arms[top]["v0"][1]
    reason = (
        "confirmation_gain"
        if score >= keep[1][0]
        else "confirmation_known_ce"
        if known > keep[1][1]
        else "accepted"
    )
    return {
        "selected": top if reason == "accepted" else None,
        "discovery_winner": top,
        "reason": reason,
    }


def edit_loss(lp, action, teacher):
    if action.kind == "move":
        loss = 0.2 * (lp[:, action.source] - lp[:, action.target]).sum()
    else:
        loss = (teacher.flip(0)[:, action.row] * lp).sum()
    return loss / (2 * teacher.shape[1])


def action_delta(backbone, tokens, h, vc, action, teacher, student_temp):
    """Differentiate only the two affected rows; every other row contributes zero."""
    n = teacher.shape[1]
    v = vc.v.detach().clone().requires_grad_()
    loc = torch.stack((h[action.row], h[n + action.row]))
    lp = (vc.logits(F.normalize(loc, dim=-1), v) / student_temp).log_softmax(-1)
    dv, dh = torch.autograd.grad(0.65 * edit_loss(lp, action, teacher), (v, loc))
    params = list(backbone.blocks[-1].parameters())
    grads = None
    for row, incoming in zip((action.row, n + action.row), dh, strict=True):
        t = tokens[row // 8][row % 8 : row % 8 + 1]
        delta = torch.autograd.grad(cls(backbone, t), params, incoming.unsqueeze(0))
        grads = (
            list(delta)
            if grads is None
            else [a + b for a, b in zip(grads, delta, strict=True)]
        )
    return dv, torch.cat([d.flatten() for d in grads])


def pool_for(records, arms, mode):
    pools = {}
    rows = {}
    for r in records:
        key = str(r["id"])
        rows[key] = {}
        for p, depth in DEPTHS.items():
            targets = (
                [r["original_target"]]
                if depth == 0
                else r["expanded"]["ranks"]["L1"][:depth]
            )
            selected = [
                i
                for i, a in enumerate(arms)
                if a["id"] == r["id"]
                and (
                    (a["kind"] == "move" and a["target"] in targets)
                    or (a["kind"] == "abstain" and mode == "mixed")
                )
            ]
            selected.sort(
                key=lambda i: (
                    arms[i]["kind"] == "abstain",
                    targets.index(arms[i]["target"])
                    if arms[i]["kind"] == "move"
                    else 0,
                )
            )
            pools.setdefault(p, {})[key] = selected
            rows[key][p] = decide(arms, selected)
    return {"pools": pools, "decisions": rows}


def predict(work, learned, frozen_proposals, scratch):
    model, train, _, center, _, _ = build()
    model.eval()
    backbone, head = model
    ds = train.dataset
    known = ds.labelled_dataset
    ids = np.load(work / "ids.npy")
    assert np.array_equal(ids, ds.unlabelled_dataset.ids)
    lookup = {int(i): r for r, i in enumerate(ids)}
    centers = copy.copy(ds.unlabelled_dataset)
    centers.transform = center
    previous = json.loads(learned.read_text())
    original = json.loads(frozen_proposals.read_text())
    assert previous["proposal_sha256"] == file_digest(frozen_proposals)
    criterion = DistillLoss(0, 200)
    out = {
        "checkpoints": {},
        "semantic_epsilon": EPS,
        "move_mass": 0.2,
        "source_hashes": {
            "frozen_proposals": file_digest(frozen_proposals),
            "global_cosine": file_digest(learned),
        },
    }
    for epoch in (100, 150):
        e = str(epoch)
        path = work / f"input/epoch-{epoch}.pt"
        assert file_digest(path) == previous["snapshots"][e]["checkpoint_sha256"]
        state = load_checkpoint(path)
        items = previous["snapshots"][e]["candidates"]
        assert len({r["id"] for r in items}) == len(items)
        # Use only pre-seal fields from previous output.
        slots = original["snapshots"][e]["predicted_slots"]
        endpoint = {
            "batches": [],
            "lr": state["optimizer"]["param_groups"][0]["lr"],
            "source_count": len(items),
        }
        out["checkpoints"][e] = endpoint
        all_sources = {r["id"] for r in items}
        for step, group in enumerate(
            [items[i : i + 4] for i in range(0, len(items), 4)]
        ):
            start = time.monotonic()
            model.load_state_dict(state["model"])
            rng = np.random.default_rng(23901 + epoch * 17 + step)
            torch.manual_seed(23901 + epoch * 17 + step)
            source_ids = [r["id"] for r in group]
            peer_ids = []
            for r in group:
                wanted = [r["original_target"], *r["expanded"]["ranks"]["L1"][:5]]
                for target in dict.fromkeys(wanted):
                    take = [
                        k
                        for k in r["expanded"]["peers"]
                        if k not in all_sources and slots[str(k)] == target
                    ][:2]
                    peer_ids.extend(take)
            unlabeled = list(dict.fromkeys([*source_ids, *peer_ids]))
            assert len(unlabeled) <= 80
            remaining = [int(k) for k in ids if k not in set(unlabeled) | all_sources]
            sampled = rng.choice(
                remaining, 80 - len(unlabeled) + 128 + 128, replace=False
            ).tolist()
            missing = 80 - len(unlabeled)
            unlabeled += sampled[:missing]
            proxy = sampled[missing : missing + 128]
            semantic_ids = sampled[-128:]
            known_rows = rng.choice(len(known), 112, replace=False).tolist()
            images, legal, batch_ids, _ = legal_batch(
                next(
                    iter(
                        DataLoader(
                            Subset(
                                ds,
                                known_rows[:48]
                                + [len(known) + lookup[k] for k in unlabeled],
                            ),
                            batch_size=128,
                        )
                    )
                )
            )
            tokens = prefix(backbone, images)
            h = encode(backbone, tokens).requires_grad_()
            x = F.normalize(h.detach(), dim=-1).reshape(2, 128, -1)
            opt = checkpoint_optimizer(model, state)
            vc = VirtualClassifier(head.last_layer, opt)
            raw = vc.logits(x, vc.v)
            teacher = (raw / 0.04).softmax(-1)
            folds = []
            fold_ids = []
            for i in range(2):
                rows = known_rows[48 + i * 32 : 80 + i * 32] + [
                    len(known) + lookup[k] for k in proxy[i * 64 : (i + 1) * 64]
                ]
                imgs, y, keys, _ = legal_batch(
                    next(iter(DataLoader(Subset(ds, rows), batch_size=96)))
                )
                feature = F.normalize(
                    encode(backbone, prefix(backbone, imgs)), dim=-1
                ).reshape(2, 96, -1)
                folds.append((feature, y, (vc.logits(feature, vc.v) / 0.1).softmax(-1)))
                fold_ids.append(keys.tolist())
            inputs, y, keys = next(
                iter(
                    DataLoader(
                        Subset(centers, [lookup[k] for k in semantic_ids]),
                        batch_size=128,
                    )
                )
            )
            assert (y == -1).all() and keys.tolist() == semantic_ids
            sets = [set(batch_ids.tolist()), *map(set, fold_ids), set(semantic_ids)]
            assert len(set.union(*sets)) == sum(map(len, sets))
            sem_tokens = prefix(backbone, inputs)
            gv, gb = next(
                gradient_components(model, tokens, h, raw, legal, criterion, [])
            )
            params = list(backbone.blocks[-1].parameters())
            sizes = [a.numel() for a in params]
            actions = [(None, None)]
            reason = {}
            lookup_batch = {int(k): i for i, k in enumerate(batch_ids)}
            for r in group:
                key, source = r["id"], r["source"]
                row = lookup_batch[key]
                ab = Action(row, "abstain", source)
                targets = list(
                    dict.fromkeys(
                        [r["original_target"], *r["expanded"]["ranks"]["L1"][:5]]
                    )
                )
                mv = [Action(row, "move", source, t) for t in targets if t != source]
                if not source_ok(teacher, row, ab, 0.2):
                    reason[str(key)] = "stale_source"
                    continue
                actions.append((key, ab))
                good = [a for a in mv if source_ok(teacher, row, a, 0.2)]
                reason[str(key)] = (
                    "eligible" if good else "insufficient_teacher_mass_or_support"
                )
                actions.extend((key, a) for a in good)
            # Every differential is calculated at checkpoint parameters before any step.
            parts = [(gv, gb.cpu())]
            for _, a in actions[1:]:
                dv, db = action_delta(
                    backbone, tokens, h, vc, a, teacher, criterion.student_temp
                )
                parts.append((gv + dv, (gb + db).cpu()))
            arms, semantic = [], []
            for (key, a), (cv, bv) in zip(actions, parts, strict=True):
                model.load_state_dict(state["model"])
                optim = checkpoint_optimizer(model, state)
                head.last_layer.weight_v.grad = cv
                for p, g in zip(params, bv.to(cv.device).split(sizes), strict=True):
                    p.grad = g.view_as(p)
                optim.step()
                optim.zero_grad()
                v = vc.step(x, raw, legal, criterion, 0, [] if a is None else [a], ARGS)
                torch.testing.assert_close(head.last_layer.weight_v, v)
                scores = [list(vc.probe(v, px, py, anchor)) for px, py, anchor in folds]
                arms.append(
                    {
                        "id": key,
                        "kind": a.kind if a else "keep",
                        "target": a.target if a else None,
                        "v0": scores,
                        "objective": [scores[0][0], scores[0][0], None],
                        "known_ce": [scores[0][1]] * 3,
                    }
                )
                semantic.append(
                    F.normalize(encode(backbone, sem_tokens), dim=-1).cpu().numpy()
                )
            f = scratch / f"post-{epoch}-{step}.npy"
            np.save(f, np.stack(semantic))
            endpoint["batches"].append(
                {
                    "batch": step,
                    "train_ids": batch_ids.tolist(),
                    "fold_ids": fold_ids,
                    "proxy_ids": functools.reduce(operator.iadd, fold_ids, []),
                    "semantic_ids": semantic_ids,
                    "semantic_sha256": file_digest(f),
                    "sources": source_ids,
                    "eligibility": reason,
                    "source_slots": {str(r["id"]): r["source"] for r in group},
                    "arms": arms,
                    "policies": {
                        m: pool_for(group, arms, m) for m in ("move", "mixed")
                    },
                    "seconds": time.monotonic() - start,
                }
            )
            print(epoch, step, len(source_ids), len(arms) - 1, flush=True)
    write_json(scratch / "frozen_predictions.json", out)


def mean_rows(rows):
    return {
        k: float(np.mean([r[k] for r in rows])) if rows else None
        for k in ("all", "old", "new")
    }


def evaluate(work, scratch, output):
    from scripts.audit_r2_evidence import sealed_labels

    intermediate = scratch / "sealed.json"
    seal(work, scratch, intermediate)
    result = json.loads(intermediate.read_text())
    frozen = json.loads((work / "e3-frozen-proposals.json").read_text())
    y = sealed_labels()
    ids = np.load(work / "ids.npy").tolist()
    zero = dict.fromkeys(("all", "old", "new"), 0.0)
    rows = []
    for e, endpoint in result["checkpoints"].items():
        slots = {
            int(k): v for k, v in frozen["snapshots"][e]["predicted_slots"].items()
        }
        mapping, alignment = align(ids, slots, y)
        endpoint["alignment"] = alignment
        for b in endpoint["batches"]:
            for arm in b["arms"][1:]:
                i = arm["id"]
                wrong = mapping.get(b["source_slots"][str(i)], -1) != y[i]
                arm["source_wrong"] = bool(wrong)
                arm["destination_correct"] = bool(
                    arm["kind"] == "move" and mapping.get(arm["target"], -1) == y[i]
                )
                arm["semantic_action_correct"] = bool(
                    wrong and (arm["kind"] == "abstain" or arm["destination_correct"])
                )
            b["sealed_source_subset"] = {
                str(i): "old" if y[i] < 98 else "new" for i in b["sources"]
            }
            b["oracle"] = {}
            for mode, policy in b["policies"].items():
                b["oracle"][mode] = {}
                for i in map(str, b["sources"]):
                    arms = b["arms"]
                    candidates = policy["pools"]["D3"][i]
                    admissible = [
                        j for j in candidates if arms[j]["semantic_action_correct"]
                    ]
                    best = max(
                        admissible,
                        key=lambda j: arms[j]["semantic_gain"]["all"],
                        default=None,
                    )
                    picked = (
                        best
                        if best is not None and arms[best]["semantic_gain"]["all"] > EPS
                        else None
                    )
                    b["oracle"][mode][i] = picked
                    rows.append((mode, i, b, picked))
    summary = {}
    for mode in ("move", "mixed"):
        selected = [(i, b, o) for m, i, b, o in rows if m == mode]
        mode_results = {}
        for p in [*DEPTHS, "D5"]:
            gains = []
            random = []
            random_action = []
            oracle_matched = []
            reasons = {}
            count = correct = harm = 0
            moves = dest = abst = 0
            correct_to_wrong = covered = wrong_covered = wrong_source = ranked_moves = (
                ranked_correct
            ) = 0
            for i, b, o in selected:
                policy = b["policies"][mode]
                arms = b["arms"]
                available = policy["pools"][p if p != "D5" else "D3"][i]
                mv = [j for j in available if arms[j]["kind"] == "move"]
                wrong = bool(arms[mv[0]]["source_wrong"]) if mv else False
                wrong_source += wrong
                hit = any(arms[j]["destination_correct"] for j in mv)
                covered += hit
                wrong_covered += int(wrong and hit)
                d = policy["decisions"][i].get(p)
                j = o if p == "D5" else d["selected"]
                reason = (
                    ("oracle_keep" if j is None else "accepted")
                    if d is None
                    else d["reason"]
                )
                reasons[reason] = reasons.get(reason, 0) + 1
                ranking = o if d is None else d["discovery_winner"]
                if ranking is not None and arms[ranking]["kind"] == "move":
                    ranked_moves += 1
                    ranked_correct += arms[ranking]["destination_correct"]
                if j is None:
                    gains.append(zero)
                    random.append(zero)
                    random_action.append(zero)
                    oracle_matched.append(zero)
                    continue
                a = arms[j]
                is_move = a["kind"] == "move"
                count += 1
                moves += is_move
                abst += not is_move
                dest += a["destination_correct"]
                correct += a["semantic_action_correct"]
                correct_to_wrong += int(
                    is_move and not a["source_wrong"] and not a["destination_correct"]
                )
                g = a["semantic_gain"]
                harm += g["all"] < -EPS
                gains.append(g)
                peers = [arms[k] for k in available if arms[k]["kind"] == a["kind"]]
                random.append(mean_rows([q["semantic_gain"] for q in peers]))
                batch_pool = policy["pools"][p if p != "D5" else "D3"]
                kind = a["kind"]
                by_source = [
                    [
                        arms[k]["semantic_gain"]
                        for k in options
                        if arms[k]["kind"] == kind
                    ]
                    for options in batch_pool.values()
                ]
                random_action.append(
                    mean_rows([mean_rows(options) for options in by_source if options])
                )
                exact = [q for q in peers if q["semantic_action_correct"]]
                # Same ID, kind and count; assigns the best semantic-valid action where it exists.
                oracle_matched.append(
                    max(exact or peers, key=lambda q: q["semantic_gain"]["all"])[
                        "semantic_gain"
                    ]
                )
            mode_results[p] = {
                "count": count,
                "move": moves,
                "abstain": abst,
                "harmful": harm,
                "semantic_correct": correct,
                "selected_destination_correct": dest,
                "selected_correct_to_wrong": correct_to_wrong,
                "discovery_move": ranked_moves,
                "discovery_destination_correct": ranked_correct,
                "ids_with_correct_destination_in_pool": covered,
                "wrong_source_with_eligible_move": wrong_source,
                "wrong_source_with_correct_destination_in_pool": wrong_covered,
                "gain": mean_rows(gains),
                "D4_matched_gain": mean_rows(random),
                "D4_random_source_and_destination_gain": mean_rows(random_action),
                "matched_oracle_gain": mean_rows(oracle_matched),
                "matched_oracle_gap": mean_rows(
                    [
                        {key: right[key] - left[key] for key in zero}
                        for left, right in zip(gains, oracle_matched, strict=True)
                    ]
                ),
                "reasons": reasons,
            }
        summary[mode] = mode_results
    result["summary"] = summary
    result = json_native(result)
    write_json(output, result)
    return result


def report(result, path):
    def g(vals):
        return " / ".join(
            f"{v * 1e6:+.3f}" if v is not None else "—" for v in vals.values()
        )

    text = [
        "# Bounded expanded-destination verifier diagnostic",
        "",
        (
            "R2 checkpoints 100/150 and the 26 previously frozen MOVE IDs. D0 uses original E3 target. "
            "D1, D2 and D3 use global cosine top 1, 3 and 5. Each ID shares its scoring context, KEEP, "
            "and fresh discovery and confirmation folds across arms. MOVE mass .2 and full ABSTAIN retain "
            "the original divisor. Unchanged E3 source and two-view source-mass checks are repeated on "
            "fresh scoring crops. Other IDs and supported destination sets come from frozen receipts. "
            "V0 ranks positive discovery gains with strict known CE, then checks only the winner on the "
            "independent confirmation fold. Rejection returns KEEP."
        ),
        "",
        (
            "This diagnostic forces the listed candidates into a fresh scoring context. It tests one action "
            "per source against shared KEEP and does not apply multiple simultaneous edits. Full joint virtual "
            "SGD uses original SimGCD including contrastive gradients, saved optimizer state, and no R2 loss. "
            "Prefix tokens are cached within each batch and differential backprop touches only the two "
            "edited rows. The unchanged functional V0 step checks that joint classifier parameters agree. "
            "Proxy inputs, scores, eligibility and all non-oracle decisions are frozen before any hidden "
            "label access. The sealed score uses a fixed center-feature bank and held-out center queries."
        ),
        "",
        (
            "D4 is the expected uniform control over destinations sharing each policy's accepted source ID "
            "and action kind, preserving intervention count and MOVE mass. In the mixed comparison it "
            "therefore also preserves the MOVE/ABSTAIN mix. A second D4 control averages uniformly over "
            "eligible source IDs of that kind in the same scoring batch, then over their destinations. "
            "D5 chooses the best positive realized semantic "
            "gain among semantically valid D3 actions, otherwise KEEP. It uses hidden labels and bypasses "
            "the guard, so it is an upper-reference diagnostic rather than a deployable policy. The "
            "matched oracle column holds each policy's IDs, action kinds and count fixed, preferring a "
            "class-correct alternative when available. A single global Hungarian mapping per checkpoint "
            "is fitted only in the sealed stage; unmatched slots receive no correctness credit."
        ),
        "",
        "| Mode / policy | MOVE / ABSTAIN accepted | Harmful / accepted | Correct destination in pool among wrong-source eligible IDs | Discovery destination correct / MOVE | Selected destination correct / MOVE | Correct→wrong MOVE | Gain All / Old / New ×10⁶ | D4 same ID | D4 random ID | Matched oracle gap |",
        "|---|---|---|---|---|---|---:|---|---|---|---|",
    ]
    for mode, policies in result["summary"].items():
        for p, r in policies.items():
            text.append(
                f"| {mode} / {p} | {r['move']} / {r['abstain']} | {r['harmful']} / {r['count']} | "
                f"{r['wrong_source_with_correct_destination_in_pool']} / {r['wrong_source_with_eligible_move']} | "
                f"{r['discovery_destination_correct']} / {r['discovery_move']} | "
                f"{r['selected_destination_correct']} / {r['move']} | {r['selected_correct_to_wrong']} | "
                f"{g(r['gain'])} | {g(r['D4_matched_gain'])} | "
                f"{g(r['D4_random_source_and_destination_gain'])} | {g(r['matched_oracle_gap'])} |"
            )
    text.extend(
        [
            "",
            (
                "Gains average over all 26 source trials, with zero for a no-op. Old/New refer to "
                "the semantic probe class. Coverage counts are recorded before V0 screening. The "
                "discovery and selection columns show what survives ranking and independent confirmation. "
                "One-step semantic margins on two checkpoints do not establish full-training effects."
            ),
            "",
        ]
    )
    for e, q in result["checkpoints"].items():
        reason = {}
        for b in q["batches"]:
            for value in b["eligibility"].values():
                reason[value] = reason.get(value, 0) + 1
        text.append(
            f"- Epoch {e} — {q['source_count']} frozen IDs, fresh eligibility {reason}, "
            f"{sum(len(b['arms']) - 1 for b in q['batches'])} non-KEEP virtual arms."
        )
    for mode, policies in result["summary"].items():
        for p, r in policies.items():
            text.append(f"- {mode} {p} — {r['reasons']}.")
    text.extend(
        [
            "",
            "[Frozen prediction hashes and per-arm sealed measurements](r2_destination_diagnostic.json)",
        ]
    )
    path.write_text("\n".join(text) + "\n", encoding="utf8")


def validate_decisions():
    arms = [
        {"v0": [[1.0, 1.0], [1.0, 1.0]]},
        {"v0": [[0.8, 0.9], [1.1, 0.9]]},
        {"v0": [[0.9, 0.9], [0.9, 0.9]]},
        {"v0": [[0.7, 1.1], [0.7, 0.9]]},
    ]
    assert decide(arms, [1, 2, 3])["reason"] == "confirmation_gain"
    assert decide(arms, [2, 3])["selected"] == 2
    assert decide(arms, [])["reason"] == "no_eligible_arm"
    criterion = DistillLoss(0, 1)
    raw = torch.tensor([[0.2, 0.0], [0.1, 0.2], [0.2, 0.1], [0.0, 0.2]])
    q = (raw / 0.04).softmax(-1).reshape(2, 2, -1)
    logits = torch.tensor(
        [[0.3, 0.4], [0.1, 0.2], [0.7, 0.1], [0.2, 0.9]], requires_grad=True
    )
    keep = criterion(logits, raw, 0)
    for a in (Action(0, "move", 0, 1), Action(0, "abstain", 0)):
        direct = corrected_distill(logits, raw, criterion, 0, [a], 0.2) - keep
        local = edit_loss((logits[[0, 2]] / 0.1).log_softmax(-1), a, q)
        torch.testing.assert_close(
            torch.autograd.grad(direct, logits, retain_graph=True)[0],
            torch.autograd.grad(local, logits)[0],
        )


if __name__ == "__main__":
    validate_decisions()
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "learned", "scratch", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    predict(a.work, a.learned, a.work / "e3-frozen-proposals.json", a.scratch)
    result = evaluate(a.work, a.scratch, a.output)
    report(result, a.output.with_name("CRISH_R2_DESTINATION_DIAGNOSTIC.md"))
