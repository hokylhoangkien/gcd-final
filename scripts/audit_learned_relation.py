"""Class-disjoint Known supervision and a sealed diagnostic of E3 destinations."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from crish.experiment import file_digest, load_checkpoint
from crish.reliability import blockwise_euclidean_knn, sparse_evidence
from relational.state import write_json
from relational.train import build
from scripts.audit_e3_proposals import align, json_native

METHODS = ("L0", "L1", "L2", "L3")
SEED = 44903


def pair_features(a, b):
    return np.concatenate((np.abs(a - b), a * b), axis=-1)


def make_pairs(x, y, rng):
    """Balanced split-local pairs; retain half uniform and half hard negatives."""
    sim = x @ x.T
    result = {}
    for i, c in enumerate(y):
        pos = np.flatnonzero((y == c) & (np.arange(len(y)) != i))
        neg = np.flatnonzero(y != c)
        if not len(pos) or not len(neg):
            continue
        hard = neg[np.argsort(-sim[i, neg], kind="stable")[:8]]
        for j, z, tag in [
            (int(rng.choice(pos)), 1, "positive"),
            (int(rng.choice(neg)), 0, "uniform"),
            (int(rng.choice(hard)), 0, "hard"),
        ]:
            edge = tuple(sorted((i, j)))
            # Keep unique unordered pairs. A hard negative overrides a duplicate uniform draw.
            if edge not in result or tag == "hard":
                result[edge] = (z, tag)
    positives = [p for p, (z, _) in result.items() if z]
    negatives = {
        t: [p for p, (z, tag) in result.items() if not z and tag == t]
        for t in ("uniform", "hard")
    }
    size = min(len(positives) // 2, *(len(v) for v in negatives.values()))
    assert size > 0
    chosen = rng.choice(len(positives), 2 * size, replace=False)
    keys = [positives[i] for i in chosen]
    for v in negatives.values():
        keys.extend(v[i] for i in rng.choice(len(v), size, replace=False))
    ij = np.asarray(keys)
    return (
        pair_features(x[ij[:, 0]], x[ij[:, 1]]),
        np.asarray([result[p][0] for p in keys]),
        ij,
    )


def fit_models(x, y, tr, va, seed):
    rng = np.random.default_rng(seed)
    a, b, ij = make_pairs(x[tr], y[tr], rng)
    c, d, vij = make_pairs(x[va], y[va], rng)
    assert np.intersect1d(tr[ij], va[vij]).size == 0
    log = LogisticRegression(C=1.0, max_iter=200, random_state=seed).fit(a, b)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1536, 128), nn.ReLU(), nn.Linear(128, 1)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    tx, ty = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda().float()
    vx = torch.from_numpy(c).cuda()
    best = -1.0
    weights = None
    selected = 0
    history = []
    for epoch in range(1, 26):
        model.train()
        for idx in np.array_split(
            rng.permutation(len(a)), max(1, (len(a) + 127) // 128)
        ):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(
                model(tx[idx]).flatten(), ty[idx]
            ).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(d, model(vx).flatten().cpu().numpy())
        history.append(auc)
        if auc > best + 1e-6:
            best, selected = auc, epoch
            weights = copy.deepcopy(model.state_dict())
        elif epoch - selected >= 4:
            break
    model.load_state_dict(weights)
    model.requires_grad_(False)
    return (
        log,
        model,
        {
            "train_pairs": len(a),
            "validation_pairs": len(c),
            "logistic_validation_auc": roc_auc_score(d, log.decision_function(c)),
            "mlp_selected_epoch": selected,
            "mlp_validation_auc": history,
            "logistic_sha256": hashlib.sha256(
                log.coef_.tobytes() + log.intercept_.tobytes()
            ).hexdigest(),
            "mlp_sha256": hashlib.sha256(
                b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())
            ).hexdigest(),
        },
    )


@torch.no_grad()
def known_features(backbone, ds, center, states):
    """Encode only missing legal centers, sharing the frozen prefix across snapshots."""
    from relational.sensitivity_metrics import prefix
    from scripts.audit_r2_virtual import encode

    original = copy.copy(ds)
    original.transform = center
    blocks = [copy.deepcopy(backbone.blocks[-1]) for _ in states]
    for block, state in zip(blocks, states, strict=True):
        block.load_state_dict(
            {
                k.removeprefix("0.blocks.11."): v
                for k, v in state.items()
                if k.startswith("0.blocks.11.")
            }
        )
    outputs = [[] for _ in blocks]
    ids, y = [], []
    active = backbone.blocks[-1]
    try:
        for images, legal, keys in DataLoader(original, batch_size=16):
            assert (legal >= 0).all()
            ids.extend(keys.tolist())
            y.extend(legal.tolist())
            tokens = prefix(backbone, images)
            for block, output in zip(blocks, outputs, strict=True):
                backbone.blocks[-1] = block
                output.append(
                    F.normalize(encode(backbone, tokens), dim=-1).cpu().numpy()
                )
    finally:
        backbone.blocks[-1] = active
    return np.asarray(ids), np.asarray(y), [np.concatenate(o) for o in outputs]


def destination_scores(
    candidate, x, ids, slots, group, log, mlp, alternatives, pair_evidence
):
    """All learned methods see exactly the same cosine-shortlisted peers per slot."""
    pool = group[group != candidate]
    sim = x[pool] @ x[candidate]
    ordered = pool[np.argsort(-sim, kind="stable")]
    per_slot = {}
    for i in ordered:
        p = per_slot.setdefault(slots[i], [])
        if len(p) < 3:
            p.append(int(i))
    peers = np.array([i for v in per_slot.values() for i in v], dtype=int)
    if not len(peers):
        return {"peers": [], "pair_scores": [], "ranks": {m: [] for m in METHODS}}
    z = pair_features(x[candidate], x[peers])
    with torch.no_grad():
        scores = np.column_stack(
            [
                [
                    pair_evidence.get(
                        tuple(sorted((int(ids[candidate]), int(ids[i])))), 0.0
                    )
                    for i in peers
                ],
                x[peers] @ x[candidate],
                log.decision_function(z),
                mlp(torch.from_numpy(z).cuda()).flatten().cpu().numpy(),
            ]
        )
    rank = {}
    rank["L0"] = [int(s) for s in alternatives if s in per_slot]
    rank["L0"] += sorted(int(s) for s in per_slot if s not in alternatives)
    offset = 0
    slot_scores = {}
    for s, p in per_slot.items():
        slot_scores[int(s)] = scores[offset : offset + len(p)].max(0)
        offset += len(p)
    for j, m in enumerate(METHODS[1:], 1):
        rank[m] = sorted(slot_scores, key=lambda s: (-slot_scores[s][j], s))
    return {
        "peers": ids[peers].tolist(),
        "pair_scores": scores.tolist(),
        "ranks": {m: v[:5] for m, v in rank.items()},
    }


def freeze(work, proposal_file, out):
    source = json.loads(proposal_file.read_text())
    model, train, _, center, _, _ = build()
    states = [
        load_checkpoint(work / f"input/epoch-{e}.pt")["model"] for e in (100, 150)
    ]
    model.eval()
    print("Encoding missing legally labeled Known centers", flush=True)
    valid_ids, legal, known = known_features(
        model[0], train.dataset.labelled_dataset, center, states
    )
    assert sorted(set(legal.tolist())) == list(range(98))
    classes = np.random.default_rng(SEED).permutation(98)
    tr = np.flatnonzero(np.isin(legal, classes[:78]))
    va = np.flatnonzero(np.isin(legal, classes[78:]))
    ids = np.load(work / "ids.npy")
    assert (
        not np.intersect1d(ids, valid_ids).size
        and not np.intersect1d(valid_ids[tr], valid_ids[va]).size
    )
    lookup = {int(k): i for i, k in enumerate(ids)}
    cached = np.load(work / "features.npy", mmap_mode="r")
    result = {
        "proposal_sha256": file_digest(proposal_file),
        "seed": SEED,
        "train_classes": sorted(classes[:78].tolist()),
        "validation_classes": sorted(classes[78:].tolist()),
        "train_ids": valid_ids[tr].tolist(),
        "validation_ids": valid_ids[va].tolist(),
        "snapshots": {},
    }
    for n, epoch in enumerate((100, 150)):
        began = time.monotonic()
        s = source["snapshots"][str(epoch)]
        p = [r for r in s["proposals"] if r["kind"] == "move" and r["score_teacher_ok"]]
        model.load_state_dict(states[n])
        log, mlp, details = fit_models(known[n], legal, tr, va, SEED)
        x = F.normalize(torch.from_numpy(cached[n + 1, 0].copy()), dim=-1).numpy()
        slots = np.array([s["predicted_slots"][str(i)] for i in ids])
        with torch.no_grad():
            old_slots = (
                model[1]
                .last_layer(torch.from_numpy(known[n]).cuda())
                .argmax(-1)
                .cpu()
                .numpy()
            )
        counts = np.zeros((196, 98), int)
        np.add.at(counts, (old_slots, legal), 1)
        rr, cc = linear_sum_assignment(counts, maximize=True)
        anchors = {
            int(r): int(c) for r, c in zip(rr, cc, strict=True) if counts[r, c] > 0
        }
        cache = np.array([lookup[k] for k in s["pool_ids"] if k in lookup])
        global_rows = np.array([lookup[k] for k in s["available_ids"] if k in lookup])
        # Only a single cached unlabeled trial is available. This is a paired L0
        # evidence proxy, while L0 destination ordering below uses frozen E3.
        aug = F.normalize(
            torch.from_numpy(cached[n + 1, 1:][:, cache].copy()).cuda(), dim=-1
        )
        evidence = {
            k: v.joint
            for k, v in sparse_evidence(
                ids[cache], [tuple(blockwise_euclidean_knn(view, 5) for view in aug)]
            ).items()
        }
        global_counts = np.bincount(slots[global_rows], minlength=196)
        records = []
        for r in p:
            i = lookup[r["id"]]
            common = cache[np.isin(slots[cache], r["alternatives"])]
            wide = global_rows[
                (slots[global_rows] != r["source"])
                & (global_counts[slots[global_rows]] >= 2)
            ]
            records.append(
                {
                    "id": r["id"],
                    "source": r["source"],
                    "original_target": r["target"],
                    "evidence": r["evidence"],
                    "common": destination_scores(
                        i, x, ids, slots, common, log, mlp, r["alternatives"], evidence
                    ),
                    "expanded": destination_scores(
                        i, x, ids, slots, wide, log, mlp, r["alternatives"], evidence
                    ),
                }
            )
        result["snapshots"][str(epoch)] = {
            "slots": dict(zip(ids.tolist(), slots.tolist(), strict=True)),
            "known_anchor": anchors,
            "training": details,
            "candidates": records,
            "checkpoint_sha256": file_digest(work / f"input/epoch-{epoch}.pt"),
            "seconds": time.monotonic() - began,
        }
        print(
            f"Frozen epoch {epoch}: {len(records)} teacher-eligible MOVE candidates",
            flush=True,
        )
    write_json(out, json_native(result))


def binary_metrics(y, s):
    if not len(y) or len(set(y)) < 2:
        return {"n": len(y), "positive": int(sum(y)), "auroc": None, "auprc": None}
    return {
        "n": len(y),
        "positive": int(sum(y)),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }


def assess(group, pool, mapping, labels, anchored):
    result = {}
    pair_y, pair_s, hard_y, hard_s = [], [], [], []
    for r in group:
        q = r[pool]
        scores = np.asarray(q["pair_scores"])
        if not len(scores):
            continue
        y = labels[q["peers"]] == labels[r["id"]]
        neg = np.flatnonzero(~y)
        hard = neg[np.argsort(-scores[neg, 1], kind="stable")[: min(5, len(neg))]]
        inds = np.r_[np.flatnonzero(y), hard]
        pair_y.extend(y.tolist())
        pair_s.extend(scores.tolist())
        hard_y.extend(y[inds].tolist())
        hard_s.extend(scores[inds].tolist())
    for j, method in enumerate(METHODS):
        # For Old-only legal alignment, unknown sources are excluded from
        # source-conditioned rates; unmatched targets remain uncredited.
        observed = [r for r in group if not anchored or r["source"] in mapping]
        wrong = [r for r in observed if mapping.get(r["source"], -1) != labels[r["id"]]]
        right = [r for r in observed if r not in wrong]

        def hit(r, k, method=method):
            return any(
                mapping.get(s, -1) == labels[r["id"]]
                for s in r[pool]["ranks"][method][:k]
            )

        def has_target(r, method=method):
            return bool(r[pool]["ranks"][method])

        n = len(observed)
        result[method] = {
            "candidates": len(group),
            "mapped_source_eval": n,
            "wrong_source": len(wrong),
            "destination_coverage": sum(map(has_target, observed)),
            "destination_accuracy": sum(hit(r, 1) for r in observed) / n if n else None,
            "move_correctness": sum(hit(r, 1) for r in wrong) / n if n else None,
            "correct_to_wrong": sum(has_target(r) and not hit(r, 1) for r in right) / n
            if n
            else None,
            "recall": {
                str(k): sum(hit(r, k) for r in wrong) / len(wrong) if wrong else None
                for k in (1, 3, 5)
            },
            "pairs": binary_metrics(pair_y, [s[j] for s in pair_s]),
            "hard_pairs": binary_metrics(hard_y, [s[j] for s in hard_s]),
        }
    return result


def seal(frozen, out):
    from scripts.audit_r2_evidence import sealed_labels

    result = json.loads(frozen.read_text())
    result["frozen_prediction_sha256"] = file_digest(frozen)
    labels = sealed_labels()
    for snap in result["snapshots"].values():
        slots = {int(k): v for k, v in snap["slots"].items()}
        hung, match = align(list(slots), slots, labels)
        anchor = {int(k): v for k, v in snap["known_anchor"].items()}
        snap["alignment"] = {
            "hungarian": hung,
            "legal_old": anchor,
            "hungarian_details": match,
        }
        snap["metrics"] = {}
        for mode, mapping in (("hungarian", hung), ("legal_old", anchor)):
            snap["metrics"][mode] = {}
            for pool in ("common", "expanded"):
                snap["metrics"][mode][pool] = {}
                for group in ("all", "old", "new") if mode == "hungarian" else ("old",):
                    rows = [
                        r
                        for r in snap["candidates"]
                        if group == "all" or (labels[r["id"]] < 98) == (group == "old")
                    ]
                    snap["metrics"][mode][pool][group] = assess(
                        rows, pool, mapping, labels, mode == "legal_old"
                    )
        raw = snap["candidates"]
        snap["frozen_e3_control"] = {
            "n": len(raw),
            "mapped_destination_correct": sum(
                bool(hung.get(r["original_target"], -1) == labels[r["id"]]) for r in raw
            ),
            "target_missing_common": sum(
                r["original_target"] not in r["common"]["ranks"]["L0"] for r in raw
            ),
        }
    result = json_native(result)
    write_json(out, result)
    return result


def report(path, result):
    def fmt(v):
        return "—" if v is None else f"{100 * v:.1f}"

    lines = [
        "# Bounded learned semantic relation audit",
        "",
        (
            "R2 checkpoints 100/150. Only missing legal Known center features were encoded. "
            "The two heads per checkpoint used the same fixed, class-disjoint 78/20 Known split. "
            "All pairs are symmetric and unique within each split, balanced positive/negative, "
            "with half of the negative pairs mined from top-eight cosine neighbors of other Known classes. "
            "L2 uses fixed L2-regularized logistic regression. L3 uses the 1536→128→1 MLP, "
            "at most 25 epochs, with patience 4 on Known validation AUROC. "
            "All predictions, training manifests and model hashes were frozen before opening unlabeled labels."
        ),
        "",
        (
            "All methods use the exact same frozen teacher-eligible E3 MOVE IDs and source slots, mass .2. "
            "Common scoring uses the original peer-backed destination slots and cached unlabeled evidence-pool peers. "
            "A frozen slot supported only by an unavailable peer has no common-pool score; the JSON includes "
            "the unchanged original E3 top-1 as a separate control. Expanded scoring uses available unlabeled "
            "global peers in alternative slots with at least two members. All methods use the same three-or-fewer "
            "cosine-shortlisted peers per slot and at most five destination ranks. Learned slot scores are max peer scores."
        ),
        "",
        (
            "L0 destination ranking preserves the frozen E3 slot order where available. Expanded slots outside the "
            "frozen order receive tied evidence and are ordered by slot ID. Exact original two-trial dyadic scores "
            "were not exported, so L0 pairwise results use a one-trial reciprocal/cross-view proxy reconstructed "
            "from cached crops in the unlabeled evidence pool. Unsupported pairs score zero. "
            "Hard pair evaluation takes up to five highest-cosine different-class test peers per candidate "
            "after freezing scores. Pairwise estimates reuse candidate IDs and are descriptive."
        ),
        "",
        "| Epoch / mapping / pool / group | Method | Candidates / source-evaluable | Pair n | AUROC / AUPRC (%) | Hard AUROC / AUPRC (%) | Dest. top-1 (%) | R@1 / 3 / 5 on wrong source (%) | MOVE correct / correct→wrong (%) |",
        "|---|---|---|---:|---|---|---:|---|---|",
    ]
    for epoch, snap in result["snapshots"].items():
        for alignment, metrics in snap["metrics"].items():
            for pool, groups in metrics.items():
                for group, methods in groups.items():
                    for m, r in methods.items():
                        pair = r["pairs"]
                        hard = r["hard_pairs"]
                        lines.append(
                            f"| {epoch} / {alignment} / {pool} / {group} | {m} | {r['candidates']} / {r['mapped_source_eval']} | {pair['n']} | "
                            f"{fmt(pair['auroc'])} / {fmt(pair['auprc'])} | {fmt(hard['auroc'])} / {fmt(hard['auprc'])} | {fmt(r['destination_accuracy'])} | "
                            f"{' / '.join(fmt(v) for v in r['recall'].values())} | {fmt(r['move_correctness'])} / {fmt(r['correct_to_wrong'])} |"
                        )
    lines.extend(
        [
            "",
            (
                "The global Hungarian mapping is a sealed one-to-one alignment on the same unlabeled panel, "
                "so class splits, merges and unmatched slots limit interpretation. The Old anchor mapping uses "
                "only legal Known labels, drops zero-support assignments and excludes unmapped source slots from "
                "source-conditioned Old rates. Unmapped targets receive no correctness credit. Novel is evaluated "
                "only with sealed Hungarian. This is a destination-ranking diagnostic, with no virtual update "
                "or accuracy claim."
            ),
            "",
        ]
    )
    for epoch, s in result["snapshots"].items():
        lines.append(
            f"- Epoch {epoch} — {s['training']['train_pairs']} training pairs, {s['training']['validation_pairs']} validation pairs, "
            f"MLP selected epoch {s['training']['mlp_selected_epoch']}, {len(s['known_anchor'])} legally anchored slots. "
            f"Actual E3 aligned top-1 {s['frozen_e3_control']['mapped_destination_correct']}/{s['frozen_e3_control']['n']}."
        )
    lines.extend(
        [
            "",
            "[Frozen prediction provenance and full metrics](r2_learned_relation.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "proposals", "frozen", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    freeze(a.work, a.proposals, a.frozen)
    report(a.output.with_name("CRISH_LEARNED_RELATION.md"), seal(a.frozen, a.output))
