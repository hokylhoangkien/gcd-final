"""Replay E3 global proposals from R2 caches, then run a sealed label join."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from crish.experiment import file_digest, load_checkpoint
from crish.global_repair import candidates, encode, eval_dataset, frozen, source_ok
from crish.ids import rows_for_ids
from crish.reliability import blockwise_euclidean_knn, sparse_evidence
from crish.sampling import balanced_known_folds, global_neighbor_pool
from crish.scoring_batch import scoring_ids
from relational.state import write_json
from relational.train import build


def json_native(value):
    """Normalize audit results without relaxing the shared strict JSON writer."""
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {json_native(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    return value


class EncodingBudget:
    """The detached watchdog enforces the hard time cap for the whole job."""

    current = "audit"

    def __init__(self):
        self.encoded = Counter()

    def check(self):
        pass


def replay(work, scratch):
    model, train, _, center, _, _ = build()
    ds = train.dataset
    cached = np.load(work / "features.npy", mmap_mode="r")
    ids_blind = torch.from_numpy(np.load(work / "ids.npy")).long()
    torch.testing.assert_close(
        ids_blind, torch.as_tensor(ds.unlabelled_dataset.ids).long()
    )
    ids = torch.cat([torch.as_tensor(ds.labelled_dataset.uq_idxs).long(), ids_blind])
    known_n = len(ds.labelled_dataset)
    device = torch.device("cuda:0")
    args = SimpleNamespace(crish_encode_batch=8, num_workers=0)
    p = {
        "cache": 512,
        "k": 5,
        "trials": 2,
        "boundaries": 64,
        "scoring": 256,
        "folds": 9,
        "known_fold": 32,
        "unlabeled_fold": 64,
        "MOVE": 0.2,
        "ABSTAIN": 1.0,
    }
    results = {"config": p, "snapshots": {}}
    for index, epoch in ((1, 100), (2, 150)):
        start = time.monotonic()
        checkpoint = work / f"input/epoch-{epoch}.pt"
        model.load_state_dict(load_checkpoint(checkpoint)["model"])
        backbone, head = model.eval()
        budget = EncodingBudget()
        seed = 43181 + epoch
        g = torch.Generator().manual_seed(seed)
        known = torch.arange(known_n)
        with frozen(backbone, seed, device):
            # A subset dataset still uses the original merged legal-label adapter.
            from torch.utils.data import Subset

            kcenter, _kids, klabel = encode(
                backbone,
                Subset(eval_dataset(ds, center), known.tolist()),
                None,
                device,
                args,
                budget,
                ids[known],
            )
            legal = torch.cat([klabel, torch.full_like(ids_blind, -1)])
            x = torch.cat(
                [
                    kcenter[0],
                    F.normalize(torch.from_numpy(cached[index, 0].copy()), dim=-1),
                ]
            )

            def logits(z, head=head):
                return head.last_layer(z.to(device)).cpu()

            with torch.no_grad():
                slots = logits(x).argmax(-1)
            state = SimpleNamespace(
                ids=ids,
                features=x,
                labels=[slots],
                labeled=legal >= 0,
                abstained=[torch.zeros_like(legal, dtype=torch.bool)],
            )
            other = torch.arange(known_n, len(ids))
            kfold = balanced_known_folds(
                known, legal[known], 32, g, per_class=2, folds=9
            )
            order = other[torch.randperm(len(other), generator=g)]
            reserved = torch.cat([*kfold, order[: 9 * 64]])
            available = (
                (~torch.isin(torch.arange(len(ids)), reserved)).nonzero().flatten()
            )
            pool = global_neighbor_pool(state, available, 512, 5, g)
            first = torch.empty(2, len(pool), x.shape[-1])
            is_known = pool < known_n
            first[:, ~is_known] = F.normalize(
                torch.from_numpy(
                    cached[index, 1:][:, pool[~is_known].numpy() - known_n].copy()
                ),
                dim=-1,
            )
            if is_known.any():
                first[:, is_known], _, _ = encode(
                    backbone,
                    ds,
                    pool[is_known],
                    device,
                    args,
                    budget,
                    ids[pool[is_known]],
                )
            second, _, _ = encode(backbone, ds, pool, device, args, budget, ids[pool])
            trials = [
                tuple(blockwise_euclidean_knn(v.to(device), 5) for v in views)
                for views in (first, second)
            ]
            edges = sparse_evidence(ids[pool], trials)
            counts = Counter(slots[available].tolist())
            options, adj = candidates(state, pool, edges, counts, 64)
            # Uncapped production proposals supply the pre-ranking audit population.
            all_options, _ = candidates(state, pool, edges, counts, len(pool))
            if not options:
                raise ValueError("No global candidates in selected snapshot")
            score_ids, _ = scoring_ids(state, options, adj, pool, reserved, 256)
            rows = rows_for_ids(ids, score_ids)
            if not legal[rows].ge(0).any():
                rows[-1] = available[legal[available] >= 0][0]
                score_ids = ids[rows]
            feature_map = {int(r): first[:, i] for i, r in enumerate(pool)}
            missing = torch.tensor(
                [int(r) for r in rows if int(r) not in feature_map], dtype=torch.long
            )
            if len(missing):
                extra, _, _ = encode(
                    backbone, ds, missing, device, args, budget, ids[missing]
                )
                feature_map.update({int(r): extra[:, i] for i, r in enumerate(missing)})
            with torch.no_grad():
                probs = (
                    logits(torch.stack([feature_map[int(r)] for r in rows], dim=1))
                    / 0.04
                ).softmax(-1)
                pool_probs = (logits(first) / 0.04).softmax(-1)
            lookup = {int(key): i for i, key in enumerate(score_ids)}
            by_id = dict(zip(ids.tolist(), slots.tolist(), strict=True))
            pindex = {int(key): i for i, key in enumerate(ids[pool])}
            selected = set(options)
            output = []
            for a in all_options:
                support = Counter()
                for peer, weight in adj[a.sample_id]:
                    support[by_id[peer]] += weight
                evidence = (sum(support.values()) - 2 * support[a.source]) / len(
                    adj[a.sample_id]
                )
                eligible_slots = (
                    sorted(
                        (s for s in support if s != a.source and counts[s] >= 2),
                        key=lambda s: (-support[s], s),
                    )
                    if counts[a.source] >= 3
                    else []
                )
                # Every top-k entry is a measured peer-backed alternative, with production tie order.
                qrow = pindex[a.sample_id]
                output.append(
                    {
                        "id": a.sample_id,
                        "kind": a.kind,
                        "source": a.source,
                        "target": a.target,
                        "evidence": evidence,
                        "alternatives": eligible_slots,
                        "in_top64": a in selected,
                        "pool_teacher_ok": source_ok(pool_probs, qrow, a, 0.2),
                        "score_teacher_ok": source_ok(
                            probs, lookup[a.sample_id], a, 0.2
                        )
                        if a in selected
                        else None,
                    }
                )
        results["snapshots"][str(epoch)] = {
            "proposals": output,
            "available_ids": ids[available].tolist(),
            "pool_ids": ids[pool].tolist(),
            "score_ids": score_ids.tolist(),
            "predicted_slots": dict(zip(ids.tolist(), slots.tolist(), strict=True)),
            "probe_ids": ids[reserved].tolist(),
            "sha256": file_digest(checkpoint),
            "seed": seed,
            "new_image_views": budget.encoded["audit"],
            "seconds": time.monotonic() - start,
        }
        write_json(scratch, json_native(results))
        print(
            f"Frozen epoch {epoch}: {len(output)} uncapped proposals; {len(options)} shortlisted",
            flush=True,
        )


def align(ids, slots, labels):
    table = np.zeros((196, 196), dtype=int)
    np.add.at(table, ([slots[i] for i in ids], labels[ids]), 1)
    row, col = linear_sum_assignment(table, maximize=True)
    mapping = {int(r): int(c) for r, c in zip(row, col, strict=True) if table[r, c] > 0}
    return mapping, {
        "unmatched_slots": sorted(set(range(196)) - mapping.keys()),
        "active_unmatched_slots": sorted(
            set(np.flatnonzero(table.sum(1))) - mapping.keys()
        ),
        "unmatched_classes": sorted(set(range(196)) - set(mapping.values())),
        "alignment_accuracy": float(table[row, col].sum() / len(ids)),
    }


def ratio(num, den):
    return num / den if den else None


def rank_stats(y, s):
    if len(set(y)) < 2:
        return {"auroc": None, "average_precision": None, "spearman": None}
    rank = float(spearmanr(y, s).statistic) if len(set(s)) > 1 else None
    return {
        "auroc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "spearman": rank,
    }


def metrics(pop, props, wrong, correct, seen):
    candidate_ids = {r["id"] for r in props if r["in_top64"]}
    actionable = {r["id"] for r in props if r["score_teacher_ok"]}
    by_id = {r["id"]: r["evidence"] for r in props}
    errors = sum(map(wrong, pop))
    detection = {}
    for kind, picked in (("boundary", candidate_ids), ("teacher_eligible", actionable)):
        n = sum(map(wrong, picked))
        detection[kind] = {
            "selected": len(picked),
            "errors": n,
            "precision": ratio(n, len(picked)),
            "recall": ratio(n, errors),
            "random_precision": ratio(errors, len(pop)),
            "random_expected_errors": len(picked) * ratio(errors, len(pop))
            if pop
            else None,
            "random_expected_recall": ratio(len(picked), len(pop)) if errors else None,
        }
    floor = min(by_id.values(), default=0.0) - 1.0
    detection.update(
        {
            "n": len(pop),
            "errors": errors,
            **rank_stats([wrong(i) for i in pop], [by_id.get(i, floor) for i in pop]),
        }
    )
    result = {"detection": detection}
    for kind in ("move", "abstain"):
        selected = [r for r in props if r["kind"] == kind and r["score_teacher_ok"]]
        pool = [r for r in props if r["kind"] == kind and r["pool_teacher_ok"]]
        labels = [correct(r) for r in selected]
        record = {
            "n": len(selected),
            "correct": sum(labels),
            "precision": ratio(sum(labels), len(labels)),
            "evidence_vs_correctness": rank_stats(
                labels, [r["evidence"] for r in selected]
            ),
            "matched_count_random_precision": ratio(sum(map(correct, pool)), len(pool)),
            "random_pool_n": len(pool),
            "eligible_pool_evidence_rank": rank_stats(
                [correct(r) for r in pool], [r["evidence"] for r in pool]
            ),
        }
        if kind == "move":
            wrong_source = [r for r in selected if wrong(r["id"])]
            correct_source = [r for r in selected if not wrong(r["id"])]
            record.update(
                {
                    "destination_accuracy": ratio(
                        sum(seen(r, r["target"]) for r in selected), len(selected)
                    ),
                    "wrong_source_n": len(wrong_source),
                    "correct_source_n": len(correct_source),
                    "correct_source_to_wrong_destination": ratio(
                        len(correct_source), len(selected)
                    ),
                    "wrong_destination_given_correct_source": 1.0
                    if correct_source
                    else None,
                    "recall_at_k_given_wrong_source": {
                        str(k): ratio(
                            sum(
                                any(seen(r, s) for s in r["alternatives"][:k])
                                for r in wrong_source
                            ),
                            len(wrong_source),
                        )
                        for k in (1, 3, 5)
                    },
                    "random_peer_destination_accuracy": ratio(
                        sum(
                            sum(seen(r, s) for s in r["alternatives"])
                            / len(r["alternatives"])
                            for r in wrong_source
                        ),
                        len(wrong_source),
                    ),
                }
            )
        result[kind] = record
    return result


def seal(work, frozen_path, out):
    from scripts.audit_r2_evidence import sealed_labels

    result = json.loads(frozen_path.read_text())
    result["frozen_proposals_sha256"] = file_digest(frozen_path)
    ids = np.load(work / "ids.npy").tolist()
    blind = set(ids)
    labels = sealed_labels()
    for s in result["snapshots"].values():
        slots = {int(k): v for k, v in s["predicted_slots"].items()}
        mapping, summary = align(ids, slots, labels)
        s["alignment"] = {"slot_to_class": mapping, **summary}

        def wrong(i, mapping=mapping, slots=slots):
            return bool(mapping.get(slots[i], -1) != labels[i])

        def seen(r, slot, mapping=mapping):
            return bool(mapping.get(slot, -1) == labels[r["id"]])

        def correct(r):
            return wrong(r["id"]) and (r["kind"] == "abstain" or seen(r, r["target"]))

        global_pop = [i for i in s["available_ids"] if i in blind]
        pool_pop = [i for i in s["pool_ids"] if i in blind]
        s["metrics"] = {}
        for group in ("all", "old", "new"):
            for pop_name, pop in (("global", global_pop), ("cache", pool_pop)):
                subset = [
                    i
                    for i in pop
                    if group == "all" or (labels[i] < 98) == (group == "old")
                ]
                use = set(subset)
                s["metrics"].setdefault(group, {})[pop_name] = metrics(
                    subset,
                    [r for r in s["proposals"] if r["id"] in use],
                    wrong,
                    correct,
                    seen,
                )
        for r in s["proposals"]:
            r.update(
                {
                    "source_wrong": bool(wrong(r["id"])),
                    "correct": bool(correct(r)),
                    "semantic_subset": "old" if labels[r["id"]] < 98 else "new",
                }
            )
    result = json_native(result)
    write_json(out, result)
    return result


def report(result, path):
    def fmt(x):
        return "—" if x is None else f"{100 * x:.1f}"

    lines = [
        "# E3 global evidence and proposal audit",
        "",
        (
            "Two checkpoint-conditioned global replays at R2 epochs 100 and 150. We run the production "
            "probe reservation, global neighbor shortlist, reciprocal/cross-view evidence, candidate ranking, "
            "peer-aware scoring batch and teacher eligibility routines. Config: 512 cached rows, k=5, two "
            "trials, 64 boundaries, 256 scoring rows and nine probe folds. MOVE mass is .2 and ABSTAIN is full. "
            "V0 is unchanged and is not stepped or scored. No virtual SGD or downstream acceptance is evaluated."
        ),
        "",
        (
            "Unlabeled center and first-trial crops come from the existing R2 audit cache. Only missing "
            "known centers, known first-trial views, second-trial evidence views and scoring fillers are "
            "encoded. This replays the global proposal mechanism on R2 checkpoints. It cannot recreate "
            "historical E3 model states, worker-specific random crops or exact historical proposals. "
            "A batch-local approximation was unnecessary because the global mechanism was executable."
        ),
        "",
        (
            "Global one-to-one Hungarian alignment uses only the sealed full unlabeled split. It maximizes "
            "accuracy on the same panel and can favor majority modes, hide splitting and force merging into one assigned class. "
            "Zero-support assignments are excluded. An unmatched source counts as wrong and an unmatched "
            "destination receives no credit. No alignment is used for evidence or proposals."
        ),
        "",
        "### Detection and action precision",
        "",
        "| Epoch / subgroup | Global errors / n | Boundaries precision / recall (%) | Eligible precision / recall (%) | Global / cache ROC AUC (%) | Global random precision (%) | ABSTAIN wrong / n · random (%) | MOVE correct / n · random (%) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for epoch, s in result["snapshots"].items():
        for group, m in s["metrics"].items():
            g, c = m["global"], m["cache"]
            d = g["detection"]
            b = d["boundary"]
            a = d["teacher_eligible"]
            abst, move = g["abstain"], g["move"]
            lines.append(
                f"| {epoch} / {group} | {d['errors']} / {d['n']} | {fmt(b['precision'])} / {fmt(b['recall'])} | "
                f"{fmt(a['precision'])} / {fmt(a['recall'])} | {fmt(d['auroc'])} / {fmt(c['detection']['auroc'])} | {fmt(a['random_precision'])} | "
                f"{abst['correct']} / {abst['n']} · {fmt(abst['matched_count_random_precision'])} | "
                f"{move['correct']} / {move['n']} · {fmt(move['matched_count_random_precision'])} |"
            )
    lines.extend(
        [
            "",
            "### MOVE destination and evidence ranking",
            "",
            "| Epoch / subgroup | Wrong / correct source | Destination accuracy (%) | Recall@1 / 3 / 5 given wrong source (%) | Correct→wrong among MOVE (%) | Random peer accuracy on same wrong-source IDs (%) | Evidence AUC ABSTAIN / MOVE in eligible pool (%) |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for epoch, s in result["snapshots"].items():
        for group, m in s["metrics"].items():
            g = m["global"]
            v = g["move"]
            recall = " / ".join(
                fmt(x) for x in v["recall_at_k_given_wrong_source"].values()
            )
            lines.append(
                f"| {epoch} / {group} | {v['wrong_source_n']} / {v['correct_source_n']} | "
                f"{fmt(v['destination_accuracy'])} | {recall} | {fmt(v['correct_source_to_wrong_destination'])} | "
                f"{fmt(v['random_peer_destination_accuracy'])} | {fmt(g['abstain']['eligible_pool_evidence_rank']['auroc'])} / "
                f"{fmt(v['eligible_pool_evidence_rank']['auroc'])} |"
            )
    lines.extend(
        [
            "",
            (
                "Scores for rows without measured support tie below every scored row. Global recall "
                "includes non-probe unlabeled errors outside the cache. A low global recall is partly a 64-row "
                "budget effect. Random controls are the exact expectations of uniform draws with the same count, "
                "conditional on snapshot/group and (for action precision) kind and teacher eligibility. The random "
                "destination control holds the MOVE ID fixed and samples from supported alternatives. Recall@k "
                "uses only peer-backed targets with sufficient slot support. Many rows have fewer than five. "
                "The chosen destination remains the original E3 top-1."
            ),
            "",
            "### Bottleneck assessment",
            "",
        ]
    )
    for ep, s in result["snapshots"].items():
        g = s["metrics"]["all"]["global"]
        m = g["move"]
        a = g["abstain"]
        d = g["detection"]["teacher_eligible"]
        r1, r5 = [m["recall_at_k_given_wrong_source"][k] for k in ("1", "5")]
        if not m["wrong_source_n"]:
            finding = "No eligible wrong-source MOVE supports a destination conclusion."
        elif r5 < 0.5 and m["wrong_source_n"] >= m["correct_source_n"]:
            finding = "Destination generation is the largest directly observed loss among MOVE proposals: a majority of wrong-source candidates lack the aligned true class even in the top five."
        elif r5 - r1 >= 0.2:
            finding = "Destination ordering loses substantial recoverable accuracy: the true slot appears below rank one on many wrong-source candidates."
        elif m["correct_source_n"] > m["wrong_source_n"]:
            finding = "Sample detection is the clearest observed MOVE bottleneck: most eligible moves would disturb an aligned correct source."
        else:
            finding = "These counts do not isolate one dominant bottleneck. The detection and destination gates both exclude or misrank some errors."
        lines.append(
            f"- Epoch {ep} — {finding} Detection precision {fmt(d['precision'])}% against random {fmt(d['random_precision'])}%. "
            f"ABSTAIN evidence AUC {fmt(a['eligible_pool_evidence_rank']['auroc'])}%. "
            f"Unmatched slots {s['alignment']['unmatched_slots']}; active unmatched slots {s['alignment']['active_unmatched_slots']}."
        )
    lines.extend(
        [
            "",
            (
                "These are descriptive, dependent samples from two snapshots. A proposal that is correct under "
                "Hungarian alignment does not guarantee improved feature geometry or cluster accuracy after an optimizer step."
            ),
            "",
            "[Per-action predictions, raw counts and sealed alignment](r2_e3_proposal_audit.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "frozen", "output"):
        p.add_argument(key, type=Path)
    p.add_argument(
        "--sealed-only",
        action="store_true",
        help="Reuse an existing frozen proposal receipt",
    )
    args = p.parse_args()
    if not args.sealed_only:
        replay(args.work, args.frozen)
    result = seal(args.work, args.frozen, args.output)
    report(result, args.output.with_name("CRISH_E3_PROPOSAL_AUDIT.md"))
