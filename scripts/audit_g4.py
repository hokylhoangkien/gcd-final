"""G4 causal-time geometry diagnostics on frozen Cars snapshots."""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from crish.experiment import file_digest
from relational.state import write_json
from scripts import audit_g1 as g1
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_g3 import COVERAGES, G1_REPORT
from scripts.audit_g7 import pack

LOCAL = g1.WORK / "g4"
REPORT = Path("reports/CRISH_TEMPORAL_GEOMETRY.md")
OUTPUT = Path("reports/r2_temporal_geometry.json")
FIELDS = ("neighbors", "pair", "wrong", "dest")


def overlap_risk(a, b):
    shared = (a[:, :, None] == b[:, None, :]).sum((1, 2))
    return 1 - shared / (a.shape[1] + b.shape[1] - shared)


def slot_identity(old, current, bank, q):
    c = np.bincount(196 * old[bank] + current[bank], minlength=196**2).reshape(196, 196)
    a, b = linear_sum_assignment(c, maximize=True)
    chosen = c[a, b] > 0
    mapping = np.full(196, -1)
    mapping[a[chosen]] = b[chosen]
    same = mapping[old[q]] == current[q]
    return mapping, {
        "mapped_slots": int(chosen.sum()),
        "identity_matches": int((a[chosen] == b[chosen]).sum()),
        "bank_raw_agreement": g1.rate(old[bank] == current[bank]),
        "bank_aligned_agreement": g1.rate(mapping[old[bank]] == current[bank]),
        "query_raw_agreement": g1.rate(old[q] == current[q]),
        "query_aligned_agreement": g1.rate(same),
        "query_mapped": int((mapping[old[q]] >= 0).sum()),
    }


def ledger_signal(q_ids, receipt):
    # Row t refers to a graph built from checkpoint t-1. No row after 150 is used.
    path = g1.WORK / "input/relationships.jsonl"
    if not path.is_file():
        return None, {"available": False, "reason": "Relationship ledger absent"}
    checks = {name: sha for sha, name in (line.split() for line in receipt["members"])}
    digest = file_digest(path)
    if digest != checks["relationships.jsonl"]:
        raise ValueError("Relationship ledger hash mismatch")
    graphs = {}
    with path.open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["epoch"] in (101, 150) and row.get("graph"):
                graph = defaultdict(set)
                for a, b, *_ in row["graph"]["edges"]:
                    graph[a].add(b)
                    graph[b].add(a)
                graphs[row["epoch"]] = graph
            if row["epoch"] == 150:
                break
    if graphs.keys() != {101, 150}:
        return None, {"available": False, "reason": "Required graph rows absent"}
    risk, neither = [], 0
    for i in q_ids:
        a, b = graphs[101][i], graphs[150][i]
        union = len(a | b)
        neither += union == 0
        # Two empty edge sets show no observed change.
        risk.append(1 - len(a & b) / union if union else 0.0)
    return np.asarray(risk), {
        "available": True,
        "sha256": digest,
        "ledger_epochs": [101, 150],
        "graph_model_epochs": [100, 149],
        "queries_isolated_in_both": neither,
    }


def risk_only(out, name, risk):
    for field in FIELDS:
        out[f"{name}_{field}"] = risk if field == "wrong" else out[f"Current_{field}"]


def freeze_flags(out, methods):
    for name in methods:
        risk = out[f"{name}_wrong"]
        assert np.isfinite(risk).all()
        q = out["q"]
        order = np.lexsort((out["ids"][q], -risk))
        for percent in COVERAGES:
            out[f"{name}_flag_{percent}"] = order[: len(q) * percent // 100]


def snapshot(epoch, info):
    meta = info["snapshots"][str(epoch)]
    path = g1.LOCAL / f"frozen-{epoch}.npz"
    if file_digest(path) != meta["frozen_sha256"]:
        raise ValueError(f"G1 manifest hash mismatch for {epoch}")
    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in ("ids", "q", "bank", "slots", "pairs")}
        out.update({f"Current_{k}": f[f"B11_{k}"] for k in FIELDS})
    assert out["ids"][out["q"]].tolist() == info["query_ids"]
    return out


def temporal(old, out, cache, receipt):
    for k in ("ids", "q", "bank"):
        np.testing.assert_array_equal(old[k], out[k])
    ids, q, bank, slots = (out[k] for k in ("ids", "q", "bank", "slots"))
    previous = old["slots"]
    mapping, identity = slot_identity(previous, slots, bank, q)
    identity["past_to_current"] = mapping.tolist()
    out["history_slots"] = previous
    out["aligned_slot_stable"] = mapping[previous[q]] == slots[q]
    before, now = old["Current_neighbors"], out["Current_neighbors"]
    risk_only(out, "Neighborhood", overlap_risk(before, now))
    risk_only(out, "SlotChange", (~out["aligned_slot_stable"]).astype(float))
    both = before[:, :, None] == now[:, None, :]
    old_outside = previous[before] != previous[q, None]
    current_outside = slots[now] != slots[q, None]
    persistent = (both & old_outside[:, :, None] & current_outside[:, None, :]).sum(
        (1, 2)
    )
    risk_only(out, "Persistent", persistent / before.shape[1])
    a, b = [
        F.normalize(torch.from_numpy(cache[i, 0].copy()), dim=1).cuda() for i in (1, 2)
    ]
    risk_only(out, "Drift", (1 - (a[q] * b[q]).sum(1)).clamp_min(0).cpu().numpy())
    c = b[q] @ b[bank].T
    inv = np.full(len(ids), -1)
    inv[bank] = np.arange(len(bank))
    pairs = inv[out["pairs"]]
    assert (pairs >= 0).all()
    baseline = pack(c, q, bank, slots, pairs)
    for k in ("neighbors", "dest"):
        np.testing.assert_array_equal(baseline[k], out[f"Current_{k}"])
    for k in ("pair", "wrong"):
        np.testing.assert_allclose(
            baseline[k], out[f"Current_{k}"], atol=3e-6, rtol=3e-6
        )
    fused = pack(((a[q] @ a[bank].T) + c) / 2, q, bank, slots, pairs)
    out.update({f"Fusion_{k}": fused[k] for k in FIELDS})
    methods = ["Current", "Neighborhood", "Drift", "SlotChange", "Persistent", "Fusion"]
    ledger, history = ledger_signal(ids[q], receipt)
    if ledger is not None:
        risk_only(out, "LedgerNeighborhood", ledger)
        methods.append("LedgerNeighborhood")
    return methods, identity, history


def sealed(out, truth, methods, *, baseline="Current"):
    ids, q, bank, slots = (out[k] for k in ("ids", "q", "bank", "slots"))
    metrics, coverage = g1.evaluate(
        out, ids, q, bank, slots, truth, methods=methods, baseline=baseline
    )
    labels = truth[ids[q]]

    def wrong_for(s):
        m, _ = align(
            ids[bank].tolist(), dict(zip(ids.tolist(), s.tolist(), strict=True)), truth
        )
        return np.array([m.get(k, -1) for k in s[q]]) != labels

    wrong = wrong_for(slots)
    has_history = "history_slots" in out
    stable = out.get("aligned_slot_stable", np.zeros(len(q), dtype=bool))
    persistent = wrong & wrong_for(out["history_slots"]) if has_history else stable
    stable_errors = persistent & stable
    for group, mask in (
        ("All", labels >= 0),
        ("Old", labels < 98),
        ("New", labels >= 98),
    ):
        for name in methods:
            row = metrics[name][group]
            risk = out[f"{name}_wrong"]
            row["stable_error_n"] = (
                int((mask & stable_errors).sum()) if has_history else None
            )
            row["persistent_error_n"] = (
                int((mask & persistent).sum()) if has_history else None
            )
            row["stable_slot_error_auc"] = g1.auc(
                wrong[mask & stable], risk[mask & stable]
            )
            row["stable_error_auc"] = g1.auc(stable_errors[mask], risk[mask])
            row["flags"] = {}
            for p in COVERAGES:
                index = out[f"{name}_flag_{p}"]
                chosen = index[mask[index]]
                tp = int(wrong[chosen].sum())
                errors = int((mask & wrong).sum())
                n_stable = row["stable_error_n"]
                row["flags"][p] = {
                    "n": len(chosen),
                    "tp": tp,
                    "errors": errors,
                    "coverage": len(chosen) / int(mask.sum()) if mask.any() else None,
                    "precision": tp / len(chosen) if len(chosen) else None,
                    "recall": tp / errors if errors else None,
                    "stable_error_recall": int(stable_errors[chosen].sum()) / n_stable
                    if n_stable
                    else None,
                }
    return metrics, coverage


def report(result):
    lines = [
        "# G4 temporal geometry on Stanford Cars",
        "",
        (
            "Exploratory evaluation on reused G1 query and bank IDs. Epoch 100 is a current-only "
            "baseline. The cache supplies snapshot CLS and classifier slots at 100 and 150. No "
            "intervening per-ID classifier logits or slot trajectory is available. The verified "
            "edge ledger supplies graph history when available."
        ),
        "",
        (
            "Current risk is the alternative-minus-current slot similarity margin. Neighborhood "
            "is one minus top-10 Jaccard overlap. Drift is one minus same-image cosine. SlotChange "
            "uses a bank-only, label-blind maximum-overlap matching of historical and current slots. "
            "Persistent measures the fraction of the ten positions occupied by peers that remained "
            "neighbors and disagreed with the query's slot at both snapshots. Fusion averages "
            "historical and current cosine and ranks current slots. All risk-only rules keep current "
            "retrieval. LedgerNeighborhood uses undirected edge-set Jaccard distance from rows "
            "101 and 150 (models 100 and 149), with two empty sets scored zero."
        ),
        "",
        (
            "Before opening labels, each rule freezes its global top 10/20/40% flags using ascending "
            "sample ID to break ties. Old/New metrics use those same global flags. Error labels use "
            "bank-only Hungarian alignment per checkpoint. A stable error is a query wrong under "
            "both semantic alignments with the same slot after label-blind matching. Stable AUC "
            "treats all other queries as negatives. JSON also gives ordinary error AUROC restricted "
            "to unchanged slots and recall of stable errors at each coverage. A single slot match "
            "can conceal splits or merges, and binary slot-change scores have large tied groups."
        ),
        "",
    ]

    def f(v):
        return "—" if v is None else f"{v:.3f}"

    s = result["snapshots"]["150"]["slot_identity"]
    lines += [
        (
            f"**Slot identity.** {s['identity_matches']} of {s['mapped_slots']} matched slot pairs "
            f"kept their raw index. Bank agreement was {f(s['bank_raw_agreement'])} raw "
            f"and {f(s['bank_aligned_agreement'])} after matching. Query agreement was "
            f"{f(s['query_raw_agreement'])} raw and {f(s['query_aligned_agreement'])} after matching."
        ),
        "",
        "| Epoch / group / rule | Error AUC | P/R @10% | P/R @20% | P/R @40% | Dest R@1/3/5 | Stable errors / AUC |",
        "|---|---:|---|---|---|---|---|",
    ]
    for epoch, snap in result["snapshots"].items():
        for group in ("All", "Old", "New"):
            for name, entries in snap["metrics"].items():
                m = entries[group]
                flags = [
                    "/".join(f(m["flags"][p][k]) for k in ("precision", "recall"))
                    for p in COVERAGES
                ]
                dest = "/".join(f(m["destination_recall"][k]) for k in (1, 3, 5))
                n = m["stable_error_n"]
                lines.append(
                    f"| {epoch} / {group} / {name} | {f(m['detection_auc'])} | "
                    + " | ".join(flags)
                    + f" | {dest} | {n if n is not None else '—'} / {f(m['stable_error_auc'])} |"
                )
    lines += [
        "",
        f"Runtime {result['seconds']:.1f} seconds.",
        "",
        "[Counts, coverage, slot matching and frozen digests](r2_temporal_geometry.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    torch.use_deterministic_algorithms(True)
    info = json.loads(G1_REPORT.read_text())
    p = g1.WORK / "provenance.json"
    receipt = json.loads(p.read_text())
    feature_path = g1.WORK / "features.npy"
    result = {
        "base": "8ebbbda",
        "exploratory": True,
        "coverage_percent": COVERAGES,
        "g1_report_sha256": file_digest(G1_REPORT),
        "snapshot_receipt_sha256": file_digest(p),
        "snapshots": {},
    }

    def persist(epoch, out, methods, **notes):
        freeze_flags(out, methods)
        path = LOCAL / f"frozen-{epoch}.npz"
        np.savez(path, **out)
        meta = info["snapshots"][str(epoch)]
        result["snapshots"][str(epoch)] = {
            "methods": methods,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "g1_manifest_sha256": meta["frozen_sha256"],
            "frozen_sha256": file_digest(path),
            **notes,
        }
        print("Frozen G4", epoch, flush=True)

    old = snapshot(100, info)
    persist(100, old, ["Current"], history_available=False)
    result["feature_cache_sha256"] = file_digest(feature_path)
    out = snapshot(150, info)
    cache = np.load(feature_path, mmap_mode="r")
    assert cache.shape == (3, 3, len(out["ids"]), 768)
    np.testing.assert_array_equal(out["ids"], np.load(g1.WORK / "ids.npy"))
    methods, slots, ledger = temporal(old, out, cache, receipt)
    persist(
        150,
        out,
        methods,
        history_available=True,
        history_epoch=100,
        slot_identity=slots,
        relationship_history=ledger,
    )
    write_json(LOCAL / "frozen-provenance.json", json_native(result))

    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    for epoch, row in result["snapshots"].items():
        path = LOCAL / f"frozen-{epoch}.npz"
        assert file_digest(path) == row["frozen_sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            row["metrics"], row["bank_alignment"] = sealed(
                dict(arrays), truth, row["methods"]
            )
    result["seconds"] = time.monotonic() - start
    report(result)
    write_json(OUTPUT, json_native(result))


if __name__ == "__main__":
    main()
