"""G3 compares cached pretrained and R2 CLS on the frozen G1 Cars queries."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from crish.experiment import file_digest
from relational.state import write_json
from scripts import audit_g1 as g1
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_g7 import pack

LOCAL = g1.WORK / "g3"
G1_REPORT = Path("reports/r2_multilayer_geometry.json")
REPORT = Path("reports/CRISH_PRETRAINED_GEOMETRY.md")
OUTPUT = Path("reports/r2_pretrained_geometry.json")
METHODS = ("FT", "Pretrained", "Equal", "DisagreementRisk", "PretrainedRisk")
COVERAGES = (10, 20, 40)


def score_records(pre, ft, base):
    ids, q, bank, slots = (base[k] for k in ("ids", "q", "bank", "slots"))
    inv = np.full(len(ids), -1)
    inv[bank] = np.arange(len(bank))
    pairs = inv[base["pairs"]]
    assert (pairs >= 0).all()
    a = pre[q] @ pre[bank].T
    b = ft[q] @ ft[bank].T
    records = {"ids": ids, "q": q, "bank": bank, "slots": slots, "pairs": base["pairs"]}
    outputs = {}
    for name, values in (("FT", b), ("Pretrained", a), ("Equal", (a + b) / 2)):
        fields = pack(values, q, bank, slots, pairs)
        for k in ("neighbors", "dest", "wrong", "pair"):
            records[f"{name}_{k}"] = fields[k]
        outputs[name] = fields
    for k in ("neighbors", "dest", "wrong", "pair"):
        check = (
            np.testing.assert_array_equal
            if k in ("neighbors", "dest")
            else np.testing.assert_allclose
        )
        if k in ("neighbors", "dest"):
            check(records[f"FT_{k}"], base[f"B11_{k}"])
        else:
            check(records[f"FT_{k}"], base[f"B11_{k}"], atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(outputs["FT"]["specific_ids"], base["pairs"][:, 256:])
    overlap = (
        outputs["FT"]["neighbors"][:, :, None]
        == outputs["Pretrained"]["neighbors"][:, None, :]
    ).sum((1, 2))
    disagreement = 1 - overlap / (20 - overlap)
    records["neighbor_overlap"] = overlap
    for method, risk in (
        ("DisagreementRisk", disagreement),
        ("PretrainedRisk", records["Pretrained_wrong"]),
    ):
        for k in ("neighbors", "dest", "pair"):
            records[f"{method}_{k}"] = records[f"FT_{k}"]
        records[f"{method}_wrong"] = risk
    for method in METHODS:
        risk = records[f"{method}_wrong"]
        assert np.isfinite(risk).all()
        # Query ID is the label-blind tie breaker.
        order = np.lexsort((ids[q], -risk))
        for percent in COVERAGES:
            records[f"{method}_flag_{percent}"] = order[: len(q) * percent // 100]
    return records


def sealed(records, truth):
    ids, q, bank, slots = (records[k] for k in ("ids", "q", "bank", "slots"))
    metrics, alignment = g1.evaluate(
        records, ids, q, bank, slots, truth, methods=METHODS, baseline="FT"
    )
    mp, _ = align(
        ids[bank].tolist(), dict(zip(ids.tolist(), slots.tolist(), strict=True)), truth
    )
    table = np.array([mp.get(k, -1) for k in range(196)] + [-1])
    y = truth[ids[q]]
    correct = table[slots[q]] == y
    ft = records["FT_dest"][:, 0]
    overlap = {}
    for group, group_mask in (("All", y >= 0), ("Old", y < 98), ("New", y >= 98)):
        intersection = records["neighbor_overlap"][group_mask]
        overlap[group] = {
            "mean_shared_of_10": g1.rate(intersection),
            "mean_jaccard": g1.rate(intersection / (20 - intersection)),
        }
        errors = group_mask & ~correct
        for method in METHODS:
            top = records[f"{method}_dest"][:, 0]
            disagree = errors & (top != ft)
            row = metrics[method][group]
            row["destination_disagreement_n"] = int(disagree.sum())
            row["method_accuracy_on_disagreement"] = g1.rate(
                table[top[disagree]] == y[disagree]
            )
            row["ft_accuracy_on_disagreement"] = g1.rate(
                table[ft[disagree]] == y[disagree]
            )
            row["flags"] = {}
            for percent in COVERAGES:
                idx = records[f"{method}_flag_{percent}"]
                selected = idx[group_mask[idx]]
                tp = int((~correct[selected]).sum())
                error_count = int(errors.sum())
                row["flags"][percent] = {
                    "selected": len(selected),
                    "true_positives": tp,
                    "errors": error_count,
                    "coverage": len(selected) / int(group_mask.sum())
                    if group_mask.any()
                    else None,
                    "precision": tp / len(selected) if len(selected) else None,
                    "recall": tp / error_count if error_count else None,
                }
    return metrics, alignment, overlap


def report(result):
    lines = [
        "# G3 frozen pretrained geometry on Stanford Cars",
        "",
        (
            "This exploratory comparison reuses G1 queries, retrieval banks, pair pools and "
            "classifier slots. The verified R2 snapshot audit cached DINOv1 center-crop CLS "
            "before fine-tuning and after epochs 100 and 150. G3 records the input hashes "
            "and reproduces the G1 fine-tuned baseline before scoring."
        ),
        "",
        (
            "Pretrained and FT use cosine. Equal averages their scores with fixed weights. "
            "DisagreementRisk uses one minus Jaccard overlap of the two top-10 neighborhoods. "
            "PretrainedRisk uses the pretrained alternative-minus-current slot support margin. "
            "Both risk-only rows retrieve destinations and neighbors using FT. No calibration "
            "is fitted. Detection risk is separate from destination ranking."
        ),
        "",
        (
            "All rules use the same G1 hard negatives. The bank alone supplies Hungarian alignment. "
            "Recovery and regression are measured relative to FT on wrong-source queries. "
            "Fixed-coverage flags take the top 10%, 20% or 40% of all queries by risk (rounded down), "
            "with ascending sample ID for ties. Old/New precision and recall condition on query class "
            "within those same flags. The table gives 20% precision/recall and realized group coverage; "
            "JSON includes all levels and counts."
        ),
        "",
        "| Epoch / group / method | Purity@1/5/10 | Pair / hard AUROC | Error AUROC | Dest R@1/3/5 | Recovery / regression | P / R at 20% | Group coverage |",
        "|---|---|---|---:|---|---|---|---:|",
    ]

    def f(x):
        return "—" if x is None else f"{x:.3f}"

    for epoch, snapshot in result["snapshots"].items():
        for group in ("All", "Old", "New"):
            for method in METHODS:
                m = snapshot["metrics"][method][group]
                purity = "/".join(f(m["purity"][k]) for k in (1, 5, 10))
                dest = "/".join(f(m["destination_recall"][k]) for k in (1, 3, 5))
                d = m["flags"][20]
                lines.append(
                    f"| {epoch} / {group} / {method} | {purity} | {f(m['pair_auc'])} / "
                    f"{f(m['hard_auc'])} | {f(m['detection_auc'])} | {dest} | "
                    f"{f(m['failure_recovery'][1])} / {f(m['correct_to_wrong_vs_final'])} | "
                    f"{f(d['precision'])} / {f(d['recall'])} | {f(d['coverage'])} |"
                )
    lines += [
        "",
        "## Neighborhood overlap",
        "",
        "| Epoch / group | Mean shared neighbors out of 10 | Mean Jaccard |",
        "|---|---:|---:|",
    ]
    for epoch, snapshot in result["snapshots"].items():
        for group, m in snapshot["overlap"].items():
            lines.append(
                f"| {epoch} / {group} | {f(m['mean_shared_of_10'])} | {f(m['mean_jaccard'])} |"
            )
    lines += [
        "",
        f"Runtime {result['seconds']:.1f} seconds.",
        "",
        "[Coverage counts, disagreement analysis and prediction provenance](r2_pretrained_geometry.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    begin = time.monotonic()
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(exist_ok=True)
    manifest = json.loads(G1_REPORT.read_text())
    path = g1.WORK / "features.npy"
    source = np.load(path, mmap_mode="r")
    ids = np.load(g1.WORK / "ids.npy")
    assert source.shape == (3, 3, len(ids), 768) and source.dtype == np.float32
    provenance = g1.WORK / "provenance.json"
    original = json.loads(provenance.read_text())
    checks = {name: sha for sha, name in (line.split() for line in original["members"])}
    result = {
        "base": "93b40eb",
        "exploratory": True,
        "coverage_percent": COVERAGES,
        "g1_report_sha256": file_digest(G1_REPORT),
        "feature_cache_sha256": file_digest(path),
        "ids_sha256": file_digest(g1.WORK / "ids.npy"),
        "snapshot_provenance_sha256": file_digest(provenance),
        "dino_weights_sha256": original["config"]["protocol"]["dino_sha256"],
        "snapshots": {},
    }
    pre = F.normalize(torch.from_numpy(source[0, 0].copy()), dim=1).cuda()
    assert torch.isfinite(pre).all()
    for i, epoch in enumerate(g1.EPOCHS, 1):
        key = str(epoch)
        file = g1.LOCAL / f"frozen-{key}.npz"
        m = manifest["snapshots"][key]
        assert file_digest(file) == m["frozen_sha256"]
        assert checks[f"epoch-{epoch}.pt"] == m["checkpoint_sha256"]
        ft = F.normalize(torch.from_numpy(source[i, 0].copy()), dim=1).cuda()
        with np.load(file, allow_pickle=False) as base:
            np.testing.assert_array_equal(base["ids"], ids)
            assert ids[base["q"]].tolist() == manifest["query_ids"]
            assert not np.intersect1d(base["q"], base["bank"]).size
            records = score_records(pre, ft, base)
        out = LOCAL / f"frozen-{epoch}.npz"
        np.savez(out, **records)
        result["snapshots"][key] = {
            "checkpoint_sha256": m["checkpoint_sha256"],
            "g1_manifest_sha256": m["frozen_sha256"],
            "frozen_sha256": file_digest(out),
        }
        print("Frozen G3", epoch, flush=True)
    write_json(LOCAL / "frozen-provenance.json", json_native(result))
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    for key, snapshot in result["snapshots"].items():
        file = LOCAL / f"frozen-{key}.npz"
        assert file_digest(file) == snapshot["frozen_sha256"]
        with np.load(file, allow_pickle=False) as frozen:
            metrics, alignment, overlap = sealed(dict(frozen), truth)
        snapshot.update(metrics=metrics, alignment=alignment, overlap=overlap)
    result["seconds"] = time.monotonic() - begin
    report(result)
    write_json(OUTPUT, json_native(result))


if __name__ == "__main__":
    main()
