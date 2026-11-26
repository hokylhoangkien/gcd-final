"""Insert measured tables from completed offline receipts into the audit report."""

import argparse
import json
from pathlib import Path
from statistics import mean


def tables(root):
    def load(name):
        return json.loads((root / f"r2_audit_{name}.json").read_text())

    h, e, v, ex = map(load, ["history", "evidence", "virtual", "examples"])

    def pct(obj):
        return f"{100 * obj['mean']:.1f} [{100 * obj['ci95'][0]:.1f}, {100 * obj['ci95'][1]:.1f}]"

    text = []

    def section(title, head):
        text.extend(
            [
                "",
                "### " + title,
                "",
                "| " + " | ".join(head) + " |",
                "|" + "---|" * len(head),
            ]
        )

    def row(*values):
        text.append("| " + " | ".join(map(str, values)) + " |")

    section(
        "Actual training graph and realized exposure",
        [
            "Epoch",
            "Action (votes)",
            "Same / edges",
            "Edge purity %, 95% CI",
            "Sampled occurrences",
            "Exposure-weighted purity %, 95% CI",
        ],
    )
    for epoch in ("100", "150"):
        for action, s in h["historical_semantics"][epoch]["all"].items():
            a, b = s["edge_purity"], s["exposure_purity"]
            row(
                epoch,
                action,
                f"{s['same_edges']} / {a['edges']}",
                pct(a),
                b["mass"],
                pct(b),
            )
    section(
        "Reconstructed graph at each model state",
        [
            "Snapshot",
            "Vote / action",
            "Same / edges",
            "Purity %, 95% CI",
            "Reweighted anchor encounters",
            "Reweighted purity %, 95% CI",
        ],
    )
    for name, r in e.items():
        for action, s in r["semantic"]["all"].items():
            a, b = s["edge_purity"], s["exposure_purity"]
            row(
                name,
                action,
                f"{s['same_edges']} / {a['edges']}",
                pct(a),
                b["mass"],
                pct(b),
            )
    text.append(
        "\nThe three reweightings use anchor draws at epochs 11, 101 and 150, respectively. The last uses a preceding epoch’s anchors because epoch 151 was not run. Vote 0 = WEAKEN, 1 = KEEP, 2 = STRENGTHEN. All mixed edges have zero same-class probability by definition."
    )
    section(
        "Conditional endpoint composition, same / total edges",
        [
            "Snapshot",
            "Endpoints",
            "WEAKEN",
            "KEEP",
            "STRENGTHEN",
            "Overall exposure-weighted purity %, 95% CI",
        ],
    )
    for name in ("epoch100", "epoch150"):
        for group in ("old_old", "novel_novel", "mixed"):
            sub = e[name]["semantic"][group]
            row(
                name,
                group,
                *[
                    f"{sub[k]['same_edges']} / {sub[k]['edge_purity']['edges']}"
                    for k in ("weaken", "keep", "strengthen")
                ],
                pct(sub["overall"]["exposure_purity"]),
            )
    section(
        "Vote versus cosine discrimination",
        [
            "Snapshot",
            "AUC cosine / votes / both",
            "Brier cosine / votes / both",
            "Paired Brier gain, 95% dyadic CI",
            "Center-support Jaccard / vote agreement on shared edges vs archived refresh",
        ],
    )
    for name, r in e.items():
        p = r["prediction"]
        gain = p["paired_brier_gain"]
        row(
            name,
            *[
                " / ".join(
                    f"{p['models'][m][key]:.4f}" for m in ("cosine", "vote", "both")
                )
                for key in ("auc", "brier")
            ],
            f"{gain['mean']:+.5f} [{gain['ci95'][0]:+.5f}, {gain['ci95'][1]:+.5f}]",
            f"{r['center_jaccard']:.3f} / {r['vote_agreement_common']:.3f} (epoch {r['compared_ledger_epoch']})",
        )
    section(
        "Full-split semantic geometry",
        [
            "Snapshot",
            "Anchors",
            "n",
            "kNN-5 purity",
            "Own LOO centroid cosine",
            "Nearest other centroid distance",
            "Class margin",
            "Silhouette",
        ],
    )
    for name, r in e.items():
        for group in ("all", "old", "novel"):
            g = r["geometry"][group]
            row(
                name,
                group,
                g["n"],
                *[
                    f"{g[k]:.4f}"
                    for k in (
                        "purity5",
                        "own_loo_cos",
                        "nearest_other_centroid_distance",
                        "semantic_margin",
                        "silhouette",
                    )
                ],
            )
    section(
        "Fragmentation of the reciprocal center graph",
        [
            "Snapshot",
            "Edges",
            "Isolated nodes",
            "Connected components (including isolates)",
            "Largest component",
            "Feature variance",
            "Mean resultant",
        ],
    )
    for name, r in e.items():
        g = r["graph"]
        row(
            name,
            g["edges"],
            g["isolated"],
            r["components"],
            r["largest_component"],
            f"{g['feature_variance']:.7f}",
            f"{g['mean_resultant_length']:.4f}",
        )
    text.append(
        f"\nA separate recomputation on the exact 980 IDs in the archive reproduced the epoch-150 geometry report with maximum absolute metric difference {e['epoch150']['archived_panel_max_abs_error']:.2g}. The full-split table uses 6,144 anchors, with the same full candidate set for Old and Novel queries."
    )
    section(
        "Checkpoint-conditioned virtual steps on independent probe images",
        [
            "Checkpoint / next LR",
            "Arm",
            "Mean + / − encounters",
            "Δ purity@5 vs B (pp)",
            "Δ semantic margin vs B ×10⁵",
            "Δ margin vs pre-step ×10⁵",
            "Unit CLS RMS vs B ×10⁵",
            "Pairwise cosine RMS vs B ×10⁵",
            "Edit cosine vs SimGCD / oracle",
        ],
    )
    for ep, r in v.items():
        for a in "ABCDE":
            values = [b["arms"][a] for b in r["batches"]]

            def avg(k, values=values):
                return mean(z[k] for z in values)

            def diff(k, ref, a=a, r=r):
                return mean(
                    b["arms"][a][k]
                    - (b["arms"]["B"] if ref == "B" else b["reference"])[k]
                    for b in r["batches"]
                )

            row(
                f"{ep} / {r['lr']:.5f}",
                a,
                f"{avg('positive_occurrences'):.1f} / {avg('negative_occurrences'):.1f}",
                f"{100 * diff('purity5', 'B'):+.4f}",
                f"{1e5 * diff('margin', 'B'):+.3f}",
                f"{1e5 * diff('margin', 'pre'):+.3f}",
                f"{1e5 * avg('probe_cls_rms_vs_B'):.3f}",
                f"{1e5 * avg('probe_pair_rms_vs_B'):.3f}",
                f"{avg('edit_cos_base'):+.3f} / {avg('edit_cos_oracle'):+.3f}"
                if a in "CDE"
                else "—",
            )
    section(
        "Ledger-backed failure cases",
        [
            "Selection",
            "Stable IDs",
            "Cars classes",
            "Historical W / K / S refreshes",
            "Actual edited encounters",
        ],
    )
    for item in ex:
        votes = item["committed_votes"]
        row(
            item["reason"],
            " / ".join(map(str, item["ids"])),
            " ↔ ".join(item["class_names"]),
            " / ".join(str(votes.count(s)) for s in (0, 1, 2)),
            item["actual_edited_occurrences"],
        )
    return "\n".join(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path)
    root = p.parse_args().reports
    path = root / "CRISH_RELATIONAL_MECHANISTIC_AUDIT.md"
    start, end = "<!-- measured tables -->", "<!-- end measured tables -->"
    s = path.read_text(encoding="utf8")
    left = s.split(start)[0]
    right = s.split(end)[-1] if end in s else ""
    path.write_text(
        left.rstrip()
        + "\n\n"
        + start
        + "\n"
        + tables(root)
        + "\n"
        + end
        + "\n"
        + right.lstrip(),
        encoding="utf8",
    )
