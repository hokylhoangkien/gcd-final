"""Summarize the fixed pre-training rule from a completed label-blind receipt."""

import argparse
import json
from pathlib import Path
from statistics import median


def summarize(receipt):
    batches = receipt["batches"]
    if receipt["completed_batches"] != 4 or [b["batch"] for b in batches] != [
        1,
        2,
        3,
        4,
    ]:
        raise ValueError("Expected four completed paired batches")
    groups = {}
    for batch in batches:
        for head, snapshot in batch["snapshots"].items():
            for r in snapshot["settings"]:
                groups.setdefault((r["lambda"], r["delta"]), {}).setdefault(
                    head, []
                ).append(r)
    reference = groups[0.1, 0.25]
    results = []
    for (lam, delta), heads in groups.items():
        by_head = {}
        for head, rows in heads.items():
            if len(rows) != 4:
                raise ValueError("Incomplete grid cell")
            displacement = median(r["pairwise_cos_rms"] for r in rows)
            by_head[head] = {
                "r1_pct": 100 * median(r["r1"]["ratio"] for r in rows),
                "r2_pct": 100 * median(r["r2"]["ratio"] for r in rows),
                "edit_pct": 100 * median(r["edit"]["ratio"] for r in rows),
                "pairwise_rms": displacement,
                "rms_multiple": displacement
                / median(r["pairwise_cos_rms"] for r in reference[head]),
            }
        all_rows = [r for group in heads.values() for r in group]
        worst_total = 100 * max(r["r2"]["ratio"] for r in all_rows)
        min_variance_change = 100 * min(
            r["panel_r2_health"]["variance"] / r["panel_keep_health"]["variance"] - 1
            for r in all_rows
        )
        passes = (
            all(
                m["edit_pct"] >= 0.10 and m["rms_multiple"] >= 3
                for m in by_head.values()
            )
            and worst_total < 5
            and min_variance_change > -1
        )
        results.append(
            {
                "coefficient": lam,
                "delta": delta,
                "heads": by_head,
                "passes": passes,
                "peak_r2_pct": worst_total,
                "min_panel_variance_change_pct": min_variance_change,
                "median_effective_mass": median(r["effective_mass"] for r in all_rows),
            }
        )
    if len(results) != 10 or any(
        set(r["heads"]) != {"fresh", "learned_head"} for r in results
    ):
        raise ValueError("Incomplete sensitivity grid")
    accepted = sorted(
        (r for r in results if r["passes"]),
        key=lambda r: (r["delta"], r["coefficient"]),
    )
    return {
        "grid": results,
        "selected": (
            [accepted[0]["coefficient"], accepted[0]["delta"]] if accepted else None
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    print(
        json.dumps(summarize(json.loads(p.parse_args().receipt.read_text())), indent=2)
    )
