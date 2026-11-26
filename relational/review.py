"""Descriptive standalone milestone report, built only from completed epoch journals."""

import json
import math
from statistics import mean

from relational.guard import evaluate


def records(path, end):
    with path.open(encoding="utf8") as stream:
        for epoch in range(1, end + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"{path.name}: missing epoch {epoch}")
            if epoch > end - 20:
                row = json.loads(line)
                if row["epoch"] != epoch:
                    raise ValueError(f"{path.name}: expected epoch {epoch}")
                yield row


def milestone(root, epoch):
    if epoch not in (100, 150):
        raise ValueError("Standalone review is defined at 100 and 150")
    folder = root / "R2"
    rows = list(records(folder / "metrics.jsonl", epoch))
    accuracy = {}
    for key in ("all", "old", "new"):
        values = [100 * float(row[key]) for row in rows]
        if not all(math.isfinite(v) and 0 <= v <= 100 for v in values):
            raise ValueError(f"Invalid {key} in R2 window")
        x = [i - 9.5 for i in range(20)]
        accuracy[key] = {
            "mean_pct": mean(values),
            "slope_pp_per_epoch": sum(t * v for t, v in zip(x, values))
            / sum(t * t for t in x),
            "first_pct": values[0],
            "last_pct": values[-1],
        }
    gradients = {}
    for term in ("relational", "edit_delta"):
        ratios = [r["gradient_first_batch"][term]["ratio"] for r in rows]
        valid = [v for v in ratios if v is not None and math.isfinite(v)]
        gradients[term] = {
            "mean_ratio": mean(valid) if valid else None,
            "measured_epochs": len(valid),
        }
    total = dict.fromkeys(
        (
            "rows",
            "graph_occurrences",
            "strengthen",
            "weaken",
            "intervention_mass",
            "effective_mass",
        ),
        0,
    )
    health = {
        k: []
        for k in (
            "edge_count",
            "isolated",
            "crossview_edge_jaccard",
            "feature_variance",
            "mean_resultant_length",
        )
    }
    anchors, edges = set(), set()
    for item in records(folder / "relationships.jsonl", epoch):
        graph, applied = item["graph"], item["applied"]
        for key, values in health.items():
            values.append(len(graph["edges"]) if key == "edge_count" else graph[key])
        for key in total:
            total[key] += applied[key]
        anchors.update(applied["edited_anchor_ids"])
        edges.update((a, b) for a, b, _ in applied["applied_edges"])
    total.update(unique_edited_anchors=len(anchors), unique_edited_edges=len(edges))
    comparative = evaluate(root, "R2", epoch, allow_missing_references=True)
    return {
        "epoch": epoch,
        "first_epoch": epoch - 19,
        "accuracy": accuracy,
        "gradient_first_batch": gradients,
        "exposure": total,
        "graph_window_means": {k: mean(v) for k, v in health.items()},
        "comparative": comparative,
        "historical_context_only": {
            "E3": {
                "all": 56.51,
                "old": 71.51,
                "new": 49.26,
                "source": "verified E3 run, final percentages",
            },
            "E3-F": {
                "all": 56.49,
                "old": 73.36,
                "new": 48.35,
                "source": "manually recovered terminal aggregates, final percentages",
            },
        },
    }
