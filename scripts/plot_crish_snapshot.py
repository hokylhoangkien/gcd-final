"""Render snapshot tables and figures from the diagnostic JSON (no GPU required)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    args = p.parse_args()
    result = json.loads(args.snapshot.read_text())
    gradients = result["gradients"]
    terms = list(gradients["weighted_loss_norms"])
    rows = gradients["rows"]
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    spaces = list(result["geometry"])
    x = np.arange(len(spaces))
    for j, group in enumerate(("old", "novel")):
        ax.bar(x + (j - .5) * .35, [result["geometry"][s]["subgroups"][group]["purity5"] for s in spaces],
               width=.35, label=group)
    ax.set(xticks=x, xticklabels=spaces, ylabel="Same-class fraction among five nearest neighbors")
    ax.legend()
    vals, groups = [], []
    print("Group, kind, N, median norm, median ratio, median cosine per loss, number negative")
    for space in ("block11", "classifier", "features"):
        print(space, terms)
        for kind in ("abstain", "move"):
            for group in ("old", "novel"):
                cells = [r["gradients"][space] for r in rows if r["kind"] == kind and r["group"] == group]
                if not cells:
                    continue
                cosine = [[c["cosine"][t] for c in cells if c["cosine"][t] is not None] for t in terms]
                medians = [np.median(v) if v else np.nan for v in cosine]
                print(group, kind, len(cells), "%.6g" % np.median([c["norm"] for c in cells]),
                      "%.6g" % np.median([c["ratio"] for c in cells]), np.round(medians, 4),
                      [sum(vv < 0 for vv in v) if v else None for v in cosine])
                if space == "block11":
                    vals.append(medians)
                    groups.append(f"{group} {kind}")
    hm = heat.imshow(vals, vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
    heat.set(xticks=np.arange(len(terms)), xticklabels=terms, yticks=np.arange(len(groups)), yticklabels=groups,
             title="Block 11 · median gradient cosine")
    for i, row in enumerate(vals):
        for j, value in enumerate(row):
            heat.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(hm, ax=heat, shrink=.65)
    feasible = sum(r["mass_supported"] for r in rows if r["kind"] == "move")
    n = sum(r["kind"] == "move" for r in rows)
    fig.suptitle(f"{result['state']} · Cars fixed panel · MOVE mass supported on {feasible}/{n} sampled rows")
    fig.savefig(args.snapshot.with_suffix(".png"), dpi=145)


if __name__ == "__main__":
    main()
