"""Render descriptive audit figures and identify ledger-backed errors without exporting images."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from data.stanford_cars import CarsDataset
from relational.state import write_json


def run(work, out):
    def load(name):
        return json.loads((out / f"r2_audit_{name}.json").read_text())

    h, e, v = load("history"), load("evidence"), load("virtual")
    if list(e) != ["DINO", "epoch100", "epoch150"] or set(v) != {"100", "150"}:
        raise ValueError("Incomplete endpoint receipts")
    for endpoint in v.values():
        assert len(endpoint["batches"]) == 3
        for batch in endpoint["batches"]:
            arms = batch["arms"]
            assert set(arms) == set("ABCDE") and len(set(batch["probe_ids"])) == 384
            assert not set(batch["batch_ids"]).intersection(batch["probe_ids"])
            assert arms["B"]["probe_pair_rms_vs_B"] == 0
            for key in ("positive_occurrences", "negative_occurrences"):
                assert arms["C"][key] == arms["D"][key]
    epochs = h["epochs"]
    x = np.arange(1, 151)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for key in ("all", "old", "new"):
        axes[0].plot(x, [100 * r[key] for r in epochs], label=key.title())
    axes[0].plot(
        x,
        [100 * r["e3"][0] for r in epochs],
        "--",
        color="grey",
        label="E3 All (context)",
    )
    axes[0].set_ylabel("ACC (%)")
    for term in ("relational", "edit_delta"):
        axes[1].plot(
            x[10:],
            [100 * r["gradient_first_batch"][term]["ratio"] for r in epochs[10:]],
            label=term,
        )
    axes[1].set_ylabel("Block-11 norm / SimGCD (%)")
    for i, label in enumerate(("WEAKEN", "KEEP", "STRENGTHEN")):
        axes[2].plot(
            x[10:],
            [
                100 * r["truth_by_vote"][i][1] / r["truth_by_vote"][i][0]
                for r in epochs[10:]
            ],
            label=label,
        )
    axes[2].set_ylabel("Same-class edges (%)")
    axes[2].set_xlabel("Completed epoch")
    for ax in axes:
        ax.legend(ncol=4)
        ax.grid(alpha=0.2)
    fig.savefig(out / "r2_audit_trajectory.png", dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for j, (key, r) in enumerate(e.items()):
        rows = [
            r["semantic"]["all"][key]["edge_purity"]
            for key in ("weaken", "keep", "strengthen")
        ]
        means = np.array([r["mean"] for r in rows])
        ci = np.array([r["ci95"] for r in rows]).T
        axes[0].errorbar(
            np.arange(3) + (j - 1) * 0.07,
            100 * means,
            yerr=100 * abs(ci - means),
            fmt="o",
            capsize=3,
            label=key,
        )
    axes[0].set(
        xticks=range(3),
        xticklabels=["WEAKEN", "KEEP", "STRENGTHEN"],
        ylabel="Same-class edges (%)",
    )
    axes[0].legend()
    for i, (key, item) in enumerate(v.items()):
        for n, b in enumerate(item["batches"]):
            vals = [
                1e5 * (b["arms"][a]["margin"] - b["arms"]["B"]["margin"])
                for a in "ABCDE"
            ]
            axes[1].plot(
                np.arange(5) + 0.025 * (n - 1),
                vals,
                "o-",
                alpha=0.7,
                color=("tab:blue", "tab:orange")[i],
                label=f"Epoch {key}" if n == 0 else None,
            )
    axes[1].set(
        xticks=range(5),
        xticklabels=list("ABCDE"),
        ylabel="Probe cosine margin change vs KEEP (×10⁻⁵)",
    )
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].legend()
    fig.savefig(out / "r2_audit_evidence_cf.png", dpi=155)
    plt.close(fig)
    dataset = CarsDataset()
    names = loadmat(
        Path(os.environ["SIMGCD_CARS_ROOT"]) / "devkit/cars_meta.mat", squeeze_me=True
    )["class_names"]
    examples = {}
    for reason, pairs in (
        ("longest wrong STRENGTHEN", h["failure_examples"][:2]),
        *e["epoch150"]["examples"].items(),
    ):
        for item in pairs[:2]:
            pair = tuple(item["ids"])
            examples[pair] = {
                **item,
                "reason": reason,
                "class_names": [str(names[dataset.target[k] - 1]) for k in pair],
                "committed_votes": [],
                "actual_edited_occurrences": 0,
            }
    with (work / "input/relationships.jsonl").open() as ledger:
        for line in ledger:
            r = json.loads(line)
            if r["graph"] is None:
                continue
            votes = {
                tuple(p[:2]): p[2]
                for p in r["graph"]["edges"]
                if tuple(p[:2]) in examples
            }
            times = {
                tuple(p[:2]): p[2]
                for p in r["applied"]["applied_edges"]
                if tuple(p[:2]) in examples
            }
            for pair, item in examples.items():
                item["committed_votes"].append(votes.get(pair, -1))
                item["actual_edited_occurrences"] += times.get(pair, 0)
    write_json(out / "r2_audit_examples.json", list(examples.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    run(args.work, args.out)
