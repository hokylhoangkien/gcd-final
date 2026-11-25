"""Offline audit of all refresh/exposure events and stdout; accepts E3 or E3-F."""
import argparse
import hashlib
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr


def distribution(xs):
    if not len(xs):
        return None
    return dict(zip(("min", "q25", "median", "q75", "max"), np.quantile(xs, [0, .25, .5, .75, 1])))


def summarize(refreshes, exposures, labels):
    ds = [d for r in refreshes for d in r["decisions"]]
    tested = [d for d in ds if "confirmation" in d]
    accepts = [d for d in tested if d["accepted"]]
    acts = [a for r in refreshes for a in r["accepted"]]
    seen = [key for x in exposures for key in x["applied_ids"]]
    encounters = sum((Counter(x["exposure"]) for x in exposures), Counter())
    applied = sum((Counter(x["applied"]) for x in exposures), Counter())
    groups = Counter()
    old_moves = Counter()
    anchored = []
    for r, x in zip(refreshes, exposures):
        ids = set(x["applied_ids"])
        winners = {(d["id"], d["kind"]): d for d in r["decisions"] if d["accepted"]}
        for a in r["accepted"]:
            key = a["sample_id"]
            group = "old" if labels[key] < 98 else "novel"
            groups[(a["kind"], group, "accepted")] += 1
            groups[(a["kind"], group, "applied_id_epoch")] += int(key in ids)
            if group == "old" and a["kind"] == "move":
                origin = a["source"] == labels[key]
                dest = a["target"] == labels[key]
                direction = "toward_legal_slot" if dest else "away_from_legal_slot" if origin else "neither"
                old_moves[direction] += 1
                trace = winners[key, "move"]
                anchored.append(dict(id=key, epoch=r["epoch"], direction=direction, applied_id=key in ids,
                                     gain=trace["gain"], confirmation=trace["confirmation"]))
    disc = [d["gain"] for d in tested]
    conf = [d["confirmation"] for d in tested]
    rho = spearmanr(disc, conf).statistic if len(set(disc)) > 1 and len(set(conf)) > 1 else None
    attempts = [a for r in refreshes for a in r.get("confirmation_attempts", [])]
    signs = [a["sign_audit"] for a in attempts if "p_bonferroni" in a.get("sign_audit", {})]
    return dict(refreshes=len(refreshes), stop=Counter(r["stop"] for r in refreshes),
                runtime_seconds=distribution([r["runtime_seconds"] for r in refreshes]),
                decisions=len(ds), positive_discovery=sum(d.get("gain", 0) > 0 for d in ds),
                zero_discovery=sum(d.get("gain") == 0 for d in ds),
                last_accept_epoch=max((r["epoch"] for r in refreshes if r["accepted"]), default=None),
                last_applied_epoch=max((r["epoch"] for r in exposures if r["applied_ids"]), default=None),
                discovery_all=distribution([d["gain"] for d in ds if "gain" in d]),
                confirmed_attempts=len(tested), accepted=len(acts), accepted_kinds=Counter(a["kind"] for a in acts),
                applied=applied, exposure=encounters, unique_accepted_ids=len({a["sample_id"] for a in acts}),
                unique_applied_ids=len(set(seen)), applied_id_epochs=len(seen),
                groups={"/".join(k): v for k, v in groups.items()}, old_moves=old_moves, old_move_rows=anchored,
                candidates_round0=Counter("old" if labels[key] < 98 else "novel" for r in refreshes
                                          for key in {d["id"] for d in r["decisions"] if d["round"] == 0}),
                tested_discovery=distribution(disc), tested_confirmation=distribution(conf),
                confirmation_spearman=rho, accepted_confirmation=distribution([d["confirmation"] for d in accepts]),
                accepted_known_delta=distribution([d["confirmation_known_delta"] for d in accepts]),
                accepted_unlabeled_gain=distribution([d["confirmation"] + d["confirmation_known_delta"] for d in accepts]),
                accepted_unlabeled_worse=sum(d["confirmation"] + d["confirmation_known_delta"] < 0 for d in accepts),
                accepted_unlabeled_equal=sum(d["confirmation"] + d["confirmation_known_delta"] == 0 for d in accepts),
                accepted_unlabeled_below_minus_1e6th=sum(d["confirmation"] + d["confirmation_known_delta"] < -1e-6 for d in accepts),
                accepted_unlabeled_above_1e6th=sum(d["confirmation"] + d["confirmation_known_delta"] > 1e-6 for d in accepts),
                rejected_nonpositive=sum(not d["accepted"] and min(d["confirmation"], d["group_confirmation"]) <= 0 for d in tested),
                rejected_positive_guard=sum(not d["accepted"] and min(d["confirmation"], d["group_confirmation"]) > 0 for d in tested),
                fallback_rank=Counter(str(a["rank"]) for a in attempts),
                accepted_rank=Counter(str(a["rank"]) for a in attempts if a["result"] == "accepted"),
                attempt_status=Counter(a["result"] for a in attempts),
                sign_diagnostics=len(signs), sign_adjusted_le_005=sum(a["p_bonferroni"] <= .05 for a in signs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("ledger", "stdout", "cars", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    recs = [json.loads(line) for line in args.ledger.read_text().splitlines()]
    rs = [r for r in recs if r["event"] == "refresh"]
    xs = [r for r in recs if r["event"] == "exposure"]
    epochs = [r["epoch"] for r in rs]
    assert epochs == sorted(set(epochs)) == [x["epoch"] for x in xs]
    for r, x in zip(rs, xs):
        assert len(r["accepted"]) == x["accepted"] and set(x["applied_ids"]) <= {a["sample_id"] for a in r["accepted"]}
        probe = [i for fold in r["probe_ids"] for i in fold]
        assert len(probe) == len(set(probe)) and not set(probe).intersection(r["score_ids"] + r["evidence_ids"])
    split = Path(__file__).resolve().parents[1] / "data/ssb_splits/scars_osr_splits.pkl"
    with split.open("rb") as handle:
        classes = pickle.load(handle)
    new = classes["unknown_classes"]
    order = list(classes["known_classes"]) + new["Hard"] + new["Medium"] + new["Easy"]
    labels = [order.index(int(row[4].item()) - 1) for row in loadmat(args.cars / "devkit/cars_train_annos.mat")["annotations"][0]]
    series = re.findall(r"Train Accuracies: All ([\d.]+) \| Old ([\d.]+) \| New ([\d.]+)", args.stdout.read_text(encoding="utf8", errors="replace"))
    acc = np.array(series, dtype=float)
    assert len(acc) and max(epochs) < len(acc)
    result = dict(ledger_sha256=hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
                  stdout_sha256=hashlib.sha256(args.stdout.read_bytes()).hexdigest(),
                  accuracy_epochs=len(acc), final_accuracy=acc[-1].tolist(), all=summarize(rs, xs, labels),
                  phases={})
    for lo, hi in [(10, 49), (50, 99), (100, 149), (150, 199)]:
        inds = [i for i, e in enumerate(epochs) if lo <= e <= hi]
        if inds:
            result["phases"][f"{lo}-{hi}"] = summarize([rs[i] for i in inds], [xs[i] for i in inds], labels)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.8), constrained_layout=True)
    for curve, label in zip(acc.T, ("All", "Old", "New")):
        axes[0].plot(100 * curve, label=label)
    axes[0].set(ylabel="Transductive accuracy (%)", xlabel="Epoch (zero based)")
    axes[0].legend(ncol=3)
    axes[1].plot(epochs, [len(r["accepted"]) for r in rs], label="Accepted actions", lw=1)
    axes[1].plot(epochs, [sum(x["applied"].values()) for x in xs], label="Applied occurrences", lw=1)
    axes[1].set(ylabel="Count per refresh", xlabel="Epoch")
    axes[1].legend()
    for flag, name in [(False, "Veto"), (True, "Accepted")]:
        points = [(d["gain"], d["confirmation"]) for r in rs for d in r["decisions"]
                  if "confirmation" in d and d["accepted"] == flag]
        if points:
            x, y = np.array(points).T * 1e6
            axes[2].scatter(x, y, s=8, alpha=.4, label=name)
    axes[2].axhline(0, color="gray", lw=.8)
    axes[2].set(xlabel="Discovery marginal gain (×10⁻⁶)", ylabel="Fresh-fold gain (×10⁻⁶)")
    axes[2].legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=145)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
