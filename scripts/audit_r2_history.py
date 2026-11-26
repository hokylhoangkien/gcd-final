"""Replay weighted sampler counts and join committed ledgers with sealed Cars labels."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from crish.experiment import load_checkpoint
from relational.state import write_json
from relational.train import build
from scripts.audit_r2_evidence import purity, sealed_labels


def run(work, e3log, out):
    _, train, _, _, _, _ = build()
    merged = train.dataset
    ids = np.r_[merged.labelled_dataset.uq_idxs, merged.unlabelled_dataset.ids]
    rng = torch.Generator().set_state(torch.get_rng_state())
    saved_rng = load_checkpoint(work / "input/epoch-100.pt")["rng"][2]
    y = sealed_labels()
    hist, endpoint, transitions = [], {}, Counter()
    seen_wrong, streak, max_streak = Counter(), {}, {}
    old_edges, old_streak, old_exposed = {}, {}, set()
    draws = np.full((150, len(y)), -1, np.int16)
    e3 = re.findall(
        r"Train Accuracies: All ([.\d]+) \| Old ([.\d]+) \| New ([.\d]+)",
        e3log.read_text(encoding="utf8"),
    )
    assert len(e3) == 200
    with (
        (work / "input/metrics.jsonl").open() as fm,
        (work / "input/relationships.jsonl").open() as fl,
    ):
        for epoch, (m, l) in enumerate(zip(fm, fl, strict=True), 1):
            m, l = json.loads(m), json.loads(l)
            assert m["epoch"] == l["epoch"] == epoch
            if epoch == 101:
                matches_mid = torch.equal(rng.get_state(), saved_rng)
                rng.set_state(saved_rng)
            torch.empty((), dtype=torch.int64).random_(generator=rng)
            chosen = torch.multinomial(
                train.sampler.weights, len(ids), True, generator=rng
            )[: len(train) * 128].numpy()
            count = np.bincount(ids[chosen], minlength=len(y))
            torch.empty((), dtype=torch.int64).random_(
                generator=rng
            )  # evaluation iterator
            row = {**m, "e3": [float(v) for v in e3[epoch - 1]]}
            hist.append(row)
            if not l["graph"]:
                continue
            g, a = l["graph"], l["applied"]
            raw = np.array(g["edges"])
            uv, vote = raw[:, :2].astype(int), raw[:, 2].astype(int)
            assert np.array_equal(raw[:, 3], (vote - 1) * 0.25)
            keys = uv[:, 0] * len(y) + uv[:, 1]
            times = {u * len(y) + v: n for u, v, n in a["applied_edges"]}
            logged = np.array([times.get(k, 0) for k in keys])
            exp = count[uv].sum(1)
            verified = (
                np.array_equal(exp[vote != 1], logged[vote != 1])
                and exp.sum() == a["graph_occurrences"]
            )
            if verified:
                draws[epoch - 1] = count
            else:
                exp = np.where(vote == 1, -1, logged)
            same = y[uv[:, 0]] == y[uv[:, 1]]
            row.update(
                {
                    "graph": {k: v for k, v in g.items() if k != "edges"},
                    "exposure": {
                        k: v
                        for k, v in a.items()
                        if k not in ("applied_edges", "edited_anchor_ids")
                    },
                    "replay_verified": bool(verified),
                    "truth_by_vote": [
                        [int((vote == i).sum()), int(same[vote == i].sum())]
                        for i in range(3)
                    ],
                }
            )
            if epoch in (11, 100, 101, 150):
                endpoint[epoch] = purity(uv, vote, y, exp)
            edges = dict(zip(keys.tolist(), vote.tolist()))
            streak = {}
            for key, v in edges.items():
                i, j = divmod(key, len(y))
                if y[i] != y[j] and v == 2:
                    streak[key] = old_streak.get(key, 0) + 1
                    seen_wrong[key] += 1
                    max_streak[key] = max(max_streak.get(key, 0), streak[key])
            for key, v in old_edges.items():
                i, j = divmod(key, len(y))
                semantic = "same" if y[i] == y[j] else "wrong"
                repeat = "repeat" if old_streak.get(key, 0) >= 2 else "other"
                applied = "exposed" if key in old_exposed else "other"
                transitions[(semantic, v, repeat, applied, edges.get(key, -1))] += 1
            old_edges, old_streak, old_exposed = edges, streak, set(times)
    assert len(hist) == 150
    np.save(work / "actual_draws.npy", draws)
    examples = [
        {
            "ids": list(divmod(k, len(y))),
            "classes": y[list(divmod(k, len(y)))].tolist(),
            "consecutive_wrong_strengthen": n,
            "total_epochs": seen_wrong[k],
        }
        for k, n in sorted(max_streak.items(), key=lambda z: (-z[1], z[0]))[:10]
    ]
    write_json(
        out,
        {
            "epochs": hist,
            "historical_semantics": endpoint,
            "rng_matches_checkpoint100": matches_mid,
            "wrong_strengthen_unique": len(seen_wrong),
            "wrong_strengthen_repeated": sum(n >= 2 for n in max_streak.values()),
            "failure_examples": examples,
            "transitions": [[*k, n] for k, n in sorted(transitions.items())],
        },
    )
    print(
        "Replay verified",
        sum(row.get("replay_verified", False) for row in hist),
        "of 140 active epochs",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "e3log", "out"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    run(a.work, a.e3log, a.out)
