"""Sealed label joins, dyadic uncertainty, full-split geometry and evidence discrimination."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from data.get_datasets import get_class_splits
from data.stanford_cars import CarsDataset
from relational.graph import Graph, evidence
from relational.offline import geometry
from relational.state import write_json


def sealed_labels():
    split = get_class_splits(SimpleNamespace(dataset_name="scars", use_ssb_splits=True))
    order = list(split.train_classes) + list(split.unlabeled_classes)
    return np.array([order.index(t - 1) for t in CarsDataset().target])


def graph_at(work, index):
    ids = np.load(work / "ids.npy")
    views = np.load(work / "features.npy", mmap_mode="r")[index].copy()
    return Graph(
        torch.from_numpy(ids), *torch.from_numpy(views).cuda().unbind(), evidence
    )


def estimate(uv, y, w=None, probability=True):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if not w.sum():
        return {"edges": len(y), "mass": 0}
    avg = float(np.average(y, weights=w))
    score = (y - avg) * w
    ns = np.bincount(uv.ravel(), weights=np.repeat(score, 2))
    # Edge self-pair contributes to both endpoint clusters. Subtract once.
    se = np.sqrt(max(0.0, ns @ ns - score @ score)) / w.sum()
    lo, hi = avg - 1.96 * se, avg + 1.96 * se
    return {
        "edges": len(y),
        "nodes": len(np.unique(uv[w > 0])),
        "mass": int(w.sum()),
        "mean": avg,
        "ci95": [
            max(0.0, lo) if probability else lo,
            min(1.0, hi) if probability else hi,
        ],
    }


def purity(uv, votes, labels, exposure=None):
    u, v = labels[uv].T
    same = u == v
    result = {}
    for key, group in (
        ("all", np.ones(len(u), bool)),
        ("old_old", (u < 98) & (v < 98)),
        ("novel_novel", (u >= 98) & (v >= 98)),
        ("mixed", (u < 98) != (v < 98)),
    ):
        result[key] = {}
        for action, choice in (
            ("overall", votes >= 0),
            ("weaken", votes == 0),
            ("keep", votes == 1),
            ("strengthen", votes == 2),
        ):
            m = group & choice
            item = {
                "same_edges": int(same[m].sum()),
                "edge_purity": estimate(uv[m], same[m]),
            }
            if exposure is not None and (exposure[m] >= 0).all():
                item["exposure_purity"] = estimate(uv[m], same[m], exposure[m])
            result[key][action] = item
    return result


def predict(uv, vote, cosine, labels):
    y = labels[uv[:, 0]] == labels[uv[:, 1]]
    # Every edge appears once in a test fold; train/test endpoint sets are disjoint.
    assignment = np.random.default_rng(837).permutation(len(labels)) % 5
    f = np.sort(assignment[uv], axis=1)
    c = (cosine - 0.8) * 10
    c = np.column_stack([c, c**2, c**3])
    v = np.column_stack([vote == 1, vote == 2])
    preds = {}
    for name, x in (("cosine", c), ("vote", v), ("both", np.column_stack([c, v]))):
        prob = np.full(len(y), np.nan)
        for i in range(5):
            for j in range(i, 5):
                test = (f[:, 0] == i) & (f[:, 1] == j)
                train = ~np.isin(f, [i, j]).any(axis=1)
                prob[test] = (
                    LogisticRegression(max_iter=400)
                    .fit(x[train], y[train])
                    .predict_proba(x[test])[:, 1]
                )
        assert np.isfinite(prob).all()
        preds[name] = prob
    return {
        "n": len(y),
        "correct_edges": int(y.sum()),
        "models": {
            k: {
                "auc": roc_auc_score(y, p),
                "log_loss": log_loss(y, p),
                "brier": np.mean((p - y) ** 2),
            }
            for k, p in preds.items()
        },
        "paired_brier_gain": estimate(
            uv, (preds["cosine"] - y) ** 2 - (preds["both"] - y) ** 2, probability=False
        ),
    }


def run(work, out):
    ids, y = np.load(work / "ids.npy"), sealed_labels()
    records = {}
    with (work / "input" / "relationships.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["epoch"] in (11, 100, 101, 150):
                records[row["epoch"]] = row
    results = {}
    for index, name, epoch in (
        (0, "DINO", 11),
        (1, "epoch100", 101),
        (2, "epoch150", 150),
    ):
        g = graph_at(work, index)
        uv, votes = g.edge_ids.numpy(), g.votes.numpy()
        x = g.keys.cpu().numpy()
        lookup = {int(k): i for i, k in enumerate(ids)}
        ij = np.array([[lookup[u], lookup[v]] for u, v in uv])
        cos = np.sum(x[ij[:, 0]] * x[ij[:, 1]], axis=1)
        links = coo_matrix((np.ones(len(ij)), ij.T), shape=(len(ids), len(ids))).tocsr()
        n, components = connected_components(links, directed=False)
        hist = {tuple(e[:2]): e[2] for e in records[epoch]["graph"]["edges"]}
        common = np.array([tuple(edge) in hist for edge in uv])
        examples = {}
        for label, mask in (
            ("wrong_strengthen", (votes == 2) & (y[uv[:, 0]] != y[uv[:, 1]])),
            ("wrong_weaken", (votes == 0) & (y[uv[:, 0]] == y[uv[:, 1]])),
        ):
            chosen = np.flatnonzero(mask)
            chosen = chosen[np.argsort(-cos[chosen], kind="stable")[:4]]
            examples[label] = [
                {
                    "ids": uv[j].tolist(),
                    "classes": y[uv[j]].tolist(),
                    "cosine": float(cos[j]),
                }
                for j in chosen
            ]
        counts = np.load(work / "actual_draws.npy")[epoch - 1]
        result = {
            "graph": g.health,
            "geometry": geometry(x, y[ids]),
            "components": n,
            "largest_component": int(np.bincount(components).max()),
            "semantic": purity(
                uv, votes, y, counts[uv].sum(1) if (counts >= 0).all() else None
            ),
            "exposure_reweight_epoch": epoch,
            "prediction": predict(uv, votes, cos, y),
            "compared_ledger_epoch": epoch,
            "center_jaccard": float(
                common.sum() / (len(hist) + len(uv) - common.sum())
            ),
            "vote_agreement_common": float(
                np.mean(
                    [
                        hist[tuple(edge)] == v
                        for edge, v in zip(uv[common], votes[common])
                    ]
                )
            ),
            "examples": examples,
        }
        if index == 2:
            archived = json.loads((work / "input/geometry.json").read_text())
            assert archived["epoch"] == 150
            panel = np.array([lookup[k] for k in archived["panel_ids"]])
            small = geometry(x[panel], y[ids[panel]])
            result["archived_panel_max_abs_error"] = max(
                abs(small[group][key] - v)
                for group, metrics in archived["groups"].items()
                for key, v in metrics.items()
            )
        results[name] = result
        write_json(out, results)
        print(name, n, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.work, args.output)
