"""G7 matched Euclidean and Poincare auxiliary heads on frozen Cars CLS."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from crish.experiment import file_digest, load_checkpoint
from relational.state import write_json
from relational.train import build
from scripts import audit_g1 as g1
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_learned_relation import known_features, make_pairs

LOCAL = g1.WORK / "g7"
OUT = Path("reports/r2_hyperbolic_geometry.json")
MD = Path("reports/CRISH_HYPERBOLIC_GEOMETRY.md")
G1 = Path("reports/r2_multilayer_geometry.json")
METHODS = tuple(f"H{i}" for i in range(6))
SEEDS = (1, 7, 19)
STEPS = 20
BATCH = 256


def tangent(raw):
    # Same smooth radial bound for both learned heads. Radius remains informative.
    return raw / torch.sqrt(1 + raw.square().sum(-1, keepdim=True) / 4)


def expmap(v):
    r = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    scale = torch.where(r > 1e-7, r.tanh() / r.clamp_min(1e-7), 1.0)
    return v * scale


def poincare_sq(a, b, delta_sq):
    denom = (1 - a.square().sum(-1)) * (1 - b.square().sum(-1))
    z = delta_sq / denom.clamp_min(1e-12)
    # asinh(sqrt(z))² tends to z with a finite derivative at coincidence.
    return 4 * torch.where(z > 1e-12, z.clamp_min(1e-12).sqrt().asinh().square(), z)


def distance_sq(a, b, hyperbolic):
    if hyperbolic:
        a, b = expmap(a), expmap(b)
    delta = (a - b).square().sum(-1)
    return poincare_sq(a, b, delta) if hyperbolic else delta


def matrix_distance(a, b, hyperbolic):
    a, b = a.double(), b.double()
    if hyperbolic:
        a, b = expmap(a), expmap(b)
    aa, bb = a.square().sum(-1), b.square().sum(-1)
    delta = (aa[:, None] + bb[None, :] - 2 * (a @ b.T)).clamp_min(0)
    return poincare_sq(a[:, None], b[None, :], delta) if hyperbolic else delta


def geometry_smoke():
    a = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.999, 0.0]], requires_grad=True, device="cuda"
    )
    b = torch.tensor([[0.0, 0.0], [0.0, 0.0], [-1.999, 0.0]], device="cuda")
    d = distance_sq(a, b, True)
    torch.testing.assert_close(
        d.sqrt(), torch.tensor([0.0, 1.0, 7.996], device="cuda"), atol=1e-5, rtol=1e-5
    )
    d.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert expmap(a).norm(dim=-1).max() < 0.965


def paired_pool(x, y, rows, seed):
    _, targets, edges = make_pairs(x[rows], y[rows], np.random.default_rng(seed))
    return rows[edges], targets


def fit_heads(x, train, val, seed, bench):
    torch.manual_seed(seed)
    template = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 128)).cuda()
    heads = [copy.deepcopy(template), copy.deepcopy(template)]
    n_parameters = sum(p.numel() for p in template.parameters())
    a, y = (torch.as_tensor(v, device="cuda") for v in train)
    b, vy = (torch.as_tensor(v, device="cuda") for v in val)
    rng = np.random.default_rng(seed + 230)
    orders = [
        list(torch.as_tensor(rng.permutation(len(y)), device="cuda").split(BATCH))
        for _ in range(STEPS)
    ]
    result, embeddings = [], []
    for hyper, model in zip((False, True), heads, strict=True):
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        best = None
        max_gradient = 0.0
        for epoch, batches in enumerate(orders, 1):
            for i in batches:
                z = tangent(model(x[a[i]]))
                sq = distance_sq(z[:, 0], z[:, 1], hyper)
                loss = torch.where(
                    y[i] > 0, sq, (1 - sq.clamp_min(1e-12).sqrt()).clamp_min(0).square()
                ).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.linalg.vector_norm(
                    torch.stack([p.grad.norm() for p in model.parameters()])
                )
                if not torch.isfinite(norm) or not torch.isfinite(loss):
                    raise ValueError("Nonfinite head training gradient or loss")
                max_gradient = max(max_gradient, norm.item())
                opt.step()
            with torch.no_grad():
                z = tangent(model(x[b]))
                scores = -distance_sq(z[:, 0], z[:, 1], hyper).cpu().numpy()
            auc = g1.auc(vy.cpu().numpy(), scores)
            if best is None or auc > best[0]:
                best = (auc, epoch, copy.deepcopy(model.state_dict()))
        model.load_state_dict(best[2])
        with torch.no_grad():
            z = tangent(model(bench))
            radii = z.norm(dim=-1)
            ball = expmap(z).norm(dim=-1) if hyper else None
            assert torch.isfinite(z).all()
            if ball is not None:
                assert ball.max() < 0.965
            embeddings.append(z)
            result.append(
                {
                    "parameters": n_parameters,
                    "validation_auc": best[0],
                    "selected_epoch": best[1],
                    "updates": sum(map(len, orders)),
                    "max_gradient": max_gradient,
                    "max_tangent_radius": radii.max().item(),
                    "min_tangent_radius": radii.min().item(),
                    "max_ball_radius": ball.max().item() if hyper else None,
                }
            )
    return embeddings, result


def percentile(score):
    ranks = score.argsort(dim=1, stable=True).argsort(dim=1)
    return 2 * ranks.float() / (score.shape[1] - 1) - 1


def pack(score, q, bank, slots, pairs):
    assert torch.isfinite(score).all()
    peers = torch.as_tensor(slots[bank], device="cuda", dtype=torch.long)
    src = torch.as_tensor(slots[q], device="cuda", dtype=torch.long)[:, None]
    support = torch.full((len(q), 196), -torch.inf, device="cuda", dtype=score.dtype)
    for s in np.unique(slots[bank]):
        support[:, s] = score[:, peers == int(s)].max(1).values
    own = support.gather(1, src).flatten().clamp_min(-1)
    support.scatter_(1, src, -torch.inf)
    top = support.argsort(descending=True, stable=True)[:, :5]
    dest = torch.where(support.gather(1, top).isfinite(), top, -1)
    indices = score.topk(64).indices.cpu().numpy()
    rows = np.arange(len(q))[:, None]
    cpu = score.cpu().numpy()
    return {
        "neighbors": bank[score.topk(10).indices.cpu().numpy()],
        "pair": cpu[rows, pairs],
        "wrong": (support.max(1).values - own).cpu().numpy(),
        "dest": dest.cpu().numpy(),
        "specific_ids": bank[indices],
        "specific_scores": cpu[rows, indices],
    }


def freeze(x, z, q, bank, slots, pairs, base):
    cos = x[q] @ x[bank].T
    # Float64 unit normalization makes the shared H1 tangent norm exact to roundoff.
    unit = F.normalize(x.double())
    direct = 2 / (1 + matrix_distance(unit[q], unit[bank], True)) - 1
    cosine64 = unit[q] @ unit[bank].T
    order = cosine64.argsort(descending=True)
    d = direct.gather(1, order)
    reverse = d[:, 1:] - d[:, :-1]
    # Check full ordering in float64 and the top-10 set against cached float32 cosine.
    assert reverse.max().item() < 1e-12
    shared = (
        cos.topk(10).indices.sort(1).values == direct.topk(10).indices.sort(1).values
    ).all(1)
    assert shared.float().mean().item() == 1.0
    euclid = (2 / (1 + matrix_distance(z[0][q], z[0][bank], False)) - 1).float()
    hyper = (2 / (1 + matrix_distance(z[1][q], z[1][bank], True)) - 1).float()
    rank = percentile(cos)
    scores = (
        cos,
        direct,
        euclid,
        hyper,
        (rank + percentile(euclid)) / 2,
        (rank + percentile(hyper)) / 2,
    )
    result = {"pairs": bank[pairs]}
    for m, score in zip(METHODS, scores, strict=True):
        data = pack(score, q, bank, slots, pairs)
        if m == "H0":
            for k in ("neighbors", "pair", "wrong", "dest"):
                old = base[f"B11_{k}"]
                if k in ("neighbors", "dest"):
                    np.testing.assert_array_equal(old, data[k])
                else:
                    np.testing.assert_allclose(old, data[k], atol=3e-6, rtol=3e-6)
            np.testing.assert_array_equal(base["pairs"][:, 256:], data["specific_ids"])
        result.update({f"{m}_{k}": v for k, v in data.items()})
    return result, {
        "top10_exact_set_fraction": shared.float().mean().item(),
        "max_adjacent_score_inversion": reverse.clamp_min(0).max().item(),
    }


def sealed(frozen, ids, q, bank, slots, truth):
    scores, coverage = g1.evaluate(
        frozen, ids, q, bank, slots, truth, methods=METHODS, baseline="H0"
    )
    mp, _ = align(
        ids[bank].tolist(), dict(zip(ids.tolist(), slots.tolist(), strict=True)), truth
    )
    table = np.array([mp.get(k, -1) for k in range(196)] + [-1])
    y = truth[ids[q]]
    wrong = table[slots[q]] != y
    base = frozen["H0_dest"][:, 0]
    positive = truth[ids[frozen["pairs"][:, :256]]] == y[:, None]
    for method in METHODS:
        p = frozen[f"{method}_dest"][:, 0]
        neg = truth[ids[frozen[f"{method}_specific_ids"]]] != y[:, None]
        for group, mask in (("All", y >= 0), ("Old", y < 98), ("New", y >= 98)):
            pos_scores = frozen[f"{method}_pair"][:, :256][positive & mask[:, None]]
            neg_scores = frozen[f"{method}_specific_scores"][neg & mask[:, None]]
            disagree = mask & wrong & (p != base)
            metrics = scores[method][group]
            metrics.update(
                method_hard_auc=g1.auc(
                    np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))],
                    np.r_[pos_scores, neg_scores],
                ),
                method_hard_count=len(neg_scores),
                disagreement_n=int(disagree.sum()),
                disagreement_accuracy=g1.rate(table[p[disagree]] == y[disagree]),
                h0_on_disagreement=g1.rate(table[base[disagree]] == y[disagree]),
                either_correct=g1.rate(
                    ((table[p] == y) | (table[base] == y))[mask & wrong]
                ),
            )
    return scores, coverage


def report(result):
    lines = [
        "# G7 dual Euclidean–hyperbolic geometry",
        "",
        (
            "The same R2-100/150 checkpoints, G1 query/bank IDs, classifier slots and 78/20 "
            "class-disjoint Known split are reused. These are exploratory comparisons on reused queries. "
            "All backbone weights stay frozen. Only missing legally labeled center CLS features are encoded."
        ),
        "",
        (
            "H0 uses cosine. H1 applies the origin exponential map at curvature c=1, "
            "x = tanh(||v||) v / ||v||. Unit input radius makes H1 pair similarity monotonic in cosine. "
            "The assignment-error score subtracts two similarities, so it can change despite equivalent ranking."
        ),
        "",
        (
            "H2/H3 share a 768–256–128 GELU architecture, initial parameters, Known pair and batch order, "
            "AdamW and 20 epochs. Both use a shared smooth tangent radius bound of 2, which preserves "
            "radial information. H2 uses Euclidean distance on those vectors and is the radial control. "
            "H3 applies the exponential map and uses Poincaré geodesic distance. The equivalent losses "
            "are squared distance for positives and squared hinge with distance margin 1 for negatives. "
            "Known validation AUROC selects the earliest best epoch per head. H4/H5 average within-query "
            "percentile similarities from H0 and the corresponding learned head. Each distance score "
            "uses 2/(1+d²)−1 before scoring. The shared radius bound caps the Poincaré norm below tanh(2)."
        ),
        "",
        (
            "Common hard AUROC uses uniform-pool same-class pairs against negatives in the G1 H0 top-64. "
            "Own hard AUROC uses negatives in each rule’s own top-64 and therefore compares different "
            "negative pools. Destination metrics use G1 bank-only Hungarian alignment and wrong-source "
            "queries. Disagreement accuracy is top-1 accuracy where the rule and H0 propose different slots. "
            "JSON includes H0 accuracy on those same disagreements, either-rule recovery, seed results and counts."
        ),
        "",
        "| Epoch / group / method | Purity@1/5/10 | Pair / common / own hard AUC | Error AUC | Destination R@1/3/5 | Recovery / regression | Disagreement ACC |",
        "|---|---|---|---:|---|---|---:|",
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    for epoch, record in result["snapshots"].items():
        for g in ("All", "Old", "New"):
            for m in METHODS:
                n = [entry["metrics"][m][g] for entry in record["seeds"]]

                def mean(key, i=None, rows=n):
                    vals = [a[key] if i is None else a[key][i] for a in rows]
                    vals = [v for v in vals if v is not None]
                    return fmt(np.mean(vals) if vals else None)

                fields = (
                    "/".join(mean("purity", i) for i in (1, 5, 10)),
                    "/".join(
                        mean(k) for k in ("pair_auc", "hard_auc", "method_hard_auc")
                    ),
                    mean("detection_auc"),
                    "/".join(mean("destination_recall", i) for i in (1, 3, 5)),
                    mean("failure_recovery", 1)
                    + "/"
                    + mean("correct_to_wrong_vs_final"),
                    mean("disagreement_accuracy"),
                )
                lines.append(f"| {epoch} / {g} / {m} | " + " | ".join(fields) + " |")
    lines += [
        "",
        "The table reports arithmetic means across seeds. Seeds share queries and are not independent replications.",
        f"Runtime {result['elapsed_seconds'] / 60:.1f} minutes.",
        "",
        "[Seed-level metrics and frozen prediction provenance](r2_hyperbolic_geometry.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    g1_report = json.loads(G1.read_text())
    model, loader, _, center, _, _ = build()
    model.eval().requires_grad_(False)
    geometry_smoke()
    paths = [g1.WORK / f"input/epoch-{e}.pt" for e in g1.EPOCHS]
    states = []
    for epoch, path in zip(g1.EPOCHS, paths, strict=True):
        assert (
            file_digest(path) == g1_report["snapshots"][str(epoch)]["checkpoint_sha256"]
        )
        states.append(load_checkpoint(path)["model"])
    live = model[0].state_dict()
    for state in states:
        for key, tensor in live.items():
            if not key.startswith("blocks.11."):
                assert torch.equal(tensor.cpu(), state[f"0.{key}"]), key
    legal_ids, labels, known = known_features(
        model[0], loader.dataset.labelled_dataset, center, states
    )
    del states, live, model
    tr = np.flatnonzero(np.isin(legal_ids, g1_report["known_train_ids"]))
    va = np.flatnonzero(np.isin(legal_ids, g1_report["known_val_ids"]))
    assert len(tr) == g1_report["known_train_n"] and len(va) == g1_report["known_val_n"]
    assert len(np.intersect1d(labels[tr], labels[va])) == 0 and len(tr) + len(
        va
    ) == len(labels)
    final = np.load(g1.WORK / "features.npy", mmap_mode="r")
    result = {
        "base": "ee9f1dd",
        "curvature": 1,
        "train_epochs": STEPS,
        "batch": BATCH,
        "g1_report_sha256": file_digest(G1),
        "query_count": g1_report["queries"],
        "bank_count": g1_report["bank"],
        "snapshots": {},
    }
    saved = {}
    for index, epoch in enumerate(g1.EPOCHS, 1):
        key = str(epoch)
        meta = g1_report["snapshots"][key]
        manifest = g1.LOCAL / f"frozen-{epoch}.npz"
        assert file_digest(manifest) == meta["frozen_sha256"]
        with np.load(manifest) as base:
            ids, q, bank, slots = (base[k] for k in ("ids", "q", "bank", "slots"))
            assert np.array_equal(ids, np.load(g1.WORK / "ids.npy"))
            assert ids[q].tolist() == g1_report["query_ids"]
            assert (
                not np.intersect1d(legal_ids, ids).size
                and not np.intersect1d(q, bank).size
            )
            inv = np.full(len(ids), -1)
            inv[bank] = np.arange(len(bank))
            pairs = inv[base["pairs"]]
            assert (pairs >= 0).all()
            x = F.normalize(torch.from_numpy(final[index, 0].copy()), dim=1).cuda()
            raw = known[index - 1]
            train = paired_pool(raw, labels, tr, 4019)
            val = paired_pool(raw, labels, va, 4029)
            known_x = torch.as_tensor(raw, device="cuda")
            row = {
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "g1_manifest_sha256": meta["frozen_sha256"],
                "train_pairs": len(train[1]),
                "validation_pairs": len(val[1]),
                "train_pairs_sha256": pair_digest(train),
                "validation_pairs_sha256": pair_digest(val),
                "seeds": [],
            }
            for seed in SEEDS:
                emb, heads = fit_heads(known_x, train, val, seed, x)
                scores, parity = freeze(x, emb, q, bank, slots, pairs, base)
                path = LOCAL / f"frozen-{epoch}-{seed}.npz"
                np.savez(path, **scores)
                saved[str(path)] = (ids, q, bank, slots)
                row["seeds"].append(
                    {
                        "seed": seed,
                        "heads": dict(zip(("H2", "H3"), heads, strict=True)),
                        "h1_parity": parity,
                        "frozen_sha256": file_digest(path),
                    }
                )
                print("Frozen", epoch, seed, parity, flush=True)
            result["snapshots"][key] = row
    write_json(LOCAL / "frozen-provenance.json", json_native(result))
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    assert np.array_equal(truth[legal_ids], labels)
    for epoch, rec in result["snapshots"].items():
        for row in rec["seeds"]:
            path = LOCAL / f"frozen-{epoch}-{row['seed']}.npz"
            assert file_digest(path) == row["frozen_sha256"]
            with np.load(path) as frozen:
                row["metrics"], row["alignment"] = sealed(
                    dict(frozen), *saved[str(path)], truth
                )
    result["elapsed_seconds"] = time.monotonic() - start
    report(result)
    write_json(OUT, json_native(result))


def pair_digest(pool):
    from hashlib import sha256

    return sha256(np.column_stack(pool).astype("<i8").tobytes()).hexdigest()


if __name__ == "__main__":
    main()
