"""Frozen Cars readouts at the O0/O5/O6 five-epoch endpoints."""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest, load_checkpoint
from relational.sensitivity_metrics import prefix
from relational.state import write_json
from relational.train import build
from scripts.audit_r2_evidence import sealed_labels

WORK = Path("dev_outputs/r2-audit")
LOCAL = WORK / "five-readout"
REPORT = Path("reports/CRISH_FROZEN_READOUT.md")
JSON = Path("reports/r2_frozen_readout.json")
ARMS = ("O0", "O5", "O6")
SEEDS = (1, 7, 19)
K = 196
MAX_ITER = 50


def digest(x):
    return hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()).hexdigest()


@torch.inference_mode()
def extract():
    history = json.loads(Path("reports/r2_oracle_move_trajectory.json").read_text())
    five = json.loads(Path("reports/r2_oracle_move_five_epochs.json").read_text())
    panel = np.asarray(history["panel_ids"])
    model, loader, _, center, _, _ = build()
    model.eval().requires_grad_(False)
    ds = loader.dataset
    ds.labelled_dataset.transform = ds.unlabelled_dataset.transform = center
    all_ids = np.r_[ds.labelled_dataset.uq_idxs, ds.unlabelled_dataset.ids]
    rows = np.flatnonzero(~np.isin(all_ids, panel))
    fit = all_ids[rows]
    assert len(np.unique(np.r_[fit, panel])) == len(all_ids) == len(fit) + 588
    assert set(panel) <= set(ds.unlabelled_dataset.ids)
    bank = torch.load(WORK / "oracle-move/step0-panel-tokens.pt", weights_only=True)
    assert bank["panel_ids"] == panel.tolist()
    assert bank["checkpoint_sha256"] == history["checkpoint_sha256"]
    origin = load_checkpoint(WORK / "input/epoch-100.pt")["model"]
    tails, classifiers, shas = {}, {}, {}
    for arm in ARMS:
        path = WORK / f"oracle-five/{arm}-epoch-5.pt"
        state = load_checkpoint(path)
        p = state["progress"]
        assert (p["arm"], p["epoch"], p["signature"]) == (arm, 5, five["verification"])
        assert p["result"] == five["epochs"]["5"][arm]
        model.load_state_dict(state["model"])
        for key, val in state["model"].items():
            if key.startswith("0.") and not key.startswith(("0.blocks.11.", "0.norm.")):
                assert torch.equal(val, origin[key]), key
        tails[arm] = copy.deepcopy(
            torch.nn.Sequential(model[0].blocks[-1], model[0].norm)
        )
        layer = model[1].last_layer
        layer(torch.zeros(1, 768, device="cuda"))  # refresh weight_norm hook
        classifiers[arm] = layer.weight.detach().clone()
        shas[arm] = file_digest(path)
        del state
    del origin
    features = {a: [] for a in ARMS}

    def append(tokens):
        for a in ARMS:
            h = torch.cat([tails[a](t.cuda())[:, 0] for t in tokens])
            features[a].append(F.normalize(h, dim=1).cpu())

    legal, seen = [], []
    for images, labels, ids, mask in DataLoader(
        Subset(ds, rows.tolist()), batch_size=32
    ):
        assert torch.equal(mask.flatten().bool(), labels >= 0)
        append(prefix(model[0], images))
        legal.extend(labels.tolist())
        seen.extend(ids.tolist())
    assert np.array_equal(seen, fit)
    for tokens in bank["tokens"]:
        append(tokens)
    x = {a: torch.cat(features[a]).cuda() for a in ARMS}
    original = {a: (x[a] @ classifiers[a].T).argmax(1).cpu().numpy() for a in ARMS}
    np.savez(
        LOCAL / "frozen-features.npz",
        **{a: v.cpu().numpy() for a, v in x.items()},
        fit=fit,
        panel=panel,
        legal=legal,
        checkpoint_sha256=list(shas.values()),
    )
    return x, original, fit, panel, np.asarray(legal), shas


def spherical(x, fit_n, legal, seed, anchored):
    train = x[:fit_n]
    known = legal >= 0
    rng = np.random.default_rng(seed)
    if anchored:
        means = torch.stack(
            [train[np.flatnonzero(legal == k)].mean(0) for k in range(98)]
        )
        init = rng.choice(np.flatnonzero(~known), 98, replace=False)
        centers = torch.cat([means, train[init]])
    else:
        centers = train[rng.choice(fit_n, K, replace=False)]
    centers = F.normalize(centers)
    previous = None
    for _ in range(MAX_ITER):
        y = (train @ centers.T).argmax(1)
        if anchored:
            anchors = torch.as_tensor(legal, dtype=torch.long, device=x.device)
            y = torch.where(anchors >= 0, anchors, y)
        if previous is not None and torch.equal(previous, y):
            break
        previous = y
        sums = F.one_hot(y, K).float().T @ train
        centers = torch.where(
            sums.norm(dim=1, keepdim=True) > 0, F.normalize(sums), centers
        )
    # Out-of-sample decision rule, also used on fit rows for evaluation alignment.
    return (x @ centers.T).argmax(1).cpu().numpy()


def oracle(x, truth):
    n = len(truth)
    targets = F.one_hot(
        torch.as_tensor(truth, device=x.device, dtype=torch.long), K
    ).double()
    assert targets.sum(0).min() > 0
    u = x.double()
    centroid = (u @ F.normalize(targets.T @ u[:n]).T).argmax(1)
    # Multiclass linear ridge, unpenalized intercept and unit L2 penalty on weights.
    xm, ym = u[:n].mean(0), targets.mean(0)
    z = u[:n] - xm
    w = torch.linalg.solve(
        z.T @ z + torch.eye(z.shape[1], device=z.device), z.T @ (targets - ym)
    )
    linear = ((u - xm) @ w + ym).argmax(1)
    return {"centroid": [centroid.cpu().numpy()], "linear": [linear.cpu().numpy()]}


def mapping(slots, truth):
    table = np.bincount(K * slots + truth, minlength=K * K).reshape(K, K)
    r, c = linear_sum_assignment(-table)
    out = np.full(K, -1)
    good = table[r, c] > 0
    out[r[good]] = c[good]
    return out


def evaluate(pred, fit_y, panel_y):
    n = len(fit_y)
    p = pred[n:]
    fixed = mapping(pred[:n], fit_y)[p]
    aligned = mapping(p, panel_y)[p]
    result = {}
    for group, mask in (
        ("All", panel_y >= 0),
        ("Old", panel_y < 98),
        ("New", panel_y >= 98),
    ):
        t = panel_y[mask]
        result[group] = {
            "fit": float(np.mean(fixed[mask] == t)),
            "rematch": float(np.mean(aligned[mask] == t)),
            "ari": adjusted_rand_score(t, p[mask]),
            "nmi": normalized_mutual_info_score(t, p[mask]),
        }
    return result


def reduce_rows(rows):
    return {
        g: {m: float(np.mean([r[g][m] for r in rows])) for m in rows[0][g]}
        for g in rows[0]
    }


def diff(left, right):
    return {
        g: {m: v - right[g][m] for m, v in metrics.items()}
        for g, metrics in left.items()
    }


def main():
    started = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    x, original, fit, panel, legal, shas = extract()
    print("Frozen features ready", len(fit), len(panel), flush=True)
    preds = {}
    for a in ARMS:
        preds[a] = {"classifier": [original[a]]}
        for name in ("spherical", "semi"):
            preds[a][name] = [
                spherical(x[a], len(fit), legal, s, name == "semi") for s in SEEDS
            ]
    # Serialize legal predictions before opening privileged labels.
    frozen = LOCAL / "legal-predictions.npz"
    np.savez(
        frozen,
        fit=fit,
        panel=panel,
        **{f"{a}_{k}": v for a in ARMS for k, v in preds[a].items()},
    )
    truth = sealed_labels()
    fit_y = truth[fit]
    assert np.array_equal(fit_y[legal >= 0], legal[legal >= 0])
    for a in ARMS:
        preds[a].update(oracle(x[a], fit_y))
    panel_y = truth[panel]
    scores = {
        a: {k: [evaluate(v, fit_y, panel_y) for v in vs] for k, vs in preds[a].items()}
        for a in ARMS
    }
    means = {a: {k: reduce_rows(v) for k, v in scores[a].items()} for a in ARMS}
    paired = {
        f"{a}-{b}": {k: diff(v, means[b][k]) for k, v in means[a].items()}
        for a, b in (("O5", "O0"), ("O6", "O0"), ("O5", "O6"))
    }
    five = json.loads(Path("reports/r2_oracle_move_five_epochs.json").read_text())
    for a in ARMS:
        for g, v in means[a]["classifier"].items():
            assert (
                abs(
                    v["rematch"]
                    - five["epochs"]["5"][a]["panel"]["rematched"][g.lower()][
                        "accuracy"
                    ]
                )
                < 1e-12
            )
    result = {
        "base": "225fc63",
        "checkpoints": shas,
        "fit_n": len(fit),
        "known_fit_n": int((legal >= 0).sum()),
        "panel_n": len(panel),
        "fit_sha256": digest(fit),
        "panel_sha256": digest(panel),
        "frozen_legal_predictions_sha256": file_digest(frozen),
        "seeds": SEEDS,
        "max_iterations": MAX_ITER,
        "scores_by_seed": scores,
        "means": means,
        "paired_arm_differences": paired,
        "seconds": time.monotonic() - started,
    }
    write_json(JSON, result)
    lines = [
        "# Frozen readouts after five Oracle MOVE diagnostic epochs",
        "",
        (
            f"All {len(fit):,} non-panel Cars IDs form the fit bank ({result['known_fit_n']:,} legal labels). "
            "All arms use identical center crops and the original 588 held-out IDs. Frozen normalized CLS "
            "features feed every readout. Legal refers to readout fitting only: O5 features were trained with oracle edits."
        ),
        "",
        (
            "Spherical Lloyd clustering uses 196 slots, seeds 1/7/19, at most 50 iterations, "
            "unchanged empty centers, and unweighted seed means. Semi-supervised clustering initializes "
            "98 centers with labeled class means and clamps labeled assignments during fitting. Other initial "
            "centers are distinct random unlabeled rows. Oracle centroid uses normalized class sums. "
            "Oracle linear uses multiclass least squares with a unit ridge penalty and an unpenalized intercept."
        ),
        "",
        (
            "Both accuracy columns use one global Hungarian assignment across Old and New. Fit alignment uses "
            "privileged fit labels only during scoring. Panel rematching is descriptive. ARI and NMI use raw slots. "
            "Differences pair the same panel and seeds; initialization repeats are not independent training runs."
        ),
    ]
    for section, keys in (
        ("Legal readouts", ("classifier", "spherical", "semi")),
        ("Oracle readouts", ("centroid", "linear")),
    ):
        lines += [
            "",
            f"## {section}",
            "",
            "| Arm / readout | Fit All/Old/New (%) | Rematched All/Old/New (%) | ARI / NMI (All) | Δ fit All vs O0 (pp) |",
            "|---|---|---|---|---:|",
        ]
        for k in keys:
            for a in ARMS:
                v = means[a][k]
                triplets = [
                    " / ".join(f"{100 * v[g][m]:.2f}" for g in ("All", "Old", "New"))
                    for m in ("fit", "rematch")
                ]
                delta = v["All"]["fit"] - means["O0"][k]["All"]["fit"]
                lines.append(
                    f"| {a} / {k} | {triplets[0]} | {triplets[1]} | {v['All']['ari']:.4f} / {v['All']['nmi']:.4f} | {100 * delta:+.2f} |"
                )
    lines += [
        "",
        "[Seed-level metrics, subgroup ARI/NMI and paired arm differences](r2_frozen_readout.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
