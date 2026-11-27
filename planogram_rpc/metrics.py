"""Final-only semi-supervised Euclidean K-means and product-reference retrieval."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def distances(x, centers):
    return (
        x.square().sum(1, keepdim=True)
        + centers.square().sum(1)[None]
        - 2 * x @ centers.T
    ).clamp_min(0)


@torch.no_grad()
def ss_kmeans(features, legal_labels, clusters, *, seed=1, n_init=10, max_iter=100):
    """Lloyd K-means with legal known assignments clamped at every iteration.

    Inputs are unit backbone features; means are ordinary Euclidean centroids,
    not re-normalized spherical centers. Unlabeled ground truth is not an input.
    This standalone evaluator is not claimed bitwise equivalent to HypCD's solver.
    """
    x = features.float()
    legal = legal_labels.to(device=x.device, dtype=torch.long)
    if x.ndim != 2 or legal.shape != (len(x),) or not torch.isfinite(x).all():
        raise ValueError("Invalid features/labels")
    known = sorted(legal[legal >= 0].unique().tolist())
    if (
        not known
        or known != list(range(len(known)))
        or clusters <= len(known)
        or clusters > len(x)
    ):
        raise ValueError(
            "Known labels must be contiguous, with at least one novel cluster"
        )
    if (
        n_init < 1
        or max_iter < 1
        or int(legal.max()) >= clusters
        or int(legal.min()) < -1
    ):
        raise ValueError("Invalid solver settings/labels")
    best = None
    fixed = legal >= 0
    candidates = (~fixed).nonzero().flatten()
    if len(candidates) < clusters - len(known):
        raise ValueError("Not enough unlabeled examples for novel centers")
    for attempt in range(n_init):
        generator = torch.Generator(device=x.device).manual_seed(seed + attempt)
        centers = torch.stack([x[legal == k].mean(0) for k in known])
        for _ in range(clusters - len(known)):
            nearest = distances(x[candidates], centers).min(1).values
            if float(nearest.sum()) <= 1e-12:
                index = int(
                    torch.randint(
                        len(candidates), (), generator=generator, device=x.device
                    )
                )
            else:
                index = int(torch.multinomial(nearest, 1, generator=generator))
            centers = torch.cat([centers, x[candidates[index]][None]])
        previous = None
        for iteration in range(max_iter):
            d = distances(x, centers)
            labels = d.argmin(1)
            labels[fixed] = legal[fixed]
            if previous is not None and torch.equal(previous, labels):
                break
            previous = labels.clone()
            sums = torch.zeros_like(centers).index_add_(0, labels, x)
            counts = torch.bincount(labels, minlength=clusters)
            occupied = counts > 0
            centers[occupied] = sums[occupied] / counts[occupied, None]
            # Empty centers use distant unlabeled rows; never gold novel labels.
            farthest = (
                distances(x[candidates], centers[occupied])
                .min(1)
                .values.argsort(descending=True)
            )
            for rank, cluster in enumerate((~occupied).nonzero().flatten()):
                centers[cluster] = x[candidates[farthest[rank % len(farthest)]]]
        d = distances(x, centers)
        labels = d.argmin(1)
        labels[fixed] = legal[fixed]
        inertia = float(d.gather(1, labels[:, None]).sum())
        if best is None or inertia < best["inertia"]:
            best = {
                "centers": centers.clone(),
                "labels": labels.clone(),
                "inertia": inertia,
                "iterations": iteration + 1,
                "initialization": attempt,
            }
    return best


def cluster_accuracy(targets, predictions, known_count):
    from scipy.optimize import linear_sum_assignment

    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if (
        not len(targets)
        or targets.shape != predictions.shape
        or min(targets.min(), predictions.min()) < 0
    ):
        raise ValueError("Invalid evaluation labels")
    k = int(max(targets.max(), predictions.max())) + 1
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (predictions, targets), 1)
    rows, cols = linear_sum_assignment(-matrix)
    mapping = dict(zip(rows.tolist(), cols.tolist()))
    correct = np.array([mapping[p] == t for p, t in zip(predictions, targets)])
    known = targets < known_count
    return {
        "all": float(correct.mean()),
        "old": float(correct[known].mean()) if known.any() else None,
        "new": float(correct[~known].mean()) if (~known).any() else None,
        "samples": len(targets),
        "protocol": "single Hungarian assignment over this evaluation pool (v2)",
    }


@torch.no_grad()
def retrieval(
    gallery,
    gallery_targets,
    queries,
    targets,
    *,
    known_count,
    threshold=0.35,
    margin=0.05,
):
    gallery, queries = (
        F.normalize(gallery.float(), dim=1),
        F.normalize(queries.float(), dim=1),
    )
    gallery_targets = torch.as_tensor(gallery_targets, device=gallery.device)
    targets = torch.as_tensor(targets, device=queries.device)
    classes = gallery_targets.unique(sorted=True)
    if len(classes) < 2:
        raise ValueError("Retrieval requires at least two products")
    similarities = queries @ gallery.T
    scores = torch.stack(
        [similarities[:, gallery_targets == k].max(1).values for k in classes], dim=1
    )
    values, indices = scores.topk(2, dim=1)
    predictions = classes[indices[:, 0]]
    correct = predictions == targets
    accepted = (values[:, 0] >= threshold) & (values[:, 0] - values[:, 1] >= margin)
    old = targets < known_count
    per_class = [float(correct[targets == k].float().mean()) for k in targets.unique()]
    return {
        "top1": float(correct.float().mean()),
        "macro_top1": float(np.mean(per_class)),
        "old_top1": float(correct[old].float().mean()) if old.any() else None,
        "new_top1": float(correct[~old].float().mean()) if (~old).any() else None,
        "coverage": float(accepted.float().mean()),
        "accepted_accuracy": float(correct[accepted].float().mean())
        if accepted.any()
        else None,
        "threshold": threshold,
        "margin": margin,
        "queries": len(targets),
        "note": "All query SKUs have gallery references. This does not measure unknown-product rejection or shelf-photo accuracy.",
    }, predictions
