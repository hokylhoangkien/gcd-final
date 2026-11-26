"""Sealed endpoint diagnostic. Outputs are never read by training or the guard."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import silhouette_samples

from crish.experiment import file_digest, load_checkpoint
from data.augmentations import get_transform
from data.get_datasets import get_class_splits
from data.stanford_cars import CarsDataset, get_scars_datasets
from model import DINOHead
from relational.state import write_json
from scripts.audit_crish_snapshot import batch_images, unit


def geometry(x, y):
    x = unit(x)
    groups = np.unique(y)
    sums = np.stack([x[y == k].sum(0) for k in groups])
    centers = unit(sums)
    own = (x * unit(sums[y] - x)).sum(1)
    sim = x @ centers.T
    sim[np.arange(len(y)), y] = -np.inf
    d = np.clip(1 - x @ x.T, 0, 2)
    np.fill_diagonal(d, 0)
    silhouette = silhouette_samples(d, y, metric="precomputed")
    np.fill_diagonal(d, np.inf)
    purity = (y[np.argsort(d, axis=1)[:, :5]] == y[:, None]).mean(1)
    sep = centers @ centers.T
    np.fill_diagonal(sep, -np.inf)
    output = {}
    for group, mask in (("old", y < 98), ("novel", y >= 98), ("all", y >= 0)):
        output[group] = {
            "n": int(mask.sum()),
            "purity5": float(purity[mask].mean()),
            "own_loo_cos": float(own[mask].mean()),
            "nearest_other_centroid_distance": float((1 - sep.max(1)[y[mask]]).mean()),
            "semantic_margin": float((own - sim.max(1))[mask].mean()),
            "silhouette": float(silhouette[mask].mean()),
        }
    return output


def run(source, out, reference=False):
    split = get_class_splits(SimpleNamespace(dataset_name="scars", use_ssb_splits=True))
    rows = get_scars_datasets(None, None, split.train_classes, 0.5)[
        "train_unlabelled"
    ].uq_idxs
    dataset = CarsDataset()
    order = list(split.train_classes) + list(split.unlabeled_classes)
    y = np.array([order.index(t - 1) for t in dataset.target])
    rng = np.random.default_rng(104729)
    panel = np.concatenate(
        [rng.choice(rows[y[rows] == k], 5, replace=False) for k in range(196)]
    )
    _, center = get_transform(
        "imagenet", 224, SimpleNamespace(interpolation=3, crop_pct=0.875)
    )
    backbone = torch.hub.load(
        os.environ["SIMGCD_DINO_REPO"], "dino_vitb16", source="local", pretrained=False
    )
    epoch = 0
    if reference:
        backbone.load_state_dict(
            torch.load(source, map_location="cpu", weights_only=True)
        )
    else:
        model = torch.nn.Sequential(backbone, DINOHead(768, 196))
        saved = load_checkpoint(source)
        model.load_state_dict(saved["model"])
        epoch = saved["epoch"]
        del saved
    backbone.eval().cuda()
    with torch.no_grad():
        x = torch.cat(
            [
                backbone(
                    batch_images(panel[i : i + 4], center, 0, dataset, "cuda")
                ).cpu()
                for i in range(0, len(panel), 4)
            ]
        ).numpy()
    write_json(
        out,
        {
            "epoch": epoch,
            "source_sha256": file_digest(source),
            "panel_ids": panel.tolist(),
            "state": "DINO initialization" if reference else "trained endpoint",
            "groups": geometry(x, y[panel]),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--reference", action="store_true")
    a = p.parse_args()
    run(a.source, a.output, a.reference)
