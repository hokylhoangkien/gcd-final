"""Frozen CLIP anchored spherical k-means on the complete transductive Cars split."""

import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dual import cache
from relational.state import write_json
from relational.train import build
from scripts import audit_g6 as g6
from scripts.audit_frozen_readout import MAX_ITER, SEEDS, spherical
from util.cluster_and_log_utils import split_cluster_acc_v2


@torch.inference_mode()
def main():
    model, loader, _, center, _, protocol = build()
    del model
    ds = loader.dataset
    clip, receipt = cache.load(
        g6.LOCAL / "clip-image.npz",
        ds.unlabelled_dataset,
        protocol["split"],
        torch.device("cuda"),
    )
    known = copy.copy(ds.labelled_dataset)
    known.transform = transforms.Compose(
        [*center.transforms[:3], transforms.Normalize(g6.MEAN, g6.STD)]
    )
    net = torch.jit.load(str(g6.weights()), map_location="cuda").eval()
    parts, labels, keys = [], [], []
    for images, y, ids in DataLoader(known, batch_size=16):
        parts.append(F.normalize(net.encode_image(images.cuda()).float(), dim=1))
        labels.extend(y.tolist())
        keys.extend(ids.tolist())
    ids = np.asarray(ds.unlabelled_dataset.ids)
    np.testing.assert_array_equal(keys, known.uq_idxs)
    assert not set(ids) & set(keys) and set(labels) == set(range(98))
    x = torch.cat([*parts, clip])
    targets = np.r_[labels, np.full(len(ids), -1)]
    pred = np.stack(
        [spherical(x, len(x), targets, s, True)[len(keys) :] for s in SEEDS]
    )
    path = Path("dev_outputs/d1-pair-diagnostics/clip-pred.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, pred=pred)
    from scripts.audit_r2_evidence import sealed_labels

    y = sealed_labels()[ids]
    rows = [list(split_cluster_acc_v2(y, p, y < 98)) for p in pred]
    write_json(
        Path("reports/D1_CLIP_SEMI_KMEANS.json"),
        {
            "seeds": SEEDS,
            "max_iter": MAX_ITER,
            "legal": len(keys),
            "unlabeled": len(ids),
            "split": protocol["split"],
            "clip_sha": receipt["sha256"],
            "pred_sha": g6.file_digest(path),
            "v2_all_old_new": rows,
            "mean": np.mean(rows, axis=0).tolist(),
        },
    )
    print(rows, flush=True)


if __name__ == "__main__":
    main()
