"""After training only: classifier, ONE SS-KMeans fit, and reference-image retrieval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from crish.experiment import load_checkpoint
from model import DINOHead

from .assets import backbone
from .common import digest, write_json
from .data import Images, load_manifest, transforms
from .metrics import cluster_accuracy, distances, retrieval, ss_kmeans


@torch.inference_mode()
def encode(model, root, rows, args):
    loader = DataLoader(
        Images(root, rows, transforms()[1], gold=True),
        batch_size=args.encode_batch,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    features, predictions = [], []
    model.eval()
    for images, _, _ in loader:
        x = model[0](images.to(args.device))
        features.append(F.normalize(x.float(), dim=1).cpu())
        predictions.append(model[1](x)[1].argmax(1).cpu())
    return torch.cat(features), torch.cat(predictions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("datasets/rpc40"))
    p.add_argument("--assets", type=Path, default=Path("assets_local"))
    p.add_argument("--run", type=Path, default=Path("runs/rpc40-e3-s1"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ss-inits", type=int, default=10)
    p.add_argument("--ss-max-iter", type=int, default=100)
    p.add_argument("--gallery-per-class", type=int, default=5)
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Diagnostic only, explicitly mark incomplete training",
    )
    args = p.parse_args()
    checkpoint = args.run / "last.pt"
    saved = load_checkpoint(checkpoint)
    manifest = load_manifest(args.data)
    if saved["identity"]["manifest_sha256"] != digest(args.data / "manifest.json"):
        raise ValueError("Evaluation manifest differs from training")
    completed = saved["epoch"] == saved["identity"]["settings"]["epochs"]
    if not completed and not args.allow_partial:
        raise ValueError("Training is incomplete; resume it before final evaluation")
    if args.gallery_per_class < 1:
        p.error("--gallery-per-class must be positive")
    model = nn.Sequential(
        backbone(args.assets), DINOHead(768, len(manifest["classes"]))
    ).to(args.device)
    model.load_state_dict(saved["model"])
    rows = {
        s: [r for r in manifest["images"] if r["split"] == s]
        for s in ("train", "val", "test")
    }
    features, classifier = {}, {}
    for split in rows:
        print(
            f"Extracting trained features: {split}, {len(rows[split])} images",
            flush=True,
        )
        features[split], classifier[split] = encode(model, args.data, rows[split], args)
    known = manifest["known_count"]
    # This tensor contains only the labels that were legally available in training.
    legal = torch.tensor([r["target"] if r["labeled"] else -1 for r in rows["train"]])
    print(
        "Final-only SS-KMeans fit on training embeddings (no test images or novel gold labels)",
        flush=True,
    )
    ss = ss_kmeans(
        features["train"].to(args.device),
        legal,
        len(manifest["classes"]),
        seed=saved["identity"]["settings"]["seed"],
        n_init=args.ss_inits,
        max_iter=args.ss_max_iter,
    )
    report = {
        "checkpoint_sha256": digest(checkpoint),
        "epoch": saved["epoch"],
        "training_complete": completed,
        "manifest_sha256": digest(args.data / "manifest.json"),
        "ss_solver": {
            "type": "Euclidean constrained Lloyd on L2-normalized backbone features",
            "n_init": args.ss_inits,
            "max_iter": args.ss_max_iter,
            "inertia": ss["inertia"],
            "iterations": ss["iterations"],
            "note": "Custom RPC holdout and solver; not a direct reproduction of HypCD benchmark numbers.",
        },
        "classifier": {},
        "ss_kmeans": {},
        "retrieval": {},
    }
    for split in ("val", "test"):
        gold = [r["target"] for r in rows[split]]
        report["classifier"][split] = cluster_accuracy(
            gold, classifier[split].numpy(), known
        )
        predicted = (
            distances(features[split].to(args.device), ss["centers"])
            .argmin(1)
            .cpu()
            .numpy()
        )
        report["ss_kmeans"][split] = cluster_accuracy(gold, predicted, known)
    unlabeled = legal < 0
    report["ss_kmeans"]["train_unlabeled_transductive"] = cluster_accuracy(
        [r["target"] for r in rows["train"] if not r["labeled"]],
        ss["labels"].cpu()[unlabeled].numpy(),
        known,
    )
    groups = defaultdict(list)
    for index, row in enumerate(rows["train"]):
        groups[row["target"]].append(index)
    gallery = [
        i
        for k in sorted(groups)
        for i in sorted(groups[k], key=lambda j: rows["train"][j]["id"])[
            : args.gallery_per_class
        ]
    ]
    gallery_targets = [rows["train"][i]["target"] for i in gallery]
    report["gallery"] = {
        "ids": [rows["train"][i]["id"] for i in gallery],
        "per_class": args.gallery_per_class,
        "policy": "Fixed train-only references for ALL SKUs, used only in evaluation, never for training or checkpoint selection",
    }
    for split in ("val", "test"):
        report["retrieval"][split], _ = retrieval(
            features["train"][gallery],
            gallery_targets,
            features[split],
            [r["target"] for r in rows[split]],
            known_count=known,
        )
    # Compare the exact same references/queries against untouched pretrained DINO.
    model[0] = backbone(args.assets).to(args.device)
    base_features = {}
    for split in rows:
        print(f"Extracting pretrained reference features: {split}", flush=True)
        base_features[split], _ = encode(model, args.data, rows[split], args)
    report["pretrained_retrieval"] = {}
    for split in ("val", "test"):
        report["pretrained_retrieval"][split], _ = retrieval(
            base_features["train"][gallery],
            gallery_targets,
            base_features[split],
            [r["target"] for r in rows[split]],
            known_count=known,
        )
    report["limitations"] = [
        "Single-product RPC crops; evaluate local shelf crops separately before deployment.",
        "Exact crop duplicates removed; correlated views of the same physical pack can remain.",
        "SS-KMeans accuracy and retrieval accuracy measure different prediction mechanisms.",
        "Reference labels for novel SKUs are used only for post-training retrieval evaluation.",
    ]
    output = args.run / "evaluation"
    write_json(output / "metrics.json", report)
    torch.save(
        {"centers": ss["centers"].cpu(), "manifest_sha256": report["manifest_sha256"]},
        output / "ss_centers.pt",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
