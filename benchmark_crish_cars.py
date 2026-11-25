"""Bounded global CRiSH refresh benchmark using the official Cars training split."""

import argparse
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from crish.runtime import arguments, global_defaults


def main():
    p = argparse.ArgumentParser(description=__doc__)
    arguments(p)
    global_defaults(p)
    p.set_defaults(crish_mode="E3", run_seed=1)
    p.add_argument("--cars", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument(
        "--checkpoint", type=Path, help="Official epoch-boundary SimGCD model/optimizer"
    )
    p.add_argument("--refreshes", type=int, choices=range(1, 6), default=3)
    p.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Real training batches per benchmark refresh",
    )
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()
    if args.crish_mode != "E3" or args.steps < 1 or args.crish_every != 1:
        p.error(
            "Benchmark requires E3 with refresh every epoch and positive training exposure"
        )
    import config

    config.car_root = str(args.cars.resolve())
    from crish.data import ImageOnly
    from data.augmentations import get_transform
    from data.get_datasets import get_class_splits, get_datasets
    from model import ContrastiveLearningViewGenerator, DINOHead
    from train import train

    random.seed(args.run_seed)
    np.random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_path = str(args.output / "model.pt")
    args.dataset_name, args.use_ssb_splits, args.prop_train_labels = "scars", True, 0.5
    args = get_class_splits(args)
    args.epochs, args.batch_size, args.n_views = 200, 128, 2
    args.lr, args.weight_decay, args.momentum = 0.1, 5e-5, 0.9
    args.sup_weight, args.memax_weight = 0.35, 1
    args.warmup_teacher_temp, args.teacher_temp, args.warmup_teacher_temp_epochs = (
        0.07,
        0.04,
        30,
    )
    args.interpolation, args.crop_pct, args.fp16 = 3, 0.875, False
    args.logger, args.print_freq = logging.getLogger("benchmark"), 1
    args.benchmark_refreshes, args.benchmark_steps = args.refreshes, args.steps
    args.benchmark_epoch = args.crish_warmup
    if args.dino_repo:
        if not args.dino_weights:
            p.error("Local DINO source requires the pretrained weights")
        backbone = torch.hub.load(
            str(args.dino_repo.resolve()),
            "dino_vitb16",
            source="local",
            pretrained=False,
        )
        backbone.load_state_dict(torch.load(args.dino_weights, map_location="cpu"))
    else:
        backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    for name, param in backbone.named_parameters():
        param.requires_grad_("block" in name and int(name.split(".")[1]) >= 11)
    trans, center = get_transform("imagenet", image_size=224, args=args)
    dataset, _, _, _ = get_datasets(
        "scars", ContrastiveLearningViewGenerator(trans, 2), center, args
    )
    dataset.unlabelled_dataset = ImageOnly(dataset.unlabelled_dataset, "scars")
    n, m = len(dataset.labelled_dataset), len(dataset.unlabelled_dataset)
    weights = torch.DoubleTensor([1] * n + [n / m] * m)
    loader = DataLoader(
        dataset,
        batch_size=128,
        sampler=WeightedRandomSampler(weights, len(dataset)),
        num_workers=args.num_workers,
        drop_last=True,
    )
    model = nn.Sequential(
        backbone,
        DINOHead(768, len(args.train_classes) + len(args.unlabeled_classes), nlayers=3),
    ).cuda()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(saved["model"])
        args.benchmark_state = saved["optimizer"]
        args.benchmark_epoch = int(saved["epoch"])
        if (
            not args.crish_warmup
            <= args.benchmark_epoch
            <= args.epochs - args.refreshes
        ):
            p.error("Checkpoint epoch outside the eligible benchmark interval")
    print(
        f"Cars {n}+{m}, DINOv1 ViT-B/16, checkpoint={args.checkpoint}, "
        f"refreshes={args.refreshes}, official SGD batches per refresh={args.steps}",
        flush=True,
    )
    train(
        model,
        loader,
        None,
        SimpleNamespace(dataset=SimpleNamespace(transform=center)),
        args,
    )


if __name__ == "__main__":
    main()
