"""Training datasets expose only legal known labels; gold labels stay in evaluation."""

from __future__ import annotations

import json
import math
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

from data.data_utils import MergedDataset
from model import ContrastiveLearningViewGenerator

from .common import image_path


def transforms():
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # Preserve printed brand/text direction: unlike Cars, no horizontal mirror.
    train = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(224),
            T.ToTensor(),
            normalize,
        ]
    )
    evaluate = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]
    )
    return ContrastiveLearningViewGenerator(train, 2), evaluate


class Images(Dataset):
    def __init__(self, root, rows, transform, *, gold=False):
        self.root, self.transform = root, transform
        # Do not retain a gold-target array in any unlabeled training dataset.
        self.rows = [
            (r["path"], r["target"] if gold or r["labeled"] else -1, r["id"])
            for r in rows
        ]
        self.uq_idxs = np.array([r[2] for r in self.rows], dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, target, key = self.rows[index]
        with Image.open(image_path(self.root, path)) as image:
            x = self.transform(image.convert("RGB"))
        return x, target, key


def training_data(root, manifest):
    train, evaluate = transforms()
    rows = [r for r in manifest["images"] if r["split"] == "train"]
    labeled = Images(root, [r for r in rows if r["labeled"]], train)
    unlabeled = Images(root, [r for r in rows if not r["labeled"]], train)
    if not len(labeled) or not len(unlabeled):
        raise ValueError("Both legal labeled and unlabeled training pools are required")
    return MergedDataset(labeled, unlabeled), evaluate


def load_manifest(root):
    return json.loads((root / "manifest.json").read_text())


class BalancedBatches(Sampler):
    """Same 50:50 expected mix as upstream; ensure legal CE targets in every batch.

    Epoch length approximates the upstream weighted sampler, without a partial
    final batch. Sampling is deterministic by seed/epoch and uses replacement.
    """

    def __init__(self, labeled, unlabeled, batch_size, seed):
        if batch_size < 4 or batch_size % 2:
            raise ValueError("Use an even batch size >=4")
        self.labeled, self.unlabeled, self.batch_size = labeled, unlabeled, batch_size
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return max(1, math.ceil((self.labeled + self.unlabeled) / self.batch_size))

    def __iter__(self):
        rng = random.Random(self.seed + 1000003 * self.epoch)
        half = self.batch_size // 2
        for _ in range(len(self)):
            batch = [rng.randrange(self.labeled) for _ in range(half)]
            batch += [self.labeled + rng.randrange(self.unlabeled) for _ in range(half)]
            rng.shuffle(batch)
            yield batch
