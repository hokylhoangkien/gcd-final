"""Verified image-only CLIP embeddings for the locked Cars split."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from crish.data import ImageOnly
from crish.experiment import file_digest

WEIGHTS = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
AUDITED = "13cc941ecfc2539fd5b2d9ba19c9fdfe3d271df1380c4931e5cccc91a46866a5"
SPLIT = "2f2b8a435f8caf5d2b7886a410484d25f8104b53f031a1d4aca13aba524931ce"
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
PREPROCESS = {
    "name": "OpenAI CLIP ViT-B/16",
    "weights_sha256": WEIGHTS,
    "short_side": 256,
    "crop": [224, 224],
    "interpolation": "bicubic",
    "color": "RGB",
    "mean": list(MEAN),
    "std": list(STD),
    "output": "L2-normalized image projection, float32",
    "encoder": "official TorchScript encode_image",
}


def check_split(blind, split):
    if not isinstance(blind, ImageOnly):
        raise TypeError("D1 requires the ImageOnly unlabeled adapter")
    ids = np.asarray(blind.ids, dtype=np.int64)
    if split != SPLIT or len(ids) != 6144 or len(np.unique(ids)) != len(ids):
        raise ValueError("Unexpected Cars seed-1 SSB split fingerprint or IDs")
    return ids


def image_digest(blind):
    h = hashlib.sha256()
    for key, p in zip(blind.ids, blind.images, strict=True):
        h.update(np.asarray([key], dtype="<i8").tobytes())
        h.update(bytes.fromhex(file_digest(p)))
    return h.hexdigest()


def load(path, blind, split, device):
    ids = check_split(blind, split)
    digest = file_digest(path)
    with np.load(path, allow_pickle=False) as f:
        if not np.array_equal(ids, f["ids"]):
            raise ValueError("CLIP cache ID ordering differs from training")
        meta = json.loads(f["provenance"].item())
        x = f["features"]
    if not all(meta.get(k) == v for k, v in PREPROCESS.items()):
        raise ValueError("CLIP weights or preprocessing provenance mismatch")
    if digest != AUDITED and (
        meta.get("split_sha256") != split
        or meta.get("image_sha256") != image_digest(blind)
    ):
        raise ValueError("Locally encoded CLIP cache split or image bytes differ")
    if x.shape != (len(ids), 512) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Malformed CLIP embedding bank")
    if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-5, rtol=0):
        raise ValueError("Expected unit CLIP embeddings")
    receipt = {
        "sha256": digest,
        "split_sha256": split,
        "ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "provenance": meta,
        "audited_without_image_hash": digest == AUDITED,
    }
    return torch.from_numpy(x).to(device), receipt


@torch.inference_mode()
def encode(path, weights, blind, split, device):
    import copy

    ids = check_split(blind, split)
    if file_digest(weights) != WEIGHTS:
        raise ValueError("Expected official OpenAI CLIP ViT-B/16 weights")
    ds = copy.copy(blind)
    ds.transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    net = torch.jit.load(str(weights), map_location=device).eval()
    assert net.visual.conv1.weight.shape == (
        768,
        3,
        16,
        16,
    ) and net.visual.proj.shape == (768, 512)
    rows, seen = [], []
    for images, legal, keys in DataLoader(ds, batch_size=16):
        if (legal != -1).any():
            raise ValueError("Label present in CLIP cache builder")
        seen.extend(keys.tolist())
        rows.append(
            F.normalize(net.encode_image(images.to(device)).float(), dim=1).cpu()
        )
    np.testing.assert_array_equal(seen, ids)
    meta = {
        **PREPROCESS,
        "split_sha256": split,
        "image_sha256": image_digest(ds),
        "torch": str(torch.__version__),
        "torchvision": torchvision.__version__,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        ids=ids,
        features=torch.cat(rows).numpy(),
        provenance=json.dumps(meta, sort_keys=True),
    )
    os.replace(tmp, path)
    return load(path, blind, split, device)
