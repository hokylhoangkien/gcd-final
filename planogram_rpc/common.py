"""Portable paths, provenance and atomic artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DINO_REVISION = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
DINO_URL = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
DINO_SHA256 = "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e"
PREPROCESSING = "simgcd-rpc-center-224-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temp, path)


def image_path(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Image path escapes the dataset root")
    return result
