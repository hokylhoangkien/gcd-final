"""Prepare the pinned official DINO code and checksum-verified weights on GPU host."""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path

from .common import DINO_REVISION, DINO_SHA256, DINO_URL, digest


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "dino"
    if not repo.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/facebookresearch/dino.git",
                str(repo),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "checkout", "--detach", DINO_REVISION], check=True
        )
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != DINO_REVISION:
        raise ValueError("DINO source revision differs; use a new assets directory")
    weight = root / "dino_vitbase16_pretrain.pth"
    if not weight.exists():
        temp = weight.with_suffix(".part")
        print("Downloading official DINO ViT-B/16 weights", flush=True)
        with (
            urllib.request.urlopen(DINO_URL, timeout=60) as response,
            temp.open("wb") as stream,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        if digest(temp) != DINO_SHA256:
            raise ValueError("Downloaded DINO weight checksum mismatch")
        temp.replace(weight)
    if digest(weight) != DINO_SHA256:
        raise ValueError("DINO weight checksum mismatch")
    print("Pinned DINO assets verified", flush=True)


def backbone(root):
    import torch

    repo, weight = root / "dino", root / "dino_vitbase16_pretrain.pth"
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != DINO_REVISION or digest(weight) != DINO_SHA256:
        raise ValueError(
            "Run python -m planogram_rpc.assets first; assets differ from pinned source/weights"
        )
    model = torch.hub.load(
        str(repo.resolve()), "dino_vitb16", source="local", pretrained=False
    )
    model.load_state_dict(
        torch.load(weight, map_location="cpu", weights_only=True), strict=True
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("assets_local"))
    prepare(parser.parse_args().root)
