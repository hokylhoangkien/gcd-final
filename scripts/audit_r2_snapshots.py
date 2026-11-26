"""Verify the R2 archive and encode three label-blind snapshots with a shared frozen prefix."""

import argparse
import copy
import hashlib
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crish.experiment import file_digest, load_checkpoint
from crish.global_repair import frozen
from relational.export import verify
from relational.state import write_json
from relational.train import build

COMMIT = "c87de5e81391720c8e020256bed72d843584c0cb"


def extract(bundle, folder):
    verify(bundle)
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:") as archive:
        if not all(m.isfile() and m.name == Path(m.name).name for m in archive):
            raise ValueError("Expected flat regular archive members")
        archive.extractall(folder, filter="data")


def run(bundle, work):
    folder = work / "input"
    extract(bundle, folder)
    config = json.loads((folder / "config.json").read_text())
    p = config["protocol"]
    assert config["variant"] == "R2" and p["commit"] == COMMIT and p["seed"] == 1
    model, train, _, center, _, local = build()
    diffs = {k: [p[k], local[k]] for k in p if local[k] != p[k]}
    assert not (set(diffs) - {"commit", "dino_source"})
    normalized = (
        Path(os.environ["SIMGCD_DINO_REPO"], "vision_transformer.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    )
    assert hashlib.sha256(normalized).hexdigest() == p["dino_source"]
    backbone = model[0].eval()
    blocks = [backbone.blocks[-1]]
    journals = {
        k: (folder / k).read_bytes().splitlines(keepends=True)
        for k in ("metrics.jsonl", "relationships.jsonl")
    }
    provenance = {
        "archive_sha256": file_digest(bundle),
        "members": (folder / "SHA256SUMS").read_text().splitlines(),
        "config": config,
        "local_differences": diffs,
        "checkpoints": {},
    }
    for epoch in (100, 150):
        state = load_checkpoint(folder / f"epoch-{epoch}.pt")
        assert state["identity"] == config and state["epoch"] == epoch
        assert (
            state["scheduler"]["last_epoch"] == epoch
            and state["scheduler"]["T_max"] == 200
        )
        for k, lines in journals.items():
            assert (
                len(lines) == 150
                and sum(map(len, lines[:epoch])) == state["journal_bytes"][k]
            )
        own = model.state_dict()
        assert state["model"].keys() == own.keys()
        assert all(
            torch.equal(v.cpu(), state["model"][k])
            for k, v in own.items()
            if k.startswith("0.") and not k.startswith("0.blocks.11.")
        )
        block = copy.deepcopy(blocks[0])
        block.load_state_dict(
            {
                k.removeprefix("0.blocks.11."): v
                for k, v in state["model"].items()
                if k.startswith("0.blocks.11.")
            }
        )
        blocks.append(block)
        provenance["checkpoints"][epoch] = {
            "optimizer_entries": len(state["optimizer"]["state"]),
            "next_lr": state["optimizer"]["param_groups"][0]["lr"],
            "journal_bytes": state["journal_bytes"],
            "rng_components": len(state["rng"]),
            "frozen_backbone_verified": True,
        }
        del state
    write_json(work / "provenance.json", provenance)
    blind = train.dataset.unlabelled_dataset
    center_data = copy.copy(blind)
    center_data.transform = center
    assert len(blind) == 6144
    store = np.lib.format.open_memmap(
        work / "features.npy", mode="w+", dtype="float32", shape=(3, 3, len(blind), 768)
    )
    np.save(work / "ids.npy", blind.ids)
    with torch.no_grad(), frozen(backbone, 2079, torch.device("cuda:0")):
        for centered, ds in ((True, center_data), (False, blind)):
            offset = 0
            for views, sentinel, ids in DataLoader(ds, batch_size=8):
                assert (sentinel == -1).all() and np.array_equal(
                    ids.numpy(), blind.ids[offset : offset + len(ids)]
                )
                for view, images in enumerate([views] if centered else views):
                    x = backbone.prepare_tokens(images.cuda())
                    for layer in list(backbone.blocks)[:-1]:
                        x = layer(x)
                    for n, block in enumerate(blocks):
                        store[
                            n, 0 if centered else 1 + view, offset : offset + len(ids)
                        ] = backbone.norm(block(x))[:, 0].cpu().numpy()
                offset += len(ids)
                if offset % 1024 == 0:
                    print("center" if centered else "crops", offset, flush=True)
            assert offset == len(blind)
    store.flush()
    print("Snapshots complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("work", type=Path)
    args = p.parse_args()
    run(args.archive, args.work)
