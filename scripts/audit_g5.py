"""G5 multi-backbone Cars geometry with a frozen DINOv2 ViT-B/14 bank."""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest
from relational.state import write_json
from relational.train import build
from scripts import audit_g1 as g1
from scripts import audit_g4 as g4
from scripts.audit_e3_proposals import json_native
from scripts.audit_g3 import COVERAGES, G1_REPORT
from scripts.audit_g7 import pack

LOCAL = g1.WORK / "g5"
WEIGHT = "dinov2_vitb14_pretrain.pth"
WEIGHT_SHA = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
URL = f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/{WEIGHT}"
METHODS = ("M0", "M1", "M2", "M3", "M4", "M5")
MD = Path("reports/CRISH_MULTIBACKBONE_GEOMETRY.md")
OUT = Path("reports/r2_multibackbone_geometry.json")
BATCH = 16


def backbone_inputs():
    hub = Path(torch.hub.get_dir())
    repo = Path(
        os.environ.get("SIMGCD_DINOV2_REPO", hub / "facebookresearch_dinov2_main")
    )
    candidates = [hub / "checkpoints" / WEIGHT, LOCAL / WEIGHT]
    if "SIMGCD_DINOV2_WEIGHTS" in os.environ:
        candidates.insert(0, Path(os.environ["SIMGCD_DINOV2_WEIGHTS"]))
    for path in candidates:
        if path.is_file() and file_digest(path) == WEIGHT_SHA:
            break
    else:
        raise ValueError("No ViT-B/14 checkpoint matching the locked release SHA256")
    digest = hashlib.sha256()
    for f in sorted([repo / "hubconf.py", *repo.joinpath("dinov2").rglob("*.py")]):
        digest.update(f.relative_to(repo).as_posix().encode())
        digest.update(bytes.fromhex(file_digest(f)))
    return repo, path, digest.hexdigest()


def describe_transform(center):
    resize, crop, _, norm = center.transforms
    assert (resize.size, tuple(crop.size)) == (256, (224, 224))
    assert resize.interpolation.value == "bicubic"
    return {
        "short_side": resize.size,
        "crop": list(crop.size),
        "interpolation": resize.interpolation.value,
        "mean": norm.mean.tolist(),
        "std": norm.std.tolist(),
        "color": "RGB",
        "torchvision": torchvision.__version__,
    }


@torch.inference_mode()
def features(ids, smoke=False):
    repo, weights, source_hash = backbone_inputs()
    baseline, loader, _, center, _, _ = build()
    del baseline
    ds = copy.copy(loader.dataset.unlabelled_dataset)
    ds.transform = center
    np.testing.assert_array_equal(ids, ds.ids)
    expected = {
        "weights_sha256": WEIGHT_SHA,
        "upstream_url": URL,
        "source_sha256": source_hash,
        "preprocessing": describe_transform(center),
        "rows": len(ids),
        "feature": "L2-normalized final CLS",
        "torch": torch.__version__,
    }
    cached = LOCAL / "dinov2-cls.npz"
    if cached.is_file() and not smoke:
        with np.load(cached, allow_pickle=False) as f:
            np.testing.assert_array_equal(ids, f["ids"])
            if json.loads(f["provenance"].item()) != expected:
                raise ValueError("DINOv2 cache provenance mismatch")
            x = f["features"]
        assert (
            x.shape == (len(ids), 768)
            and x.dtype == np.float32
            and np.isfinite(x).all()
        )
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=2e-5)
        return torch.from_numpy(x).cuda(), {
            **expected,
            "cache_sha256": file_digest(cached),
            "reused": True,
        }

    model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=False)
    state = torch.load(weights, weights_only=True, map_location="cpu")
    assert tuple(state["patch_embed.proj.weight"].shape) == (768, 3, 14, 14)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False).cuda()
    data = Subset(ds, list(range(BATCH))) if smoke else ds
    outputs, offset = [], 0
    for images, y, keys in DataLoader(data, batch_size=BATCH, pin_memory=True):
        assert (y == -1).all()
        h = F.normalize(model.forward_features(images.cuda())["x_norm_clstoken"], dim=1)
        assert h.shape == (len(images), 768) and torch.isfinite(h).all()
        end = offset + len(images)
        np.testing.assert_array_equal(ids[offset:end], keys.numpy())
        outputs.append(h.cpu())
        offset = end
        if offset % 1024 == 0:
            print("DINOv2 cache rows", offset, flush=True)
    x = torch.cat(outputs).numpy()
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=2e-5)
    if smoke:
        assert offset == BATCH
        print(
            "Verified strict ViT-B/14 weights, real Cars center crops and finite normalized CLS",
            flush=True,
        )
        return None, expected
    assert offset == len(ids)
    temp = cached.with_suffix(".tmp.npz")
    np.savez(temp, ids=ids, features=x, provenance=json.dumps(expected, sort_keys=True))
    os.replace(temp, cached)
    return torch.from_numpy(x).cuda(), {
        **expected,
        "cache_sha256": file_digest(cached),
        "reused": False,
    }


def freeze(pre, ft, v2, records, history):
    ids, q, bank, slots = (records[k] for k in ("ids", "q", "bank", "slots"))
    inv = np.full(len(ids), -1)
    inv[bank] = np.arange(len(bank))
    pairs = inv[records["pairs"]]
    assert (pairs >= 0).all()
    current, original, other = (x[q] @ x[bank].T for x in (ft, pre, v2))
    out = {k: records[k] for k in ("ids", "q", "bank", "slots", "pairs")}
    for name, sim in zip(
        METHODS[:4], (current, original, other, (current + other) / 2), strict=True
    ):
        p = pack(sim, q, bank, slots, pairs)
        if name == "M0":
            np.testing.assert_array_equal(p["specific_ids"], records["pairs"][:, 256:])
            for k in ("dest", "neighbors"):
                np.testing.assert_array_equal(p[k], records[f"Current_{k}"])
            for k in ("wrong", "pair"):
                np.testing.assert_allclose(
                    p[k], records[f"Current_{k}"], atol=3e-6, rtol=3e-6
                )
        out.update({f"{name}_{k}": p[k] for k in g4.FIELDS})
    out["backbone_disagreement"] = g4.overlap_risk(
        out["M0_neighbors"], out["M2_neighbors"]
    )
    for name, signal in (("M4", out["M2_wrong"]), ("M5", out["backbone_disagreement"])):
        for k in g4.FIELDS:
            out[f"{name}_{k}"] = signal if k == "wrong" else out[f"M0_{k}"]
    if history is not None:
        for k in ("ids", "q", "bank"):
            np.testing.assert_array_equal(out[k], history[k])
        out["history_slots"] = history["slots"]
        mapping, _ = g4.slot_identity(history["slots"], slots, bank, q)
        out["aligned_slot_stable"] = mapping[history["slots"][q]] == slots[q]
    g4.freeze_flags(out, METHODS)
    return out


def report(result):
    lines = [
        "# G5 multi-backbone geometry on Stanford Cars",
        "",
        (
            "This exploratory analysis reuses the G1 query and bank IDs, common pair pools and "
            "checkpoint-specific classifier slots. DINOv2 ViT-B/14 uses the same RGB images, "
            "bicubic short-edge resize to 256 and center crop to 224 as DINOv1. Both use "
            "ImageNet RGB normalization. DINOv2 final CLS is L2-normalized and cached once "
            "for both checkpoints. JSON records the official release URL, locked SHA256, "
            "local source digest and feature-cache provenance."
        ),
        "",
        (
            "M0 uses fine-tuned DINOv1 cosine. M1 uses frozen pretrained DINOv1 cosine. "
            "M2 uses frozen DINOv2 cosine. M3 averages the M0 and M2 cosine matrices at equal "
            "weight. M4 uses the M2 alternative-minus-current slot support margin for detection. "
            "M5 uses one minus top-10 neighborhood Jaccard agreement between M0 and M2. "
            "M4 and M5 keep M0 retrieval."
        ),
        "",
        (
            "Flags use fixed global 10/20/40% coverage with query-ID tie breaking. Old/New "
            "precision and recall are conditional on the same global flags. Destinations use "
            "bank-only Hungarian semantic alignment. Recovery and regression are versus M0 "
            "top-1 on wrong-source queries. For epoch 150, stable errors are wrong under both "
            "checkpoint alignments with unchanged slots after label-blind bank matching, as in G4. "
            "Stable-error AUROC includes all other queries as negatives. JSON also gives error "
            "AUROC restricted to unchanged slots and stable-error recall at fixed coverage."
        ),
        "",
        "| Epoch / group / method | Purity@1/5/10 | Pair / hard AUC | Error AUC | P/R @10% | P/R @20% | P/R @40% | Dest R@1/3/5 | Recovery / regression | Stable error AUC |",
        "|---|---|---|---:|---|---|---|---|---|---:|",
    ]

    def f(x):
        return "—" if x is None else f"{x:.3f}"

    for epoch, snapshot in result["snapshots"].items():
        for group in ("All", "Old", "New"):
            for name in METHODS:
                m = snapshot["metrics"][name][group]
                purity = "/".join(f(m["purity"][k]) for k in (1, 5, 10))
                dest = "/".join(f(m["destination_recall"][k]) for k in (1, 3, 5))
                flags = [
                    " / ".join(f(m["flags"][p][k]) for k in ("precision", "recall"))
                    for p in COVERAGES
                ]
                lines.append(
                    f"| {epoch} / {group} / {name} | {purity} | {f(m['pair_auc'])} / "
                    f"{f(m['hard_auc'])} | {f(m['detection_auc'])} | "
                    + " | ".join(flags)
                    + f" | {dest} | {f(m['failure_recovery'][1])} / {f(m['correct_to_wrong_vs_final'])} | "
                    f"{f(m['stable_error_auc'])} |"
                )
    lines += [
        "",
        f"Runtime {result['seconds']:.1f} seconds.",
        "",
        "[Counts, subgroup coverage and frozen prediction provenance](r2_multibackbone_geometry.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def main(smoke):
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    ids = np.load(g1.WORK / "ids.npy")
    v2, v2_meta = features(ids, smoke=smoke)
    if smoke:
        return
    info = json.loads(G1_REPORT.read_text())
    feature_path = g1.WORK / "features.npy"
    cache = np.load(feature_path, mmap_mode="r")
    assert cache.shape == (3, 3, len(ids), 768)
    pre = F.normalize(torch.from_numpy(cache[0, 0].copy()), dim=1).cuda()
    result = {
        "base": "d9e37ce",
        "exploratory": True,
        "dinov2": v2_meta,
        "coverage_percent": COVERAGES,
        "g1_sha256": file_digest(G1_REPORT),
        "dinov1_cache_sha256": file_digest(feature_path),
        "snapshots": {},
    }
    history = None
    for index, epoch in enumerate(g1.EPOCHS, 1):
        key = str(epoch)
        records = g4.snapshot(epoch, info)
        np.testing.assert_array_equal(records["ids"], ids)
        ft = F.normalize(torch.from_numpy(cache[index, 0].copy()), dim=1).cuda()
        out = freeze(pre, ft, v2, records, history)
        history = records
        path = LOCAL / f"frozen-{epoch}.npz"
        np.savez(path, **out)
        record = info["snapshots"][key]
        result["snapshots"][key] = {
            "checkpoint_sha256": record["checkpoint_sha256"],
            "g1_manifest_sha256": record["frozen_sha256"],
            "frozen_sha256": file_digest(path),
        }
        print("Frozen G5", epoch, flush=True)
    write_json(LOCAL / "frozen-provenance.json", json_native(result))
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    for key, snapshot in result["snapshots"].items():
        path = LOCAL / f"frozen-{key}.npz"
        assert file_digest(path) == snapshot["frozen_sha256"]
        with np.load(path, allow_pickle=False) as f:
            metrics, alignment = g4.sealed(dict(f), truth, METHODS, baseline="M0")
        snapshot.update(metrics=metrics, bank_alignment=alignment)
    result["seconds"] = time.monotonic() - start
    report(result)
    write_json(OUT, json_native(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
