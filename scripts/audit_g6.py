"""G6 image-only CLIP ViT-B/16 and frozen DINO geometry on Cars."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from crish.experiment import file_digest
from relational.state import write_json
from relational.train import build
from scripts import audit_g1 as g1
from scripts import audit_g4 as g4
from scripts import audit_g5 as g5
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_g3 import COVERAGES, G1_REPORT
from scripts.audit_g7 import pack

LOCAL = g1.WORK / "g6"
G5_REPORT = Path("reports/r2_multibackbone_geometry.json")
MD = Path("reports/CRISH_VLM_GEOMETRY.md")
OUT = Path("reports/r2_vlm_geometry.json")
SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
WEIGHT = "ViT-B-16.pt"
URL = f"https://openaipublic.azureedge.net/clip/models/{SHA}/{WEIGHT}"
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
BATCH = 16
RETRIEVERS = ("D1", "D1Frozen", "D2", "D1+D2", "CLIP", "D1+CLIP", "D2+CLIP", "All3")
METHODS = (*RETRIEVERS, "CLIPRisk", "Disagreement")


def weights():
    paths = [
        Path.home() / ".cache/clip" / WEIGHT,
        Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT,
        LOCAL / WEIGHT,
    ]
    if "SIMGCD_CLIP_WEIGHTS" in os.environ:
        paths.insert(0, Path(os.environ["SIMGCD_CLIP_WEIGHTS"]))
    for path in paths:
        if path.is_file() and file_digest(path) == SHA:
            return path
    raise ValueError("No weights matching OpenAI CLIP ViT-B/16 SHA256")


@torch.inference_mode()
def extract(ids, smoke):
    path = weights()
    model, loader, _, center, _, _ = build()
    del model
    ds = copy.copy(loader.dataset.unlabelled_dataset)
    np.testing.assert_array_equal(ds.ids, ids)
    transforms_v1 = center.transforms
    assert transforms_v1[0].size == 256 and tuple(transforms_v1[1].size) == (224, 224)
    assert transforms_v1[0].interpolation.value == "bicubic"
    ds.transform = transforms.Compose(
        [*transforms_v1[:3], transforms.Normalize(MEAN, STD)]
    )
    meta = {
        "name": "OpenAI CLIP ViT-B/16",
        "weights_sha256": SHA,
        "upstream_url": URL,
        "rows": len(ids),
        "short_side": 256,
        "crop": [224, 224],
        "interpolation": "bicubic",
        "color": "RGB",
        "mean": list(MEAN),
        "std": list(STD),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "output": "L2-normalized image projection, float32",
        "encoder": "official TorchScript encode_image",
    }
    cache = LOCAL / "clip-image.npz"
    if cache.is_file() and not smoke:
        with np.load(cache, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["ids"], ids)
            if json.loads(f["provenance"].item()) != meta:
                raise ValueError("CLIP cache provenance mismatch")
            x = f["features"]
    else:
        model = torch.jit.load(str(path), map_location="cuda:0").eval()
        assert int(model.input_resolution) == 224
        state = model.state_dict()
        assert state["visual.conv1.weight"].shape == (768, 3, 16, 16)
        assert state["visual.proj"].shape == (768, 512)
        del state
        parts, offset = [], 0
        data = Subset(ds, list(range(BATCH))) if smoke else ds
        for pixels, legal, keys in DataLoader(data, batch_size=BATCH, pin_memory=True):
            assert (legal == -1).all()
            end = offset + len(pixels)
            np.testing.assert_array_equal(keys.numpy(), ids[offset:end])
            parts.append(
                F.normalize(model.encode_image(pixels.cuda()).float(), dim=1).cpu()
            )
            offset = end
            if offset % 1024 == 0:
                print("CLIP cache rows", offset, flush=True)
        x = torch.cat(parts).numpy()
    assert x.shape == (BATCH if smoke else len(ids), 512) and x.dtype == np.float32
    assert np.isfinite(x).all()
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=2e-5)
    if smoke:
        print(
            "CLIP architecture, official SHA256 and first Cars batch verified",
            flush=True,
        )
        return None, meta
    if not cache.exists():
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, ids=ids, features=x, provenance=json.dumps(meta, sort_keys=True))
        os.replace(tmp, cache)
    return torch.from_numpy(x).cuda(), {**meta, "cache_sha256": file_digest(cache)}


def freeze(d1, pre, d2, clip, rec, past):
    ids, q, bank, slots = (rec[k] for k in ("ids", "q", "bank", "slots"))
    inv = np.full(len(ids), -1)
    inv[bank] = np.arange(len(bank))
    pairs = inv[rec["pairs"]]
    assert (pairs >= 0).all()
    a, p, b, c = (x[q] @ x[bank].T for x in (d1, pre, d2, clip))
    sims = (a, p, b, (a + b) / 2, c, (a + c) / 2, (b + c) / 2, (a + b + c) / 3)
    out = {k: rec[k] for k in ("ids", "q", "bank", "slots", "pairs")}
    for name, sim in zip(RETRIEVERS, sims, strict=True):
        r = pack(sim, q, bank, slots, pairs)
        if name == "D1":
            np.testing.assert_array_equal(r["specific_ids"], rec["pairs"][:, 256:])
            for k in ("dest", "neighbors"):
                np.testing.assert_array_equal(r[k], rec[f"Current_{k}"])
            for k in ("wrong", "pair"):
                np.testing.assert_allclose(
                    r[k], rec[f"Current_{k}"], atol=3e-6, rtol=3e-6
                )
        out.update({f"{name}_{k}": r[k] for k in g4.FIELDS})
    out["backbone_disagreement"] = g4.overlap_risk(
        out["D1_neighbors"], out["CLIP_neighbors"]
    )
    for name, risk in (
        ("CLIPRisk", out["CLIP_wrong"]),
        ("Disagreement", out["backbone_disagreement"]),
    ):
        for k in g4.FIELDS:
            out[f"{name}_{k}"] = risk if k == "wrong" else out[f"D1_{k}"]
    if past is not None:
        for k in ("ids", "q", "bank"):
            np.testing.assert_array_equal(out[k], past[k])
        out["history_slots"] = past["slots"]
        mapping, _ = g4.slot_identity(past["slots"], slots, bank, q)
        out["aligned_slot_stable"] = mapping[past["slots"][q]] == slots[q]
    g4.freeze_flags(out, METHODS)
    return out


def actionable(out, truth):
    ids, q, bank, slots = (out[k] for k in ("ids", "q", "bank", "slots"))
    mp, _ = align(
        ids[bank].tolist(), dict(zip(ids.tolist(), slots.tolist(), strict=True)), truth
    )
    table = np.array([mp.get(k, -1) for k in range(196)] + [-1])
    y = truth[ids[q]]
    before = table[slots[q]] == y
    result = {}
    for det in METHODS:
        result[det] = {}
        for ret in RETRIEVERS:
            result[det][ret] = {}
            d = out[f"{ret}_dest"][:, 0]
            move = (d >= 0) & (d != slots[q])
            correct = table[d] == y
            for p in COVERAGES:
                indices = out[f"{det}_flag_{p}"]
                row = {}
                for group, mask in (("All", y >= 0), ("Old", y < 98), ("New", y >= 98)):
                    flags = indices[mask[indices]]
                    edits = flags[move[flags]]
                    repairs = int((~before[edits] & correct[edits]).sum())
                    harmful = int((before[edits] & ~correct[edits]).sum())
                    wrong = int((~before[edits] & ~correct[edits]).sum())
                    row[group] = {
                        "selected": len(flags),
                        "edits": len(edits),
                        "no_action": len(flags) - len(edits),
                        "correct_repairs": repairs,
                        "harmful_edits": harmful,
                        "wrong_to_wrong": wrong,
                        "correct_to_correct": len(edits) - repairs - harmful - wrong,
                        "net_correct": repairs - harmful,
                        "net_gain_pp": 100 * (repairs - harmful) / int(mask.sum())
                        if mask.any()
                        else None,
                    }
                result[det][ret][p] = row
    return result


def report(result):
    lines = [
        "# G6 image-only CLIP geometry on Stanford Cars",
        "",
        (
            "This exploratory comparison reuses G1 queries, bank and common pairs. Frozen "
            "OpenAI CLIP ViT-B/16 receives only RGB center crops and uses CLIP RGB normalization. "
            "The same bicubic 256-to-224 center crop geometry as DINO keeps image content aligned. "
            "CLIP runs once for both checkpoints. No categories, prompts, taxonomy or text embeddings "
            "enter the model or scoring rules."
        ),
        "",
        (
            "D1 denotes fine-tuned DINOv1. D1Frozen is the pretrained DINOv1 control. "
            "D2 denotes frozen DINOv2. Each fusion equally averages the indicated cosine matrices. "
            "CLIPRisk uses the CLIP alternative-minus-current slot margin. Disagreement uses one minus "
            "top-10 Jaccard similarity between D1 and CLIP. Both risk-only rules retrieve with D1."
        ),
        "",
        (
            "Flags target global coverage of 10%, 20% and 40%, rounded down with query-ID tie breaking. "
            "Subgroup metrics use the same global flags. Recovery and regression compare top-1 "
            "destinations on wrong sources against D1. Stable errors at 150 use G4's two-checkpoint "
            "definition, with label-blind bank matching of slots and bank-only semantic alignment."
        ),
        "",
        "| Epoch / group / method | Purity@1/5/10 | Pair / hard AUC | Error AUC | P/R @10% | P/R @20% | P/R @40% | Dest R@1/3/5 | Recovery / regression | Stable error AUC |",
        "|---|---|---|---:|---|---|---|---|---|---:|",
    ]

    def f(v):
        return "—" if v is None else f"{v:.3f}"

    for epoch, s in result["snapshots"].items():
        for group in ("All", "Old", "New"):
            for name in METHODS:
                m = s["metrics"][name][group]
                purity = "/".join(f(m["purity"][k]) for k in (1, 5, 10))
                dest = "/".join(f(m["destination_recall"][k]) for k in (1, 3, 5))
                flags = [
                    "/".join(f(m["flags"][p][k]) for k in ("precision", "recall"))
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
        "## Joint correction at 20% global coverage",
        "",
        (
            "Each cell gives correct repairs / harmful edits / wrong-to-wrong edits / net gain "
            "in percentage points across all queries. Each flagged query attempts one top-1 "
            "alternative-slot assignment. Missing alternatives abstain. This is an offline "
            "assignment counterfactual; it does not model an optimizer update or subsequent training."
        ),
        "",
        "| Epoch / detector | D1 | D1+D2 | D1+CLIP | D2+CLIP | All3 |",
        "|---|---|---|---|---|---|",
    ]
    for epoch, s in result["snapshots"].items():
        for det in METHODS:
            cells = []
            for ret in ("D1", "D1+D2", "D1+CLIP", "D2+CLIP", "All3"):
                v = s["actionable"][det][ret][20]["All"]
                cells.append(
                    f"{v['correct_repairs']} / {v['harmful_edits']} / {v['wrong_to_wrong']} / {v['net_gain_pp']:+.2f}"
                )
            lines.append(f"| {epoch} / {det} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Runtime {result['seconds']:.1f} seconds.",
        "",
        "[Full detector–retriever grid, subgroup results and cache provenance](r2_vlm_geometry.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def main(smoke):
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    ids = np.load(g1.WORK / "ids.npy")
    clip, meta = extract(ids, smoke)
    if smoke:
        return
    info = json.loads(G1_REPORT.read_text())
    g5_receipt = json.loads(G5_REPORT.read_text())
    assert file_digest(G1_REPORT) == g5_receipt["g1_sha256"]
    p = g5.LOCAL / "dinov2-cls.npz"
    v2_digest = file_digest(p)
    assert v2_digest == g5_receipt["dinov2"]["cache_sha256"]
    with np.load(p, allow_pickle=False) as v2:
        np.testing.assert_array_equal(v2["ids"], ids)
        d2 = torch.from_numpy(v2["features"]).cuda()
    path = g1.WORK / "features.npy"
    d1_digest = file_digest(path)
    assert d1_digest == g5_receipt["dinov1_cache_sha256"]
    cache = np.load(path, mmap_mode="r")
    assert cache.shape == (3, 3, len(ids), 768)
    pre = F.normalize(torch.from_numpy(cache[0, 0].copy()), dim=1).cuda()
    result = {
        "base": "668aa64",
        "exploratory": True,
        "clip": meta,
        "coverage_percent": COVERAGES,
        "g1_sha256": file_digest(G1_REPORT),
        "g5_sha256": file_digest(G5_REPORT),
        "dinov1_cache_sha256": d1_digest,
        "dinov2_cache_sha256": v2_digest,
        "snapshots": {},
    }
    past = None
    for i, epoch in enumerate(g1.EPOCHS, 1):
        k = str(epoch)
        cur = g4.snapshot(epoch, info)
        np.testing.assert_array_equal(cur["ids"], ids)
        ft = F.normalize(torch.from_numpy(cache[i, 0].copy()), dim=1).cuda()
        out = freeze(ft, pre, d2, clip, cur, past)
        past = cur
        p = LOCAL / f"frozen-{k}.npz"
        np.savez(p, **out)
        result["snapshots"][k] = {
            "frozen_sha256": file_digest(p),
            "g1_manifest_sha256": info["snapshots"][k]["frozen_sha256"],
            "checkpoint_sha256": info["snapshots"][k]["checkpoint_sha256"],
        }
        print("Frozen G6", epoch, flush=True)
    write_json(LOCAL / "frozen-provenance.json", json_native(result))
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    for k, row in result["snapshots"].items():
        p = LOCAL / f"frozen-{k}.npz"
        assert file_digest(p) == row["frozen_sha256"]
        with np.load(p, allow_pickle=False) as arrays:
            r = dict(arrays)
        metrics, alignment = g4.sealed(r, truth, METHODS, baseline="D1")
        row.update(
            metrics=metrics, bank_alignment=alignment, actionable=actionable(r, truth)
        )
    result["seconds"] = time.monotonic() - start
    report(result)
    write_json(OUT, json_native(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
