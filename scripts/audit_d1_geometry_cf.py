"""Paired one-step geometry verification at four historical Cars snapshots."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest
from crish.repair import Action
from dual.policy import scores
from model import DistillLoss
from relational.state import write_json
from scripts import audit_r2_v3 as v3
from scripts.audit_r2_evidence import sealed_labels

WORK = Path("dev_outputs/r2-audit")
OUT = Path("dev_outputs/d1-pair-diagnostics")
TAGS = ("D1-100", "D1-150", "D1-200", "E0-200")


def checked(path, digest):
    if file_digest(path) != digest:
        raise ValueError(f"Artifact digest mismatch at {path}")
    return path


def pick(records):
    if not records:
        return None
    top = max(range(len(records)), key=lambda i: records[i]["gain"][0])
    r = records[top]
    if not np.isfinite([r["gain"], r["ce_delta"]]).all():
        return None
    return top if min(r["gain"]) > 1e-7 and max(r["ce_delta"]) <= 0 else None


def predict(tag, model, train, center, ids, clip, d2, proof):
    epoch = int(tag.split("-")[1])
    d1 = tag.startswith("D1")
    directory = WORK / ("d1-final" if d1 else "e0-baseline")
    if d1:
        path = directory / f"epoch-{epoch}.pt"
        sha = proof["archives"]["200"]["members"][path.name]
        frozen_sha = proof["frozen_sha256"][tag]
    else:
        (path,) = Path("../log/E0").rglob("model.pt")
        sha = proof["e0_archive"]["members_sha256"]["model.pt"]
        frozen_sha = json.loads(Path("reports/r2_e0_baseline_audit.json").read_text())[
            "provenance"
        ]["frozen_sha256"][tag]
    state = torch.load(
        checked(path, sha), map_location="cpu", weights_only=False, mmap=True
    )
    assert state["epoch"] == epoch
    with np.load(checked(directory / f"frozen-{tag}.npz", frozen_sha)) as f:
        np.testing.assert_array_equal(f["ids"], ids)
        slots = f["slots"]
        anchor = torch.from_numpy(f["feature"]).cuda()
    model.eval()
    backbone, head = model
    known = copy.copy(train.dataset.labelled_dataset)
    blind = copy.copy(train.dataset.unlabelled_dataset)
    known.transform = blind.transform = center
    torch.manual_seed(72151)
    import random

    random.seed(72151)
    np.random.seed(72151)
    batches = iter(DataLoader(train.dataset, batch_size=128, sampler=train.sampler))
    results = []
    for number in range(2):
        rng = np.random.default_rng(73901 + number)
        model.load_state_dict(state["model"])
        images, y, keys, _ = v3.legal_batch(next(batches))
        y = y.long()
        keys = keys.numpy()
        unknown = np.flatnonzero(~np.isin(ids, keys))
        bank, dis, confirm, sem = np.split(
            rng.choice(unknown, 3072 + 32 + 32 + 64, replace=False), [3072, 3104, 3136]
        )
        known_rows = rng.choice(
            np.flatnonzero(~np.isin(known.uq_idxs, keys)), 32, replace=False
        )
        q = np.flatnonzero(
            ~np.isin(np.arange(len(ids)), np.r_[bank, dis, confirm, sem])
        )
        risk, dst = scores(clip, slots, ids, [(q, bank)])
        valid = q[np.isfinite(risk[q]) & (risk[q] > 0) & (dst[q] >= 0)]
        ranked = valid[np.lexsort((ids[valid], -risk[valid]))][: len(q) // 10]
        tokens = v3.prefix(backbone, images)
        h = v3.encode(backbone, tokens).requires_grad_()
        with torch.no_grad():
            raw = head(h)[1].reshape(2, 128, -1)
            teacher = (raw / 0.04).softmax(-1)
        actions = []
        for i in ranked:
            rows = np.flatnonzero(keys == ids[i])
            if len(rows) != 1:
                continue
            row = int(rows[0])
            src = int(slots[i])
            if (
                y[row] < 0
                and teacher[:, row].mean(0).argmax() == src
                and teacher[:, row, src].min() >= 0.2
            ):
                actions.append(Action(row, "move", src, int(dst[i])))
                if len(actions) == 4:
                    break

        def get(ds, rows):
            pixels, labels, returned = next(
                iter(DataLoader(Subset(ds, rows.tolist()), batch_size=len(rows)))
            )
            expected = known.uq_idxs[rows] if ds is known else ids[rows]
            np.testing.assert_array_equal(returned.numpy(), expected)
            return v3.prefix(backbone, pixels), labels.cuda().long()

        probes = []
        for u, k in zip((dis, confirm), np.split(known_rows, 2), strict=True):
            neighbors = (d2[u] @ d2[bank].T).topk(10).indices.cpu().numpy()
            a = anchor[bank[neighbors]].double()
            direction = a[:, :5].mean(1) - a[:, 5:].mean(1)
            t, mask = get(blind, u)
            assert (mask == -1).all()
            kt, ky = get(known, k)
            probes.append((t, direction, kt, ky))
        st, mask = get(blind, sem)
        assert (mask == -1).all()
        bases = list(
            v3.gradients(model, tokens, h, raw, y, DistillLoss(0, 200), actions)
        )
        scores_by_arm = []
        features = []
        params = list(backbone.blocks[-1].parameters())
        for gv, gb in bases:
            model.load_state_dict(state["model"])
            opt = v3.checkpoint_optimizer(model, state)
            opt.zero_grad(set_to_none=True)
            head.last_layer.weight_v.grad = gv
            for p, g in zip(params, gb.split([p.numel() for p in params]), strict=True):
                p.grad = g.view_as(p)
            opt.step()
            with torch.no_grad():
                pair = []
                for t, direction, kt, ky in probes:
                    z = F.normalize(v3.encode(backbone, t).double(), dim=-1)
                    ce = F.cross_entropy(
                        head(v3.encode(backbone, kt))[1].double() / 0.1, ky
                    )
                    pair.append([float((z * direction).sum(1).mean()), float(ce)])
                features.append(
                    F.normalize(v3.encode(backbone, st).double(), dim=-1).cpu().numpy()
                )
                scores_by_arm.append(pair)
        diff = np.asarray(scores_by_arm) - scores_by_arm[0]
        records = [
            {
                "id": int(keys[a.row]),
                "source": a.source,
                "destination": a.target,
                "gain": v[:, 0].tolist(),
                "ce_delta": v[:, 1].tolist(),
            }
            for a, v in zip(actions, diff[1:], strict=True)
        ]
        chosen = pick(records)
        choices = {
            "KEEP": None,
            "CF": chosen,
            "Direct": 0 if chosen is not None else None,
            "Random": int(rng.integers(len(actions))) if chosen is not None else None,
        }
        fold = {
            "checkpoint_sha": sha,
            "batch": number,
            "candidates": records,
            "choices": choices,
            "train_ids": keys.tolist(),
            "known_probe_ids": known.uq_idxs[known_rows].tolist(),
            "discovery_ids": ids[dis].tolist(),
            "confirmation_ids": ids[confirm].tolist(),
        }
        path = OUT / f"geometry-{tag}-{number}.npz"
        np.savez(
            path,
            sem_ids=ids[sem],
            bank_ids=ids[bank],
            anchor=anchor[bank].cpu().numpy(),
            features=np.stack(features),
        )
        fold["sha256"] = file_digest(path)
        results.append(fold)
        print(tag, number, "eligible", len(actions), "selected", chosen, flush=True)
    return results


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    model, train, _, center, _, _ = v3.build()
    ids = np.asarray(train.dataset.unlabelled_dataset.ids)
    hashes = json.loads(Path("reports/r2_vlm_heldout_confirmation.json").read_text())[
        "feature_sha256"
    ]
    features = []
    for p, key in (("g6/clip-image.npz", "CLIP"), ("g5/dinov2-cls.npz", "D2")):
        with np.load(checked(WORK / p, hashes[key])) as f:
            np.testing.assert_array_equal(f["ids"], ids)
            features.append(torch.from_numpy(f["features"]).cuda())
    proof = json.loads(Path("reports/CRISH_D1_FINAL_AUDIT.json").read_text())[
        "provenance"
    ]
    result = {
        tag: predict(tag, model, train, center, ids, *features, proof) for tag in TAGS
    }
    write_json(OUT / "cf-decisions.json", result)
    y = sealed_labels()
    for tag, folds in result.items():
        for fold in folds:
            with np.load(
                checked(OUT / f"geometry-{tag}-{fold['batch']}.npz", fold["sha256"])
            ) as f:
                bank = y[f["bank_ids"]]
                truth = y[f["sem_ids"]]
                classes = np.unique(bank)
                centers = np.stack(
                    [f["anchor"][bank == c].mean(0) for c in classes]
                ).astype(float)
                centers /= np.linalg.norm(centers, axis=1, keepdims=True)
                cos = f["features"] @ centers.T
                target = truth[:, None] == classes
                covered = target.any(1)
                margins = np.where(target, cos, -np.inf).max(-1) - np.where(
                    target, -np.inf, cos
                ).max(-1)
                delta = margins[:, covered] - margins[:1, covered]
                fold["semantic_coverage"] = int(covered.sum())
                fold["semantic_gain_vs_keep"] = {
                    k: float(delta[0 if i is None else i + 1].mean())
                    for k, i in fold["choices"].items()
                }
                fold["candidate_semantic_gains"] = delta[1:].mean(1).tolist()
    write_json(Path("reports/D1_GEOMETRY_CF.json"), result)


if __name__ == "__main__":
    main()
