"""G2 patch-token geometry on the frozen R2 checkpoints and G1 Cars IDs."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest, load_checkpoint
from relational.sensitivity_metrics import prefix
from relational.state import write_json
from relational.train import build
from scripts import audit_g1 as g1
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_g7 import pack

LOCAL = g1.WORK / "g2"
REPORT = Path("reports/CRISH_PATCH_GEOMETRY.md")
OUT = Path("reports/r2_patch_geometry.json")
G1_REPORT = Path("reports/r2_multilayer_geometry.json")
FULL = ("CLS", "Mean", "Spatial", "CLS+Mean", "CLS+Spatial")
SHORT = ("CLS@64", "Local@64", "CLS+Local@64")
PATCH_CHUNK = 32


def pool(patches):
    grid = patches.reshape(-1, 2, 7, 2, 7, 768)
    spatial = F.normalize(grid.mean(dim=(2, 4)), dim=-1).flatten(1)
    return F.normalize(patches.mean(1)), F.normalize(spatial)


@torch.inference_mode()
def extract(ids, info):
    model, loader, _, center, _, _ = build()
    model.eval().requires_grad_(False)
    backbone = model[0]
    ds = copy.copy(loader.dataset.unlabelled_dataset)
    ds.transform = center
    assert np.array_equal(ids, ds.ids)
    cache_path = g1.WORK / "oracle-move/step0-panel-tokens.pt"
    cache = torch.load(cache_path, weights_only=True, map_location="cpu")
    origin = g1.WORK / "input/epoch-100.pt"
    assert cache["checkpoint_sha256"] == file_digest(origin)
    past = json.loads(Path("reports/r2_oracle_move_trajectory.json").read_text())
    assert cache["panel_ids"] == past["panel_ids"] and len(cache["panel_ids"]) == 588
    shared = backbone.state_dict()
    tails = {}
    for e in g1.EPOCHS:
        path = g1.WORK / f"input/epoch-{e}.pt"
        assert file_digest(path) == info["snapshots"][str(e)]["checkpoint_sha256"]
        state = load_checkpoint(path)["model"]
        for k, v in shared.items():
            if not k.startswith("blocks.11."):
                assert torch.equal(v.cpu(), state[f"0.{k}"]), k
        tail = copy.deepcopy(backbone.blocks[-1])
        tail.load_state_dict({k: state[f"0.blocks.11.{k}"] for k in tail.state_dict()})
        tails[e] = tail
        del state
    final = np.load(g1.WORK / "features.npy", mmap_mode="r")
    arrays = {}
    for e in g1.EPOCHS:
        arrays[e] = (
            np.lib.format.open_memmap(
                LOCAL / f"scratch-patches-{e}.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(ids), 196, 768),
            ),
            np.empty((len(ids), 768), dtype=np.float32),
            np.empty((len(ids), 3072), dtype=np.float32),
        )

    def save(tokens, rows):
        for j, (e, tail) in enumerate(tails.items(), 1):
            h = backbone.norm(tail(tokens.cuda()))
            torch.testing.assert_close(
                h[:, 0].cpu(),
                torch.from_numpy(final[j, 0, rows].copy()),
                atol=6e-5,
                rtol=6e-5,
            )
            patch = F.normalize(h[:, 1:], dim=-1)
            mean, spatial = pool(patch)
            disk, m, s = arrays[e]
            disk[rows] = patch.half().cpu().numpy()
            m[rows], s[rows] = mean.cpu().numpy(), spatial.cpu().numpy()

    lookup = {k: i for i, k in enumerate(ids.tolist())}
    panel_rows = np.asarray([lookup[k] for k in cache["panel_ids"]])
    offset = 0
    for batch in cache["tokens"]:
        for tokens in batch:
            end = offset + len(tokens)
            save(tokens, panel_rows[offset:end])
            offset = end
    assert offset == len(panel_rows)
    remaining = np.setdiff1d(np.arange(len(ids)), panel_rows)
    offset = 0
    for images, labels, keys in DataLoader(
        Subset(ds, remaining.tolist()), batch_size=8
    ):
        assert (labels == -1).all()
        end = offset + len(images)
        rows = remaining[offset:end]
        np.testing.assert_array_equal(keys.numpy(), ids[rows])
        (tokens,) = prefix(backbone, images)
        save(tokens, rows)
        offset = end
        if offset % 1024 == 0:
            print("New shared prefix rows", offset, flush=True)
    assert offset == len(remaining)
    for disk, _, _ in arrays.values():
        disk.flush()
    print("Patch extraction and CLS parity complete", flush=True)
    return arrays, final, len(panel_rows)


def symmetric(query, peers):
    # Pairwise patch matrix bounded by PATCH_CHUNK × 196 × 196.
    sim = peers @ query.T
    sim = sim.float()
    return (sim.max(1).values.mean(1) + sim.max(2).values.mean(1)) / 2


def match(disk, q, shortlist):
    scores = np.empty(shortlist.shape, dtype=np.float32)
    for i, row in enumerate(q):
        query = torch.from_numpy(disk[row].copy()).cuda()
        for start in range(0, shortlist.shape[1], PATCH_CHUNK):
            stop = start + PATCH_CHUNK
            peers = torch.from_numpy(disk[shortlist[i, start:stop]]).cuda()
            scores[i, start:stop] = symmetric(query, peers).cpu().numpy()
    assert np.isfinite(scores).all() and np.abs(scores).max() <= 1.001
    return scores


def short_pack(score, q, shortlist, slots):
    rows = np.arange(len(q))[:, None]
    order = np.argsort(-score, axis=1, kind="stable")
    rank = shortlist[rows, order]
    support = np.full((len(q), 196), -np.inf, dtype=np.float32)
    np.maximum.at(support, (rows, slots[shortlist]), score)
    own = np.maximum(support[rows.ravel(), slots[q]], -1)
    support[rows.ravel(), slots[q]] = -np.inf
    alt = np.argsort(-support, axis=1, kind="stable")[:, :5]
    dest = np.where(np.isfinite(support[rows, alt]), alt, -1)
    return {
        "neighbors": rank[:, :10],
        "pair": np.tile(score, (1, 2)),
        "wrong": support.max(1) - own,
        "dest": dest,
    }


def freeze(final, mean, spatial, disk, base, q, bank, ids, slots):
    tensors = [
        F.normalize(torch.from_numpy(final.copy())).cuda(),
        torch.from_numpy(mean).cuda(),
        torch.from_numpy(spatial).cuda(),
    ]
    cos, avg, sp = [x[q] @ x[bank].T for x in tensors]
    source = base["pairs"]
    inv = np.full(len(ids), -1)
    inv[bank] = np.arange(len(bank))
    pairs = inv[source]
    assert (pairs >= 0).all()
    full = {"pairs": source}
    for method, score in zip(
        FULL, (cos, avg, sp, (cos + avg) / 2, (cos + sp) / 2), strict=True
    ):
        out = pack(score, q, bank, slots, pairs)
        for key in ("neighbors", "pair", "wrong", "dest"):
            if method == "CLS":
                expected = base[f"B11_{key}"]
                if key in ("neighbors", "dest"):
                    np.testing.assert_array_equal(out[key], expected)
                else:
                    np.testing.assert_allclose(out[key], expected, atol=3e-6, rtol=3e-6)
            full[f"{method}_{key}"] = out[key]
    shortlist = source[:, 256:]
    assert shortlist.shape[1] == 64
    cls = cos.cpu().numpy()[np.arange(len(q))[:, None], inv[shortlist]]
    # The G1 top-64 score order is also the shared deterministic shortlist tie order.
    local = match(disk, q, shortlist)
    short = {"pairs": np.tile(shortlist, (1, 2))}
    for method, score in zip(SHORT, (cls, local, (cls + local) / 2), strict=True):
        short.update(
            {
                f"{method}_{k}": v
                for k, v in short_pack(score, q, shortlist, slots).items()
            }
        )
    return full, short


def evaluate(full, short, ids, q, bank, slots, truth):
    result = {}
    for scope, records, methods, n in (
        ("full", full, FULL, 256),
        ("shortlist", short, SHORT, 64),
    ):
        result[scope], coverage = g1.evaluate(
            records,
            ids,
            q,
            bank,
            slots,
            truth,
            methods=methods,
            baseline=methods[0],
            uniform_count=n,
        )
    mp, _ = align(
        ids[bank].tolist(), dict(zip(ids.tolist(), slots.tolist(), strict=True)), truth
    )
    table = np.array([mp.get(k, -1) for k in range(196)] + [-1])
    y = truth[ids[q]]
    wrong = table[slots[q]] != y
    available = (table[slots[short["pairs"][:, :64]]] == y[:, None]).any(1)
    ceiling = {}
    for group, group_mask in (("All", y >= 0), ("Old", y < 98), ("New", y >= 98)):
        mask = group_mask & wrong
        ceiling[group] = {
            "wrong_sources": int(mask.sum()),
            "reachable": int((mask & available).sum()),
            "recall_ceiling": g1.rate(available[mask]),
        }
        for scope, records, methods in (
            ("full", full, FULL),
            ("shortlist", short, SHORT),
        ):
            base = records[f"{methods[0]}_dest"][:, 0]
            for method in methods:
                p = records[f"{method}_dest"][:, 0]
                disagree = mask & (p != base)
                result[scope][method][group].update(
                    disagreement_n=int(disagree.sum()),
                    disagreement_accuracy=g1.rate(table[p[disagree]] == y[disagree]),
                    cls_accuracy_on_disagreement=g1.rate(
                        table[base[disagree]] == y[disagree]
                    ),
                    either_top1_correct=g1.rate(
                        ((table[p] == y) | (table[base] == y))[mask]
                    ),
                )
    return result, coverage, ceiling


def report(result):
    lines = [
        "# G2 patch-token geometry",
        "",
        (
            "This exploratory evaluation reuses the G1 Cars query and bank manifests. "
            "Neither the backbone nor an auxiliary head is trained. Every token comes from "
            "block 11 followed by final LayerNorm. Each patch is L2-normalized before pooling."
        ),
        "",
        (
            "Mean pools all 196 normalized patch tokens and normalizes the resulting vector. "
            "Spatial averages four fixed 7×7 quadrants, normalizes each quadrant, then concatenates "
            "and normalizes the descriptor. The three full-bank similarity matrices use cosine. "
            "Each CLS fusion averages its two component similarities with equal fixed weights."
        ),
        "",
        (
            "Local matching averages the mean best patch cosine in each direction. It uses the "
            "same fixed 64 CLS peers as CLS@64. Similarity uses float16 patch storage and matmul "
            "with float32 reductions. Matrix allocation is bounded by 32 peers per chunk. "
            "All shortlist variants use the G1 top-64 as their only pair pool; their pair AUC and "
            "common hard AUC therefore describe the same restricted discrimination task. "
            "Full-bank variants use the original G1 uniform positives and common CLS hard negatives."
        ),
        "",
        (
            "Destination recall is conditioned on a wrong source under G1 bank-only Hungarian "
            "alignment. The shortlist ceiling is the fraction of such queries whose true slot "
            "occurs among the 64 peers. Recovery and regression condition on top-1 failure and "
            "success of the CLS comparator from the same scope. Either-correct is the fraction "
            "of wrong-source queries with a correct top-1 destination from either scoring rule."
        ),
        "",
        "| Epoch / group / rule | Purity@1/5/10 | Pair / hard AUC | Error AUC | Destination R@1/3/5 | Recovery / regression | Either-correct |",
        "|---|---|---|---:|---|---|---:|",
    ]

    def f(v):
        return "—" if v is None else f"{v:.3f}"

    for epoch, snapshot in result["snapshots"].items():
        for group in ("All", "Old", "New"):
            for scope, methods in (("full", FULL), ("shortlist", SHORT)):
                for method in methods:
                    m = snapshot["metrics"][scope][method][group]
                    purity = "/".join(f(m["purity"][k]) for k in (1, 5, 10))
                    recall = "/".join(f(m["destination_recall"][k]) for k in (1, 3, 5))
                    lines.append(
                        f"| {epoch} / {group} / {method} | {purity} | "
                        f"{f(m['pair_auc'])} / {f(m['hard_auc'])} | {f(m['detection_auc'])} | "
                        f"{recall} | {f(m['failure_recovery'][1])} / {f(m['correct_to_wrong_vs_final'])} | "
                        f"{f(m['either_top1_correct'])} |"
                    )
    lines += [
        "",
        "## Shortlist destination coverage",
        "",
        "| Epoch / group | Reachable / wrong sources | Recall ceiling |",
        "|---|---:|---:|",
    ]
    for epoch, snapshot in result["snapshots"].items():
        for group, m in snapshot["coverage_ceiling"].items():
            lines.append(
                f"| {epoch} / {group} | {m['reachable']} / {m['wrong_sources']} | {f(m['recall_ceiling'])} |"
            )
    lines += [
        "",
        (
            f"Runtime {result['seconds'] / 60:.1f} minutes. Exact prefix cache reused for "
            f"{result['prefix_cached_rows']} images; the shared frozen prefix ran once per remaining image."
        ),
        "",
        "[Counts, disagreement metrics and frozen prediction digests](r2_patch_geometry.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    t0 = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    info = json.loads(G1_REPORT.read_text())
    ids = np.load(g1.WORK / "ids.npy")
    patch, final, reused = extract(ids, info)
    result = {
        "base": "f820fbb",
        "g1_report_sha256": file_digest(G1_REPORT),
        "g1_queries": info["queries"],
        "g1_bank": info["bank"],
        "prefix_cached_rows": reused,
        "new_prefix_rows": len(ids) - reused,
        "patches_per_image": 196,
        "local_chunk": PATCH_CHUNK,
        "snapshots": {},
    }
    inputs = {}
    for idx, epoch in enumerate(g1.EPOCHS, 1):
        key = str(epoch)
        manifest = g1.LOCAL / f"frozen-{epoch}.npz"
        meta = info["snapshots"][key]
        assert file_digest(manifest) == meta["frozen_sha256"]
        with np.load(manifest) as base:
            q, bank, slots = (base[k] for k in ("q", "bank", "slots"))
            assert np.array_equal(base["ids"], ids)
            assert ids[q].tolist() == info["query_ids"]
            assert len(bank) == info["bank"] and not np.intersect1d(q, bank).size
            inputs[key] = (q, bank, slots)
            disk, mean, spatial = patch[epoch]
            full, short = freeze(
                final[idx, 0], mean, spatial, disk, base, q, bank, ids, slots
            )
        path = LOCAL / f"frozen-{key}.npz"
        np.savez(
            path,
            q=q,
            bank=bank,
            ids=ids,
            slots=slots,
            **{f"full_{k}": v for k, v in full.items()},
            **{f"short_{k}": v for k, v in short.items()},
        )
        result["snapshots"][key] = {
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "g1_manifest_sha256": meta["frozen_sha256"],
            "frozen_sha256": file_digest(path),
        }
        print("Frozen G2", epoch, flush=True)
    write_json(LOCAL / "frozen-provenance.json", result)
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    for key, snapshot in result["snapshots"].items():
        path = LOCAL / f"frozen-{key}.npz"
        assert file_digest(path) == snapshot["frozen_sha256"]
        with np.load(path) as frozen:
            full = {
                k.removeprefix("full_"): frozen[k]
                for k in frozen.files
                if k.startswith("full_")
            }
            short = {
                k.removeprefix("short_"): frozen[k]
                for k in frozen.files
                if k.startswith("short_")
            }
        metrics, coverage, ceiling = evaluate(full, short, ids, *inputs[key], truth)
        snapshot.update(
            metrics=metrics, bank_alignment=coverage, coverage_ceiling=ceiling
        )
    result["seconds"] = time.monotonic() - t0
    report(result)
    write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
