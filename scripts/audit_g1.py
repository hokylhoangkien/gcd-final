"""G1 frozen multi-layer geometry on Stanford Cars R2 snapshots."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from crish.experiment import file_digest, load_checkpoint
from relational.state import write_json
from relational.train import build
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_r2_evidence import sealed_labels

WORK = Path("dev_outputs/r2-audit")
LOCAL = WORK / "g1"
LAYERS = (3, 7, 9, 11)
EPOCHS = (100, 150)
METHODS = ("B3", "B7", "B9", "B11", "Equal", "Learned")
SEED = 473606
REPORT = Path("reports/CRISH_MULTILAYER_GEOMETRY.md")
OUTPUT = Path("reports/r2_multilayer_geometry.json")


@torch.inference_mode()
def extract():
    model, train, _, center, _, _ = build()
    model.eval().requires_grad_(False)
    backbone = model[0]
    ds = train.dataset
    blind = copy.copy(ds.unlabelled_dataset)
    legal_ds = copy.copy(ds.labelled_dataset)
    blind.transform = legal_ds.transform = center
    ids = np.load(WORK / "ids.npy")
    cached = np.load(WORK / "features.npy", mmap_mode="r")
    assert np.array_equal(ids, blind.ids) and cached.shape == (3, 3, len(ids), 768)
    tails, hashes = {}, {}
    checksums = {
        name: sha
        for sha, name in (
            line.split()
            for line in (WORK / "input/SHA256SUMS").read_text().splitlines()
        )
    }
    shared = backbone.state_dict()
    for epoch in EPOCHS:
        path = WORK / f"input/epoch-{epoch}.pt"
        hashes[epoch] = file_digest(path)
        assert hashes[epoch] == checksums[path.name]
        state = load_checkpoint(path)
        assert state["scheduler"]["last_epoch"] == state["epoch"] == epoch
        tail = copy.deepcopy(backbone.blocks[11])
        for key, value in shared.items():
            if not key.startswith("blocks.11."):
                assert torch.equal(value.cpu(), state["model"][f"0.{key}"]), key
        tail.load_state_dict(
            {key: state["model"][f"0.blocks.11.{key}"] for key in tail.state_dict()}
        )
        tails[epoch] = tail

    def collect(dataset, is_legal):
        early = []
        final = {e: [] for e in EPOCHS}
        label, seen = [], []
        for i, (images, y, keys) in enumerate(DataLoader(dataset, batch_size=8)):
            label.extend(y.tolist())
            seen.extend(keys.tolist())
            tokens = backbone.prepare_tokens(images.cuda())
            per_layer = []
            for j, block in enumerate(backbone.blocks[:11]):
                tokens = block(tokens)
                if j in LAYERS:
                    per_layer.append(F.normalize(tokens[:, 0], dim=1).cpu())
            early.append(torch.stack(per_layer))
            if is_legal or i == 0:
                for e, tail in tails.items():
                    h = backbone.norm(tail(tokens))[:, 0]
                    if not is_legal:
                        j = EPOCHS.index(e) + 1
                        torch.testing.assert_close(
                            h.cpu(),
                            torch.from_numpy(cached[j, 0, : len(y)].copy()),
                            atol=3e-5,
                            rtol=3e-5,
                        )
                    final[e].append(F.normalize(h, dim=1).cpu())
            if (i + 1) % 128 == 0:
                print("Known" if is_legal else "Unlabeled", (i + 1) * 8, flush=True)
        return torch.cat(early, dim=1), final, np.asarray(seen), np.asarray(label)

    known, finals, legal_ids, legal = collect(legal_ds, True)
    earlier, _, seen, sentinel = collect(blind, False)
    assert np.array_equal(seen, ids) and np.all(sentinel == -1)
    assert not np.intersect1d(ids, legal_ids).size
    print("Final CLS cache parity passed for both checkpoints", flush=True)
    return known, finals, legal_ids, legal, earlier, ids, cached, hashes


def legal_pairs(x, y, seed):
    """Balanced pairs with half the negatives sampled from the closest 32 wrong peers."""
    rng = np.random.default_rng(seed)
    sims = (x @ x.transpose(1, 2)).cpu().numpy()
    same = y[:, None] == y
    np.fill_diagonal(same, False)
    hard = np.where(y[:, None] != y, sims[-1], -np.inf).argsort(axis=1)[:, -32:]
    a, b, c = [], [], []
    for i in range(len(y)):
        pos = np.flatnonzero(same[i])
        if not len(pos):
            continue
        p = rng.choice(pos, 4, replace=True)
        n = np.r_[rng.choice(hard[i], 2), rng.choice(np.flatnonzero(y != y[i]), 2)]
        a.extend([i] * 8)
        b.extend(np.r_[p, n])
        c.extend([1] * 4 + [0] * 4)
    return sims[:, a, b].T.astype(np.float64), np.asarray(c)


def learn(x, labels, tr, va):
    xx, y = legal_pairs(x[:, tr], labels[tr], SEED + 1)
    val, vy = legal_pairs(x[:, va], labels[va], SEED + 2)
    candidates = []
    for reg in (0.0, 0.01, 0.1):

        def loss(p, reg=reg):
            z = 10 * (xx @ p[:4]) + p[4]
            delta = p[:4] - 0.25
            cost = np.mean(np.logaddexp(0, z) - y * z) + reg * (delta @ delta)
            res = (expit(z) - y) / len(y)
            return cost, np.r_[10 * xx.T @ res + 2 * reg * delta, res.sum()]

        fit = minimize(
            loss,
            [0.25] * 4 + [0],
            jac=True,
            method="SLSQP",
            bounds=[(0, 1)] * 4 + [(-20, 20)],
            constraints={"type": "eq", "fun": lambda p: p[:4].sum() - 1},
        )
        if not fit.success:
            raise ValueError(fit.message)
        w = fit.x[:4]
        candidates.append(
            {
                "ridge": reg,
                "weights": w.tolist(),
                "known_validation_auc": roc_auc_score(vy, val @ w),
            }
        )
    return max(candidates, key=lambda r: r["known_validation_auc"]), candidates


def freeze(x, q, bank, slots, random_pairs, weights):
    f = x.cuda()
    scores = torch.stack([f[j, q] @ f[j, bank].T for j in range(4)])
    bank_slots = torch.as_tensor(slots[bank], device="cuda", dtype=torch.long)
    src = torch.as_tensor(slots[q], device="cuda", dtype=torch.long)
    hard = scores[-1].topk(64).indices.cpu().numpy()
    pairs = np.concatenate([random_pairs, hard], axis=1)
    all_rows = np.arange(len(q))[:, None]
    rules = np.r_[np.eye(4), np.full((1, 4), 0.25), [weights]]
    frozen = {"pairs": bank[pairs]}
    for name, w in zip(METHODS, rules, strict=True):
        sim = (
            scores
            * torch.as_tensor(w, device="cuda", dtype=scores.dtype)[:, None, None]
        ).sum(0)
        support = torch.full((len(q), 196), -torch.inf, device="cuda")
        # Slot maximum, independent of peer ordering and with fixed slot tie order.
        for s in np.unique(slots[bank]):
            support[:, s] = sim[:, bank_slots == int(s)].max(1).values
        own = support.gather(1, src[:, None]).flatten()
        support.scatter_(1, src[:, None], -torch.inf)
        order = support.argsort(descending=True, stable=True)
        dest = order[:, :5]
        dest = torch.where(support.gather(1, dest).isfinite(), dest, -1)
        # A missing own-slot peer supplies the cosine floor.
        wrong = support.max(1).values - own.clamp_min(-1)
        frozen.update(
            {
                f"{name}_neighbors": bank[sim.topk(10).indices.cpu().numpy()],
                f"{name}_pair": sim.cpu().numpy()[all_rows, pairs],
                f"{name}_wrong": wrong.cpu().numpy(),
                f"{name}_dest": dest.cpu().numpy(),
            }
        )
    return frozen


def auc(y, score):
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None


def rate(v):
    return float(v.mean()) if v.size else None


def evaluate(frozen, ids, q, bank, slots, truth, *, methods=METHODS, baseline="B11"):
    mp, coverage = align(
        ids[bank].tolist(), dict(zip(ids.tolist(), slots.tolist(), strict=True)), truth
    )
    slot_class = np.array([mp.get(k, -1) for k in range(196)] + [-1])
    y = truth[ids[q]]
    wrong = slot_class[slots[q]] != y
    pair_same = truth[ids[frozen["pairs"]]] == y[:, None]
    target_exists = np.isin(y, list(mp.values())) & wrong
    preds = {m: slot_class[frozen[f"{m}_dest"]] == y[:, None] for m in methods}
    base = preds[baseline][:, 0]
    out = {}
    for m in methods:
        hit = preds[m]
        knn = truth[ids[frozen[f"{m}_neighbors"]]] == y[:, None]
        score = frozen[f"{m}_pair"]
        out[m] = {}
        for g, mask in (("All", y >= 0), ("Old", y < 98), ("New", y >= 98)):
            p = pair_same[mask, :256]
            h = pair_same[mask, 256:]
            ordinary = score[mask, :256]
            hard = score[mask, 256:]
            hard_scores = np.r_[ordinary[p], hard[~h]]
            failure = mask & wrong & ~base
            success = mask & wrong & base
            out[m][g] = {
                "queries": int(mask.sum()),
                "wrong_source": int((mask & wrong).sum()),
                "mapped_wrong": int((mask & target_exists).sum()),
                "pair_positive": int(p.sum()),
                "pair_negative": int((~p).sum()),
                "hard_negative": int((~h).sum()),
                "purity": {k: rate(knn[mask, :k]) for k in (1, 5, 10)},
                "pair_auc": auc(p.ravel(), ordinary.ravel()),
                "hard_auc": auc(
                    np.r_[np.ones(p.sum()), np.zeros((~h).sum())], hard_scores
                ),
                "detection_auc": auc(wrong[mask], frozen[f"{m}_wrong"][mask]),
                "destination_recall": {
                    k: rate(hit[mask & wrong, :k].any(1)) for k in (1, 3, 5)
                },
                "destination_recall_mapped": {
                    k: rate(hit[mask & target_exists, :k].any(1)) for k in (1, 3, 5)
                },
                "final_failure_n": int(failure.sum()),
                "failure_recovery": {
                    k: rate(hit[failure, :k].any(1)) for k in (1, 3, 5)
                },
                "final_success_n": int(success.sum()),
                "correct_to_wrong_vs_final": rate(~hit[success, 0]),
                "move_from_correct_source_to_wrong": rate(~hit[mask & ~wrong, 0]),
                "purity1_on_final_misses": rate(
                    knn[
                        mask & (truth[ids[frozen[f"{baseline}_neighbors"][:, 0]]] != y),
                        0,
                    ]
                ),
            }
    return out, coverage


def report(result):
    lines = [
        "# G1 multi-layer geometry on Stanford Cars",
        "",
        (
            "Blocks 3, 7 and 9 use the L2-normalized post-block CLS token. "
            "Block 11 applies the checkpoint's ViT final LayerNorm before L2 normalization. "
            "The frozen prefix is identical at epochs 100 and 150. The final CLS cache is "
            "reused after exact-protocol forward parity on the first eight unlabeled IDs."
        ),
        "",
        (
            f"Both checkpoints share {result['queries']:,} fresh label-blind queries and "
            f"{result['bank']:,} other unlabeled retrieval IDs. Query IDs exclude the prior "
            "588-ID Oracle MOVE panel and frozen E3 proposal IDs. Earlier split-wide diagnostics "
            "included these images, so this is a new held-out query partition within Cars. "
            "The fit data consists only of separate legally labeled images, split 78/20 by class."
        ),
        "",
        (
            "Equal fusion averages cosine scores. Learned fusion minimizes logistic loss under "
            "nonnegative weights summing to one. Known validation AUROC selects the ridge term "
            "from 0, .01 and .1. Every scoring rule shares 256 uniform peers per query and the "
            "same top-64 B11 neighbors for hard negatives. Hard AUROC uses uniform positives "
            "against the wrong-class hard peers. All groups are grouped by query class. Pairs "
            "sharing IDs are dependent, so the table is descriptive."
        ),
        "",
        (
            "The sealed evaluator maps frozen classifier slots to classes by Hungarian alignment "
            "on bank IDs. Candidate destinations are the best alternative slots by maximum "
            "peer cosine across that entire bank. Wrong-assignment risk is the best alternative "
            "support minus current-slot support. Recall includes all wrong assignments, including "
            "ones with no mapped destination. Recovery conditions on B11 top-1 failure on wrong "
            "sources. Regression conditions on B11 top-1 success on wrong sources. JSON also "
            "reports forced edits on correct sources, mapped-only recall and purity on B11 misses. "
            "Coverage and mapping depend on the checkpoint. These metrics do not measure training effects."
        ),
        "",
        "| Epoch / group / rule | Purity@1/5/10 | Pair / hard AUC | Wrong AUC | Destination R@1/3/5 | Recovery@1 | Regression@1 |",
        "|---|---|---|---:|---|---:|---:|",
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    for epoch, snapshot in result["snapshots"].items():
        for group in ("All", "Old", "New"):
            for method in METHODS:
                m = snapshot["metrics"][method][group]
                purity = "/".join(fmt(m["purity"][k]) for k in (1, 5, 10))
                recall = "/".join(fmt(m["destination_recall"][k]) for k in (1, 3, 5))
                lines.append(
                    f"| {epoch} / {group} / {method} | {purity} | "
                    f"{fmt(m['pair_auc'])} / {fmt(m['hard_auc'])} | {fmt(m['detection_auc'])} | "
                    f"{recall} | {fmt(m['failure_recovery'][1])} | "
                    f"{fmt(m['correct_to_wrong_vs_final'])} |"
                )
    for epoch, snap in result["snapshots"].items():
        w = ", ".join(f"{v:.3f}" for v in snap["learned"]["weights"])
        lines.append(f"\nEpoch {epoch} learned weights for B3/B7/B9/B11 were {w}.")
    lines += [
        "",
        f"Runtime {result['elapsed_seconds'] / 60:.1f} minutes.",
        "",
        "[Counts, subgroup metrics and frozen prediction provenance](r2_multilayer_geometry.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    started = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    known, finals, legal_ids, legal, early, ids, cache, hashes = extract()
    assert np.array_equal(np.unique(legal), np.arange(98))
    rng = np.random.default_rng(SEED)
    classes = rng.permutation(98)
    tr = np.flatnonzero(np.isin(legal, classes[:78]))
    va = np.flatnonzero(np.isin(legal, classes[78:]))
    original = json.loads((WORK / "e3-frozen-proposals.json").read_text())
    snaps = original["snapshots"]
    past = json.loads(Path("reports/r2_oracle_move_trajectory.json").read_text())
    excluded = set(past["panel_ids"])
    excluded.update(p["id"] for e in EPOCHS for p in snaps[str(e)]["proposals"])
    eligible = np.flatnonzero(~np.isin(ids, list(excluded)))
    q = np.sort(rng.choice(eligible, 1024, replace=False))
    bank = np.setdiff1d(np.arange(len(ids)), q)
    random_pairs = np.stack([rng.choice(len(bank), 256, replace=False) for _ in q])
    result = {
        "base": "4736d6d",
        "seed": SEED,
        "layers": LAYERS,
        "queries": len(q),
        "bank": len(bank),
        "query_ids": ids[q].tolist(),
        "excluded_from_query": len(excluded),
        "known_train_classes": classes[:78].tolist(),
        "known_val_classes": classes[78:].tolist(),
        "known_train_n": len(tr),
        "known_val_n": len(va),
        "known_train_ids": legal_ids[tr].tolist(),
        "known_val_ids": legal_ids[va].tolist(),
        "selection_objective": "Known validation AUROC on balanced pairs",
        "snapshots": {},
    }
    frozen_all = {}
    for j, epoch in enumerate(EPOCHS, 1):
        k = str(epoch)
        fit = torch.cat([known, torch.cat(finals[epoch])[None]])
        learned, alternatives = learn(fit, legal, tr, va)
        x = torch.cat(
            [early, F.normalize(torch.from_numpy(cache[j, 0].copy()), dim=1)[None]]
        )
        slots = np.asarray([snaps[k]["predicted_slots"][str(i)] for i in ids])
        frozen = freeze(x, q, bank, slots, random_pairs, learned["weights"])
        frozen_path = LOCAL / f"frozen-{epoch}.npz"
        np.savez(frozen_path, ids=ids, q=q, bank=bank, slots=slots, **frozen)
        result["snapshots"][k] = {
            "checkpoint_sha256": hashes[epoch],
            "learned": learned,
            "alternatives": alternatives,
            "frozen_sha256": file_digest(frozen_path),
        }
        frozen_all[k] = (frozen, slots)
        print("Frozen scoring", epoch, learned, flush=True)
    write_json(LOCAL / "frozen-provenance.json", result)
    truth = sealed_labels()
    assert np.array_equal(truth[legal_ids], legal)
    for k, (frozen, slots) in frozen_all.items():
        metrics, coverage = evaluate(frozen, ids, q, bank, slots, truth)
        result["snapshots"][k].update(metrics=metrics, bank_alignment=coverage)
    result["elapsed_seconds"] = time.monotonic() - started
    report(result)
    write_json(OUTPUT, json_native(result))


if __name__ == "__main__":
    main()
