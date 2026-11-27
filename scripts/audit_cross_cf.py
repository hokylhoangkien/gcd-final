"""Locked R2-150 CLIP proposals and paired classifier-only cross-geometry CF."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import file_digest, load_checkpoint
from crish.repair import Action, VirtualClassifier
from model import DistillLoss
from relational.sensitivity_metrics import prefix
from relational.state import write_json
from relational.train import build
from scripts import audit_g1 as g1
from scripts import audit_g5 as g5
from scripts import audit_g6 as g6
from scripts import audit_g6_confirm as heldout
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_r2_verifier import ARGS, checkpoint_optimizer
from scripts.audit_r2_virtual import encode

LOCAL = g1.WORK / "cross-cf"
MD = Path("reports/CRISH_CROSS_GEOMETRY_CF.md")
OUT = Path("reports/r2_cross_geometry_cf.json")
SEED = 762150
MASS = 0.2
BUDGETS = (25, 50)
METHODS = ("Direct", "CLIPConfidence", "V0", "CrossCF", "StaticD2")
EPOCH = 150
TOPK = 10
TEMP = 0.07


def split(ids, manifest, confirmed):
    proposals = json.loads((g1.WORK / "e3-frozen-proposals.json").read_text())
    oracle = json.loads(Path("reports/r2_oracle_move_trajectory.json").read_text())
    excluded = (
        set(manifest["query_ids"])
        | set(confirmed["query_ids"])
        | set(oracle["panel_ids"])
    )
    excluded.update(
        row["id"]
        for item in proposals["snapshots"].values()
        for row in item["proposals"]
    )
    fresh = np.random.default_rng(SEED).permutation(
        np.flatnonzero(~np.isin(ids, list(excluded)))
    )
    groups = {
        "candidate": np.sort(fresh[:1024]),
        "scoring_context": fresh[1024:1248],
        "discovery": fresh[1248:1312],
        "confirmation": fresh[1312:1376],
        "sealed_probe": fresh[1376:1440],
    }
    rest = np.concatenate(list(groups.values()))
    groups["reference"] = np.setdiff1d(np.arange(len(ids)), rest)
    assert len(np.unique(np.concatenate(list(groups.values())))) == len(ids)
    assert len(groups["sealed_probe"]) == 64 and not set(ids[rest]) & excluded
    return groups, len(excluded)


def slot_support(x, q, bank, slots):
    sim = x[q] @ x[bank].T
    s = torch.as_tensor(slots[bank], device=x.device)
    out = torch.full((len(q), 196), -torch.inf, device=x.device)
    for k in np.unique(slots[bank]):
        out[:, k] = sim[:, s == int(k)].max(1).values
    return out.cpu().numpy()


def clip_proposals(clip, q, bank, slots, ids):
    support = slot_support(clip, q, bank, slots)
    rows = np.arange(len(q))
    original = np.maximum(-1, support[rows, slots[q]])
    support[rows, slots[q]] = -np.inf
    order = np.argsort(-support, axis=1, kind="stable")
    best, runner = support[rows, order[:, 0]], support[rows, order[:, 1]]
    detection = best - original
    priority = np.lexsort((ids[q], -detection))[: len(q) // 10]
    assert np.isfinite(best[priority]).all() and np.isfinite(runner[priority]).all()
    return priority, order[priority, 0], detection[priority], (best - runner)[priority]


@torch.no_grad()
def known_views(model, dataset):
    rng = np.random.default_rng(SEED + 1)
    pick = rng.choice(len(dataset), 96, replace=False)
    torch.manual_seed(SEED + 2)
    np.random.seed(SEED + 2)
    random.seed(SEED + 2)
    x, y, ids = [], [], []
    for views, legal, keys in DataLoader(Subset(dataset, pick.tolist()), batch_size=16):
        y.extend(legal.tolist())
        ids.extend(keys.tolist())
        out = encode(model, prefix(model, torch.cat(views))).reshape(2, len(legal), -1)
        x.append(F.normalize(out, dim=-1))
    assert min(y) >= 0
    return (
        torch.cat(x, dim=1),
        torch.as_tensor(y, dtype=torch.long, device="cuda"),
        np.asarray(ids),
    )


@torch.no_grad()
def geometry_targets(vc, features, d2, query, bank):
    sim, index = (d2[query] @ d2[bank].T).topk(TOPK)
    probs = (vc.logits(features[0, bank], vc.v) / 0.1).softmax(-1)
    q = (probs[index] * (sim / TEMP).softmax(-1)[..., None]).sum(1)
    torch.testing.assert_close(q.sum(1), torch.ones(len(query), device=q.device))
    return q


def operands(known, legal, start, features, rows):
    x = torch.cat([known[:, start : start + 32], features[1:, rows]], dim=1)
    y = torch.cat(
        [
            legal[start : start + 32],
            torch.full((len(rows),), -1, dtype=torch.long, device=x.device),
        ]
    )
    return x, y


def evaluate_virtual(vc, features, d2, folds, slots, shown, target, known, legal):
    context = folds["scoring_context"][: 224 - len(shown)]
    x, y = operands(known, legal, 0, features, np.r_[shown, context])
    assert x.shape[1] == 256
    teacher = vc.logits(x, vc.v)
    prob = (teacher / 0.04).softmax(-1)
    prob = prob[:, 32 : 32 + len(shown)]
    source = torch.as_tensor(slots[shown], device="cuda", dtype=torch.long)
    match = (prob.mean(0).argmax(1) == source).cpu().numpy()
    mass = (
        (prob.gather(-1, source[None, :, None].expand(2, -1, 1)) >= MASS)
        .all(0)
        .flatten()
        .cpu()
        .numpy()
    )
    indices = np.flatnonzero(match & mass)
    probes = []
    for start, k in ((32, "discovery"), (64, "confirmation")):
        rows = folds[k]
        px, py = operands(known, legal, start, features, rows)
        anchor = (vc.logits(px, vc.v) / 0.1).softmax(-1)
        geom = anchor.clone()
        geom[:, 32:] = geometry_targets(vc, features, d2, rows, folds["reference"])
        probes.append((px, py, anchor, geom))
    criterion = DistillLoss(0, 200)
    initial_v = vc.v.clone()
    initial_buffer = None if vc.buffer is None else vc.buffer.clone()
    keep = vc.step(x, teacher, y, criterion, EPOCH, [], ARGS)
    torch.testing.assert_close(
        keep, vc.step(x, teacher, y, criterion, EPOCH, [], ARGS), rtol=0, atol=0
    )

    def proxy(w):
        return np.asarray(
            [
                [vc.probe(w, px, py, a), vc.probe(w, px, py, b)]
                for px, py, a, b in probes
            ]
        )

    baseline = proxy(keep)
    sealed = features[0, folds["sealed_probe"]]
    keep_logp = (vc.logits(sealed, keep) / 0.1).log_softmax(-1)
    pairs, transfer = [], []
    for j in indices:
        action = Action(32 + int(j), "move", int(slots[shown[j]]), int(target[j]))
        w = vc.step(x, teacher, y, criterion, EPOCH, [action], ARGS)
        pairs.append(proxy(w))
        transfer.append(
            ((vc.logits(sealed, w) / 0.1).log_softmax(-1) - keep_logp).cpu().numpy()
        )
    torch.testing.assert_close(vc.v, initial_v, rtol=0, atol=0)
    if initial_buffer is not None:
        torch.testing.assert_close(vc.buffer, initial_buffer, rtol=0, atol=0)
    scores = np.asarray(pairs).reshape(-1, 2, 2, 2)
    assert np.isfinite(baseline).all() and np.isfinite(scores).all()
    return (
        indices,
        baseline - scores,
        np.asarray(transfer, dtype=np.float32).reshape(-1, len(sealed), 196),
        {
            "source_mismatch": int((~match).sum()),
            "low_mass": int((~mass).sum()),
            "both": int((~match & ~mass).sum()),
            "eligible": len(indices),
        },
    )


def freeze_policies(ids, score, clip, static):
    rankings = {}
    for method, signal in (
        ("Direct", np.arange(len(ids))[::-1]),
        ("CLIPConfidence", clip),
        ("V0", score[:, 0, 0, 0]),
        ("CrossCF", score[:, 0, 1, 0]),
        ("StaticD2", static),
    ):
        order = np.lexsort((ids, -signal))
        if method in ("V0", "CrossCF"):
            m = int(method == "CrossCF")
            passes = (score[:, :, m, 0] > 0).all(1) & (score[:, :, m, 1] >= 0).all(1)
            order = order[passes[order]]
        rankings[method] = order
    rng = np.random.default_rng(SEED + 3)
    random_draws = np.stack([rng.permutation(len(ids)) for _ in range(64)])
    cases = {}
    for b in BUDGETS:
        cases[f"cap_{b}"] = {m: order[:b].tolist() for m, order in rankings.items()}
    for b in sorted(
        {min(len(rankings[m]), b) for m in ("V0", "CrossCF") for b in BUDGETS}
    ):
        cases[f"matched_{b}"] = {
            m: order[:b].tolist() for m, order in rankings.items() if len(order) >= b
        }
    return cases, random_draws, {m: len(o) for m, o in rankings.items()}


def paired_rank(x, y):
    if len(x) == 0 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    return float(spearmanr(x, y).statistic)


def seal(out, settings, truth):
    ids, slots = out["ids"], out["slots"]
    reference, q, shown = (out[k] for k in ("reference", "candidate", "shown"))
    mp, alignment = align(
        ids[reference].tolist(),
        dict(zip(ids.tolist(), slots.tolist(), strict=True)),
        truth,
    )
    table = np.array([mp.get(k, -1) for k in range(196)])
    y = truth[ids[q]]
    before = table[slots[q]] == y
    pos = np.searchsorted(q, shown)
    proposed = table[out["target"]] == y[pos]
    net = proposed.astype(np.float32) - before[pos].astype(np.float32)
    weights = heldout.bootstrap_counts(y)
    inverse = {c: s for s, c in mp.items()}
    probe_y = truth[ids[out["sealed_probe"]]]
    true_slots = np.array([inverse.get(c, -1) for c in probe_y])
    valid = true_slots >= 0
    transfer = out["transfer"][:, np.flatnonzero(valid), true_slots[valid]]
    ranks, cases = {}, {}
    for group, mask in (("All", y >= 0), ("Old", y < 98), ("New", y >= 98)):
        selected = mask[pos]
        cols = valid & (
            (probe_y < 98)
            if group == "Old"
            else (probe_y >= 98)
            if group == "New"
            else True
        )
        independent = (
            transfer[:, cols[valid]].mean(1) if cols.any() else np.zeros(len(pos))
        )
        ranks[group] = {}
        for m, signal in (
            ("CLIPDetection", out["detection"]),
            ("CLIPConfidence", out["confidence"]),
            ("StaticD2", out["static"]),
            ("V0", out["gain"][:, 0, 0, 0]),
            ("CrossCF", out["gain"][:, 0, 1, 0]),
        ):
            ranks[group][m] = {
                "proposals": int(selected.sum()),
                "probe_rows": int(cols.sum()),
                "repair_auroc": g1.auc(net[selected] > 0, signal[selected]),
                "spearman_net": paired_rank(signal[selected], net[selected]),
                "spearman_independent_logp_gain": paired_rank(signal, independent),
            }
        boot = weights[:, mask].copy()
        for case, plans in settings.items():
            rows = cases.setdefault(case, {}).setdefault(group, {})
            for m, plan in {"KEEP": [], **plans}.items():
                plan = np.asarray(plan, dtype=int)
                attempted = plan[selected[plan]]
                repaired = int((net[attempted] > 0).sum())
                harmed = int((net[attempted] < 0).sum())
                delta = np.zeros(len(q), dtype=np.float32)
                delta[pos[plan]] = net[plan]
                row = {
                    "total_accepted": len(plan),
                    "edits": len(attempted),
                    "repairs": repaired,
                    "harmful": harmed,
                    "wrong_to_wrong": int(
                        (~before[pos[attempted]] & ~proposed[attempted]).sum()
                    ),
                    "vs_keep": heldout.estimate(delta[mask], boot),
                }
                if m != "KEEP":
                    random = out["random"][:, : len(plan)]
                    fraction = np.bincount(random.ravel(), minlength=len(pos)) / len(
                        random
                    )
                    expected = np.zeros_like(delta)
                    expected[pos] = fraction * net
                    row["random"] = {
                        "total_edits": len(plan),
                        "mean_edits_in_group": float(fraction[selected].sum()),
                        "mean_repairs": float(fraction[selected & (net > 0)].sum()),
                        "mean_harmful": float(fraction[selected & (net < 0)].sum()),
                        "mean_wrong_to_wrong": float(
                            fraction[selected & ~before[pos] & ~proposed].sum()
                        ),
                        "vs_keep": heldout.estimate(expected[mask], boot),
                    }
                    row["vs_random"] = heldout.estimate((delta - expected)[mask], boot)
                rows[m] = row
    return {"policies": cases, "ranking": ranks, "alignment": alignment}


def report(result):
    lines = [
        "# Cross-geometry classifier counterfactual on Stanford Cars",
        "",
        (
            "R2-150 only. The 1,024 fresh query IDs exclude G1, held-out G6, Oracle MOVE and frozen E3 proposal IDs. "
            "The reference bank, discovery, confirmation, scoring context and independent sealed probe are disjoint from "
            "the candidate query pool and each other. Candidate labels are opened only to score the frozen edits. "
            "An independent 64-image probe measures transfer of the virtual update."
        ),
        "",
        (
            "CLIP locks the top 10% by alternative-minus-source support and fixes the best alternative slot. "
            "All policies share the exact pair and move 0.2 teacher mass. Both views must have enough mass "
            "and their mean teacher argmax must match the frozen source. Direct retains detection order. "
            "CLIPConfidence ranks by the gap to the next alternative. StaticD2 ranks the proposed slot's "
            "support minus source support in DINOv2."
        ),
        "",
        (
            "V0 reuses the original weight-normalized classifier SGD and opposite-view teacher probe. "
            "CrossCF changes only the unlabeled target: the ten nearest DINOv2 reference images contribute "
            "frozen SimGCD slot distributions, weighted by softmax cosine at temperature .07. Both "
            "require positive total gain and nonincreasing known CE on discovery and confirmation. "
            "All KEEP/MOVE steps start from the same saved classifier and SGD momentum. Candidates are evaluated "
            "one edit at a time from that common state; ranking and capping do not simulate a sequence of updates. "
            "Confirmation is ID-disjoint from discovery, shared across candidates and descriptive under repeated screening."
        ),
        "",
        (
            "Net gain counts slot corrections on the full query panel. Pointwise 95% paired query bootstrap "
            "intervals condition on the fixed reference and flags. Random filtering averages 64 frozen "
            "matched-count draws. Counts below cap are reported without padding. Matched tiers omit "
            "policies unable to fill that tier. Candidate ranking uses sealed repair correctness and "
            "transfer is mean true-slot log-probability gain on the independent probe. These are offline "
            "slot edits and one virtual classifier step, with no measurement of training effects."
        ),
        "",
        (
            f"CLIP proposed {result['proposed']} edits; {result['eligibility']['eligible']} were eligible. "
            f"V0 passed {result['available']['V0']}, CrossCF passed {result['available']['CrossCF']}."
        ),
        "",
        "| Tier / group / policy | Edits | Repairs / harmful / wrong→wrong | Net gain pp [95% CI] | Δ vs random pp [95% CI] |",
        "|---|---:|---|---|---|",
    ]

    def fmt(x):
        if x is None:
            return "—"
        lo, hi = x["ci95_pp"]
        return f"{x['pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    for case, groups in result["sealed"]["policies"].items():
        for g, rows in groups.items():
            if g != "All" and case.startswith("matched"):
                continue
            for m, v in rows.items():
                lines.append(
                    f"| {case} / {g} / {m} | {v['edits']} | {v['repairs']} / {v['harmful']} / "
                    f"{v['wrong_to_wrong']} | {fmt(v['vs_keep'])} | {fmt(v.get('vs_random'))} |"
                )
    lines += [
        "",
        "| Ranking rule (All) | Repair AUROC | Spearman net | Spearman independent gain |",
        "|---|---:|---:|---:|",
    ]
    for m, v in result["sealed"]["ranking"]["All"].items():
        cells = [
            "—" if v[k] is None else f"{v[k]:.3f}"
            for k in ("repair_auroc", "spearman_net", "spearman_independent_logp_gain")
        ]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Runtime {result['seconds']:.1f} seconds, including {result['virtual_seconds']:.1f} seconds for virtual updates.",
        "",
        "[Full counts, ranks and frozen prediction provenance](r2_cross_geometry_cf.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    smoke = parser.parse_args().smoke
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    g1_manifest = json.loads(g6.G1_REPORT.read_text())
    g6_receipt = json.loads(g6.OUT.read_text())
    assert file_digest(g6.G1_REPORT) == g6_receipt["g1_sha256"]
    previous = json.loads(heldout.OUT.read_text())
    assert previous["g6_sha256"] == file_digest(g6.OUT)
    ids = np.load(g1.WORK / "ids.npy")
    folds, excluded = split(ids, g1_manifest, previous)
    clip = heldout.checked_cache(
        g6.LOCAL / "clip-image.npz", g6_receipt["clip"]["cache_sha256"], ids
    )
    d2 = heldout.checked_cache(
        g5.LOCAL / "dinov2-cls.npz", g6_receipt["dinov2_cache_sha256"], ids
    )
    path = g1.WORK / "features.npy"
    digest = file_digest(path)
    assert digest == g6_receipt["dinov1_cache_sha256"]
    cache = np.load(path, mmap_mode="r")
    assert cache.shape == (3, 3, len(ids), 768)
    feat = F.normalize(torch.from_numpy(cache[2].copy()), dim=2).cuda()
    current = g4_snapshot_slots(g1_manifest, ids)
    q, ref = (folds[k] for k in ("candidate", "reference"))
    priority, target, detection, confidence = clip_proposals(clip, q, ref, current, ids)
    shown = q[priority]
    if smoke:
        shown, target, detection, confidence = (
            v[:8] for v in (shown, target, detection, confidence)
        )
    support = slot_support(d2, shown, ref, current)
    n = np.arange(len(shown))
    static = support[n, target] - np.maximum(-1, support[n, current[shown]])
    assert np.isfinite(static).all()
    model, loader, _, _, _, _ = build()
    model.eval()
    path = g1.WORK / "input/epoch-150.pt"
    sha = file_digest(path)
    assert sha == g1_manifest["snapshots"]["150"]["checkpoint_sha256"]
    ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == EPOCH
    model.load_state_dict(ckpt["model"])
    vc = VirtualClassifier(model[1].last_layer, checkpoint_optimizer(model, ckpt))
    lab_x, legal, known_ids = known_views(model[0], loader.dataset.labelled_dataset)
    assert not np.intersect1d(ids, known_ids).size and len(np.unique(known_ids)) == 96
    torch.cuda.synchronize()
    started = time.monotonic()
    eligible, gain, transfer, eligibility = evaluate_virtual(
        vc, feat, d2, folds, current, shown, target, lab_x, legal
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    lists, random_draws, available = freeze_policies(
        ids[shown[eligible]], gain, confidence[eligible], static[eligible]
    )
    if smoke:
        assert len(eligible) > 0
        print(
            "Actual checkpoint paired SGD smoke passed",
            eligibility,
            available,
            flush=True,
        )
        return
    data = {
        "ids": ids,
        "slots": current,
        **folds,
        "shown": shown[eligible],
        "target": target[eligible],
        "locked_proposals": shown,
        "locked_targets": target,
        "gain": gain,
        "transfer": transfer,
        "random": random_draws,
        "detection": detection[eligible],
        "confidence": confidence[eligible],
        "static": static[eligible],
        "known_scoring": known_ids[:32],
        "known_discovery": known_ids[32:64],
        "known_confirmation": known_ids[64:],
    }
    frozen = LOCAL / "frozen.npz"
    np.savez(frozen, **data)
    result = {
        "base": "cde8525",
        "checkpoint_sha256": sha,
        "frozen_sha256": file_digest(frozen),
        "g1_manifest_sha256": g1_manifest["snapshots"]["150"]["frozen_sha256"],
        "g6_confirm_sha256": file_digest(heldout.OUT),
        "seed": SEED,
        "move_mass": MASS,
        "proposed": len(shown),
        "eligibility": eligibility,
        "available": available,
        "excluded_query_ids": excluded,
        "fold_sizes": {k: len(v) for k, v in folds.items()},
        "virtual_lr": vc.group["lr"],
        "virtual_seconds": elapsed,
        "frozen_policies": lists,
    }
    receipt = LOCAL / "frozen-provenance.json"
    write_json(receipt, json_native(result))
    print(
        "Frozen cross-geometry proposals, scores and policies",
        availability_message(available),
        flush=True,
    )

    from scripts.audit_r2_evidence import sealed_labels

    result = json.loads(receipt.read_text())
    assert file_digest(frozen) == result["frozen_sha256"]
    with np.load(frozen, allow_pickle=False) as f:
        result["sealed"] = seal(dict(f), result["frozen_policies"], sealed_labels())
    result["seconds"] = time.monotonic() - start
    report(result)
    write_json(OUT, json_native(result))


def g4_snapshot_slots(info, ids):
    path = g1.LOCAL / "frozen-150.npz"
    assert file_digest(path) == info["snapshots"]["150"]["frozen_sha256"]
    with np.load(path, allow_pickle=False) as f:
        np.testing.assert_array_equal(f["ids"], ids)
        return f["slots"]


def availability_message(available):
    return ", ".join(f"{k}={v}" for k, v in available.items())


if __name__ == "__main__":
    main()
