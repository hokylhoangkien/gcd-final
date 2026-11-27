"""Locked held-out confirmation of the five G6 assignment policies."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from crish.experiment import file_digest
from relational.state import write_json
from scripts import audit_g1 as g1
from scripts import audit_g4 as g4
from scripts import audit_g5 as g5
from scripts import audit_g6 as g6
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_g7 import pack

LOCAL = g1.WORK / "g6-confirm"
MD = Path("reports/CRISH_VLM_HELDOUT_CONFIRMATION.md")
OUT = Path("reports/r2_vlm_heldout_confirmation.json")
COVERAGES = (10, 20, 40)
POLICIES = {"D1": "D1", "D2": "D1+D2", "CLIP": "CLIP", "All3": "All3"}
QUERY_SEED = 890819
RANDOM_SEED = 803116
BOOTSTRAP_SEED = 190628
RANDOM_DRAWS = 64
BOOTSTRAPS = 2000
QUERY_N = 1024


def partition(ids, g1_report):
    prior = json.loads((g1.WORK / "e3-frozen-proposals.json").read_text())
    oracle = json.loads(Path("reports/r2_oracle_move_trajectory.json").read_text())
    sources = {
        "g1": set(g1_report["query_ids"]),
        "oracle_panel": set(oracle["panel_ids"]),
        "e3_proposals": {
            row["id"] for s in prior["snapshots"].values() for row in s["proposals"]
        },
    }
    excluded = set().union(*sources.values())
    rng = np.random.default_rng(QUERY_SEED)
    eligible = np.flatnonzero(~np.isin(ids, sorted(excluded)))
    q = np.sort(rng.choice(eligible, QUERY_N, replace=False))
    bank = np.setdiff1d(np.arange(len(ids)), q)
    assert not (set(ids[q]) & excluded) and not np.intersect1d(q, bank).size
    uniform = np.stack([rng.choice(len(bank), 256, replace=False) for _ in q])
    return (
        q,
        bank,
        uniform,
        {**{k: len(v) for k, v in sources.items()}, "union": len(excluded)},
    )


def checked_cache(path, sha, ids):
    if file_digest(path) != sha:
        raise ValueError(f"Cache digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["ids"], ids)
        values = saved["features"]
    assert values.shape[0] == len(ids) and np.isfinite(values).all()
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-5)
    return torch.from_numpy(values).cuda()


def freeze(ids, q, bank, uniform, slots, past, d1, d2, clip, rng):
    a, b, c = (x[q] @ x[bank].T for x in (d1, d2, clip))
    pairs = np.c_[uniform, a.topk(64).indices.cpu().numpy()]
    out = {"ids": ids, "q": q, "bank": bank, "slots": slots, "pairs": bank[pairs]}
    for method, sim in (
        ("D1", a),
        ("D2", b),
        ("CLIP", c),
        ("All3", (a + b + c) / 3),
        ("D1+D2", (a + b) / 2),
    ):
        fields = pack(sim, q, bank, slots, pairs)
        out.update({f"{method}_{k}": fields[k] for k in g4.FIELDS})
    if past is not None:
        mapping, _ = g4.slot_identity(past, slots, bank, q)
        out["history_slots"] = past
        out["aligned_slot_stable"] = mapping[past[q]] == slots[q]
    g4.freeze_flags(out, POLICIES)
    # Reuse one random priority per replicate across policies and coverage levels.
    priorities = np.stack([rng.permutation(len(q)) for _ in range(RANDOM_DRAWS)])
    for det, ret in POLICIES.items():
        dest = out[f"{ret}_dest"][:, 0]
        allowed = (dest >= 0) & (dest != slots[q])
        random_order = priorities[allowed[priorities]].reshape(
            RANDOM_DRAWS, int(allowed.sum())
        )
        for p in COVERAGES:
            flags = out[f"{det}_flag_{p}"]
            count = int(allowed[flags].sum())
            out[f"{det}_random_{p}"] = random_order[:, :count]
    return out


def edits(before, after, flags, valid):
    chosen = np.zeros(len(before), dtype=bool)
    chosen[flags] = True
    applied = chosen & valid
    repaired = applied & ~before & after
    harmed = applied & before & ~after
    return {
        "selected": chosen,
        "edits": applied,
        "correct_repairs": repaired,
        "harmful_edits": harmed,
        "wrong_to_wrong": applied & ~before & ~after,
        "net": repaired.astype(np.float32) - harmed.astype(np.float32),
    }


def bootstrap_counts(labels):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.zeros((BOOTSTRAPS, len(labels)), dtype=np.float32)
    for mask in (labels < 98, labels >= 98):
        count = int(mask.sum())
        if count:
            weights[:, mask] = rng.multinomial(
                count, np.full(count, 1 / count), size=BOOTSTRAPS
            )
    return weights


def estimate(delta, weights):
    if not delta.size:
        return None
    values = weights @ delta * (100 / len(delta))
    return {
        "pp": float(delta.mean() * 100),
        "ci95_pp": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def corrections(out, truth, *, policies=POLICIES, baseline="D1"):
    ids, q, bank, slots = (out[k] for k in ("ids", "q", "bank", "slots"))
    mp, _ = align(
        ids[bank].tolist(), dict(zip(ids.tolist(), slots.tolist(), strict=True)), truth
    )
    table = np.array([mp.get(i, -1) for i in range(196)] + [-1])
    y = truth[ids[q]]
    before = table[slots[q]] == y
    weights = bootstrap_counts(y)
    result = {}
    for p in COVERAGES:
        actual, controls = {"KEEP": edits(before, before, [], before)}, {}
        for det, ret in policies.items():
            dest = out[f"{ret}_dest"][:, 0]
            allowed = (dest >= 0) & (dest != slots[q])
            after = table[dest] == y
            actual[det] = edits(before, after, out[f"{det}_flag_{p}"], allowed)
            random = [
                edits(before, after, flags, allowed)
                for flags in out[f"{det}_random_{p}"]
            ]
            counts = [int(v["edits"].sum()) for v in random]
            assert all(n == int(actual[det]["edits"].sum()) for n in counts)
            controls[det] = {
                k: np.mean([v[k] for v in random], axis=0, dtype=np.float32)
                for k in actual[det]
            }
        result[p] = {}
        for group, mask in (("All", y >= 0), ("Old", y < 98), ("New", y >= 98)):
            n = int(mask.sum())
            boot = weights[:, mask].copy()
            rows = {}
            for name, a in actual.items():
                d = a["net"][mask]
                rows[name] = {
                    **{k: int(v[mask].sum()) for k, v in a.items() if k != "net"},
                    "n": n,
                    "net_correct": int(d.sum()),
                    "accuracy_after": float(before[mask].mean() + d.mean())
                    if n
                    else None,
                    "vs_keep": estimate(d, boot),
                    f"vs_{baseline.lower()}": estimate(
                        d - actual[baseline]["net"][mask], boot
                    ),
                }
                if name in controls:
                    control = controls[name]
                    rows[name]["matched_random"] = {
                        **{
                            k: float(v[mask].sum())
                            for k, v in control.items()
                            if k != "net"
                        },
                        "vs_keep": estimate(control["net"][mask], boot),
                    }
                    rows[name]["vs_matched_random"] = estimate(
                        d - control["net"][mask], boot
                    )
            result[p][group] = rows
    return result


def report(result):
    lines = [
        "# Locked held-out G6 confirmation on Stanford Cars",
        "",
        (
            "The primary endpoint is CLIP detection plus CLIP retrieval at 10% coverage "
            "on R2-150. Epoch 100 and 20%/40% coverage are secondary. The query sample excludes "
            "all G1 queries, the Oracle MOVE panel and previous frozen E3 proposal IDs. "
            "This is a fresh query partition within the same Cars trajectory. Independent dataset "
            "or training-seed replication would require separate artifacts."
        ),
        "",
        (
            f"The {QUERY_N:,} queries are disjoint from the retrieval bank, which contains all other "
            "unlabeled Cars IDs. Alternative destinations use maximum peer cosine per slot. Each risk "
            "score compares best alternative support against current-slot support. D1 means fine-tuned "
            "DINOv1. D2 detection routes to equal D1+D2 retrieval. All3 averages fine-tuned D1, frozen D2 "
            "and CLIP cosine scores. Exact flags, destinations and random draws were stored before opening "
            "labels. Bank-only Hungarian alignment maps slots to classes during evaluation."
        ),
        "",
        (
            f"Random controls match each policy's number of actionable edits, use its same retriever "
            f"and average {RANDOM_DRAWS} uniform label-blind draws. Net gain is repairs minus harmful edits "
            "as a percentage of all queries in the group. The 95% paired query-bootstrap intervals "
            f"use {BOOTSTRAPS:,} Old/New-stratified resamples with fixed bank, alignment, scores and flags. "
            "Intervals are pointwise and conditional on these snapshots and frozen random controls. "
            "Shared bank dependencies and training-run variability are outside these intervals."
        ),
        "",
        "| Epoch / group / policy (10%) | Edits | Repairs / harmful / wrong-to-wrong | Gain vs KEEP pp [95% CI] | Gain vs random pp [95% CI] | Stable error AUC / recall |",
        "|---|---:|---|---|---|---|",
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    def interval(v):
        if v is None:
            return "—"
        lo, hi = v["ci95_pp"]
        return f"{v['pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    for epoch, snapshot in result["snapshots"].items():
        for group, policies in snapshot["corrections"][10].items():
            for name, row in policies.items():
                m = snapshot["detection"].get(name, {}).get(group, {})
                stable = f"{fmt(m.get('stable_error_auc'))} / {fmt(m.get('flags', {}).get(10, {}).get('stable_error_recall'))}"
                lines.append(
                    f"| {epoch} / {group} / {name} | {row['edits']} | "
                    f"{row['correct_repairs']} / {row['harmful_edits']} / {row['wrong_to_wrong']} | "
                    f"{interval(row['vs_keep'])} | {interval(row.get('vs_matched_random'))} | {stable} |"
                )
    p = result["snapshots"]["150"]["corrections"][10]["All"]["CLIP"]
    lines.extend(
        [
            "",
            f"Primary CLIP difference against D1 was {interval(p['vs_d1'])} pp.",
            "",
            "## Secondary global net gains",
            "",
            "| Epoch / coverage | D1 vs KEEP | D2 vs KEEP | CLIP vs KEEP | All3 vs KEEP |",
            "|---|---|---|---|---|",
        ]
    )
    for epoch, snap in result["snapshots"].items():
        for p in (20, 40):
            vals = [
                interval(snap["corrections"][p]["All"][name]["vs_keep"])
                for name in POLICIES
            ]
            lines.append(f"| {epoch} / {p}% | " + " | ".join(vals) + " |")
    lines += [
        "",
        (
            "Stable errors follow G4: both checkpoint assignments are wrong and the assigned "
            "slot is unchanged under label-blind bank matching. Epoch 100 has no historical stable-error outcome."
        ),
        "",
        f"Runtime {result['elapsed_seconds']:.1f} seconds.",
        "",
        "[All subgroup and coverage metrics, matched random means and frozen provenance](r2_vlm_heldout_confirmation.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    start = time.monotonic()
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(exist_ok=True)
    g1_report = json.loads(g6.G1_REPORT.read_text())
    g6_report = json.loads(g6.OUT.read_text())
    assert file_digest(g6.G1_REPORT) == g6_report["g1_sha256"]
    path = g1.WORK / "features.npy"
    d1_sha = file_digest(path)
    assert d1_sha == g6_report["dinov1_cache_sha256"]
    ids = np.load(g1.WORK / "ids.npy")
    cs = np.load(path, mmap_mode="r")
    assert cs.shape == (3, 3, len(ids), 768)
    d2 = checked_cache(
        g5.LOCAL / "dinov2-cls.npz", g6_report["dinov2_cache_sha256"], ids
    )
    clip = checked_cache(
        g6.LOCAL / "clip-image.npz", g6_report["clip"]["cache_sha256"], ids
    )
    q, bank, uniform, excluded = partition(ids, g1_report)
    result = {
        "base": "e0898d9",
        "primary": {
            "checkpoint": 150,
            "detector": "CLIP",
            "retriever": "CLIP",
            "coverage": 10,
        },
        "policies": {"KEEP": None, **POLICIES},
        "coverage_percent": COVERAGES,
        "query_ids": ids[q].tolist(),
        "bank_n": len(bank),
        "excluded_id_counts": excluded,
        "seeds": {
            "query": QUERY_SEED,
            "random": RANDOM_SEED,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "random_draws": RANDOM_DRAWS,
        "bootstrap_replicates": BOOTSTRAPS,
        "g1_sha256": g6_report["g1_sha256"],
        "g6_sha256": file_digest(g6.OUT),
        "feature_sha256": {
            "D1": d1_sha,
            "D2": g6_report["dinov2_cache_sha256"],
            "CLIP": g6_report["clip"]["cache_sha256"],
        },
        "snapshots": {},
    }
    previous = None
    for i, epoch in enumerate(g1.EPOCHS, 1):
        key = str(epoch)
        old_manifest = g4.snapshot(epoch, g1_report)
        np.testing.assert_array_equal(old_manifest["ids"], ids)
        slots = old_manifest["slots"]
        d1 = F.normalize(torch.from_numpy(cs[i, 0].copy()), dim=1).cuda()
        out = freeze(
            ids,
            q,
            bank,
            uniform,
            slots,
            previous,
            d1,
            d2,
            clip,
            np.random.default_rng(RANDOM_SEED),
        )
        previous = slots
        frozen = LOCAL / f"frozen-{epoch}.npz"
        np.savez(frozen, **out)
        result["snapshots"][key] = {
            "frozen_sha256": file_digest(frozen),
            "checkpoint_sha256": g1_report["snapshots"][key]["checkpoint_sha256"],
        }
        print("Frozen held-out G6", epoch, flush=True)
    write_json(LOCAL / "frozen-provenance.json", json_native(result))

    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    for key, row in result["snapshots"].items():
        frozen = LOCAL / f"frozen-{key}.npz"
        assert file_digest(frozen) == row["frozen_sha256"]
        with np.load(frozen, allow_pickle=False) as f:
            out = dict(f)
        routed = {**out, "D2_dest": out["D1+D2_dest"]}
        detection, coverage = g4.sealed(routed, truth, POLICIES, baseline="D1")
        row.update(
            detection=detection,
            bank_alignment=coverage,
            corrections=corrections(out, truth),
        )
        print("Sealed held-out G6", key, flush=True)
    result["elapsed_seconds"] = time.monotonic() - start
    report(result)
    write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
