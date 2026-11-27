"""Matched CLIP MOVE continuation at mass 0.8 from the R2-100 snapshot."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts import audit_clip_repair as repair
from scripts import audit_g4 as g4
from scripts import audit_g6_confirm as holdout
from scripts import audit_move_strength_training as match
from scripts import audit_oracle_move as audit
from scripts.audit_e3_proposals import json_native

LOCAL = repair.g1.WORK / "early-move"
REPORT = Path("reports/CRISH_EARLY_MOVE_TRAINING.md")
OUT = Path("reports/r2_early_move_training.json")
ARMS = ("E0", "E1", "E2")
MASSES = {"E0": 0.0, "E1": 0.8, "E2": 0.8}
START, END = 100, 102


def setup():
    model, loader, _, center, _, _ = audit.build()
    ds = loader.dataset
    blind = ds.unlabelled_dataset.ids
    ids = np.r_[ds.labelled_dataset.uq_idxs, blind]
    prior = json.loads(repair.HISTORY.read_text())
    panel = np.asarray(prior["panel_ids"])
    assert len(np.unique(panel)) == 588 and set(panel) <= set(blind)
    archive = repair.g1.WORK / "input/epoch-100.pt"
    sha = audit.file_digest(archive)
    report = json.loads(repair.g6.G1_REPORT.read_text())
    assert (
        sha
        == report["snapshots"]["100"]["checkpoint_sha256"]
        == prior["checkpoint_sha256"]
    )
    base = audit.load_checkpoint(archive)
    assert base["epoch"] == base["scheduler"]["last_epoch"] == START
    assert base["scheduler"]["T_max"] == 200
    for k, v in model[0].state_dict().items():
        if not k.startswith("blocks.11."):
            assert torch.equal(v.cpu(), base["model"][f"0.{k}"]), k
    ck = g4.snapshot(START, report)
    np.testing.assert_array_equal(blind, ck["ids"])
    clip_ref = json.loads(repair.g6.OUT.read_text())
    clip_sha = clip_ref["clip"]["cache_sha256"]
    vectors = holdout.checked_cache(repair.g6.LOCAL / "clip-image.npz", clip_sha, blind)
    path = repair.g1.WORK / "features.npy"
    d1_sha = audit.file_digest(path)
    assert d1_sha == clip_ref["dinov1_cache_sha256"]
    features = np.load(path, mmap_mode="r")
    assert features.shape == (3, 3, len(blind), 768)
    model.load_state_dict(base["model"])
    model.eval()
    with torch.no_grad():
        x = F.normalize(torch.from_numpy(features[1, 0].copy()), dim=1).cuda()
        np.testing.assert_array_equal(
            model[1].last_layer(x).argmax(1).cpu().numpy(), ck["slots"]
        )
    aug = F.normalize(torch.from_numpy(features[1, 1:].copy()), dim=2).cuda()
    proposals, n, ref = repair.lock_policies(
        model, blind, ck["slots"], vectors, aug, panel
    )
    assert n and not set(ref) & set(panel)
    pool = {k: (s, d, i) for i, (k, s, d, _) in enumerate(proposals)}
    direct = {k: v for k, v in pool.items() if v[-1] < n}
    assert len(direct) == n and not set(pool) & set(panel)
    cached = torch.load(
        repair.g1.WORK / "oracle-move/step0-panel-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert cached["panel_ids"] == panel.tolist() and cached["checkpoint_sha256"] == sha
    plan = repair.schedule(loader, ids, panel, base, start_epoch=START, end_epoch=END)
    sig = {
        "base": "45ad693",
        "checkpoint_sha256": sha,
        "clip_cache_sha256": clip_sha,
        "dinov1_cache_sha256": d1_sha,
        "panel_ids": panel.tolist(),
        "reference_ids": ref,
        "pool_n": len(pool),
        "locked_n": len(direct),
        "masses": MASSES,
        "init_eligibility_mass": 0.2,
        "classifier_manifest_sha256": report["snapshots"]["100"]["frozen_sha256"],
        "draws": {
            str(e): {"rows": repair.digest_array(r), "seeds": repair.digest_array(s)}
            for e, (r, s) in plan.items()
        },
    }
    return (
        model,
        ds,
        center,
        ids,
        base,
        plan,
        panel,
        cached["tokens"],
        pool,
        direct,
        ck,
        sig,
        proposals,
    )


def paired_batch(models, batch, pool, direct, seeds, epoch, criterion):
    candidates, counters = {}, {}
    for a, lookup in (("E1", direct), ("E2", pool)):
        repair.seed_all(seeds[1])
        candidates[a], counters[a] = repair.live_pool(
            models[a][0], batch, lookup, strength=MASSES[a]
        )
    n = min(len(candidates["E1"]), len(candidates["E2"]))
    rng = np.random.default_rng(int(seeds[2]))
    random = [candidates["E2"][i] for i in rng.permutation(len(candidates["E2"]))[:n]]
    details = {
        "matching": {
            "E1_eligible": len(candidates["E1"]),
            "E2_eligible": len(candidates["E2"]),
            "random_supply_shortfall": len(candidates["E1"]) - n,
            "applied": n,
        }
    }
    for a, actions in zip(ARMS, ([], candidates["E1"][:n], random), strict=True):
        repair.seed_all(seeds[1])
        model, opt, _ = models[a]
        moves, row = audit.training_step(
            model,
            opt,
            batch,
            criterion,
            match.selector(actions, pool if a == "E2" else direct, MASSES[a]),
            epoch=epoch - 1,
            mass=MASSES[a],
        )
        assert len(moves) == (0 if a == "E0" else n)
        details[a] = {"moves": moves, "eligibility": dict(counters.get(a, {})), **row}
    return details


def contrast(early, late):
    result = {}
    for dataset, result_data, before, keep, repair_arm, random_arm in (
        ("R2-100", early, 100, "E0", "E1", "E2"),
        ("R2-150", late, 150, "S0", "S2", "S3"),
    ):
        result[dataset] = {}
        for offset in (1, 2):
            epoch = before + offset
            train = result_data["training"][str(epoch)]
            a, b, c = (
                result_data["endpoint"][f"{epoch}:{arm}"]["panel"]
                for arm in (repair_arm, keep, random_arm)
            )
            result[dataset][str(offset)] = {
                "completed_epoch": epoch,
                "clip_edits": train[repair_arm]["edits"],
                "random_edits": train[random_arm]["edits"],
                "exposure": train[repair_arm]["exposure"],
                "target_mass": train[repair_arm]["mass_total"],
                "clip_minus_keep": differences(a, b),
                "clip_minus_random": differences(a, c),
            }
    return result


def differences(a, b):
    result = {}
    for g in match.GROUPS:
        result[g] = {
            "fixed_pp": 100 * (a["fixed"][g]["accuracy"] - b["fixed"][g]["accuracy"]),
            "rematched_pp": 100
            * (a["rematched"][g]["accuracy"] - b["rematched"][g]["accuracy"]),
            "purity5": a["geometry"][g]["purity5"] - b["geometry"][g]["purity5"],
            "centroid_margin": a["geometry"][g]["centroid_margin"]
            - b["geometry"][g]["centroid_margin"],
        }
    return result


def render(result):
    lines = [
        "# Early CLIP MOVE continuation on Stanford Cars",
        "",
        (
            "E0 keeps supervision. E1 moves 0.8 teacher mass using R2-100 CLIP proposals. "
            "E2 samples eligible source occurrences uniformly within each batch and uses the "
            "same frozen CLIP retriever and mass. The top 5% ranking is recalculated using R2-100 "
            "classifier slots and the shared CLIP image cache, with the previous .2 initial eligibility "
            "screen. Each applied edit requires the frozen source to match the live teacher and "
            "at least .8 source mass in both views. E1 is capped by the available E2 rows per batch."
        ),
        "",
        (
            "All arms use identical conditioned weighted draws, crops and SimGCD losses, with R2 "
            "relational loss disabled as in the late strength diagnostic. SGD and cosine scheduling "
            "resume from R2-100. The fixed panel has 588 images from the original transductive "
            "training split. During continuation, it is excluded from draws, proposals and reference "
            "neighbors. Actual edited IDs are reported separately. Labels are first opened after "
            "the two checkpoints and endpoint predictions are complete."
        ),
        "",
        "| Epoch / arm | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure | Target mass |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for key, row in result["endpoint"].items():
        e, arm = key.split(":")
        p = row["panel"]
        fixed, rematch = [
            " / ".join(f"{100 * p[k][g]['accuracy']:.2f}" for g in match.GROUPS)
            for k in ("fixed", "rematched")
        ]
        g = p["geometry"]["all"]
        v = (
            result["training"][e][arm]
            if e != "100"
            else {"edits": 0, "exposure": 0.0, "mass_total": 0.0}
        )
        lines.append(
            f"| {e} / {arm} | {fixed} | {rematch} | {g['purity5']:.4f} | {g['centroid_margin']:+.4f} | "
            f"{v['edits']} / {100 * v['exposure']:.3f}% | {v['mass_total']:.1f} |"
        )
    lines.extend(
        [
            "",
            "| Epoch | CLIP live | Random live | Random supply shortfall | Matched |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for epoch, row in result["training"].items():
        m = row["matching"]
        vals = [
            m[k]
            for k in (
                "E1_eligible",
                "E2_eligible",
                "random_supply_shortfall",
                "applied",
            )
        ]
        lines.append(f"| {epoch} | " + " | ".join(map(str, vals)) + " |")
    lines += [
        "",
        "| Epoch / arm | Correct / harmful / wrong-to-wrong / unresolved | Pre-step teacher source / destination | CLIP source cohort true-slot rank |",
        "|---|---|---|---:|",
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.4f}"

    for e in ("101", "102"):
        for a in ARMS[1:]:
            r = result["applied_semantics"].get(e, {}).get(a, {}).get("all", {})
            counts = " / ".join(
                str(r.get("categories", {}).get(k, 0))
                for k in ("correct", "harmful", "wrong_to_wrong", "unresolved")
            )
            p = " / ".join(
                fmt(r.get(k)) for k in ("teacher_source", "teacher_destination")
            )
            rank = result["endpoint"][f"{e}:{a}"]["cohorts"]["CLIP_sources"]["all"][
                "true_slot_mean_rank"
            ]
            lines.append(f"| {e} / {a} | {counts} | {p} | {fmt(rank)} |")
    lines += [
        "",
        "## Descriptive comparison with R2-150",
        "",
        (
            "The late comparison uses S2/S3 at mass .8 and S0 KEEP. It conditions on the late "
            "experiment's shared .2/.8 live eligibility. The schedules, initial weights, proposals and realized "
            "doses differ between starting checkpoints. These contrasts describe the runs and do "
            "not identify an effect of starting earlier."
        ),
        "",
        "| Start / completed offset | Matched edits | Fixed All / New Δ vs KEEP (pp) | Fixed All / New Δ vs random (pp) | kNN-5 / margin Δ vs KEEP |",
        "|---|---:|---|---|---|",
    ]
    for base, rows in result["early_vs_late"].items():
        for offset, row in rows.items():
            k, r = row["clip_minus_keep"], row["clip_minus_random"]
            vs_keep, vs_random = (
                " / ".join(f"{d[g]['fixed_pp']:+.2f}" for g in ("all", "new"))
                for d in (k, r)
            )
            lines.append(
                f"| {base} / {offset} | {row['clip_edits']} | {vs_keep} | {vs_random} | "
                f"{k['all']['purity5']:+.4f} / {k['all']['centroid_margin']:+.4f} |"
            )
    lines += [
        "",
        f"Elapsed in this invocation {result['seconds'] / 60:.1f} minutes. Paired checkpoints and per-step ledgers are local.",
        "",
        "[Subgroup probabilities, ranks, prediction transitions, and checkpoint digests](r2_early_move_training.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--sealed-only", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    (
        model,
        ds,
        center,
        ids,
        base,
        plan,
        panel,
        tokens,
        pool,
        direct,
        ck,
        sig,
        proposals,
    ) = setup()
    if not args.smoke:
        policy = LOCAL / "frozen-policy.json"
        lock = {"signature": sig, "proposals": proposals}
        if policy.is_file():
            assert json.loads(policy.read_text()) == json_native(lock)
        else:
            audit.write_json(policy, json_native(lock))
        sig = {**sig, "policy_sha256": audit.file_digest(policy)}
        print("Locked R2-100 CLIP proposals", len(direct), "of", len(pool), flush=True)
    if not args.sealed_only:
        match.train(
            model,
            ds,
            ids,
            base,
            plan,
            pool,
            direct,
            sig,
            args.smoke,
            names=ARMS,
            masses=MASSES,
            local=LOCAL,
            paired_step=paired_batch,
        )
    if args.smoke:
        return
    saved, hashes, events = match.load_finished(
        sig, plan, ids, pool, direct, names=ARMS, masses=MASSES, local=LOCAL
    )
    out, frozen = match.predictions(
        model,
        ds,
        center,
        base,
        saved,
        panel,
        tokens,
        events,
        sig,
        names=ARMS,
        local=LOCAL,
    )
    assert (
        audit.file_digest(LOCAL / "frozen-endpoint-predictions.npz")
        == frozen["predictions_sha256"]
    )

    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    fixed, coverage = audit.align(
        sig["reference_ids"],
        dict(zip(ck["ids"].tolist(), ck["slots"].tolist(), strict=True)),
        truth,
    )
    result = {
        "panel_n": len(panel),
        "reference_n": len(sig["reference_ids"]),
        "provenance": {
            k: v
            for k, v in frozen.items()
            if k not in ("panel_ids", "reference_ids", "source_ids")
        },
        "checkpoint_sha256": hashes,
        "alignment": coverage,
        "training": saved[str(END)]["records"],
        "endpoint": match.endpoints(
            out, fixed, truth, events, initial_epoch=START, names=ARMS
        ),
        "applied_semantics": match.applied_semantics(events, truth, fixed),
        "late_report_sha256": audit.file_digest(match.OUT),
    }
    result["early_vs_late"] = contrast(result, json.loads(match.OUT.read_text()))
    result["seconds"] = time.monotonic() - start
    render(result)
    audit.write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
