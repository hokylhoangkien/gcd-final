"""Extend the sealed, matched early CLIP MOVE continuation to epoch 105."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from crish.experiment import rng_state
from scripts import audit_clip_repair as repair
from scripts import audit_early_move_training as early
from scripts import audit_g4 as g4
from scripts import audit_move_strength_training as match
from scripts import audit_oracle_move as audit
from scripts.audit_e3_proposals import json_native

LOCAL = repair.g1.WORK / "sustained-move"
MD = Path("reports/CRISH_SUSTAINED_MOVE_TRAINING.md")
OUT = Path("reports/r2_sustained_move_training.json")
START, RESUME, END = 100, 102, 105
ARMS, MASSES = early.ARMS, early.MASSES


def setup():
    previous = json.loads(early.OUT.read_text())
    p = early.LOCAL / "frozen-policy.json"
    digest = audit.file_digest(p)
    assert digest == previous["provenance"]["policy_sha256"]
    policy = json.loads(p.read_text())
    original = {**policy["signature"], "policy_sha256": digest}
    assert all(
        original[k] == v
        for k, v in previous["provenance"].items()
        if k != "predictions_sha256"
    )
    assert original["masses"] == MASSES
    model, loader, _, center, _, _ = audit.build()
    ds = loader.dataset
    blind = ds.unlabelled_dataset.ids
    ids = np.r_[ds.labelled_dataset.uq_idxs, blind]
    panel = np.asarray(original["panel_ids"])
    original_oracle = json.loads(repair.HISTORY.read_text())
    assert panel.tolist() == original_oracle["panel_ids"] and len(panel) == 588
    assert not set(panel) & set(original["reference_ids"])
    assert set(original["reference_ids"]) == set(blind) - set(panel)
    f = repair.g1.WORK / "input/epoch-100.pt"
    assert audit.file_digest(f) == original["checkpoint_sha256"]
    base = audit.load_checkpoint(f)
    assert base["epoch"] == base["scheduler"]["last_epoch"] == START
    for k, v in model[0].state_dict().items():
        if not k.startswith("blocks.11."):
            assert torch.equal(v.cpu(), base["model"][f"0.{k}"]), k
    info = json.loads(repair.g6.G1_REPORT.read_text())
    assert (
        original["classifier_manifest_sha256"]
        == info["snapshots"]["100"]["frozen_sha256"]
    )
    ck = g4.snapshot(START, info)
    np.testing.assert_array_equal(ck["ids"], blind)
    tokens = torch.load(
        repair.g1.WORK / "oracle-move/step0-panel-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert (
        tokens["checkpoint_sha256"] == original["checkpoint_sha256"]
        and tokens["panel_ids"] == panel.tolist()
    )
    pool = {k: (s, d, i) for i, (k, s, d, _) in enumerate(policy["proposals"])}
    direct = {k: v for k, v in pool.items() if v[-1] < original["locked_n"]}
    assert len(direct) == original["locked_n"] and len(pool) == original["pool_n"]
    assert not set(panel) & set(pool)
    sampled = repair.schedule(
        loader, ids, panel, base, start_epoch=START, end_epoch=END
    )
    receipts = {
        str(e): {"rows": repair.digest_array(r), "seeds": repair.digest_array(s)}
        for e, (r, s) in sampled.items()
    }
    assert all(receipts[k] == v for k, v in original["draws"].items())
    past = {e: p for e, p in sampled.items() if e <= RESUME}
    next_plan = {e: p for e, p in sampled.items() if e > RESUME}
    completed, hashes, events = match.load_finished(
        original, past, ids, pool, direct, names=ARMS, masses=MASSES, local=early.LOCAL
    )
    assert hashes == previous["checkpoint_sha256"]
    initial = completed[str(RESUME)]
    assert initial["records"] == previous["training"]
    assert all(initial["state"][a]["scheduler"]["last_epoch"] == RESUME for a in ARMS)
    sig = {
        **original,
        "base": "c439df7",
        "draws": receipts,
        "resume_epoch": RESUME,
        "resume_sha256": hashes[str(RESUME)],
        "prior_report_sha256": audit.file_digest(early.OUT),
    }
    return (
        model,
        ds,
        center,
        ids,
        base,
        next_plan,
        panel,
        tokens["tokens"],
        pool,
        direct,
        ck,
        sig,
        previous,
        completed,
        hashes,
        events,
    )


def verify(model, resume):
    for name, state in resume["state"].items():
        opt, sched = audit.restore(model, state)
        now = opt.state_dict()
        old = state["optimizer"]
        assert (
            sched.state_dict() == state["scheduler"]
            and now["param_groups"] == old["param_groups"]
        )
        assert now["state"].keys() == old["state"].keys()
        for key, item in old["state"].items():
            assert item.keys() == now["state"][key].keys()
            for field, value in item.items():
                copy = now["state"][key][field]
                if torch.is_tensor(value):
                    assert torch.equal(copy.cpu(), value)
                else:
                    assert copy == value
        expected = state["rng"]
        current = rng_state()
        assert current[0] == expected[0] and torch.equal(current[2], expected[2])
        assert current[1][0] == expected[1][0] and current[1][2:] == expected[1][2:]
        np.testing.assert_array_equal(current[1][1], expected[1][1])
        assert all(
            torch.equal(x, y) for x, y in zip(current[3], expected[3], strict=True)
        )
        print("Verified epoch-102 optimizer, scheduler and RNG", name, flush=True)


def cumulative(training):
    out = {}
    totals = {
        a: {"edits": 0, "mass": 0.0, "seen": 0, "stale": 0, "low_mass": 0, "rows": 0}
        for a in ARMS
    }
    for epoch, row in sorted(training.items()):
        out[epoch] = {}
        for name, record in ((a, row[a]) for a in ARMS):
            stats = record["eligibility"]
            t = totals[name]
            t["edits"] += record["edits"]
            t["mass"] += record["mass_total"]
            t["seen"] += stats.get("encounters", 0)
            t["stale"] += stats.get("stale", 0)
            t["low_mass"] += stats.get("low_mass", 0)
            t["rows"] += row["rows"]
            out[epoch][name] = {**t, "exposure": t["edits"] / t["rows"]}
    return out


def report(r):
    lines = [
        "# Sustained early CLIP MOVE through epoch 105",
        "",
        (
            "E0/E1/E2 resume the exact saved optimizer, scheduler and RNG at epoch 102. "
            "The original R2-100 CLIP proposal file is read unchanged. E1 uses MOVE mass .8 "
            "and E2 uses the same retriever and mass with random eligible sources. Every batch matches "
            "the number of applied edits. Targeted edits are truncated when random supply is insufficient. "
            "The continuation extends the original checkpoint-derived shared sampler and crop streams. "
            "All arms keep the full SimGCD objective with R2 relational loss disabled, as in epochs 101–102."
        ),
        "",
        (
            "The fixed 588-ID panel is excluded from all continuation draws and the original proposal and "
            "reference banks. It belongs to the original transductive split. Source cohorts consist of IDs "
            "ever edited in E1 or E2 by epoch 105, evaluated retrospectively at each endpoint. The cohorts "
            "can overlap each other and are disjoint from the panel. True-slot ranks require an aligned class. "
            "Panel improvements measure transfer under this protocol."
        ),
        "",
        "| Epoch / arm | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure | Cumulative edits / exposure |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for tag, row in r["endpoint"].items():
        e, a = tag.split(":")
        p = row["panel"]
        fixed, rematch = [
            " / ".join(f"{100 * p[k][g]['accuracy']:.2f}" for g in match.GROUPS)
            for k in ("fixed", "rematched")
        ]
        geo = p["geometry"]["all"]
        t = r["training"][e][a] if e != "100" else {"edits": 0, "exposure": 0.0}
        c = r["cumulative"][e][a] if e != "100" else t
        lines.append(
            f"| {e} / {a} | {fixed} | {rematch} | {geo['purity5']:.4f} | {geo['centroid_margin']:+.4f} | "
            f"{t['edits']} / {100 * t['exposure']:.3f}% | {c['edits']} / {100 * c['exposure']:.3f}% |"
        )
    lines += [
        "",
        "## Live eligibility",
        "",
        (
            "A changed source takes precedence. Low mass counts rows that retain their frozen source but "
            "fall below .8 in at least one view. Random supply shortfalls are counted after eligibility."
        ),
        "",
        "| Epoch | E1 changed source / low mass | E2 changed source / low mass | E1 live | E2 live | Random supply shortfall |",
        "|---|---|---|---:|---:|---:|",
    ]
    for e, row in r["training"].items():
        a, b = [row[k]["eligibility"] for k in ("E1", "E2")]

        def fail(s):
            return f"{s.get('stale', 0)} / {s.get('low_mass', 0)}"

        m = row["matching"]
        lines.append(
            f"| {e} | {fail(a)} | {fail(b)} | {m['E1_eligible']} | {m['E2_eligible']} | {m['random_supply_shortfall']} |"
        )
    lines += [
        "",
        "## Source changes and panel transfer",
        "",
        "Each row compares to epoch-matched KEEP under the original fixed mapping.",
        "",
        "| Epoch / arm / cohort | IDs | Wrong→correct / correct→wrong | Mean rank change |",
        "|---|---:|---|---:|",
    ]
    for e in r["training"]:
        for a in ARMS[1:]:
            tag = f"{e}:{a}"
            for cohort, by in r["endpoint"][tag]["cohorts"].items():
                c = by["all"]
                t = c["transitions"][f"{e}:E0"]
                rank = t["mean_rank_delta"]
                fmt = "—" if rank is None else f"{rank:+.3f}"
                lines.append(
                    f"| {e} / {a} / {cohort} | {c['n']} | {t['wrong_to_correct']} / {t['correct_to_wrong']} | {fmt} |"
                )
    lines += [
        "",
        (
            f"Elapsed this invocation {r['elapsed_seconds'] / 60:.1f} minutes. "
            "Endpoint predictions were frozen before opening labels. Checkpoints and full per-step ledgers are local."
        ),
        "",
        "[All/Old/New transition and geometry metrics, exposure and checkpoint provenance](r2_sustained_move_training.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def verify_past_metrics(endpoints, previous):
    for tag, row in previous["endpoint"].items():
        expected = row["panel"]
        actual = endpoints[tag]["panel"]
        for key in ("fixed", "rematched"):
            assert actual[key] == expected[key], (tag, key)
        for g in match.GROUPS:
            for metric in ("purity5", "centroid_margin", "margin_rows"):
                np.testing.assert_allclose(
                    actual["geometry"][g][metric],
                    expected["geometry"][g][metric],
                    atol=2e-6,
                    rtol=0,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify", action="store_true")
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
        previous,
        past,
        old_hash,
        old_events,
    ) = setup()
    if args.verify:
        verify(model, past[str(RESUME)])
        print(
            "Locked proposals and original 101/102 draws/ledgers verified", flush=True
        )
        return
    audit.write_json(LOCAL / "protocol.json", json_native(sig))
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
            False,
            names=ARMS,
            masses=MASSES,
            local=LOCAL,
            paired_step=early.paired_batch,
            resume=past[str(RESUME)],
        )
    fresh, hashes, events = match.load_finished(
        sig, plan, ids, pool, direct, names=ARMS, masses=MASSES, local=LOCAL
    )
    events = {**old_events, **events}
    out, frozen = match.predictions(
        model,
        ds,
        center,
        base,
        {**past, **fresh},
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
    fixed, alignment = audit.align(
        sig["reference_ids"],
        dict(zip(ck["ids"].tolist(), ck["slots"].tolist(), strict=True)),
        truth,
    )
    endpoint = match.endpoints(
        out, fixed, truth, events, initial_epoch=START, names=ARMS
    )
    verify_past_metrics(endpoint, previous)
    training = fresh[str(END)]["records"]
    for e in ("101", "102"):
        assert training[e] == previous["training"][e]
    result = {
        "provenance": {
            k: v
            for k, v in frozen.items()
            if k not in ("panel_ids", "reference_ids", "source_ids")
        },
        "checkpoint_sha256": {**old_hash, **hashes},
        "panel_n": len(panel),
        "alignment": alignment,
        "training": training,
        "cumulative": cumulative(training),
        "endpoint": endpoint,
        "applied_semantics": match.applied_semantics(events, truth, fixed),
        "paired_panel": {
            e: {
                f"{a}-{b}": early.differences(
                    endpoint[f"{e}:{a}"]["panel"], endpoint[f"{e}:{b}"]["panel"]
                )
                for a, b in (("E1", "E0"), ("E2", "E0"), ("E1", "E2"))
            }
            for e in training
        },
        "elapsed_seconds": time.monotonic() - start,
    }
    report(result)
    audit.write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
