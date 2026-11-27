"""Matched hard and top-three soft CLIP MOVE continuation using frozen schedules."""

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from crish.repair import corrected_distill
from scripts import audit_clip_repair as repair
from scripts import audit_dynamic_move_training as dynamic
from scripts import audit_early_move_training as early
from scripts import audit_g6_confirm as holdout
from scripts import audit_move_strength_training as match
from scripts import audit_oracle_move as audit
from scripts import audit_sustained_move_training as sustained
from scripts.audit_e3_proposals import json_native

LOCAL = repair.g1.WORK / "soft-target"
REPORT = Path("reports/CRISH_SOFT_TARGET_TRAINING.md")
OUT = Path("reports/r2_soft_target_training.json")
ARMS = ("A", "B", "C")
NAMES = ("KEEP", *ARMS)
START = 102
EPOCHS = (103, 104, 105)
MASS = 0.8
TEMP = 0.07


@dataclass(frozen=True)
class SoftMove:
    row: int
    source: int
    destinations: tuple[int, ...]
    fractions: tuple[float, ...]


def mixed_distill(logits, teacher, criterion, epoch, actions, mass):
    q = (teacher.detach().float() / criterion.teacher_temp_schedule[epoch]).softmax(-1)
    q = q.reshape(2, -1, q.shape[-1])
    for move in actions:
        if (q[:, move.row, move.source] < mass).any():
            raise ValueError("Insufficient source mass for soft MOVE")
        q[:, move.row, move.source] -= mass
        q[:, move.row, list(move.destinations)] += mass * q.new_tensor(move.fractions)
    logp = (logits.float().reshape_as(q) / criterion.student_temp).log_softmax(-1)
    return -(q.flip(0) * logp).sum(-1).mean()


@torch.inference_mode()
def reconstruct(ids, vectors, refresh):
    entries, locked = dynamic.pool_from(refresh)
    positions = {int(key): i for i, key in enumerate(ids)}
    rows = np.asarray(sorted(positions[k] for k in entries))
    slots = torch.as_tensor(refresh["slots"], dtype=torch.long, device=vectors.device)
    assert len(slots) == len(ids) and len(refresh["slots"]) == len(ids)
    risks = {k: r for k, _, _, r in refresh["proposals"]}
    mixes = {}
    for group in np.array_split(rows, (len(rows) + 255) // 256):
        sim = vectors[group] @ vectors.T
        index = torch.arange(len(group), device=vectors.device)
        sim[index, group] = -torch.inf
        support = torch.full((len(group), 196), -torch.inf, device=vectors.device)
        for s in np.unique(refresh["slots"]):
            support[:, s] = sim[:, slots == int(s)].max(1).values
        own = support[index, slots[group]].clamp_min(-1)
        support[index, slots[group]] = -torch.inf
        top = support.argsort(descending=True, stable=True)[:, :3]
        vals = support.gather(1, top)
        assert torch.isfinite(vals).all()
        probs = (vals / TEMP).softmax(1)
        source = slots[group].cpu().tolist()
        for key, src, picks, p, values, risk in zip(
            ids[group].tolist(),
            source,
            top.cpu().tolist(),
            probs.cpu().tolist(),
            vals.cpu().tolist(),
            (vals[:, 0] - own).cpu().tolist(),
            strict=True,
        ):
            assert (src, picks[0]) == entries[key][:2]
            assert src not in picks and len(set(picks)) == 3
            assert abs(risk - risks[key]) < 3e-6 and abs(sum(p) - 1) < 1e-6
            mixes[key] = {"slots": picks, "fractions": p, "support": values}
    assert mixes.keys() == entries.keys()
    return entries, locked, mixes


def selector(rows, pool, mix, arm):
    original = match.selector(rows, pool, MASS)

    def apply(prob, ids, legal):
        actions, events, stats = original(prob, ids, legal)
        if not events:
            return actions, events, stats
        specs = [
            {"slots": [int(dst)], "fractions": [1.0]}
            if arm == "A"
            else {
                "slots": mix[int(key)]["slots"],
                "fractions": mix[int(key)]["fractions"],
            }
            for _, key, _, dst in events
        ]
        stats["teacher_before"] = prob[:, [e[0] for e in events]].transpose(0, 1).cpu()
        stats["distribution"] = specs
        if arm != "A":
            actions = [
                SoftMove(int(row), int(src), tuple(s["slots"]), tuple(s["fractions"]))
                for (row, _, src, _), s in zip(events, specs, strict=True)
            ]
        return actions, events, stats

    return apply


def batch_step(models, batch, pool, direct, mix, seeds, epoch, criterion):
    candidates, counters = {}, {}
    for name in ARMS:
        repair.seed_all(seeds[1])
        candidates[name], counters[name] = repair.live_pool(
            models[name][0], batch, pool if name == "C" else direct, strength=MASS
        )
    second = set(candidates["B"])
    common = [v for v in candidates["A"] if v in second]
    n = min(len(common), len(candidates["C"]))
    rng = np.random.default_rng(int(seeds[2]))
    random = [candidates["C"][i] for i in rng.permutation(len(candidates["C"]))[:n]]
    out = {
        "matching": {
            "A_live": len(candidates["A"]),
            "B_live": len(candidates["B"]),
            "C_live": len(candidates["C"]),
            "common": len(common),
            "random_shortfall": len(common) - n,
            "applied": n,
        }
    }
    for a, rows in zip(ARMS, (common[:n], common[:n], random), strict=True):
        model, opt, _ = models[a]
        repair.seed_all(seeds[1])
        edits, receipt = audit.training_step(
            model,
            opt,
            batch,
            criterion,
            selector(rows, pool if a == "C" else direct, mix, a),
            epoch=epoch - 1,
            mass=MASS,
            correction=corrected_distill if a == "A" else mixed_distill,
        )
        assert len(edits) == n
        out[a] = {"moves": edits, "eligibility": dict(counters[a]), **receipt}
    assert out["A"]["moves"] == out["B"]["moves"]
    return out


def schedules(ds, sig):
    past = json.loads(dynamic.OUT.read_text())
    published = json.loads((dynamic.LOCAL / "protocol.json").read_text())
    assert all(
        published[k] == v
        for k, v in past["provenance"].items()
        if k != "predictions_sha256"
    )
    assert (
        published["resume_sha256"] == sig["resume_sha256"]
        and published["draws"] == sig["draws"]
    )
    clip = holdout.checked_cache(
        repair.g6.LOCAL / "clip-image.npz",
        sig["clip_cache_sha256"],
        ds.unlabelled_dataset.ids,
    )
    chosen = np.flatnonzero(np.isin(ds.unlabelled_dataset.ids, sig["reference_ids"]))
    ids = ds.unlabelled_dataset.ids[chosen]
    assert ids.tolist() == sig["reference_ids"]
    hashes, proposals = {}, {}
    previous = published["resume_sha256"]
    for e in EPOCHS:
        path = dynamic.LOCAL / f"refresh-{e}.json"
        hashes[str(e)] = audit.file_digest(path)
        assert hashes[str(e)] == past["training"][str(e)]["refresh_sha256"]
        raw = json.loads(path.read_text())
        assert raw["epoch"] == e and raw["input_sha256"] == previous
        assert raw["reference_sha256"] == published["reference_prefix_sha256"]
        proposals[e] = reconstruct(ids, clip[chosen], raw)
        previous = past["checkpoint_sha256"][str(e)]
        print(
            "Verified saved refresh and reconstructed top-three",
            e,
            len(proposals[e][0]),
            flush=True,
        )
    return proposals, hashes


def train(model, ds, ids, plan, previous, sig, schedule, fixed, smoke):
    completed = (
        [e for e in EPOCHS if (LOCAL / f"epoch-{e}.pt").is_file()] if not smoke else []
    )
    assert completed == list(EPOCHS[: len(completed)])
    last = max(completed, default=START)
    if last == EPOCHS[-1]:
        return
    if last == START:
        states = {
            "A": previous["state"]["E1"],
            "B": previous["state"]["E1"],
            "C": previous["state"]["E2"],
        }
        records = {}
    else:
        saved = audit.load_checkpoint(LOCAL / f"epoch-{last}.pt")
        assert saved["signature"] == sig and saved["epoch"] == last
        states, records = saved["state"], saved["records"]
    assert all(states[a]["scheduler"]["last_epoch"] == last for a in ARMS)
    models = repair.make_arms(model, states, names=ARMS)
    criterion = audit.DistillLoss(30, 200)
    for epoch, (rows, seeds) in plan.items():
        if epoch <= last:
            continue
        torch.cuda.reset_peak_memory_stats()
        pool, direct, mix = schedule[epoch]
        ledger = []
        for j, (rr, ss) in enumerate(zip(rows, seeds, strict=True), 1):
            batch = repair.load_batch(model, ds, rr, ss[0])
            np.testing.assert_array_equal(batch["ids"].numpy(), ids[rr])
            result = batch_step(models, batch, pool, direct, mix, ss, epoch, criterion)
            if smoke:
                if result["matching"]["applied"]:
                    assert all(result[a]["cls_edit_norm"] > 0 for a in ARMS)
                    print(
                        "Hard/soft actual paired batch passed",
                        result["matching"],
                        "peak bytes",
                        torch.cuda.max_memory_allocated(),
                        flush=True,
                    )
                    return
                if j == 5:
                    raise ValueError(
                        "No common .8 teacher-eligible MOVE in first five batches"
                    )
            result["KEEP"] = {"moves": []}
            ledger.append(result)
            if j % 10 == 0:
                print(
                    "Epoch",
                    epoch,
                    "steps",
                    j,
                    "matched edits",
                    sum(x["matching"]["applied"] for x in ledger),
                    flush=True,
                )
        matching = Counter()
        for step in ledger:
            matching.update(step["matching"])
        row = {
            "rows": int(rows.size),
            "matching": dict(matching),
            "KEEP": fixed["training"][str(epoch)]["E0"],
        }
        for a, (_, _, sched) in models.items():
            sched.step()
            assert sched.last_epoch == epoch
            blocks = [step[a] for step in ledger]
            elig = Counter()
            for step in blocks:
                elig.update(step["eligibility"])
            n = sum(len(step["moves"]) for step in blocks)
            assert n == matching["applied"]
            row[a] = {
                "edits": n,
                "exposure": n / rows.size,
                "mass_total": n * MASS,
                "unique_ids": len(
                    {event[1] for step in blocks for event in step["moves"]}
                ),
                "eligibility": dict(elig),
                "lr": sched.get_last_lr()[0],
                "mean_loss": float(np.mean([step["loss"] for step in blocks])),
                "peak_vram_bytes": max(step["peak_vram"] for step in blocks),
            }
        records[str(epoch)] = row
        repair.save(models, LOCAL / f"epoch-{epoch}.pt", sig, epoch, records, ledger)
        audit.write_json(LOCAL / "training-progress.json", json_native(records))
        print("Saved hard/soft checkpoint", epoch, flush=True)


def check_ledger(checkpoints, schedule):
    data = {}
    for epoch, cp in checkpoints.items():
        _, _, mix = schedule[int(epoch)]
        rows = {a: [] for a in ARMS}
        for entry in cp["batch_ledger"]:
            assert entry["A"]["moves"] == entry["B"]["moves"]
            for a in ARMS:
                rec = entry[a]
                if not rec["moves"]:
                    continue
                p = rec["teacher_before"].numpy()
                assert p.shape == (len(rec["moves"]), 2, 196)
                np.testing.assert_allclose(p.sum(2), 1, atol=2e-6)
                for i, ((_, key, src, top), dist) in enumerate(
                    zip(rec["moves"], rec["distribution"], strict=True)
                ):
                    expected = (
                        {"slots": [top], "fractions": [1.0]}
                        if a == "A"
                        else {k: mix[key][k] for k in ("slots", "fractions")}
                    )
                    assert dist == expected
                    assert (
                        src not in dist["slots"]
                        and abs(sum(dist["fractions"]) - 1) < 1e-6
                    )
                    assert (p[i, :, src] >= MASS).all()
                    rows[a].append((key, src, dist, p[i]))
        data[epoch] = rows
    return data


def true_class_mass(rows, mapping, truth):
    true_slot = {v: k for k, v in mapping.items()}
    summary, per_epoch = {}, {}
    for epoch, arms in rows.items():
        means, values = {}, {}
        for a, records in arms.items():
            items = []
            for key, src, dist, p in records:
                label = int(truth[key])
                target = true_slot.get(label)
                if target is None:
                    items.append((key, label, None, None))
                    continue
                dests = dict(zip(dist["slots"], dist["fractions"], strict=True))
                delta = MASS * (dests.get(target, 0) - int(target == src))
                prev = float(p[:, target].mean())
                assert -2e-6 <= prev + delta <= 1 + 2e-6
                items.append((key, label, prev, delta))
            values[a] = items
            groups = {}
            for group in match.GROUPS:
                selected = [
                    r
                    for r in items
                    if group == "all" or (r[1] < 98) == (group == "old")
                ]
                valid = [r for r in selected if r[2] is not None]
                delta = [r[3] for r in valid]
                groups[group] = {
                    "edits": len(selected),
                    "mapped_true_class": len(valid),
                    "unmapped": len(selected) - len(valid),
                    "mean_before": match.mean([r[2] for r in valid]),
                    "mean_after": match.mean([r[2] + r[3] for r in valid]),
                    "mean_delta": match.mean(delta),
                    "sum_delta": sum(delta),
                    "positive": sum(v > 0 for v in delta),
                    "negative": sum(v < 0 for v in delta),
                    "zero": sum(v == 0 for v in delta),
                }
            means[a] = groups
        paired = {}
        for group in match.GROUPS:
            diffs = []
            for a, b in zip(values["A"], values["B"], strict=True):
                assert a[:2] == b[:2] and (a[2] is None) == (b[2] is None)
                if a[2] is not None and (
                    group == "all" or (a[1] < 98) == (group == "old")
                ):
                    diffs.append(b[3] - a[3])
            paired[group] = {
                "n": len(diffs),
                "mean_delta_B_minus_A": match.mean(diffs),
                "sum_delta_B_minus_A": sum(diffs),
            }
        summary[epoch] = {"arms": means, "paired_A_B": paired}
        per_epoch[epoch] = values
    return summary, per_epoch


def report(r):
    lines = [
        "# Matched CLIP hard versus soft MOVE on Stanford Cars",
        "",
        (
            "A and B resume the same E1 checkpoint at epoch 102. C resumes E2. The detector, "
            "source slots, top-1 destinations and ranking come from the previously saved dynamic "
            "trajectory for each epoch and remain fixed for this experiment. Top-3 retrieval uses "
            "the same maximum peer cosine per alternative slot, excludes the source and self, "
            "and verifies that the top-1 support and risk match the saved proposal."
        ),
        "",
        (
            "A moves .8 to the top slot. B and C divide .8 among the top three using softmax "
            "over CLIP support at temperature .07. Each arm uses its own live teacher gate. "
            "A and B share exactly the same applied source occurrences. Their shared list is "
            "trimmed to C's supply when needed. C chooses uniformly among live eligible source "
            "occurrences in the same batch. All arms use full SimGCD with R2 relational loss off, "
            "identical draws and crops. KEEP reuses the prior checkpoint at each epoch."
        ),
        "",
        (
            "The panel's 588 images were seen before the continuation and excluded from its draws "
            "and retriever bank. Source transitions use separate ever-edited cohorts and the same "
            "fixed slot alignment. Endpoint source/destination probability summaries refer to the last "
            "applied top-1 pair for each ID. The sealed true-class mass analysis uses the complete "
            "target mixture from each occurrence and the pre-update teacher probabilities. "
            "Unmapped true classes are counted separately. All predictions precede semantic scoring."
        ),
        "",
        "| Epoch / arm | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure |",
        "|---|---|---|---:|---:|---|",
    ]

    def f(v):
        return "—" if v is None else f"{v:+.4f}"

    for tag, record in r["endpoint"].items():
        e, a = tag.split(":")
        panel = record["panel"]
        fixed, rematch = [
            " / ".join(f"{100 * panel[mode][g]['accuracy']:.2f}" for g in match.GROUPS)
            for mode in ("fixed", "rematched")
        ]
        geo = panel["geometry"]["all"]
        dose = r["training"].get(e, {}).get(a)
        detail = (
            "—" if dose is None else f"{dose['edits']} / {100 * dose['exposure']:.3f}%"
        )
        lines.append(
            f"| {e} / {a} | {fixed} | {rematch} | {geo['purity5']:.4f} | {geo['centroid_margin']:+.4f} | {detail} |"
        )
    lines += [
        "",
        "## Matched exposure and sealed target mass",
        "",
        "| Epoch | A / B / C live | Common A/B / random shortfall | Applied per arm / moved mass | A / B / C mean true-mass Δ | Paired B−A mean Δ |",
        "|---|---|---|---|---|---:|",
    ]
    for e, row in r["training"].items():
        m = row["matching"]
        v = r["true_class_mass"][e]
        live = " / ".join(str(m[f"{a}_live"]) for a in ARMS)
        delta = " / ".join(f(v["arms"][a]["all"]["mean_delta"]) for a in ARMS)
        lines.append(
            f"| {e} | {live} | {m['common']} / {m['random_shortfall']} | {m['applied']} / {row['A']['mass_total']:.1f} | "
            f"{delta} | {f(v['paired_A_B']['all']['mean_delta_B_minus_A'])} |"
        )
    lines += [
        "",
        "## Transfer and source transitions",
        "",
        "| Epoch / arm | CLIP-source fixes / regressions | Random-source fixes / regressions | Panel fixes / regressions |",
        "|---|---|---|---|",
    ]
    for e in r["training"]:
        for a in ARMS:
            cohorts = r["endpoint"][f"{e}:{a}"]["cohorts"]
            line = []
            for cohort in ("CLIP_sources", "random_sources", "panel"):
                delta = cohorts[cohort]["all"]["transitions"][f"{e}:KEEP"]
                line.append(
                    f"{delta['wrong_to_correct']} / {delta['correct_to_wrong']}"
                )
            lines.append(f"| {e} / {a} | " + " | ".join(line) + " |")
    lines += [
        "",
        f"Elapsed {r['seconds'] / 60:.1f} minutes in this invocation. Paired checkpoints and full ledgers remain local.",
        "",
        "[All/Old/New outcomes, eligibility, target mass and checkpoint digests](r2_soft_target_training.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def verify_reference_panels(endpoint, fixed):
    references = {
        "100:shared": "100:shared",
        "102:KEEP": "102:E0",
        "102:A": "102:E1",
        "102:B": "102:E1",
        "102:C": "102:E2",
        **{f"{e}:KEEP": f"{e}:E0" for e in EPOCHS},
    }
    for key, ref in references.items():
        a, b = (item["panel"] for item in (endpoint[key], fixed["endpoint"][ref]))
        for field in ("fixed", "rematched"):
            assert a[field] == b[field], (key, field)
        for g in match.GROUPS:
            for name in ("purity5", "centroid_margin", "margin_rows"):
                np.testing.assert_allclose(
                    a["geometry"][g][name], b["geometry"][g][name], atol=2e-6, rtol=0
                )


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
        old_pool,
        old_direct,
        ck,
        old_sig,
        _,
        early_cp,
        old_hash,
        _,
    ) = sustained.setup()
    fixed = json.loads(sustained.OUT.read_text())
    schedule, hashes = schedules(ds, old_sig)
    sig = {
        **old_sig,
        "base": "7efa773",
        "masses": {a: MASS for a in ARMS},
        "soft_temperature": TEMP,
        "soft_slots": 3,
        "refresh_sha256": hashes,
        "dynamic_report_sha256": audit.file_digest(dynamic.OUT),
        "keep_report_sha256": audit.file_digest(sustained.OUT),
    }
    if not args.smoke:
        specs = {str(e): schedule[e][2] for e in EPOCHS}
        lock = LOCAL / "fixed-mixtures.json"
        value = json.loads(
            json.dumps(json_native({"signature": sig, "mixtures": specs}))
        )
        if lock.is_file():
            assert json.loads(lock.read_text()) == value
        else:
            audit.write_json(lock, value)
        sig = {**sig, "locked_mixtures_sha256": audit.file_digest(lock)}
        audit.write_json(LOCAL / "protocol.json", json_native(sig))
    if not args.sealed_only:
        train(
            model, ds, ids, plan, early_cp[str(START)], sig, schedule, fixed, args.smoke
        )
    if args.smoke:
        return
    original, keep_hash, _ = match.load_finished(
        old_sig,
        plan,
        ids,
        old_pool,
        old_direct,
        names=early.ARMS,
        masses=early.MASSES,
        local=sustained.LOCAL,
    )
    assert {**old_hash, **keep_hash} == fixed["checkpoint_sha256"]
    saved, digests, events = {}, {}, {}
    for e, (rows, seeds) in plan.items():
        pool, direct, _ = schedule[e]
        x, y, z = match.load_finished(
            sig,
            {e: (rows, seeds)},
            ids,
            pool,
            direct,
            names=NAMES,
            masses={a: MASS for a in ARMS},
            local=LOCAL,
        )
        x[str(e)]["state"]["KEEP"] = original[str(e)]["state"]["E0"]
        saved.update(x)
        digests.update(y)
        events.update(z)
    ledger = check_ledger(saved, schedule)
    cp = early_cp["102"]["state"]
    common = {
        "102": {
            "state": {"KEEP": cp["E0"], "A": cp["E1"], "B": cp["E1"], "C": cp["E2"]}
        }
    }
    out, frozen = match.predictions(
        model,
        ds,
        center,
        base,
        {**common, **saved},
        panel,
        tokens,
        events,
        sig,
        names=NAMES,
        local=LOCAL,
    )
    assert (
        audit.file_digest(LOCAL / "frozen-endpoint-predictions.npz")
        == frozen["predictions_sha256"]
    )

    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    mapping, coverage = audit.align(
        sig["reference_ids"],
        dict(zip(ck["ids"].tolist(), ck["slots"].tolist(), strict=True)),
        truth,
    )
    endpoint = match.endpoints(
        out, mapping, truth, events, initial_epoch=100, names=NAMES
    )
    verify_reference_panels(endpoint, fixed)
    true_mass, raw_mass = true_class_mass(ledger, mapping, truth)
    audit.write_json(
        LOCAL / "sealed-true-mass-by-occurrence.json", json_native(raw_mass)
    )
    paired = {}
    for e in map(str, EPOCHS):
        paired[e] = {
            f"{a}-{b}": early.differences(
                endpoint[f"{e}:{a}"]["panel"], endpoint[f"{e}:{b}"]["panel"]
            )
            for a, b in (
                ("A", "KEEP"),
                ("B", "KEEP"),
                ("C", "KEEP"),
                ("B", "A"),
                ("B", "C"),
            )
        }
    result = {
        "provenance": {
            k: v
            for k, v in frozen.items()
            if k not in ("panel_ids", "reference_ids", "source_ids")
        },
        "panel_n": len(panel),
        "checkpoint_sha256": digests,
        "alignment": coverage,
        "training": saved[str(EPOCHS[-1])]["records"],
        "endpoint": endpoint,
        "true_class_mass": true_mass,
        "paired_panel": paired,
        "seconds": time.monotonic() - start,
    }
    report(result)
    audit.write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
