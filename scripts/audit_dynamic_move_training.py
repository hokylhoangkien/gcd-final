"""Dynamic CLIP proposals for the matched continuation from epoch 102."""

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import restore_rng, rng_state
from scripts import audit_clip_repair as repair
from scripts import audit_early_move_training as early
from scripts import audit_g6_confirm as holdout
from scripts import audit_move_strength_training as match
from scripts import audit_oracle_move as audit
from scripts import audit_sustained_move_training as sustained
from scripts.audit_e3_proposals import json_native

LOCAL = repair.g1.WORK / "dynamic-move"
REPORT = Path("reports/CRISH_DYNAMIC_MOVE_TRAINING.md")
OUT = Path("reports/r2_dynamic_move_training.json")
ACTIVE = ("E1", "E2")
EPOCHS = (103, 104, 105)
REFRESH_CROP_SEED = 534541


@torch.inference_mode()
def reference_cache(model, blind, center, rows, ids, root, checkpoint_sha):
    """Lossless shared prefix for center and two fixed augmentation views."""
    file = root / "reference-prefix.npy"
    meta = root / "reference-prefix.json"
    identity = {
        "ids": ids.tolist(),
        "crop_seed": REFRESH_CROP_SEED,
        "prefix_checkpoint_sha256": checkpoint_sha,
        "shape": [3, len(rows), 197, 768],
    }
    if file.is_file() and meta.is_file():
        saved = json.loads(meta.read_text())
        assert saved["identity"] == identity
        assert audit.file_digest(file) == saved["sha256"]
        cache = np.load(file, mmap_mode="r")
        assert list(cache.shape) == identity["shape"] and cache.dtype == np.float32
        return cache, saved["sha256"]

    temp = root / "reference-prefix.tmp.npy"
    store = np.lib.format.open_memmap(
        temp, mode="w+", shape=tuple(identity["shape"]), dtype=np.float32
    )
    center_ds = copy.copy(blind)
    center_ds.transform = center
    original_rng = rng_state()
    try:
        repair.seed_all(REFRESH_CROP_SEED)
        model.eval()
        for ds, views in ((center_ds, (0,)), (blind, (1, 2))):
            offset = 0
            for images, legal, keys in DataLoader(
                Subset(ds, rows.tolist()), batch_size=8
            ):
                end = offset + len(keys)
                assert (legal == -1).all()
                np.testing.assert_array_equal(keys.numpy(), ids[offset:end])
                for view, image in zip(
                    views, [images] if views == (0,) else images, strict=True
                ):
                    (tokens,) = audit.prefix(model[0], image)
                    store[view, offset:end] = tokens.numpy()
                offset = end
                if offset % 1024 == 0:
                    print("Reference prefix", views, offset, flush=True)
            assert offset == len(ids)
    finally:
        restore_rng(original_rng)
    store.flush()
    del store
    temp.replace(file)
    digest = audit.file_digest(file)
    audit.write_json(meta, {"identity": identity, "sha256": digest})
    return np.load(file, mmap_mode="r"), digest


@torch.inference_mode()
def rank(model, ids, clip, cache):
    model.eval()
    encoded = []
    for view in cache:
        x = torch.cat(
            [
                audit.encode(model[0], [torch.from_numpy(view[i : i + 8].copy())])
                for i in range(0, len(ids), 8)
            ]
        )
        encoded.append(F.normalize(x, dim=1))
    slots = model[1].last_layer(encoded[0]).argmax(1).cpu().numpy()
    proposals, limit, ref = repair.lock_policies(
        model, ids, slots, clip, torch.stack(encoded[1:]), np.empty(0, dtype=np.int64)
    )
    assert ref == ids.tolist() and limit == len(proposals) // 20 and limit > 0
    return proposals, limit, slots.tolist()


def pool_from(refresh):
    pool = {
        k: (src, dst, rank)
        for rank, (k, src, dst, _) in enumerate(refresh["proposals"])
    }
    direct = {k: value for k, value in pool.items() if value[-1] < refresh["top_n"]}
    assert len(direct) == refresh["top_n"]
    return pool, direct


def dynamic_policy(epoch, head, input_sha, sig, ids, clip, cache):
    path = LOCAL / f"refresh-{epoch}.json"
    if path.is_file():
        refresh = json.loads(path.read_text())
        assert (
            refresh["input_sha256"] == input_sha
            and refresh["reference_sha256"] == sig["reference_prefix_sha256"]
        )
    else:
        begin = time.monotonic()
        proposals, count, slots = rank(head, ids, clip, cache)
        refresh = {
            "epoch": epoch,
            "input_sha256": input_sha,
            "reference_sha256": sig["reference_prefix_sha256"],
            "top_n": count,
            "slots": slots,
            "proposals": proposals,
            "seconds": time.monotonic() - begin,
        }
        audit.write_json(path, json_native(refresh))
    assert refresh["epoch"] == epoch
    print(
        "Refreshed from E1",
        epoch - 1,
        "eligible IDs",
        len(refresh["proposals"]),
        "top",
        refresh["top_n"],
        flush=True,
    )
    return refresh, audit.file_digest(path)


def train(model, ds, all_ids, plan, start, sig, ids, clip, cache, fixed):
    done = [e for e in EPOCHS if (LOCAL / f"epoch-{e}.pt").is_file()]
    assert done == list(EPOCHS[: len(done)])
    prev = max(done, default=102)
    if prev == 105:
        return
    latest = LOCAL / f"epoch-{prev}.pt"
    state = audit.load_checkpoint(latest) if done else start
    if done:
        assert state["signature"] == sig and state["epoch"] == prev
    assert all(state["state"][a]["scheduler"]["last_epoch"] == prev for a in ACTIVE)
    models = repair.make_arms(model, state["state"], names=ACTIVE)
    records = dict(state["records"])
    criterion = audit.DistillLoss(30, 200)
    source = audit.file_digest(latest) if done else sig["resume_sha256"]
    for epoch, (rows, seeds) in plan.items():
        if epoch <= prev:
            continue
        refresh, receipt = dynamic_policy(
            epoch, models["E1"][0], source, sig, ids, clip, cache
        )
        pool, direct = pool_from(refresh)
        ledger = []
        torch.cuda.reset_peak_memory_stats()
        for step, (rr, ss) in enumerate(zip(rows, seeds, strict=True), 1):
            batch = repair.load_batch(model, ds, rr, ss[0])
            np.testing.assert_array_equal(batch["ids"].numpy(), all_ids[rr])
            details = early.paired_batch(
                models, batch, pool, direct, ss, epoch, criterion, names=ACTIVE
            )
            details["E0"] = {"moves": []}
            ledger.append(details)
            if step % 10 == 0:
                print(
                    "Epoch",
                    epoch,
                    "steps",
                    step,
                    "matched",
                    sum(x["matching"]["applied"] for x in ledger),
                    flush=True,
                )
        count, matching = sum(x["matching"]["applied"] for x in ledger), Counter()
        for x in ledger:
            matching.update(x["matching"])
        row = {
            "rows": int(rows.size),
            "matching": dict(matching),
            "refresh_sha256": receipt,
            "E0": fixed["training"][str(epoch)]["E0"],
        }
        for a, (_, _, sched) in models.items():
            sched.step()
            assert sched.last_epoch == epoch
            steps = [x[a] for x in ledger]
            elig = Counter()
            for s in steps:
                elig.update(s["eligibility"])
            assert sum(len(x["moves"]) for x in steps) == count
            row[a] = {
                "edits": count,
                "exposure": count / rows.size,
                "mass_total": count * 0.8,
                "weighted_batch_normalized_mass": count * 0.8 * 0.65 / 128,
                "lr": sched.get_last_lr()[0],
                "eligibility": dict(elig),
                "unique_source_ids": len({m[1] for s in steps for m in s["moves"]}),
                "mean_loss": float(np.mean([s["loss"] for s in steps])),
                "peak_vram_bytes": max(s["peak_vram"] for s in steps),
            }
        records[str(epoch)] = row
        path = LOCAL / f"epoch-{epoch}.pt"
        repair.save(models, path, sig, epoch, records, ledger)
        audit.write_json(LOCAL / "training-progress.json", json_native(records))
        source = audit.file_digest(path)
        print("Saved dynamic paired checkpoint", epoch, flush=True)


def smoke_test(model, ds, center, all_ids, plan, panel, prior, sig, clip):
    rows, seeds = plan[103]
    blind = ds.unlabelled_dataset
    # A bounded bank with actual IDs from the first four batches exercises the refresh path.
    selected = np.unique(np.r_[all_ids[rows[:4]].flatten(), blind.ids[:320]])
    indices = np.flatnonzero(np.isin(blind.ids, selected) & ~np.isin(blind.ids, panel))
    ids = blind.ids[indices]
    model.load_state_dict(prior["state"]["E1"]["model"])
    root = LOCAL / "smoke"
    root.mkdir(exist_ok=True)
    cache, _ = reference_cache(
        model, blind, center, indices, ids, root, sig["checkpoint_sha256"]
    )
    p, n, slots = rank(model, ids, clip[indices], cache)
    assert len(slots) == len(ids)
    pool, direct = pool_from({"proposals": p, "top_n": n})
    models = repair.make_arms(model, prior["state"], names=ACTIVE)
    for rr, ss in zip(rows[:4], seeds[:4], strict=True):
        batch = repair.load_batch(model, ds, rr, ss[0])
        detail = early.paired_batch(
            models,
            batch,
            pool,
            direct,
            ss,
            103,
            audit.DistillLoss(30, 200),
            names=ACTIVE,
        )
        if detail["matching"]["applied"]:
            assert len(detail["E1"]["moves"]) == len(detail["E2"]["moves"])
            print(
                "Dynamic ranking and a real paired batch passed",
                detail["matching"],
                flush=True,
            )
            return
    raise ValueError("Smoke found no real matched edits in the bounded bank")


def turnover(old, new):
    previous, _ = pool_from(old)
    current, selected = pool_from(new)
    old_ids = {k for k, v in previous.items() if v[-1] < old["top_n"]}
    new_ids = set(selected)
    common = old_ids & new_ids
    full = previous.keys() & current.keys()
    return {
        "selected": len(new_ids),
        "eligible_pool": len(current),
        "retained": len(common),
        "added": len(new_ids - old_ids),
        "dropped": len(old_ids - new_ids),
        "jaccard": len(common) / len(old_ids | new_ids),
        "changed_source_retained": sum(previous[k][0] != current[k][0] for k in common),
        "changed_destination_retained": sum(
            previous[k][1] != current[k][1] for k in common
        ),
        "changed_destination_common_pool": sum(
            previous[k][1] != current[k][1] for k in full
        ),
        "common_pool": len(full),
        "reference_slot_changes": int(
            np.count_nonzero(np.asarray(old["slots"]) != new["slots"])
        ),
        "refresh_seconds": new["seconds"],
    }


def edits_summary(past, incoming):
    seen = {
        a: {r[2] for (_, tag), v in past.items() if tag == a for r in v} for a in ACTIVE
    }
    out = {}
    for epoch in EPOCHS:
        out[str(epoch)] = {}
        for a in ACTIVE:
            rows = incoming.get((str(epoch), a), [])
            counts = Counter(r[2] for r in rows)
            added = counts.keys() - seen[a]
            out[str(epoch)][a] = {
                "occurrences": len(rows),
                "distinct_ids": len(counts),
                "new_ids": len(added),
                "previously_edited_occurrences": sum(
                    counts[k] for k in counts.keys() & seen[a]
                ),
                "repeat_occurrences_in_epoch": len(rows) - len(counts),
                "ids_repeated_in_epoch": sum(n > 1 for n in counts.values()),
                "cumulative_distinct_ids": len(seen[a] | counts.keys()),
            }
            seen[a].update(counts)
    return out


def render(r):
    lines = [
        "# Dynamic CLIP proposal refresh through epoch 105",
        "",
        (
            "Both dynamic arms resume the original epoch-102 checkpoints. At the start of epochs "
            "103, 104 and 105, current E1 features and classifier assign slots on all non-panel "
            "unlabeled reference IDs. CLIP cosine uses the existing image embeddings, excludes self "
            "and ranks best alternative support minus current-slot support. The previous .2 initial two-view "
            "eligibility screen and top 5% rule are applied anew. Center views and two deterministic "
            "refresh crops are fixed across refreshes and cached through the last frozen DINO block."
        ),
        "",
        (
            "E1 and E2 use the same refreshed source/destination map. E2 samples random eligible sources "
            "and has its own live teacher check. Both require the refreshed source to match their own "
            "live prediction and have .8 mass in both shared training crops. Per-batch edits are matched "
            "without padding. The fixed E0 and fixed-proposal E1/E2 come from the published run. "
            "The excluded panel had prior transductive exposure before continuation."
        ),
        "",
        (
            "Panel geometry and native accuracy use the original fixed alignment plus the descriptive "
            "panel Hungarian rematch. Same-ID transitions use the fixed alignment. Source cohorts are "
            "retrospective sets of ever-edited IDs, including shared epochs 101 and 102. For an ID whose "
            "proposal changes, endpoint source/destination probability summaries use its last applied pair. "
            "The semantic ledger scores the actual pair on each occurrence."
        ),
        "",
        "| Epoch / policy | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure |",
        "|---|---|---|---:|---:|---|",
    ]

    def triplet(p, k):
        return " / ".join(f"{100 * p[k][g]['accuracy']:.2f}" for g in match.GROUPS)

    for e, row in r["comparison"].items():
        for policy, record in row.items():
            p = record["panel"]
            geom = p["geometry"]["all"]
            lines.append(
                f"| {e} / {policy} | {triplet(p, 'fixed')} | {triplet(p, 'rematched')} | "
                f"{geom['purity5']:.4f} | {geom['centroid_margin']:+.4f} | "
                f"{record['edits']} / {100 * record['exposure']:.3f}% |"
            )
    lines += [
        "",
        "## Turnover and realized dose",
        "",
        "| Epoch | Top 5% IDs | Retained / added / dropped | Destination changes among retained | Dynamic / fixed edits per edited arm | Δ exposure (pp) |",
        "|---|---:|---|---:|---|---:|",
    ]
    for e, t in r["turnover"].items():
        d = r["comparison"][e]["dynamic E1"]
        f = r["comparison"][e]["fixed E1"]
        lines.append(
            f"| {e} | {t['selected']} | {t['retained']} / {t['added']} / {t['dropped']} | "
            f"{t['changed_destination_retained']} | {d['edits']} / {f['edits']} | "
            f"{100 * (d['exposure'] - f['exposure']):+.3f} |"
        )
    lines += [
        "",
        (
            "Dynamic versus fixed changes both the selected rows and the realized intervention dose. "
            "Differences in panel performance describe the two trajectories. When exposure differs, "
            "these comparisons do not isolate the effect of refreshing proposals."
        ),
        "",
        "| Epoch / dynamic arm | Changed source / low mass | New edited IDs / within-epoch repeat edits | Correct / harmful / wrong-to-wrong / unresolved | Panel fixes / regressions vs KEEP |",
        "|---|---|---|---|---|",
    ]
    for e, summary in r["edits"].items():
        for a in ACTIVE:
            s = summary[a]
            t = r["training"][e][a]["eligibility"]
            c = (
                r["applied_semantics"]
                .get(e, {})
                .get(a, {})
                .get("all", {})
                .get("categories", {})
            )
            kinds = " / ".join(
                str(c.get(k, 0))
                for k in ("correct", "harmful", "wrong_to_wrong", "unresolved")
            )
            tr = r["endpoint"][f"{e}:{a}"]["cohorts"]["panel"]["all"]["transitions"][
                f"{e}:E0"
            ]
            lines.append(
                f"| {e} / {a} | {t.get('stale', 0)} / {t.get('low_mass', 0)} | {s['new_ids']} / "
                f"{s['repeat_occurrences_in_epoch']} | {kinds} | "
                f"{tr['wrong_to_correct']} / {tr['correct_to_wrong']} |"
            )
    lines += [
        "",
        f"Runtime this invocation {r['seconds'] / 60:.1f} minutes.",
        "",
        "[All/Old/New outcomes, panel contrasts, ledgers and provenance](r2_dynamic_move_training.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    modes = arg.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--sealed-only", action="store_true")
    args = arg.parse_args()
    begin = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    (
        model,
        ds,
        center,
        all_ids,
        base,
        plan,
        panel,
        tokens,
        old_pool,
        old_direct,
        ck,
        fixed_sig,
        prior,
        past,
        past_hash,
        old_events,
    ) = sustained.setup()
    fixed = json.loads(sustained.OUT.read_text())
    fixed_sha = audit.file_digest(sustained.OUT)
    assert all(
        fixed_sig[k] == v
        for k, v in fixed["provenance"].items()
        if k != "predictions_sha256"
    )
    saved_102 = past["102"]
    for state in saved_102["state"].values():
        for k in model[0].state_dict():
            if not k.startswith("blocks.11."):
                assert torch.equal(state["model"][f"0.{k}"], base["model"][f"0.{k}"]), k
    blind = ds.unlabelled_dataset
    clip = holdout.checked_cache(
        repair.g6.LOCAL / "clip-image.npz", fixed_sig["clip_cache_sha256"], blind.ids
    )
    if args.smoke:
        smoke_test(model, ds, center, all_ids, plan, panel, saved_102, fixed_sig, clip)
        return
    ref = np.flatnonzero(~np.isin(blind.ids, panel))
    ids = blind.ids[ref]
    assert ids.tolist() == fixed_sig["reference_ids"]
    cache, digest = reference_cache(
        model, blind, center, ref, ids, LOCAL, fixed_sig["checkpoint_sha256"]
    )
    sig = {
        **fixed_sig,
        "base": "ff53454",
        "reference_prefix_sha256": digest,
        "refresh_crop_seed": REFRESH_CROP_SEED,
        "fixed_report_sha256": fixed_sha,
        "refresh_model": "E1",
        "refresh_epochs": list(EPOCHS),
    }
    audit.write_json(LOCAL / "protocol.json", json_native(sig))
    if not args.sealed_only:
        train(model, ds, all_ids, plan, saved_102, sig, ids, clip[ref], cache, fixed)
    fixed_states, fixed_hashes, fixed_events = match.load_finished(
        fixed_sig,
        plan,
        all_ids,
        old_pool,
        old_direct,
        names=early.ARMS,
        masses=early.MASSES,
        local=sustained.LOCAL,
    )
    assert {**past_hash, **fixed_hashes} == fixed["checkpoint_sha256"]
    all_ckpt, new_hash, events, changes = {}, {}, {}, {}
    original = json.loads((early.LOCAL / "frozen-policy.json").read_text())
    old = {
        "proposals": original["proposals"],
        "top_n": fixed_sig["locked_n"],
        "slots": ck["slots"][ref].tolist(),
    }
    prev_sha = past_hash["102"]
    for e in EPOCHS:
        key = str(e)
        file = LOCAL / f"refresh-{e}.json"
        refresh = json.loads(file.read_text())
        assert (
            refresh["input_sha256"] == prev_sha
            and refresh["reference_sha256"] == digest
            and refresh["epoch"] == e
        )
        pool, direct = pool_from(refresh)
        states, hashes, ledgers = match.load_finished(
            sig,
            {e: plan[e]},
            all_ids,
            pool,
            direct,
            names=early.ARMS,
            masses=early.MASSES,
            local=LOCAL,
        )
        saved = states[key]
        assert saved["records"][key]["refresh_sha256"] == audit.file_digest(file)
        saved["state"]["E0"] = fixed_states[key]["state"]["E0"]
        all_ckpt[key] = saved
        events.update(ledgers)
        new_hash.update(hashes)
        prev_sha = hashes[key]
        changes[key] = turnover(old, refresh)
        old = refresh
    out, frozen = match.predictions(
        model,
        ds,
        center,
        base,
        {**past, **all_ckpt},
        panel,
        tokens,
        {**old_events, **events},
        sig,
        names=early.ARMS,
        local=LOCAL,
    )
    assert (
        audit.file_digest(LOCAL / "frozen-endpoint-predictions.npz")
        == frozen["predictions_sha256"]
    )

    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    mapping, alignment = audit.align(
        sig["reference_ids"],
        dict(zip(ck["ids"].tolist(), ck["slots"].tolist(), strict=True)),
        truth,
    )
    metrics = match.endpoints(
        out,
        mapping,
        truth,
        {**old_events, **events},
        initial_epoch=100,
        names=early.ARMS,
    )
    sustained.verify_past_metrics(metrics, prior)
    training = all_ckpt["105"]["records"]
    comparison, differences = {}, {}
    for epoch in map(str, EPOCHS):
        comparison[epoch], differences[epoch] = {}, {}
        for flavor, results, records in (
            ("fixed", fixed["endpoint"], fixed["training"]),
            ("dynamic", metrics, training),
        ):
            for a in early.ARMS:
                score = results[f"{epoch}:{a}"]["panel"]
                comparison[epoch][f"{flavor} {a}"] = {
                    "panel": score,
                    **{
                        k: records[epoch][a][k]
                        for k in ("edits", "exposure", "mass_total")
                    },
                }
        for a in ACTIVE:
            left, right = (comparison[epoch][f"{x} {a}"] for x in ("dynamic", "fixed"))
            differences[epoch][a] = {
                "delta_edits": left["edits"] - right["edits"],
                "delta_exposure": left["exposure"] - right["exposure"],
                "panel": early.differences(left["panel"], right["panel"]),
            }
        for field in ("fixed", "rematched"):
            assert (
                comparison[epoch]["fixed E0"]["panel"][field]
                == comparison[epoch]["dynamic E0"]["panel"][field]
            )
        for group in match.GROUPS:
            np.testing.assert_allclose(
                list(
                    comparison[epoch]["fixed E0"]["panel"]["geometry"][group].values()
                ),
                list(
                    comparison[epoch]["dynamic E0"]["panel"]["geometry"][group].values()
                ),
                atol=2e-6,
                rtol=0,
            )
        comparison[epoch]["reused KEEP"] = comparison[epoch].pop("fixed E0")
        comparison[epoch].pop("dynamic E0")
    result = {
        "provenance": {
            k: v
            for k, v in frozen.items()
            if k not in ("panel_ids", "reference_ids", "source_ids")
        },
        "checkpoint_sha256": new_hash,
        "fixed_checkpoint_sha256": fixed["checkpoint_sha256"],
        "alignment": alignment,
        "endpoint": metrics,
        "training": training,
        "turnover": changes,
        "comparison": comparison,
        "dynamic_minus_fixed": differences,
        "edits": edits_summary(old_events, events),
        "fixed_edits": edits_summary(old_events, fixed_events),
        "fixed_applied_semantics": match.applied_semantics(
            fixed_events, truth, mapping
        ),
        "applied_semantics": match.applied_semantics(events, truth, mapping),
        "seconds": time.monotonic() - begin,
    }
    render(result)
    audit.write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
