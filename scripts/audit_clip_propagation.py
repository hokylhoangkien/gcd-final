"""Sealed propagation audit of the five-epoch CLIP repair continuation."""

import argparse
import copy
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from scripts import audit_clip_repair as repair
from scripts import audit_frozen_readout as readout
from scripts import audit_g6_confirm as stats
from scripts import audit_oracle_move as audit
from scripts.audit_e3_proposals import align, json_native

LOCAL = repair.g1.WORK / "clip-propagation"
MD = Path("reports/CRISH_CLIP_PROPAGATION_AUDIT.md")
OUT = Path("reports/r2_clip_propagation_audit.json")
ARMS = repair.ARMS
SNAPSHOTS = ("R150", *ARMS)
SEED = 240155
BOOT = 2000
GROUPS = ("All", "Old", "New")


def masks(y):
    return dict(zip(GROUPS, (y >= 0, y < 98, y >= 98), strict=True))


def load_inputs():
    initial_path = repair.g1.WORK / "input/epoch-150.pt"
    result = json.loads(repair.OUT.read_text())
    protocol = result["protocol"]
    assert audit.file_digest(initial_path) == protocol["checkpoint_sha256"]
    initial = audit.load_checkpoint(initial_path)
    assert initial["epoch"] == 150
    file = repair.LOCAL / "frozen-policy.json"
    assert audit.file_digest(file) == protocol["policy_sha256"]
    frozen = json.loads(file.read_text())
    sig = {**frozen["signature"], "policy_sha256": protocol["policy_sha256"]}
    assert all(sig[k] == v for k, v in protocol.items())
    model, loader, _, center, _, _ = audit.build()
    model.eval().requires_grad_(False)
    ds = loader.dataset
    ids = np.r_[ds.labelled_dataset.uq_idxs, ds.unlabelled_dataset.ids]
    panel = np.asarray(sig["panel_ids"])
    assert len(np.unique(panel)) == 588 and not set(panel) & set(sig["reference_ids"])
    pool = {key: (src, dst) for key, src, dst, _ in frozen["proposals"]}
    direct = {r[0] for r in frozen["proposals"][: sig["locked_n"]]}
    assert not set(pool) & set(panel)
    plans = repair.schedule(loader, ids, panel, initial)
    events, hashes = [], {"150": protocol["checkpoint_sha256"]}
    final = None
    for e, (rows, seeds) in plans.items():
        assert repair.digest_array(rows) == sig["draws"][str(e)]["rows"]
        assert repair.digest_array(seeds) == sig["draws"][str(e)]["seeds"]
        path = repair.LOCAL / f"epoch-{e}.pt"
        hashes[str(e)] = audit.file_digest(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert ckpt["signature"] == sig and ckpt["epoch"] == e
        assert ckpt["records"][str(e)] == result["epochs"][str(e)]
        assert len(ckpt["batch_ledger"]) == len(rows)
        seen = Counter()
        for batch, (steps, selected) in enumerate(
            zip(ckpt["batch_ledger"], rows, strict=True), 1
        ):
            assert len(steps["T1"]["moves"]) == len(steps["T2"]["moves"])
            assert not steps["T0"]["moves"]
            for arm in ("T1", "T2"):
                moves = steps[arm]["moves"]
                assert len({move[0] for move in moves}) == len(moves)
                for row, key, src, dst in moves:
                    assert 0 <= row < len(selected) and key == ids[selected[row]]
                    assert (src, dst) == pool[key] and src != dst
                    assert arm != "T1" or key in direct
                    events.append((e, batch, arm, row, key, src, dst))
                    seen[arm] += 1
        for a in ARMS:
            assert seen[a] == result["epochs"][str(e)][a]["applied"]
        if e == 155:
            final = ckpt["state"]
        del ckpt
    assert final is not None and events
    cache = torch.load(
        repair.g1.WORK / "oracle-move/step0-panel-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert cache["panel_ids"] == panel.tolist()
    hist = json.loads(repair.HISTORY.read_text())
    assert cache["checkpoint_sha256"] == hist["checkpoint_sha256"]
    result_meta = {
        "base": "240cdbf",
        "checkpoint_sha256": hashes,
        "training_report_sha256": audit.file_digest(repair.OUT),
        "source_policy_sha256": protocol["policy_sha256"],
        "mass": protocol["move_mass"],
        "rows_per_epoch": protocol["rows_per_epoch"],
        "panel_ids": panel.tolist(),
        "fit_n": len(ids) - len(panel),
        "crop_seed": SEED,
        "bootstrap_draws": BOOT,
        "readout_seeds": readout.SEEDS,
        "readout_max_iterations": readout.MAX_ITER,
        "oracle_ridge": 1.0,
    }
    return (
        model,
        ds,
        center,
        ids,
        panel,
        cache["tokens"],
        initial["model"],
        final,
        events,
        sig,
        result_meta,
        result,
    )


@torch.inference_mode()
def extract(model, ds, center, ids, panel, tokens, initial, final, sources, smoke):
    tails, weights = {}, {}
    for tag in SNAPSHOTS:
        state = initial if tag == "R150" else final[tag]["model"]
        for k, value in model[0].state_dict().items():
            if not k.startswith("blocks.11."):
                assert torch.equal(value.cpu(), state[f"0.{k}"]), k
        tail = copy.deepcopy(model[0].blocks[-1])
        tail.load_state_dict({k: state[f"0.blocks.11.{k}"] for k in tail.state_dict()})
        tails[tag] = torch.nn.Sequential(tail, model[0].norm).eval()
        v, g = (state[f"1.last_layer.weight_{k}"].cuda() for k in ("v", "g"))
        weights[tag] = F.normalize(v, dim=1) * g

    def apply(parts):
        return {
            a: F.normalize(
                torch.cat([tails[a](t.cuda())[:, 0] for t in parts]), dim=1
            ).cpu()
            for a in SNAPSHOTS
        }

    augmented = {}
    source_rows = np.flatnonzero(np.isin(ids, sources))
    repair.seed_all(SEED)
    source_offset = 0
    for views, legal, keys, mask in DataLoader(
        Subset(ds, source_rows.tolist()), batch_size=16
    ):
        assert (legal == -1).all() and not mask.any()
        end = source_offset + len(keys)
        np.testing.assert_array_equal(keys.numpy(), ids[source_rows[source_offset:end]])
        source_offset = end
        batch = apply(audit.prefix(model[0], torch.cat(views)))
        for a, value in batch.items():
            augmented.setdefault(a, []).append(value.reshape(2, len(keys), 768))
        if smoke:
            break
    assert smoke or source_offset == len(source_rows)
    if smoke:
        assert all(torch.isfinite(p[0]).all() for p in augmented.values())

    ds.labelled_dataset.transform = ds.unlabelled_dataset.transform = center
    rows = np.flatnonzero(~np.isin(ids, panel))
    fitted, labels, seen = {}, [], []
    for images, legal, keys, mask in DataLoader(
        Subset(ds, rows.tolist()), batch_size=32
    ):
        assert torch.equal(mask.flatten().bool(), legal >= 0)
        for a, value in apply(audit.prefix(model[0], images)).items():
            fitted.setdefault(a, []).append(value)
        labels.extend(legal.tolist())
        seen.extend(keys.tolist())
        if smoke:
            break
        if len(seen) % 2048 == 0:
            print("Shared center prefix rows", len(seen), flush=True)
    fit = ids[rows[: len(seen)]]
    np.testing.assert_array_equal(fit, seen)
    for batch in tokens if not smoke else tokens[:1]:
        for a, value in apply(batch).items():
            fitted[a].append(value)
    x = {a: torch.cat(v).cuda() for a, v in fitted.items()}
    aug_logits = {
        a: (torch.cat(v, dim=1).cuda() @ weights[a].T).cpu().numpy()
        for a, v in augmented.items()
    }
    return x, weights, fit, np.asarray(labels), ids[source_rows], aug_logits


def freeze(x, w, fit, labels, panel, source_ids, crops, meta):
    out = {"fit": fit, "panel": panel, "legal": labels, "source_ids": source_ids}
    for f in SNAPSHOTS:
        for h in SNAPSHOTS if f == "R150" else ARMS:
            logits = (x[f] @ w[h].T).cpu().numpy()
            out[f"pred_{f}_{h}"] = logits.argmax(1)
            if f == h:
                out[f"logits_{f}"] = logits
        out[f"aug_{f}"] = crops[f]
        out[f"weight_{f}"] = w[f].cpu().numpy()
        out[f"feature_{f}"] = x[f].cpu().numpy()
        for method in ("spherical", "semi"):
            out[f"{method}_{f}"] = np.stack(
                [
                    readout.spherical(x[f], len(fit), labels, seed, method == "semi")
                    for seed in readout.SEEDS
                ]
            )
        print("Frozen endpoint and legal readouts", f, flush=True)
    frozen = LOCAL / "frozen.npz"
    tmp = frozen.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(frozen)
    meta["frozen_sha256"] = audit.file_digest(frozen)
    audit.write_json(LOCAL / "frozen-provenance.json", json_native(meta))
    return out


def interval(values, draws):
    if not len(values):
        return None
    v = np.asarray(values, dtype=np.float32)
    dist = draws @ v / len(v)
    return {"mean": float(v.mean()), "ci95": np.quantile(dist, [0.025, 0.975]).tolist()}


def rate_interval(counts, total, draws):
    if not total.sum():
        return None
    den = draws @ total
    valid = den > 0
    rate = (draws @ counts)[valid] / den[valid]
    return {
        "mean": float(counts.sum() / total.sum()),
        "ci95": np.quantile(rate, [0.025, 0.975]).tolist(),
    }


def semantics(events, ids, table, truth, meta):
    counts = {a: np.zeros((len(ids), 5), dtype=np.float32) for a in ("T1", "T2")}
    lookup = {int(k): i for i, k in enumerate(ids)}
    for _, _, a, _, key, src, dst in events:
        y, s, d = truth[key], table[src], table[dst]
        category = 4 if min(s, d) < 0 else 0 if d == y else 1 if s == y else 2
        counts[a][lookup[key], category] += 1
        counts[a][lookup[key], 3] += 1
    out = {}
    labels = truth[ids]
    for g, mask in masks(labels).items():
        selected = mask & ((counts["T1"][:, 3] + counts["T2"][:, 3]) > 0)
        weights = stats.bootstrap_counts(labels[selected])
        out[g] = {}
        for a in counts:
            repairs, harms, wrong, total, unresolved = counts[a][selected].T
            categories = {
                "correct_repair": repairs,
                "harmful": harms,
                "wrong_to_wrong": wrong,
                "unresolved_slot": unresolved,
            }
            out[g][a] = {
                "occurrences": int(total.sum()),
                "unique_ids": int((total > 0).sum()),
                "repeated_ids": int((total > 1).sum()),
                "extra_occurrences": int(np.maximum(total - 1, 0).sum()),
                "max_repeats": int(total.max()) if len(total) else 0,
                "moved_mass": float(total.sum() * meta["mass"]),
                "total_wt_loss_target_mass": float(
                    total.sum() * meta["mass"] * 0.65 / 128
                ),
                "categories": {
                    k: {
                        "occurrences": int(v.sum()),
                        "moved_mass": float(v.sum() * meta["mass"]),
                        "occurrence_rate": rate_interval(v, total, weights),
                    }
                    for k, v in categories.items()
                },
            }
        a, b = (counts[k][selected] for k in ("T1", "T2"))
        ar, br = weights @ a[:, 3], weights @ b[:, 3]
        keep = (ar > 0) & (br > 0)
        if keep.any():
            gap = (weights @ a[:, 0])[keep] / ar[keep] - (weights @ b[:, 0])[keep] / br[
                keep
            ]
            out[g]["repair_rate_T1_minus_T2"] = {
                "mean": float(
                    a[:, 0].sum() / a[:, 3].sum() - b[:, 0].sum() / b[:, 3].sum()
                ),
                "ci95": np.quantile(gap, [0.025, 0.975]).tolist(),
            }
    return out


def softmax(x, temp):
    v = x / temp
    out = np.exp(v - v.max(axis=-1, keepdims=True))
    return out / out.sum(axis=-1, keepdims=True)


def endpoint(out, events, truth, table, true_slots, relative_ids):
    key = {int(i): j for j, i in enumerate(relative_ids)}
    fits = out["source_ids"]
    pairs = {event[4]: (event[5], event[6]) for event in events}
    sources = np.asarray([pairs[int(i)][0] for i in fits])
    targets = np.asarray([pairs[int(i)][1] for i in fits])
    loc = {int(i): j for j, i in enumerate(fits)}
    base_w = out["weight_R150"]
    result, raw = {}, {}
    for tag in SNAPSHOTS:
        logit = out[f"logits_{tag}"]
        aug = out[f"aug_{tag}"]
        source_p = softmax(aug, 0.04).mean(0)
        student = softmax(aug, 0.1).mean(0)
        rows = np.arange(len(fits))
        true = true_slots[truth[relative_ids]]
        ok = true >= 0
        rank = np.full(len(relative_ids), np.nan)
        rank[ok] = 1 + (logit[ok] > logit[ok, true[ok], None]).sum(1)
        x = out[f"feature_{tag}"]
        w = out[f"weight_{tag}"]
        preds = logit.argmax(1)
        source_raw = aug[:, rows, sources]
        dest_raw = aug[:, rows, targets]
        raw[tag] = {
            "slot": preds,
            "correct": table[preds] == truth[relative_ids],
            "true_rank": rank,
            "teacher_confidence": softmax(logit, 0.04).max(1),
            "student_confidence": softmax(logit, 0.1).max(1),
            "source_raw": source_raw.mean(0),
            "destination_raw": dest_raw.mean(0),
            "teacher_source_p": source_p[rows, sources],
            "teacher_destination_p": source_p[rows, targets],
            "student_source_p": student[rows, sources],
            "student_destination_p": student[rows, targets],
            "teacher_margin": (dest_raw - source_raw).mean(0) / 0.04,
            "student_margin": (dest_raw - source_raw).mean(0) / 0.1,
            "feature_drift": np.maximum(0, 1 - (x * out["feature_R150"]).sum(1))
            if tag != "R150"
            else np.zeros(len(x)),
        }
        w_cos = 1 - (w * base_w).sum(1) / (
            np.linalg.norm(w, axis=1) * np.linalg.norm(base_w, axis=1)
        )
        result[tag] = {
            "relative_weight_fro": float(
                np.linalg.norm(w - base_w) / np.linalg.norm(base_w)
            ),
            "weight_slot_drift": {
                **{
                    g: finite_mean(np.maximum(w_cos[m & (table >= 0)], 0))
                    for g, m in masks(table).items()
                },
                "all_196": float(np.maximum(w_cos, 0).mean()),
                "mapped_slots": int((table >= 0).sum()),
            },
        }
    cohorts = {
        "panel": out["panel"],
        **{
            f"{arm}_sources": np.asarray(
                sorted({row[4] for row in events if row[2] == arm})
            )
            for arm in ("T1", "T2")
        },
    }
    fields = (
        "correct",
        "teacher_confidence",
        "student_confidence",
        "true_rank",
        "feature_drift",
    )
    edit_fields = (
        "source_raw",
        "destination_raw",
        "teacher_source_p",
        "teacher_destination_p",
        "student_source_p",
        "student_destination_p",
        "teacher_margin",
        "student_margin",
    )
    summaries = {}
    for cohort, keys in cohorts.items():
        labels = truth[keys]
        indices = np.asarray([key[i] for i in keys])
        summaries[cohort] = {}
        for g, mask in masks(labels).items():
            i = indices[mask]
            weights = stats.bootstrap_counts(labels[mask])
            edit_i = np.asarray([loc[k] for k in keys[mask] if k in loc], dtype=int)
            row = {"ids": len(i), "states": {}, "paired": {}}
            for tag in SNAPSHOTS:
                row["states"][tag] = {
                    **{k: finite_mean(raw[tag][k][i]) for k in fields},
                    **{k: finite_mean(raw[tag][k][edit_i]) for k in edit_fields},
                    "mapped_true_rank_n": int(
                        np.isfinite(raw[tag]["true_rank"][i]).sum()
                    ),
                }
            for a, b in (
                ("T0", "R150"),
                ("T1", "R150"),
                ("T2", "R150"),
                ("T1", "T0"),
                ("T2", "T0"),
                ("T1", "T2"),
            ):
                ra, rb = raw[a], raw[b]
                row["paired"][f"{a}-{b}"] = {
                    k: interval(
                        ra[k][i].astype(float) - rb[k][i].astype(float), weights
                    )
                    for k in fields
                    if k != "true_rank"
                }
                row["paired"][f"{a}-{b}"]["prediction_changed"] = interval(
                    ra["slot"][i] != rb["slot"][i], weights
                )
                rank_mask = mask & np.isfinite(ra["true_rank"][indices])
                j = indices[rank_mask]
                row["paired"][f"{a}-{b}"]["true_rank"] = interval(
                    ra["true_rank"][j] - rb["true_rank"][j],
                    stats.bootstrap_counts(labels[rank_mask]),
                )
                if len(edit_i):
                    row["paired"][f"{a}-{b}"].update(
                        {
                            k: interval(ra[k][edit_i] - rb[k][edit_i], weights)
                            for k in edit_fields
                        }
                    )
            summaries[cohort][g] = row
    return result, summaries, raw, cohorts, pairs


def finite_mean(v):
    good = v[np.isfinite(v)]
    return float(good.mean()) if len(good) else None


def panel_readouts(out, truth, table):
    fit, panel = out["fit"], out["panel"]
    yf, yp = truth[fit], truth[panel]
    oracle = {}
    for tag in SNAPSHOTS:
        x = torch.from_numpy(out[f"feature_{tag}"]).cuda()
        oracle[tag] = readout.oracle(x, yf)
    n = len(fit)
    swaps, predictions = {}, {}
    for f in ARMS:
        for h in ARMS:
            method = f"{f}_feature__{h}_head"
            pred = out[f"pred_{f}_{h}"][n:]
            predictions[method] = table[pred] == yp
            swaps[method] = {
                g: finite_mean(predictions[method][m]) for g, m in masks(yp).items()
            }
    comparisons = {}
    for g, m in masks(yp).items():
        w = stats.bootstrap_counts(yp[m])
        comparisons[g] = {
            k: interval(
                p[m].astype(float)
                - predictions["T0_feature__T0_head"][m].astype(float),
                w,
            )
            for k, p in predictions.items()
        }
    scores, paired = {}, {}
    outcomes = {}
    for tag in SNAPSHOTS:
        legal = {name: out[f"{name}_{tag}"] for name in ("spherical", "semi")}
        all_pred = {**legal, **oracle[tag]}
        scores[tag] = {
            name: {
                "per_seed": (vals := [readout.evaluate(p, yf, yp) for p in preds]),
                "mean": readout.reduce_rows(vals),
            }
            for name, preds in all_pred.items()
        }
        outcomes[tag] = {
            name: np.mean(
                [readout.mapping(p[:n], yf)[p[n:]] == yp for p in preds], axis=0
            )
            for name, preds in all_pred.items()
        }
    for a, b in (
        ("T0", "R150"),
        ("T1", "R150"),
        ("T2", "R150"),
        ("T1", "T0"),
        ("T2", "T0"),
        ("T1", "T2"),
    ):
        paired[f"{a}-{b}"] = {}
        for g, m in masks(yp).items():
            w = stats.bootstrap_counts(yp[m])
            paired[f"{a}-{b}"][g] = {
                k: interval(v[m] - outcomes[b][k][m], w) for k, v in outcomes[a].items()
            }
    return {
        "swaps": swaps,
        "swaps_paired_vs_T0_native": comparisons,
        "readouts": scores,
        "readout_paired_accuracy": paired,
    }


def export_ids(out, all_ids, truth, raw, cohorts, pairs, events):
    index = {int(v): i for i, v in enumerate(all_ids)}
    loc = {int(v): i for i, v in enumerate(out["source_ids"])}
    counts = Counter((a, key) for _, _, a, _, key, _, _ in events)
    ids = np.unique(np.concatenate(list(cohorts.values())))
    path = LOCAL / "per-id.csv"
    with path.open("w", newline="", encoding="utf8") as f:
        edit = (
            "source_raw",
            "destination_raw",
            "teacher_source_p",
            "teacher_destination_p",
            "student_source_p",
            "student_destination_p",
            "teacher_margin",
            "student_margin",
        )
        main = (
            "slot",
            "correct",
            "true_rank",
            "teacher_confidence",
            "student_confidence",
            "feature_drift",
        )
        writer = csv.writer(f)
        writer.writerow(
            (
                "id",
                "cohort",
                "group",
                "snapshot",
                "T1_occurrences",
                "T2_occurrences",
                "source",
                "target",
                *main,
                *edit,
            )
        )
        for k in ids:
            channel = "panel" if k not in pairs else "intervention_source"
            src, dst = pairs.get(k, ("", ""))
            for tag in SNAPSHOTS:
                r = raw[tag]
                i = index[k]
                values = [r[v][i] for v in main]
                if not np.isfinite(values[2]):
                    values[2] = ""
                writer.writerow(
                    (
                        k,
                        channel,
                        "Old" if truth[k] < 98 else "New",
                        tag,
                        counts[("T1", k)],
                        counts[("T2", k)],
                        src,
                        dst,
                        *values,
                        *[r[v][loc[k]] if k in loc else "" for v in edit],
                    )
                )
    return audit.file_digest(path)


def seal(out, events, sig, meta, training):
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    all_ids = np.r_[out["fit"], out["panel"]]
    assert np.array_equal(
        out["legal"][out["legal"] >= 0], truth[out["fit"][out["legal"] >= 0]]
    )
    mp, coverage = align(
        sig["reference_ids"],
        dict(zip(all_ids.tolist(), out["pred_R150_R150"].tolist(), strict=True)),
        truth,
    )
    table, inverse = np.full(196, -1), np.full(196, -1)
    for slot, label in mp.items():
        table[slot], inverse[label] = label, slot
    drift, change, raw, cohorts, pairs = endpoint(
        out, events, truth, table, inverse, all_ids
    )
    for arm in ARMS:
        for g in GROUPS:
            got = change["panel"][g]["states"][arm]["correct"]
            want = training["epochs"]["155"][arm]["panel"]["fixed"][g.lower()][
                "accuracy"
            ]
            assert abs(got - want) < 1e-12, (arm, g, got, want)
    meta["per_id_csv_sha256"] = export_ids(
        out, all_ids, truth, raw, cohorts, pairs, events
    )
    return {
        "alignment": coverage,
        "moves": semantics(events, out["source_ids"], table, truth, meta),
        "weight_drift": drift,
        "endpoint_change": change,
        **panel_readouts(out, truth, table),
    }


def report(meta, result):
    lines = [
        "# CLIP repair propagation on Stanford Cars",
        "",
        (
            "This sealed audit reuses R2-150, the three R2-155 continuation endpoints and all five "
            "intervention ledgers. The fixed 588-ID panel was seen before the continuation and "
            "excluded throughout the five new epochs. Source cohorts contain actual T1/T2 MOVE IDs, "
            "can overlap one another, and are disjoint from the panel."
        ),
        "",
        (
            "The initial slot alignment is fitted on the same non-panel reference IDs used in "
            "the continuation report. An occurrence is unresolved when either slot has no mapping. "
            "Moved mass is 0.2 per occurrence. The batch-normalized contribution to the weighted "
            "distillation target is 0.65 × 0.2 / 128 per occurrence, with the two-view mean already "
            "accounted for. This is target mass rather than a measurement of gradient or parameter change."
        ),
        "",
        (
            "Per-ID margins compare the same frozen CLIP source and destination on one reproducible "
            "pair of image augmentations across the four states. Teacher and student share raw logits "
            "and use temperatures .04 and .1. Endpoint crops do not reproduce historical batch logits. "
            "Confidence, true-slot rank and slot changes use center views. Ranks require a mapped true slot. "
            "Feature drift is one minus normalized CLS cosine. Effective classifier weights use frozen-scale "
            "weight normalization and are compared to R2-150 without permuting native slots."
        ),
        "",
        (
            "Intervals are pointwise 95% paired query bootstraps, stratified by Old/New. Ledger intervals "
            "resample IDs with all repeated occurrences together. Inference conditions on the fixed checkpoints, "
            "mapping and cohorts. Source cohorts are selected by observed exposure, so their changes do not "
            "establish a causal effect on those IDs."
        ),
        "",
        "| Group / arm | Edits / distinct IDs / repeated IDs | Repairs / harmful / wrong-to-wrong / unresolved | Moved mass |",
        "|---|---|---|---:|",
    ]

    def f(v):
        return "—" if v is None else f"{v:.4f}"

    for g, row in result["moves"].items():
        for arm in ("T1", "T2"):
            a = row[arm]
            c = " / ".join(str(k["occurrences"]) for k in a["categories"].values())
            lines.append(
                f"| {g} / {arm} | {a['occurrences']} / {a['unique_ids']} / {a['repeated_ids']} | {c} | {a['moved_mass']:.1f} |"
            )
    lines += [
        "",
        "| Cohort / snapshot | Accuracy All / Old / New (%) | Mean teacher margin (All) | True-slot rank (All) | CLS drift (All) |",
        "|---|---|---:|---:|---:|",
    ]
    for c, groups in result["endpoint_change"].items():
        for a in SNAPSHOTS:
            r = groups["All"]["states"][a]
            acc = " / ".join(
                f(100 * groups[g]["states"][a]["correct"]) if groups[g]["ids"] else "—"
                for g in GROUPS
            )
            lines.append(
                f"| {c} / {a} | {acc} | {f(r['teacher_margin'])} | {f(r['true_rank'])} | {f(r['feature_drift'])} |"
            )
    lines += [
        "",
        "| Snapshot | Relative effective weight L2 | Mean cosine drift All / Old / New |",
        "|---|---:|---|",
    ]
    for tag, row in result["weight_drift"].items():
        groups = " / ".join(f(row["weight_slot_drift"][g]) for g in GROUPS)
        lines.append(f"| {tag} | {row['relative_weight_fro']:.6f} | {groups} |")
    lines += ["", "| Feature / head | Fixed panel All / Old / New (%) |", "|---|---|"]
    for k, m in result["swaps"].items():
        triplet = " / ".join(f"{100 * m[g]:.2f}" for g in GROUPS)
        lines.append(
            f"| {k.replace('_feature__', ' / ').removesuffix('_head')} | {triplet} |"
        )
    lines += [
        "",
        (
            "Swaps hold the initial semantic alignment fixed across all nine combinations. The previous "
            "readout protocol fits spherical and semisupervised K-means on all non-panel IDs using seeds 1/7/19. "
            "Oracle centroid and unit-ridge linear readouts use privileged fit-set labels as diagnostic references "
            "and cannot be deployed under GCD label access. "
            "Panel labels are used only for scoring and optional descriptive rematching."
        ),
        "",
        "| Snapshot / readout | Fit-aligned All / Old / New (%) | Rematched All / Old / New (%) |",
        "|---|---|---|",
    ]
    for a, methods in result["readouts"].items():
        for k, m in methods.items():
            groups = m["mean"]
            acc = [
                " / ".join(f"{100 * groups[g][method]:.2f}" for g in GROUPS)
                for method in ("fit", "rematch")
            ]
            lines.append(f"| {a} / {k} | {acc[0]} | {acc[1]} |")
    lines += [
        "",
        f"Elapsed {meta['elapsed_seconds']:.1f} seconds. Per-ID predictions, ranks and margins are in the local per-id.csv.",
        "",
        "[Paired intervals, subgroup details and digests](r2_clip_propagation_audit.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("--smoke", action="store_true")
    smoke = cmd.parse_args().smoke
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    (
        model,
        ds,
        center,
        ids,
        panel,
        tokens,
        initial,
        final,
        events,
        sig,
        meta,
        completed,
    ) = load_inputs()
    sources = sorted({e[4] for e in events})
    assert not set(sources) & set(panel)
    x, w, fit, y, a_ids, crops = extract(
        model, ds, center, ids, panel, tokens, initial, final, sources, smoke
    )
    if smoke:
        assert all(crops[a].shape == (2, 16, 196) for a in SNAPSHOTS)
        assert all(torch.isfinite(x[a]).all() for a in SNAPSHOTS)
        print(
            "Verified all ledgers and real paired center/augmentation encoding",
            len(events),
            flush=True,
        )
        return
    out = freeze(x, w, fit, y, panel, a_ids, crops, meta)
    assert audit.file_digest(LOCAL / "frozen.npz") == meta["frozen_sha256"]
    result = seal(out, events, sig, meta, completed)
    meta["elapsed_seconds"] = time.monotonic() - start
    report(meta, result)
    audit.write_json(OUT, json_native({"provenance": meta, **result}))


if __name__ == "__main__":
    main()
