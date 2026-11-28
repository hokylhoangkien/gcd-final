"""Verify D1 Cars archives and replay the locked frozen-checkpoint audit."""

import argparse
import copy
import itertools
import json
import math
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dual import cache
from dual.policy import Settings
from relational.export import digest_stream
from relational.state import JOURNALS, MILESTONES, write_json
from scripts import audit_e0_baseline as e0
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_oracle_move import accuracy as fixed_accuracy

WORK = e0.g1.WORK / "d1-final"
COMMIT = "94683693ee74ecd0cebcdd32b79d4be903782910"
TAGS = tuple(f"D1-{n}" for n in MILESTONES)
GROUPS = e0.GROUPS


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return e0.audit.file_digest(path)


def jsonl(raw):
    return [json.loads(line) for line in raw.splitlines()]


def archive(folder, work):
    e0.require_git(COMMIT)
    receipts, recent_journals, states = {}, None, {}
    metadata = None
    for epoch in reversed(MILESTONES):
        path = folder / f"D1-epoch-{epoch}.tar"
        expected = path.with_suffix(".tar.sha256").read_text().split()[0]
        observed = digest(path)
        trusted = expected == observed
        if epoch == MILESTONES[-1] and not trusted:
            raise ValueError("Final D1 archive checksum mismatch")
        with tarfile.open(path, "r:") as tar:
            if any(
                not item.isfile() or Path(item.name).name != item.name for item in tar
            ):
                raise ValueError("D1 archive contains a nonregular or nested member")
            lines = tar.extractfile("SHA256SUMS").read().decode().splitlines()
            members = {name: sha for sha, name in (row.split() for row in lines)}
            assert len(lines) == len(members)
            assert sorted(tar.getnames()) == sorted([*members, "SHA256SUMS"])
            mismatches = []
            for name, sha in members.items():
                with tar.extractfile(name) as stream:
                    if digest_stream(stream) != sha:
                        mismatches.append(name)
            if trusted and mismatches:
                raise ValueError(f"Archive {epoch} member mismatches: {mismatches}")
            for name, sha in members.items():
                if name.startswith("epoch-"):
                    if name in states:
                        assert states[name] == sha
                    elif trusted:
                        dest = work / name
                        if not dest.exists() or digest(dest) != sha:
                            tmp = dest.with_suffix(".pt.tmp")
                            with tar.extractfile(name) as src, tmp.open("wb") as dst:
                                shutil.copyfileobj(src, dst, 1024 * 1024)
                            assert digest(tmp) == sha
                            tmp.replace(dest)
                        states[name] = sha
            status = json.load(tar.extractfile("status.json"))
            assert status["epoch"] == epoch and status["variant"] == "D1"
            assert status["checkpoint"] == f"epoch-{epoch}.pt"
            config = json.load(tar.extractfile("config.json"))
            env = json.load(tar.extractfile("environment.json"))
            present = {name: tar.extractfile(name).read() for name in JOURNALS}
            if metadata is None:
                metadata = config, env
                recent_journals = present
            assert metadata == (config, env)
            assert all(
                recent_journals[name].startswith(present[name]) for name in JOURNALS
            )
            assert all(len(jsonl(raw)) == epoch for raw in present.values())
            receipts[str(epoch)] = {
                "archive_sha256": observed,
                "expected_archive_sha256": expected,
                "sidecar_matches": trusted,
                "member_digest_mismatches": mismatches,
                "members": members,
                "status": status,
                "journal_bytes": {k: len(v) for k, v in present.items()},
                "checkpoint_extraction_source": "D1-epoch-200.tar",
            }
    receipts = {str(epoch): receipts[str(epoch)] for epoch in MILESTONES}
    metrics, ledgers = (jsonl(recent_journals[name]) for name in JOURNALS)
    config, env = metadata
    assert config["variant"] == "D1" and config["protocol"]["commit"] == COMMIT
    assert config["settings"] == vars(Settings()) and config["microbatch8"] is False
    checkpoints = {}
    for epoch in MILESTONES:
        path = work / f"epoch-{epoch}.pt"
        saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        assert saved["epoch"] == epoch and saved["identity"] == config
        assert saved["journal_bytes"] == receipts[str(epoch)]["journal_bytes"]
        sched, opt = saved["scheduler"], saved["optimizer"]
        assert sched["T_max"] == 200 and sched["eta_min"] == 0.0001
        assert sched["last_epoch"] == epoch and sched["_step_count"] == epoch + 1
        assert sched["base_lrs"] == [0.1, 0.1]
        assert all(
            math.isclose(lr, cosine(epoch), abs_tol=1e-12) for lr in sched["_last_lr"]
        )
        assert [g["lr"] for g in opt["param_groups"]] == sched["_last_lr"]
        assert sorted(g["weight_decay"] for g in opt["param_groups"]) == [0, 5e-5]
        assert all(g["momentum"] == 0.9 for g in opt["param_groups"])
        parameters = [p for group in opt["param_groups"] for p in group["params"]]
        assert set(parameters) == set(opt["state"])
        assert all(
            r["momentum_buffer"].numel() and torch.isfinite(r["momentum_buffer"]).all()
            for r in opt["state"].values()
        )
        assert (
            len(saved["rng"]) == 4
            and saved["rng"][2].numel()
            and len(saved["rng"][3]) == 1
        )
        receipts[str(epoch)]["optimizer_entries"] = len(opt["state"])
        receipts[str(epoch)]["scheduler"] = sched
        checkpoints[f"D1-{epoch}"] = saved["model"]
    return (
        {"archives": receipts, "identity": config, "environment": env},
        metrics,
        ledgers,
        checkpoints,
    )


def cosine(epoch):
    return 0.0001 + (0.1 - 0.0001) * (1 + math.cos(math.pi * epoch / 200)) / 2


def history(metrics, ledgers, ids):
    epochs, all_keys = [], set(ids.tolist())
    applications = Counter()
    statuses, reasons = Counter(), Counter()
    prior = None
    opportunity = []
    for ep, (m, row) in enumerate(zip(metrics, ledgers, strict=True), 1):
        assert m["epoch"] == row["epoch"] == ep and math.isclose(
            m["lr"], cosine(ep - 1), abs_tol=1e-12
        )
        score = [m[k] for k in GROUPS]
        assert all(0 <= v <= 1 for v in score) and all(
            math.isfinite(m[k]) for k in ("loss", "baseline_loss", "seconds")
        )
        assert row["rows"] == 8064
        proposals = {key: (src, dst) for key, src, dst, _ in row.get("proposals", [])}
        actual = row.get("ledger", [])
        counts = row.get("counts", {})
        status = row["status"]
        assert status in ("ok", "timeout", "warmup") and (status == "warmup") == (
            ep <= 10
        )
        assert len(proposals) == row["proposed"] <= 614 and set(proposals) <= all_keys
        assert all(
            0 <= src < 196 and 0 <= dst < 196 and dst != src
            for src, dst in proposals.values()
        )
        if status == "ok":
            assert list(proposals) == row["candidate_ids"] and sum(
                row["slot_counts"]
            ) == len(ids)
            assert row["eligible_positive"] >= len(proposals) and row["budget"] == 322
        assert len(actual) == row["applied"] <= row.get("budget", 0)
        assert len({a[2] for a in actual}) == row["unique_applied"]
        assert len({tuple(a[:2]) for a in actual}) == len(actual)
        assert all(
            1 <= step <= 63 and 0 <= r < 128 and proposals[k] == (s, d)
            for step, r, k, s, d in actual
        )
        assert max(Counter(a[0] for a in actual).values(), default=0) <= 8
        assert math.isclose(row["mass"], 0.2 * len(actual)) and math.isclose(
            row["exposure"], len(actual) / row["rows"]
        )
        assert counts.get("encounters", 0) == len(actual) + sum(
            n for k, n in counts.items() if k != "encounters"
        )
        assert bool(m["gradient_first_edited_batch"]) == bool(actual)
        applications.update(a[2] for a in actual)
        statuses[status] += 1
        reasons.update(counts)
        shared = proposals.keys() & prior.keys() if prior is not None else set()
        union = proposals.keys() | prior.keys() if prior is not None else set()
        opportunity.append(
            {
                "epoch": ep,
                "same_id": len(shared),
                "union_ids": len(union),
                "same_source": sum(proposals[k][0] == prior[k][0] for k in shared),
                "same_destination": sum(proposals[k][1] == prior[k][1] for k in shared),
            }
        )
        prior = proposals
        epochs.append(
            {
                "epoch": ep,
                "accuracy": score,
                "loss": m["loss"],
                "baseline_loss": m["baseline_loss"],
                "lr": m["lr"],
                "epoch_seconds": m["seconds"],
                "peak_vram_bytes": m["peak_vram_bytes"],
                "first_edit_gradients": m["gradient_first_edited_batch"],
                **{
                    k: row.get(k)
                    for k in (
                        "status",
                        "rows",
                        "proposed",
                        "applied",
                        "unique_applied",
                        "mass",
                        "exposure",
                        "counts",
                        "seconds",
                        "runtime",
                        "eligible_positive",
                        "failed_destination",
                        "feature_variance",
                        "slot_counts",
                    )
                },
            }
        )
    return epochs, {
        "statuses": statuses,
        "refresh_destination_missing_sum": sum(
            r.get("failed_destination", 0) for r in ledgers
        ),
        "reasons": reasons,
        "applied": applications.total(),
        "globally_unique_edited": len(applications),
        "repeated_occurrences": applications.total() - len(applications),
        "epoch_unique_sum": sum(r["unique_applied"] for r in ledgers),
        "moved_mass": 0.2 * applications.total(),
        "mixed_rows": sum(r["rows"] for r in ledgers),
        "matched_epoch_budget_hits": sum(r["applied"] == 322 for r in ledgers),
        "id_application_count_histogram": Counter(applications.values()),
        "consecutive_refresh_overlap": opportunity,
    }


@torch.inference_mode()
def freeze(model, ds, center, ids, states, work, provenance):
    model.eval().requires_grad_(False)
    for state in states.values():
        assert set(state) == set(model.state_dict())
        for k, value in model[0].state_dict().items():
            if not k.startswith("blocks.11."):
                assert torch.equal(value.cpu(), state[f"0.{k}"]), k
    tails, weights = {}, {}
    for tag, state in states.items():
        tail = copy.deepcopy(model[0].blocks[-1])
        tail.load_state_dict({k: state[f"0.blocks.11.{k}"] for k in tail.state_dict()})
        tails[tag] = tail
        weights[tag] = (
            F.normalize(state["1.last_layer.weight_v"].cuda(), dim=1)
            * state["1.last_layer.weight_g"].cuda()
        )
    panel_record = read(Path("reports/r2_oracle_move_trajectory.json"))
    panel = np.asarray(panel_record["panel_ids"])
    keys_to_rows = {int(key): i for i, key in enumerate(ids)}
    panel_idx = np.asarray([keys_to_rows[k] for k in panel])
    panel_path = e0.g1.WORK / "oracle-move/step0-panel-tokens.pt"
    panel_cache = torch.load(panel_path, map_location="cpu", weights_only=True)
    assert (
        panel.tolist() == panel_cache["panel_ids"]
        and panel_cache["checkpoint_sha256"] == panel_record["checkpoint_sha256"]
    )
    ref_path = e0.dynamic.LOCAL / "reference-prefix.npy"
    info = read(e0.dynamic.LOCAL / "reference-prefix.json")
    assert (
        info["identity"]["prefix_checkpoint_sha256"]
        == panel_record["checkpoint_sha256"]
    )
    assert (
        info["sha256"]
        == read(e0.dynamic.OUT)["provenance"]["reference_prefix_sha256"]
        == digest(ref_path)
    )
    ref_idx = np.asarray([keys_to_rows[k] for k in info["identity"]["ids"]])
    assert (
        len(np.unique(np.r_[panel_idx, ref_idx]))
        == len(panel_idx) + len(ref_idx)
        == len(ids)
    )
    bank = np.load(ref_path, mmap_mode="r")
    assert list(bank.shape) == info["identity"]["shape"] and bank.dtype == np.float32
    features = {a: torch.empty((len(ids), 768), device="cuda") for a in TAGS}

    def final(tail, tokens):
        return F.normalize(model[0].norm(tail(tokens))[:, 0], dim=1)

    def consume(pieces, rows):
        offset = 0
        for part in pieces:
            tokens = part.cuda()
            selected = rows[offset : offset + len(tokens)]
            for tag in TAGS:
                features[tag][selected] = final(tails[tag], tokens)
            offset += len(tokens)
        assert offset == len(rows)

    consume((piece for batch in panel_cache["tokens"] for piece in batch), panel_idx)
    consume(
        (
            torch.from_numpy(bank[0, i : i + 8].copy())
            for i in range(0, len(ref_idx), 8)
        ),
        ref_idx,
    )
    labelled = copy.copy(ds.labelled_dataset)
    labelled.transform = center
    known, legal = [], []
    labeled_x = {a: [] for a in TAGS}
    for images, targets, batch_ids in DataLoader(labelled, batch_size=16):
        assert (targets >= 0).all() and (targets < 98).all()
        tokens = [part.cuda() for part in e0.audit.prefix(model[0], images)]
        for tag in TAGS:
            labeled_x[tag].append(torch.cat([final(tails[tag], t) for t in tokens]))
        known.extend(batch_ids.tolist())
        legal.extend(targets.tolist())
    np.testing.assert_array_equal(known, labelled.uq_idxs)
    assert not set(known) & set(ids)
    rest = np.flatnonzero(~np.isin(ids, panel))
    fit = np.r_[known, ids[rest]]
    fit_legal = np.r_[legal, np.full(len(rest), -1)]
    q, ref, pairs, clip, random_report = e0.confirmed_manifest(ids)
    frozen = {}
    for tag in TAGS:
        x = features[tag]
        logits = (x @ weights[tag].T).cpu().numpy()
        out = e0.policies(ids, q, ref, pairs, x, clip, logits, random_report)
        fit_x = torch.cat([torch.cat(labeled_x[tag]), x[rest], x[panel_idx]])
        out.update(
            feature=x.cpu().numpy(),
            logits=logits,
            fit_feature=fit_x.cpu().numpy(),
            fit_ids=fit,
            fit_legal=fit_legal,
            panel=panel,
        )
        out["classifier"] = (fit_x @ weights[tag].T).argmax(1).cpu().numpy()[None]
        for other in states:
            out[f"swap_{other}"] = (fit_x @ weights[other].T).argmax(1).cpu().numpy()
        for name in ("spherical", "semi"):
            out[name] = np.stack(
                [
                    e0.readout.spherical(
                        fit_x, len(fit), fit_legal, seed, name == "semi"
                    )
                    for seed in e0.readout.SEEDS
                ]
            )
        path = work / f"frozen-{tag}.npz"
        np.savez(path, **out)
        frozen[tag] = digest(path)
        print(f"Frozen geometry, retrieval, readouts and swaps for {tag}", flush=True)
    provenance.update(
        {
            "frozen_sha256": frozen,
            "reference_prefix_sha256": info["sha256"],
            "panel_tokens_sha256": digest(panel_path),
            "query_ids_sha256": e0.readout.digest(ids[q]),
            "bank_ids_sha256": e0.readout.digest(ids[ref]),
            "paired_queries": len(q),
            "common_pair_pool_sha256": e0.readout.digest(pairs),
            "frozen_prefix_verified": True,
            "readout_panel": len(panel),
            "readout_seeds": e0.readout.SEEDS,
            "readout_fit": len(fit),
            "query_policy_sha256": digest(e0.heldout.OUT),
        }
    )
    write_json(work / "frozen-provenance.json", json_native(provenance))


def actual_semantics(ledgers, truth, mappings):
    """Score observed edits under snapshot alignments with explicit sensitivity."""
    cohort = {}
    repeated = {}
    out = {}
    windows = ((11, 100), (101, 150), (151, 200))
    for lo, hi in windows:
        item = {}
        for tag, mapping in mappings.items():
            counts = {g: Counter() for g in GROUPS}
            for r in ledgers[lo - 1 : hi]:
                for _, _, key, src, dst in r.get("ledger", []):
                    y = truth[key]
                    s, d = mapping.get(src, -1), mapping.get(dst, -1)
                    if s < 0 or d < 0:
                        outcome = "unmatched"
                    elif s == y:
                        outcome = "harmful"
                    elif d == y:
                        outcome = "repair"
                    else:
                        outcome = "wrong_to_wrong"
                    for g in ("all", "old" if y < 98 else "new"):
                        counts[g][outcome] += 1
            item[tag] = counts
        out[f"{lo}-{hi}"] = item
    for r in ledgers:
        for _, _, key, src, dst in r.get("ledger", []):
            if key in cohort:
                ep, s, d = cohort[key]
                repeated["n"] = repeated.get("n", 0) + 1
                for field, val in (
                    ("same_source", src == s),
                    ("same_destination", dst == d),
                    ("later_epoch", ep != r["epoch"]),
                ):
                    repeated[field] = repeated.get(field, 0) + int(val)
            cohort[key] = (r["epoch"], src, dst)
    return {
        "fixed_snapshot_mapping_sensitivity": out,
        "consecutive_observed_edits_for_same_id": repeated,
    }


def source_cohorts(ledgers, ids, truth, arrays, mapping):
    first = {}
    for row in ledgers:
        for _, _, key, _, _ in row.get("ledger", []):
            first.setdefault(key, row["epoch"])
    exposed = np.array([first.get(i, 201) for i in ids])
    y = truth[ids]
    label_map = np.array([mapping.get(i, -1) for i in range(196)])
    ok = {a: label_map[p["slots"]] == y for a, p in arrays.items()}
    start = ok[TAGS[0]]
    z = e0.match_prob(arrays[TAGS[0]]["logits"], 0.04).max(1)
    bins = np.searchsorted(np.quantile(z, np.arange(1, 10) / 10), z, side="right")
    estimates = {}
    for t, upper in zip(TAGS[1:], MILESTONES[1:], strict=True):
        change = ok[t].astype(int) - start.astype(int)
        edited, never = (exposed > 100) & (exposed <= upper), exposed == 201
        endpoint = {}
        for g, group in zip(GROUPS, (y >= 0, y < 98, y >= 98), strict=True):
            a, b = edited & group, never & group
            matched = []
            for cls in np.unique(y[a]):
                for was in (False, True):
                    for decile in range(10):
                        stratum = (y == cls) & (start == was) & (bins == decile)
                        ad, bd = change[a & stratum], change[b & stratum]
                        if len(ad) and len(bd):
                            matched.append(
                                (len(ad), len(bd), float(ad.mean() - bd.mean()))
                            )
            n = sum(z[0] for z in matched)
            endpoint[g] = {
                "edited": int(a.sum()),
                "never": int(b.sum()),
                "edited_change": float(change[a].mean()) if a.any() else None,
                "never_change": float(change[b].mean()) if b.any() else None,
                "matched_edited": n,
                "matched_never": sum(z[1] for z in matched),
                "matched_strata": len(matched),
                "edited_weighted_adjusted_difference": sum(
                    n1 * d for n1, _, d in matched
                )
                / n
                if n
                else None,
            }
        estimates[t] = endpoint
    return estimates


def seal(work, frozen, series, ledgers, comparisons):
    from scripts.audit_r2_evidence import sealed_labels

    verified = {}
    for tag, value in frozen.items():
        path = work / f"frozen-{tag}.npz"
        assert digest(path) == value
        with np.load(path, allow_pickle=False) as file:
            verified[tag] = dict(file)
    owners = {tag: p["classifier"][0] for tag, p in verified.items()}
    prior = read(e0.OUT)["provenance"]["frozen_sha256"]
    for tag in e0.TAGS:
        path = e0.LOCAL / f"frozen-{tag}.npz"
        assert digest(path) == prior[tag]
        with np.load(path, allow_pickle=False) as file:
            for key in ("fit_ids", "panel"):
                np.testing.assert_array_equal(file[key], verified[TAGS[0]][key])
            owners[tag] = file["classifier"][0]
    truth = sealed_labels()
    results, maps = {}, {}
    for tag, p in verified.items():
        ids, slots = p["ids"], p["slots"]
        y = truth[ids]
        maps[tag], align_meta = align(
            ids.tolist(), dict(zip(ids.tolist(), slots.tolist(), strict=True)), truth
        )
        base = e0.split_cluster_acc_v2(y, slots, y < 98)
        logged = series["D1"][int(tag.split("-")[1]) - 1]["accuracy"]
        panel, _ = e0.audit.assess_features(
            torch.from_numpy(p["feature"]).cuda(), slots, y, maps[tag]
        )
        detection, bank = e0.g4.sealed(p, truth, e0.POLICIES, baseline="DINO")
        readouts = e0.sealed_fit(p, truth)
        swaps = {
            head: e0.readout.evaluate(
                p[f"swap_{head}"], truth[p["fit_ids"]], truth[p["panel"]]
            )
            for head in (*TAGS, *e0.TAGS)
        }
        n_fit = len(p["fit_ids"])
        fit_y, panel_y = truth[p["fit_ids"]], truth[p["panel"]]
        owner_fixed = {}
        for h, own in owners.items():
            pred = p[f"swap_{h}"]
            mapping = e0.readout.mapping(own[:n_fit], fit_y)
            group = fixed_accuracy(pred[n_fit:], panel_y, dict(enumerate(mapping)))
            owner_fixed[h] = {
                "accuracy": group,
                "fit_slot_agreement": float(np.mean(pred[:n_fit] == own[:n_fit])),
            }
        results[tag] = {
            "v2": list(base),
            "v2_recorded": logged,
            "replay_minus_logged_pp": (100 * (np.asarray(base) - logged)).tolist(),
            "geometry": panel["geometry"],
            "official_alignment": align_meta,
            "prediction_bias": e0.bias(p, y, maps[tag]),
            "readouts": readouts,
            "detection": detection,
            "bank_alignment": bank,
            "correction": e0.heldout.corrections(
                p, truth, policies=e0.POLICIES, baseline="Confidence"
            ),
            "head_swaps": swaps,
            "head_swaps_owner_fixed": owner_fixed,
            "swap_differences_vs_own_head": {
                h: e0.readout.diff(s, swaps[tag]) for h, s in swaps.items()
            },
        }
        print(f"Sealed {tag}: replay {base}; logged {logged}", flush=True)
    p = verified[TAGS[-1]]
    ids = p["ids"]
    slot_identity = {}
    for a, b in itertools.pairwise(TAGS):
        _, slot_identity[f"{a}_to_{b}"] = e0.g4.slot_identity(
            verified[a]["slots"], verified[b]["slots"], p["bank"], p["q"]
        )
    for name, value in comparisons.items():
        # Only previously verified frozen metrics. Cross-run numbered heads lack a shared learned basis.
        results[name] = {
            k: value[k]
            for k in (
                "v2",
                "geometry",
                "prediction_bias",
                "readouts",
                "detection",
                "correction",
            )
        }
    return results, {
        "slot_identity": slot_identity,
        "actual_move": actual_semantics(ledgers, truth, maps),
        "source_cohorts": source_cohorts(ledgers, ids, truth, verified, maps[TAGS[0]]),
    }


def report(result, path):
    p, history = result["provenance"], result["exposure"]
    series = result["trajectory"]
    summary = result["training_summary"]
    peak = summary["D1"]["peak"]
    last = result["frozen"]["D1-200"]
    base = result["frozen"]["E0-200"]
    replay = last["correction"]["10"]["All"]["CLIP"]
    association = result["interventions"]["source_cohorts"]["D1-200"]["all"]
    warmup = result["same_epoch_d1_minus_other_pp"]["E0"][:10]
    warmup_gap = max(abs(x) for row in warmup for x in row["difference_pp"])

    def acc(run, end, window=1):
        values = [
            row["accuracy"] for row in series[run] if end - window < row["epoch"] <= end
        ]
        return e0.accuracy(np.mean(values, axis=0)) if len(values) == window else "—"

    def values(row):
        return " / ".join(f"{row[g]:.4f}" for g in GROUPS)

    def signed_pp(value):
        return f"{value:+.2f}" if value is not None else "—"

    def percent(x):
        return f"{100 * x:+.2f}" if x is not None else "—"

    lines = [
        "# CRiSH-D1 final audit on Stanford Cars",
        "",
        f"D1 completed 200 epochs at `{COMMIT}`. Its archived final v2 All / Old / New was **{e0.accuracy(summary['D1']['last'])}%**. The same-epoch D1 minus E0 difference was {values(dict(zip(GROUPS, result['same_epoch_d1_minus_other_pp']['E0'][-1]['difference_pp'])))} percentage points. Runs are single-seed trajectories and these comparisons do not isolate the effect of CLIP evidence.",
        "",
        f"The All peak was {100 * peak['accuracy'][0]:.2f}% at epoch {peak['epoch']}. The final New advantage over E0 coincides with higher New kNN-5 purity ({last['geometry']['new']['purity5']:.3f} vs {base['geometry']['new']['purity5']:.3f}) and a positive New centroid margin ({last['geometry']['new']['centroid_margin']:+.3f} vs {base['geometry']['new']['centroid_margin']:+.3f}).",
        "",
        f"The locked D1-200 CLIP offline policy made {replay['correct_repairs']} repairs and {replay['harmful_edits']} harmful edits at 10% coverage ({replay['vs_keep']['pp']:+.2f} pp vs KEEP). The difficulty-adjusted edited-versus-never association was {100 * association['edited_weighted_adjusted_difference']:+.2f} pp at epoch 200 in the common-support cohort. These findings leave the contributions of CLIP destination choice and exposure unresolved.",
        "",
        "## Provenance and training continuity",
        "",
        "Archive SHA-256 receipts were checked before extraction. All analysis checkpoints came from the externally verified epoch-200 TAR. Earlier milestone checkpoint digests and byte-exact journal prefixes were corroborated against that archive. The checkpoint identity, recorded byte boundaries, RNG payloads, populated finite SGD momentum buffers, per-epoch LRs and milestone scheduler position agree. This checks retained state and epoch-level continuity. Individual historical optimizer updates, draws and augmentation RNG cannot be replay-verified from these artifacts.",
        "",
        f"D1 and E0 report {p['environment']['gpu']} and D1 records torch {p['identity']['protocol']['torch']}. Frozen replay used {p['local_runtime']['gpu']} and torch {p['local_runtime']['torch']}. The D1/E0/R2 split reconstruction uses the locked Cars SSB seed-1 IDs, verified DINO prefix, archive parameters and compatible v2 protocol. E0 has no external TAR sidecar. Its locally computed SHA-256 and all archived members matched the prior audit. E0 does not contain every historical optimizer/scheduler state, and none of these runs have archived remote image-byte hashes for the full split.",
        "",
        "| Completed epochs | Computed TAR SHA-256 | Sidecar | Mismatched members | Momentum entries | Saved next-epoch LR |",
        "|---:|---|---|---|---:|---:|",
    ]
    for epoch, r in p["archives"].items():
        lines.append(
            f"| {epoch} | `{r['archive_sha256']}` | {'match' if r['sidecar_matches'] else 'MISMATCH'} | {', '.join(r['member_digest_mismatches']) or 'none'} | {r['optimizer_entries']} | {r['scheduler']['_last_lr'][0]:.8f} |"
        )
    for epoch, row in p["archives"].items():
        if not row["sidecar_matches"]:
            lines += [
                "",
                f"Epoch {epoch} declared SHA-256 `{row['expected_archive_sha256']}`. The archive and sidecar are intact on disk. The cause of the sidecar discrepancy is unknown.",
            ]
    lines += [
        "",
        "## Accuracy trajectories",
        "",
        f"All numbers are percentages, ordered All / Old / New. Last-20 windows end at the marked epoch. In the first ten warmup epochs the largest absolute D1–E0 difference is {warmup_gap:.4f} pp, within E0 stdout rounding. Raw curves and same-epoch differences are in the JSON.",
        "",
        "| Epoch | Run | Endpoint | Mean of last 20 |",
        "|---:|---|---|---|",
    ]
    for end in MILESTONES:
        for run in ("D1", "E0", "E3", "R2"):
            lines.append(f"| {end} | {run} | {acc(run, end)} | {acc(run, end, 20)} |")
    lines += [
        "",
        "## Realized exposure",
        "",
        f"Refresh outcomes {dict(history['statuses'])}. There were {history['applied']:,} observed applications across {history['globally_unique_edited']:,} unique IDs ({history['repeated_occurrences']:,} additional occurrences), {history['moved_mass']:,.1f} total moved mass and {history['mixed_rows']:,} mixed rows. The 322-occurrence cap was reached in {history['matched_epoch_budget_hits']} epochs. Ledger totals and exclusions reconcile to training encounters.",
        "",
        f"Eligibility and budget counters {dict(history['reasons'])}. Missing alternative-slot detections summed to {history['refresh_destination_missing_sum']}. The JSON contains refresh overlap by ID, native source and destination agreement, slot counts, feature variance, epoch timings and first-edited-batch gradient receipts.",
        "",
        "| Window | Applications | Exposure (%) | Median refresh (s) | Median first-edit classifier / block-11 ratio |",
        "|---|---:|---:|---:|---|",
    ]
    for lo, hi in ((11, 100), (101, 150), (151, 200)):
        rows = series["D1"][lo - 1 : hi]
        edited = sum(r["applied"] for r in rows)
        refresh = np.median([r["seconds"] for r in rows])
        ratio = [
            np.median(
                [
                    r["first_edit_gradients"][g]["ratio"]
                    for r in rows
                    if r["first_edit_gradients"]
                    and r["first_edit_gradients"][g]["ratio"] is not None
                ]
            )
            for g in ("classifier", "block11")
        ]
        lines.append(
            f"| {lo}–{hi} | {edited:,} | {100 * edited / sum(r['rows'] for r in rows):.3f} | {refresh:.2f} | {ratio[0]:.3f} / {ratio[1]:.3f} |"
        )
    lines += [
        "",
        "## Frozen checkpoint geometry and classifier bias",
        "",
        "Geometry uses unit-normalized final CLS, kNN-5 and leave-one-out true-centroid minus hardest-other cosine margin. Bias is class-group misrouting under a full split alignment. Archived E0/R2 rows reuse verified frozen results. The D1 and local DINO source-file digests differ. The replay directly verified identical frozen tensors and the same locked weights, and reports any metric differences below.",
        "",
        "| Snapshot | Replayed v2 | kNN-5 All / Old / New | Margin All / Old / New | Active / effective slots | Old→New / New→Old (%) |",
        "|---|---|---|---|---|---|",
    ]
    for a, r in result["frozen"].items():
        b = r["prediction_bias"]
        knn = values({g: r["geometry"][g]["purity5"] for g in GROUPS})
        margin = values({g: r["geometry"][g]["centroid_margin"] for g in GROUPS})
        lines.append(
            f"| {a} | {e0.accuracy(r['v2'])} | {knn} | {margin} | {b['active_slots']} / {b['entropy_effective_slots']:.1f} | {b['groups']['old']['mapped_new_rate'] * 100:.2f} / {b['groups']['new']['mapped_old_rate'] * 100:.2f} |"
        )
    lines += [
        "",
        "Frozen v2 discrepancies from the archived D1 metrics (pp): "
        + "; ".join(
            f"{tag} {result['frozen'][tag]['replay_minus_logged_pp']}" for tag in TAGS
        )
        + ".",
        "",
        "## Frozen readouts and swaps",
        "",
        "The 588-ID reference panel and fit split are unchanged. Legal semi-supervised K-means uses only legally labeled anchors. Oracle nearest centroid and linear readouts are privileged diagnostics. Fit-aligned and panel-rematched subgroup scores, ARI and NMI are in JSON. The panel is transductive for D1 and does not constitute independent test data.",
        "",
        "| Snapshot | Classifier | Spherical | Legal semi-supervised | Oracle centroid | Oracle linear |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a, r in result["frozen"].items():
        choices = [
            100 * r["readouts"][x]["estimate"]["All"]["fit"]
            for x in ("classifier", "spherical", "semi", "centroid", "linear")
        ]
        lines.append(f"| {a} | " + " | ".join(f"{x:.2f}" for x in choices) + " |")
    lines += [
        "",
        "Swap rows keep the feature snapshot and change only the normalized classifier head. Accuracy is fit aligned. JSON also reports owner-head fixed alignment and slot agreement on the readout fit set. Cross-run swaps are mechanics diagnostics with jointly trained embeddings and classifier weights, so compatibility of dimensions and preprocessing does not guarantee semantic interoperability.",
        "",
        "| Feature snapshot | Own head All / Old / New | D1-100 head | D1-150 head | D1-200 head | E0-200 head | R2-100 head | R2-150 head |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        swaps = result["frozen"][tag]["head_swaps"]

        def fmt(h, swaps=swaps):
            return e0.accuracy([swaps[h][g.capitalize()]["fit"] for g in GROUPS])

        lines.append(
            f"| {tag} | {fmt(tag)} | "
            + " | ".join(fmt(h) for h in (*TAGS, *e0.TAGS))
            + " |"
        )
    lines += [
        "",
        "## Locked CLIP correction",
        "",
        "G6-confirm's fixed, disjoint 1,024 queries and 5,120 retrieval-bank IDs, the common R2-150 hard pairs, bank-only Hungarian mapping, thresholds and matched random draws were reused. These are offline assignment edits. The checkpoints have previously trained transductively on these images.",
        "",
        "| Snapshot / detector | Error AUROC All / Old / New | Destination recall@1/3/5 All | Novel recall@1/3/5 |",
        "|---|---|---|---|",
    ]
    for tag, row in result["frozen"].items():
        for name in e0.POLICIES:
            r = row["detection"][name]
            au = [r[g.capitalize()]["detection_auc"] for g in GROUPS]
            all_r = r["All"]["destination_recall"]
            new_r = r["New"]["destination_recall"]
            lines.append(
                f"| {tag} / {name} | "
                + " / ".join(f"{x:.3f}" for x in au)
                + " | "
                + " / ".join(f"{all_r[str(k)]:.3f}" for k in (1, 3, 5))
                + " | "
                + " / ".join(f"{new_r[str(k)]:.3f}" for k in (1, 3, 5))
                + " |"
            )
    lines += [
        "",
        "| Snapshot / 10% policy | Repairs / Harmful / Wrong-to-wrong | Δ vs KEEP All / Old / New (pp) | Δ vs confidence (All pp) | Δ vs matched random (All pp) |",
        "|---|---|---|---|---|",
    ]
    for tag, r in result["frozen"].items():
        x = r["correction"].get(10, r["correction"].get("10"))
        for name in ("DINO", "CLIP", "Confidence"):
            item = x["All"][name]
            lines.append(
                f"| {tag} / {name} | {item['correct_repairs']} / {item['harmful_edits']} / {item['wrong_to_wrong']} | "
                + " / ".join(
                    signed_pp(x[g.capitalize()][name]["vs_keep"]["pp"]) for g in GROUPS
                )
                + f" | {signed_pp(item['vs_confidence']['pp'])} | {signed_pp(item['vs_matched_random']['pp'])} |"
            )
    lines += [
        "",
        "## Historical applied edits and cohort changes",
        "",
        "The reported edit counts use actual ledger occurrences. Repair/harm describes where the mass went under a retrospective fixed snapshot alignment. A full contemporaneous assignment map and live teacher probabilities were not retained, so the exact per-epoch semantic correctness and soft-target effectiveness cannot be verified. JSON includes sensitivity to all three mappings and Old/New breakdowns. The window-end retained snapshot mapping is shown per window.",
        "",
        "| Applied window | Alignment snapshot | Repair / harmful / wrong-to-wrong / unmatched | Repair / harm (% of all applied) |",
        "|---|---|---|---|",
    ]
    for window, tag in zip(("11-100", "101-150", "151-200"), TAGS, strict=True):
        c = result["interventions"]["actual_move"][
            "fixed_snapshot_mapping_sensitivity"
        ][window][tag]["all"]
        total = sum(c.values())
        lines.append(
            f"| {window} | {tag} | "
            + " / ".join(
                str(c.get(k, 0))
                for k in ("repair", "harmful", "wrong_to_wrong", "unmatched")
            )
            + f" | {100 * c.get('repair', 0) / total:.1f} / {100 * c.get('harmful', 0) / total:.1f} |"
        )
    repeats = result["interventions"]["actual_move"][
        "consecutive_observed_edits_for_same_id"
    ]
    lines += [
        "",
        f"Among consecutive observed occurrences of the same ID, native slot repeat counts were {repeats}. These are observed edit pairs. Missing non-edited trajectories prevent identifying a continuously stable teacher assignment or mapping intervening proposal drift.",
        "",
        "Cohorts use D1-100 mapping for both baseline and endpoint. The exposed cohort has its first edit after epoch 100 and by the endpoint. Comparators were never edited in the retained 200 epochs. The adjusted difference post-stratifies by true class, epoch-100 correctness and epoch-100 max-teacher confidence decile, weighted to edited IDs with shared support. Selection bias, future-defined controls, unrecorded trajectories and transductive exposure preclude a causal propagation interpretation.",
        "",
        "| Endpoint / group | Edited / never | Edited / never change (pp) | Matched edited / never | Adjusted difference (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, rows in result["interventions"]["source_cohorts"].items():
        for g, row in rows.items():
            lines.append(
                f"| {tag} / {g} | {row['edited']} / {row['never']} | {percent(row['edited_change'])} / {percent(row['never_change'])} | {row['matched_edited']} / {row['matched_never']} | {percent(row['edited_weighted_adjusted_difference'])} |"
            )
    lines += [
        "",
        "## Evidence limits",
        "",
        "- E3 has 200 logged epochs and an exposure ledger. Its checkpoints are unavailable, so no E3 frozen geometry, readout, swap or locked correction was measured.",
        "- E0 retains only epoch 200, R2 retains epochs 100 and 150, and E3-F retains manually recovered aggregate metrics. Missing matrix cells remain unmeasured.",
        "- The ablations need shared realized exposure before attributing differences to destination choice or evidence source. See [ablation design](CRISH_D1_ABLATION_DESIGN.md).",
        "",
        f"Report replay runtime {result['seconds']:.1f} s. [Complete machine-readable audit](CRISH_D1_FINAL_AUDIT.json).",
    ]
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=e0.LOG)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument(
        "--reuse-frozen",
        action="store_true",
        help="Verify and score the existing frozen predictions",
    )
    args = parser.parse_args()
    started = time.monotonic()
    args.work.mkdir(parents=True, exist_ok=True)
    e0.LOG = args.logs
    provenance, metrics, ledgers, states = archive(args.logs / "D1", args.work)
    print(
        "D1 archive receipts checked, checkpoint and journal provenance verified",
        flush=True,
    )
    previous = read(e0.OUT)
    files, e0_receipt = e0.archive_receipt()
    archived = previous["provenance"]
    assert e0_receipt["members_sha256"] == archived["members_sha256"]
    assert e0_receipt["archive_sha256"] == archived["archive_sha256"]
    cfg, series = e0.trajectory(files)
    assert all(series[k] == previous["trajectory"][k] for k in series)
    model, ds, center, ids, reference_states = e0.context(files, cfg, e0_receipt)
    protocol = provenance["identity"]["protocol"]
    local = e0_receipt["local_protocol"]
    assert all(protocol[k] == v for k, v in local.items() if k != "dino_source")
    provenance["dino_source_file_sha256_d1_vs_local"] = [
        protocol["dino_source"],
        local["dino_source"],
    ]
    _, receipt = cache.load(
        e0.g6.LOCAL / "clip-image.npz",
        ds.unlabelled_dataset,
        local["split"],
        torch.device("cuda"),
    )
    assert receipt == provenance["identity"]["clip"]
    provenance["e0_archive"] = e0_receipt
    provenance["local_runtime"] = e0_receipt["runtime"]
    provenance["e0_audit_json_sha256"] = digest(e0.OUT)
    series["D1"], exposure = history(metrics, ledgers, ids)
    print(
        f"Reconciled {len(metrics)} D1 epochs and {exposure['applied']} applied occurrences",
        flush=True,
    )
    if args.reuse_frozen:
        frozen_receipt = read(args.work / "frozen-provenance.json")
        for field in ("archives", "identity", "e0_audit_json_sha256"):
            if json_native(provenance[field]) != frozen_receipt[field]:
                raise ValueError(f"Frozen D1 provenance changed: {field}")
        if frozen_receipt["query_policy_sha256"] != digest(e0.heldout.OUT):
            raise ValueError("Frozen D1 query protocol changed")
        provenance.update(frozen_receipt)
        print(
            "Reusing frozen D1 predictions and readouts; no feature extraction",
            flush=True,
        )
    else:
        freeze(
            model,
            ds,
            center,
            ids,
            {**states, **reference_states},
            args.work,
            provenance,
        )
    frozen, interventions = seal(
        args.work, provenance["frozen_sha256"], series, ledgers, previous["frozen"]
    )
    result = {
        "provenance": provenance,
        "trajectory": series,
        "training_summary": e0.summary_rows(series),
        "exposure": exposure,
        "interventions": interventions,
        "frozen": frozen,
        "seconds": time.monotonic() - started,
        "same_epoch_d1_minus_other_pp": {
            ref: [
                {
                    "epoch": a["epoch"],
                    "difference_pp": (
                        100 * (np.asarray(a["accuracy"]) - b["accuracy"])
                    ).tolist(),
                }
                for a, b in zip(series["D1"], series[ref])
                if a["epoch"] == b["epoch"]
            ]
            for ref in ("E0", "E3", "R2")
        },
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    json_path = args.reports / "CRISH_D1_FINAL_AUDIT.json"
    markdown = args.reports / "CRISH_D1_FINAL_AUDIT.md"
    result = json_native(result)
    write_json(json_path, result)
    report(read(json_path), markdown)
    print(f"Published local reports {markdown} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
