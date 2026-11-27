"""Two-epoch paired CLIP MOVE strength diagnostic starting at R2-150."""

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts import audit_clip_repair as repair
from scripts import audit_g4 as g4
from scripts import audit_oracle_move as audit
from scripts.audit_e3_proposals import json_native

LOCAL = repair.g1.WORK / "move-strength-training"
MD = Path("reports/CRISH_MOVE_STRENGTH_TRAINING.md")
OUT = Path("reports/r2_move_strength_training.json")
ARMS = ("S0", "S1", "S2", "S3")
MASS = dict(zip(ARMS, (0.0, 0.2, 0.8, 0.8), strict=True))
EPOCHS = (151, 152)
GROUPS = ("all", "old", "new")


def setup():
    model, loader, _, center, _, _ = audit.build()
    ds = loader.dataset
    blind = ds.unlabelled_dataset.ids
    ids = np.r_[ds.labelled_dataset.uq_idxs, blind]
    old = json.loads(repair.OUT.read_text())
    policy = repair.LOCAL / "frozen-policy.json"
    sha = audit.file_digest(policy)
    assert sha == old["protocol"]["policy_sha256"]
    receipt = json.loads(policy.read_text())
    original = receipt["signature"]
    panel = np.asarray(original["panel_ids"])
    assert len(np.unique(panel)) == 588 and not set(panel) & set(
        original["reference_ids"]
    )
    assert set(panel) <= set(blind)
    cache = torch.load(
        repair.g1.WORK / "oracle-move/step0-panel-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert cache["panel_ids"] == panel.tolist()
    assert (
        cache["checkpoint_sha256"]
        == json.loads(repair.HISTORY.read_text())["checkpoint_sha256"]
    )
    path = repair.g1.WORK / "input/epoch-150.pt"
    assert audit.file_digest(path) == original["checkpoint_sha256"]
    base = audit.load_checkpoint(path)
    assert base["epoch"] == base["scheduler"]["last_epoch"] == 150
    assert base["scheduler"]["T_max"] == 200
    for k, v in model[0].state_dict().items():
        if not k.startswith("blocks.11."):
            assert torch.equal(v.cpu(), base["model"][f"0.{k}"]), k
    plan = {
        e: p
        for e, p in repair.schedule(loader, ids, panel, base).items()
        if e in EPOCHS
    }
    draws = {
        str(e): {"rows": repair.digest_array(r), "seeds": repair.digest_array(s)}
        for e, (r, s) in plan.items()
    }
    assert all(draws[e] == original["draws"][e] for e in draws)
    pool = {k: (src, dst, i) for i, (k, src, dst, _) in enumerate(receipt["proposals"])}
    assert len(pool) == original["pool_n"] and not set(panel) & set(pool)
    direct = {k: p for k, p in pool.items() if p[-1] < original["locked_n"]}
    assert len(direct) == original["locked_n"]
    info = json.loads(repair.g6.G1_REPORT.read_text())
    classifier = g4.snapshot(150, info)
    np.testing.assert_array_equal(classifier["ids"], blind)
    sig = {
        "base": "e537a16",
        "checkpoint_sha256": original["checkpoint_sha256"],
        "policy_sha256": sha,
        "panel_ids": panel.tolist(),
        "reference_ids": original["reference_ids"],
        "proposal_count": len(direct),
        "random_pool_count": len(pool),
        "mass": MASS,
        "batch_size": 128,
        "draws": draws,
        "classifier_sha256": info["snapshots"]["150"]["frozen_sha256"],
    }
    return (
        model,
        ds,
        center,
        ids,
        base,
        plan,
        panel,
        cache["tokens"],
        pool,
        direct,
        classifier,
        sig,
    )


def select_rows(first, second, random_pool, seed):
    eligible = set(second)
    paired = [v for v in first if v in eligible]
    count = min(len(paired), len(random_pool))
    rng = np.random.default_rng(int(seed))
    matched = [random_pool[i] for i in rng.permutation(len(random_pool))[:count]]
    return (
        paired[:count],
        matched,
        {
            "S1_eligible": len(first),
            "S2_eligible": len(second),
            "S3_eligible": len(random_pool),
            "common_eligible": len(paired),
            "eligible_only_S1": len(first) - len(paired),
            "eligible_only_S2": len(second) - len(paired),
            "random_supply_shortfall": len(paired) - count,
            "against_S1_eligible_shortfall": len(first) - count,
            "applied": count,
        },
    )


def selector(moves, pool, strength):
    def apply(prob, ids, legal):
        if moves and not set(moves).issubset(
            repair.options(prob, ids, legal, pool, strength=strength)[0]
        ):
            raise ValueError("Live eligibility changed between preview and loss")
        rows = [list(a[:4]) for a in moves]
        action = [audit.Action(i, "move", src, dst) for i, _, src, dst in rows]
        # Fixed two-view teacher and student distributions of each actually edited occurrence.
        if rows:
            i, _, src, dst = np.asarray(rows, dtype=int).T
            student = (prob.log() * 0.4).softmax(-1)
            record = {
                "teacher_source": prob[:, i, src].T.tolist(),
                "teacher_destination": prob[:, i, dst].T.tolist(),
                "student_source": student[:, i, src].T.tolist(),
                "student_destination": student[:, i, dst].T.tolist(),
            }
        else:
            record = {}
        return action, rows, record

    return apply


def run_batch(models, batch, pool, direct, seeds, epoch, criterion):
    options, counters = {}, {}
    for name in ARMS[1:]:
        repair.seed_all(seeds[1])
        options[name], counters[name] = repair.live_pool(
            models[name][0],
            batch,
            pool if name == "S3" else direct,
            strength=MASS[name],
        )
    moves, random, stats = select_rows(
        options["S1"], options["S2"], options["S3"], seeds[2]
    )
    details = {"matching": stats}
    for name, chosen in zip(ARMS, ([], moves, moves, random), strict=True):
        repair.seed_all(seeds[1])
        net, opt, _ = models[name]
        records, info = audit.training_step(
            net,
            opt,
            batch,
            criterion,
            selector(chosen, pool if name == "S3" else direct, MASS[name]),
            epoch=epoch - 1,
            mass=MASS[name],
        )
        details[name] = {
            "moves": records,
            "eligibility": dict(counters.get(name, {})),
            **info,
        }
    assert details["S1"]["moves"] == details["S2"]["moves"]
    assert (
        len(details["S1"]["moves"]) == len(details["S3"]["moves"]) == stats["applied"]
    )
    return details


def train(model, ds, ids, base, plan, pool, direct, sig, smoke):
    done = (
        [e for e in EPOCHS if (LOCAL / f"epoch-{e}.pt").is_file()] if not smoke else []
    )
    prev = max(done, default=150)
    if prev == EPOCHS[-1]:
        if len(done) != len(EPOCHS):
            raise ValueError(
                "Missing earlier paired checkpoint needed for sealed analysis"
            )
        return
    if prev == 150:
        models = repair.make_arms(model, dict.fromkeys(ARMS, base), names=ARMS)
        results = {}
    else:
        saved = audit.load_checkpoint(LOCAL / f"epoch-{prev}.pt")
        assert saved["epoch"] == prev and saved["signature"] == sig
        models = repair.make_arms(model, saved["state"], names=ARMS)
        results = saved["records"]
        del saved
    criterion = audit.DistillLoss(30, 200)
    for e, (rows, seeds) in plan.items():
        if e <= prev:
            continue
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        ledger = []
        for j, (rr, ss) in enumerate(zip(rows, seeds, strict=True), 1):
            batch = repair.load_batch(model, ds, rr, ss[0])
            np.testing.assert_array_equal(batch["ids"].numpy(), ids[rr])
            detail = run_batch(models, batch, pool, direct, ss, e, criterion)
            ledger.append(detail)
            if smoke:
                if detail["matching"]["applied"]:
                    assert all(detail[a]["cls_edit_norm"] > 0 for a in ARMS[1:])
                    print(
                        "Exact matched strengths on real rows",
                        detail["matching"],
                        "peak bytes",
                        torch.cuda.max_memory_allocated(),
                        flush=True,
                    )
                    return
                if j == 5:
                    raise ValueError(
                        "No common actual eligible MOVE in the first five batches"
                    )
            if j % 10 == 0:
                print(
                    "Epoch",
                    e,
                    "steps",
                    j,
                    "matched edits",
                    sum(x["matching"]["applied"] for x in ledger),
                    flush=True,
                )
        results[str(e)] = {"rows": int(rows.size)}
        matching = Counter()
        for row in ledger:
            matching.update(row["matching"])
        results[str(e)]["matching"] = dict(matching)
        for name, (_, _, sched) in models.items():
            sched.step()
            assert sched.last_epoch == e
            steps = [s[name] for s in ledger]
            count = sum(len(s["moves"]) for s in steps)
            elig = Counter()
            for s in steps:
                elig.update(s["eligibility"])
            results[str(e)][name] = {
                "edits": count,
                "exposure": count / rows.size,
                "mass_total": count * MASS[name],
                "weighted_batch_normalized_mass": count * MASS[name] * 0.65 / 128,
                "lr": sched.get_last_lr()[0],
                "eligibility": dict(elig),
                "unique_source_ids": len({m[1] for s in steps for m in s["moves"]}),
                "mean_loss": float(np.mean([s["loss"] for s in steps])),
                "peak_vram_bytes": max(s["peak_vram"] for s in steps),
            }
        repair.save(models, LOCAL / f"epoch-{e}.pt", sig, e, results, ledger)
        audit.write_json(LOCAL / "training-progress.json", json_native(results))
        print(
            "Saved paired checkpoint",
            e,
            "seconds",
            round(time.monotonic() - start, 1),
            flush=True,
        )


def load_finished(sig, plan, ids, pool, direct):
    out, hashes, events = {}, {}, {}
    for e, (rows, _) in plan.items():
        path = LOCAL / f"epoch-{e}.pt"
        hashes[str(e)] = audit.file_digest(path)
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert saved["signature"] == sig and saved["epoch"] == e
        steps = saved["batch_ledger"]
        assert len(steps) == len(rows)
        for j, (s, rr) in enumerate(zip(steps, rows, strict=True), 1):
            assert s["S1"]["moves"] == s["S2"]["moves"] and not s["S0"]["moves"]
            assert (
                len(s["S1"]["moves"])
                == len(s["S3"]["moves"])
                == s["matching"]["applied"]
            )
            for name in ARMS[1:]:
                edits = s[name]["moves"]
                if not edits:
                    continue
                step_prob = s[name]
                for k, (row, key, src, dst) in enumerate(edits):
                    assert int(ids[rr[row]]) == key and pool[key][:2] == (src, dst)
                    assert name == "S3" or key in direct
                    assert all(p >= MASS[name] for p in step_prob["teacher_source"][k])
                    events.setdefault((str(e), name), []).append(
                        (
                            j,
                            row,
                            key,
                            src,
                            dst,
                            step_prob["teacher_source"][k],
                            step_prob["teacher_destination"][k],
                            step_prob["student_source"][k],
                            step_prob["student_destination"][k],
                        )
                    )
        for name in ARMS:
            assert (
                len(events.get((str(e), name), []))
                == saved["records"][str(e)][name]["edits"]
            )
        out[str(e)] = saved
    return out, hashes, events


@torch.no_grad()
def predictions(model, ds, center, base, saved, panel, tokens, events, sig):
    panel_set = set(panel)
    pairs = {r[2]: (r[3], r[4]) for moves in events.values() for r in moves}
    sources = np.asarray(sorted(pairs), dtype=np.int64)
    assert not panel_set.intersection(sources)
    blind = copy.copy(ds.unlabelled_dataset)
    blind.transform = center
    index = {int(k): i for i, k in enumerate(blind.ids)}
    model.load_state_dict(base["model"])
    model.eval()
    source_tok = audit.center_cache(model, blind, [index[k] for k in sources])
    values = {
        "ids": np.r_[panel, sources],
        "panel_n": len(panel),
        "pairs": np.asarray([pairs[k] for k in sources], dtype=np.int64).reshape(-1, 2),
    }
    tags = {"150:shared": base["model"]}
    tags.update(
        {f"{e}:{a}": cp["state"][a]["model"] for e, cp in saved.items() for a in ARMS}
    )
    for tag, state in tags.items():
        for k in model[0].state_dict():
            if not k.startswith("blocks.11."):
                assert torch.equal(base["model"][f"0.{k}"], state[f"0.{k}"]), k
        model.load_state_dict(state)
        h = F.normalize(
            torch.cat([audit.encode(model[0], t) for t in tokens + source_tok]), dim=1
        )
        values[f"{tag}_features"] = h[: len(panel)].cpu().numpy()
        values[f"{tag}_logits"] = model[1].last_layer(h).cpu().numpy()
    f = LOCAL / "frozen-endpoint-predictions.npz"
    np.savez(f, **values)
    provenance = {
        **sig,
        "predictions_sha256": audit.file_digest(f),
        "source_ids": sources.tolist(),
    }
    audit.write_json(LOCAL / "sealed-input-provenance.json", json_native(provenance))
    print("Frozen predictions for panel and MOVE sources", len(sources), flush=True)
    return values, provenance


def grouped(y):
    return zip(GROUPS, (y >= 0, y < 98, y >= 98), strict=True)


def probabilities(logit, temp):
    z = logit / temp
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def endpoints(out, table, truth, events):
    ids = out["ids"]
    n = int(out["panel_n"])
    y = truth[ids]
    rev = {v: k for k, v in table.items()}
    mapped = np.asarray([table.get(i, -1) for i in range(196)])
    true = np.asarray([rev.get(v, -1) for v in y])
    valid = true >= 0
    pairs = out["pairs"]
    pos = np.arange(len(pairs)) + n
    records, raw = {}, {}
    for key, data in out.items():
        if not key.endswith("_logits"):
            continue
        tag = key.removesuffix("_logits")
        pred = data.argmax(1)
        h = torch.as_tensor(out[f"{tag}_features"], device="cuda")
        panel_scores, _ = audit.assess_features(h, pred[:n], y[:n], table)
        tt, ss = probabilities(data, 0.04), probabilities(data, 0.1)
        rank = np.zeros(len(y), dtype=int)
        rank[valid] = 1 + (data[valid] > data[valid, true[valid], None]).sum(1)
        raw[tag] = {
            "slot": pred,
            "correct": mapped[pred] == y,
            "rank": rank,
            "teacher": tt,
            "student": ss,
        }
        records[tag] = {"panel": panel_scores, "cohorts": {}}
    cohorts = {
        "panel": ids[:n],
        "CLIP_sources": np.asarray(
            sorted(
                {r[2] for (_, a), moves in events.items() if a == "S1" for r in moves}
            ),
            dtype=int,
        ),
        "random_sources": np.asarray(
            sorted(
                {r[2] for (_, a), moves in events.items() if a == "S3" for r in moves}
            ),
            dtype=int,
        ),
    }
    for tag, r in raw.items():
        for name, keys in cohorts.items():
            loc = np.isin(ids, keys)
            groups = {}
            for g, member in grouped(y):
                mask = loc & member
                ix = pos[mask[n:]]
                val = {
                    "n": int(mask.sum()),
                    "correct": int(r["correct"][mask].sum()),
                    "true_rank_n": int((mask & valid).sum()),
                    "true_slot_mean_rank": mean(r["rank"][mask & valid]),
                }
                for h in ("teacher", "student"):
                    p = r[h]
                    val[f"{h}_confidence"] = mean(p[mask].max(1))
                    val[f"{h}_source"] = mean(p[ix, pairs[mask[n:], 0]])
                    val[f"{h}_destination"] = mean(p[ix, pairs[mask[n:], 1]])
                transitions = {}
                for anchor in ("150:shared", f"{tag.split(':')[0]}:S0"):
                    if anchor not in raw or anchor == tag:
                        continue
                    a = raw[anchor]
                    transitions[anchor] = {
                        "changed_slot": int((mask & (r["slot"] != a["slot"])).sum()),
                        "wrong_to_correct": int(
                            (mask & ~a["correct"] & r["correct"]).sum()
                        ),
                        "correct_to_wrong": int(
                            (mask & a["correct"] & ~r["correct"]).sum()
                        ),
                        "wrong_to_wrong_changed": int(
                            (
                                mask
                                & ~a["correct"]
                                & ~r["correct"]
                                & (r["slot"] != a["slot"])
                            ).sum()
                        ),
                        "mean_rank_delta": mean(
                            r["rank"][mask & valid] - a["rank"][mask & valid]
                        ),
                    }
                val["transitions"] = transitions
                groups[g] = val
            records[tag]["cohorts"][name] = groups
    return records


def mean(v):
    return float(np.mean(v)) if len(v) else None


def applied_semantics(events, truth, mapping):
    result = {}
    for (epoch, name), rows in events.items():
        yy = truth[[r[2] for r in rows]]
        out = {}
        for g, mask in grouped(yy):
            picked = [r for take, r in zip(mask, rows, strict=True) if take]
            counts = Counter()
            for row in picked:
                src, dst = (mapping.get(k, -1) for k in row[3:5])
                y = truth[row[2]]
                kind = (
                    "unresolved"
                    if min(src, dst) < 0
                    else "correct"
                    if dst == y
                    else "harmful"
                    if src == y
                    else "wrong_to_wrong"
                )
                counts[kind] += 1
            out[g] = {
                "edits": len(picked),
                "categories": dict(counts),
                **{
                    k: mean([row[i] for row in picked])
                    for i, k in enumerate(
                        (
                            "teacher_source",
                            "teacher_destination",
                            "student_source",
                            "student_destination",
                        ),
                        5,
                    )
                },
            }
        result.setdefault(epoch, {})[name] = out
    return result


def report(result):
    lines = [
        "# Matched two-epoch CLIP MOVE strength continuation",
        "",
        (
            "All arms start at R2-150 with the saved SGD momentum, cosine scheduler and RNG. "
            "The two batch schedules and paired crops match the previous CLIP diagnostic. The shared "
            "objective keeps all SimGCD terms and leaves R2 relational loss off. The locked CLIP "
            "proposal order, source slot and destination retriever are unchanged."
        ),
        "",
        (
            "S1 and S2 use the intersection of same-occurrence live eligibility at masses .2 and .8. "
            "S3 samples eligible occurrences uniformly in the same batch at mass .8 and uses the same "
            "CLIP retriever. If S3 cannot match the intersection count, S1 and S2 drop the same trailing "
            "ranked rows. S0 keeps supervision. Exposure divides applied occurrences by all mixed training "
            "rows. Target mass sums the moved amount once per occurrence after the two-view averaging convention."
        ),
        "",
        (
            "The original R2-150 checkpoint previously saw the fixed 588-ID panel in transductive training. "
            "Through continuation, panel IDs have zero sampler weight and are excluded from proposal and "
            "reference banks. Semantic scoring opened only after both paired epoch checkpoints and the "
            "endpoint predictions had been saved. The fixed bank-only alignment and descriptive panel "
            "rematch follow the prior continuation. CLIP and random source cohorts use distinct IDs that "
            "received at least one edit in the corresponding arm and can overlap each other."
        ),
        "",
        "| Epoch / arm | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure | Target mass |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for tag, r in result["endpoint"].items():
        epoch, arm = tag.split(":")
        p = r["panel"]
        fixed, rematched = [
            " / ".join(f"{100 * p[k][g]['accuracy']:.2f}" for g in GROUPS)
            for k in ("fixed", "rematched")
        ]
        geom = p["geometry"]["all"]
        m = (
            result["training"][epoch][arm]
            if epoch != "150"
            else {"edits": 0, "exposure": 0, "mass_total": 0}
        )
        lines.append(
            f"| {epoch} / {arm} | {fixed} | {rematched} | {geom['purity5']:.4f} | {geom['centroid_margin']:+.4f} | "
            f"{m['edits']} / {100 * m['exposure']:.3f}% | {m['mass_total']:.1f} |"
        )
    lines += [
        "",
        "## Eligibility and matched supply",
        "",
        "| Epoch | S1 live | S2 live | Common | S3 live | S3 supply shortfall | Matched |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for e, info in result["training"].items():
        m = info["matching"]
        v = [
            m[k]
            for k in (
                "S1_eligible",
                "S2_eligible",
                "common_eligible",
                "S3_eligible",
                "random_supply_shortfall",
                "applied",
            )
        ]
        lines.append(f"| {e} | " + " | ".join(map(str, v)) + " |")
    lines += [
        "",
        "## Applied edits and endpoint response",
        "",
        "| Epoch / arm | Correct / harmful / wrong-to-wrong / unresolved | Mean pre-step teacher source / destination | CLIP source cohort true-slot rank |",
        "|---|---|---|---:|",
    ]

    def val(v):
        return "—" if v is None else f"{v:.4f}"

    for e in result["training"]:
        for a in ARMS[1:]:
            s = result["applied_semantics"].get(e, {}).get(a, {}).get("all", {})
            counts = s.get("categories", {})
            categories = " / ".join(
                str(counts.get(k, 0))
                for k in ("correct", "harmful", "wrong_to_wrong", "unresolved")
            )
            p = " / ".join(
                val(s.get(k)) for k in ("teacher_source", "teacher_destination")
            )
            rank = result["endpoint"][f"{e}:{a}"]["cohorts"]["CLIP_sources"]["all"][
                "true_slot_mean_rank"
            ]
            lines.append(f"| {e} / {a} | {categories} | {p} | {val(rank)} |")
    lines += [
        "",
        (
            "Probability summaries use pre-update two-view posteriors for applied rows and center-view "
            "posteriors for the fixed cohorts at endpoints. Prediction transitions use the fixed slot alignment. "
            "This is one paired continuation from one Cars checkpoint."
        ),
        "",
        f"Elapsed in this invocation {result['seconds'] / 60:.1f} minutes. Full per-step ledgers remain in local paired checkpoints.",
        "",
        "[All/Old/New transitions, probabilities, matched exposure and checkpoint digests](r2_move_strength_training.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--sealed-only", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    model, ds, center, ids, base, plan, panel, tokens, pool, direct, classifier, sig = (
        setup()
    )
    if not args.sealed_only:
        train(model, ds, ids, base, plan, pool, direct, sig, args.smoke)
    if args.smoke:
        return
    saved, hashes, events = load_finished(sig, plan, ids, pool, direct)
    out, frozen = predictions(
        model, ds, center, base, saved, panel, tokens, events, sig
    )
    assert (
        audit.file_digest(LOCAL / "frozen-endpoint-predictions.npz")
        == frozen["predictions_sha256"]
    )
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    baseline_slots = dict(
        zip(classifier["ids"].tolist(), classifier["slots"].tolist(), strict=True)
    )
    mapping, alignment = audit.align(sig["reference_ids"], baseline_slots, truth)
    values = endpoints(out, mapping, truth, events)
    prior = json.loads(repair.OUT.read_text())["epochs"]["150"]["shared"]["panel"]
    initial_panel = values["150:shared"]["panel"]
    for key in ("fixed", "rematched"):
        assert initial_panel[key] == prior[key]
    for g in GROUPS:
        for metric in ("purity5", "centroid_margin", "margin_rows"):
            np.testing.assert_allclose(
                initial_panel["geometry"][g][metric],
                prior["geometry"][g][metric],
                atol=2e-6,
                rtol=0,
            )
    result = {
        "provenance": {
            k: v
            for k, v in frozen.items()
            if k not in ("panel_ids", "reference_ids", "source_ids")
        },
        "panel_n": len(panel),
        "reference_n": len(sig["reference_ids"]),
        "checkpoint_sha256": hashes,
        "alignment": alignment,
        "training": saved[str(EPOCHS[-1])]["records"],
        "endpoint": values,
        "applied_semantics": applied_semantics(events, truth, mapping),
        "seconds": time.monotonic() - start,
    }
    report(result)
    audit.write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
