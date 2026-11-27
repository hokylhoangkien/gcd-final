"""Five paired Cars continuation epochs with locked CLIP MOVE and random-dose control."""

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import rng_state
from model import DINOHead
from scripts import audit_g1 as g1
from scripts import audit_g4 as g4
from scripts import audit_g6 as g6
from scripts import audit_g6_confirm as confirmation
from scripts import audit_oracle_move as oracle
from scripts.audit_oracle_five import digest_array

LOCAL = g1.WORK / "clip-repair"
MD = Path("reports/CRISH_CLIP_REPAIR_TRAINING.md")
OUT = Path("reports/r2_clip_repair_training.json")
HISTORY = Path("reports/r2_oracle_move_trajectory.json")
ARMS = ("T0", "T1", "T2")
START = 150
END = 155


def schedule(loader, ids, panel, initial):
    weights = loader.sampler.weights.clone()
    weights[np.isin(ids, panel)] = 0
    sampler = torch.Generator().set_state(initial["rng"][2])
    augmentation = torch.Generator().manual_seed(
        torch.empty((), dtype=torch.int64)
        .random_(generator=torch.Generator().set_state(initial["rng"][2]))
        .item()
        ^ 529541
    )
    plans = {}
    for epoch in range(START + 1, END + 1):
        torch.empty((), dtype=torch.int64).random_(generator=sampler)
        draw = torch.multinomial(weights, len(loader.dataset), True, generator=sampler)
        torch.empty((), dtype=torch.int64).random_(
            generator=sampler
        )  # historical evaluation iterator
        rows = draw[: len(loader) * 128].reshape(-1, 128).numpy()
        seeds = torch.randint(0, 2**32, (len(rows), 3), generator=augmentation).numpy()
        assert not np.isin(ids[rows], panel).any()
        plans[epoch] = (rows, seeds)
    return plans


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.inference_mode()
def lock_policies(model, ids, slots, clip, aug, panel):
    nonpanel = np.flatnonzero(~np.isin(ids, panel))
    assert len(nonpanel) + len(panel) == len(ids)
    prob = (model[1].last_layer(aug) / 0.04).softmax(-1)
    source = torch.as_tensor(slots, dtype=torch.long, device="cuda")
    same = prob.mean(0).argmax(-1) == source
    mass = (
        (prob.gather(2, source[None, :, None].expand(2, -1, 1)) >= oracle.MASS)
        .all(0)
        .flatten()
    )
    eligible = set(nonpanel[(same & mass).cpu().numpy()[nonpanel]].tolist())
    rows = np.array(sorted(eligible))
    ref_slots = torch.as_tensor(slots[nonpanel], device="cuda")
    ref_vectors = clip[nonpanel]
    candidates = []
    for q in np.array_split(rows, (len(rows) + 255) // 256):
        sim = clip[q] @ ref_vectors.T
        self_cols = np.searchsorted(nonpanel, q)
        sim[torch.arange(len(q), device="cuda"), self_cols] = -torch.inf
        support = torch.full((len(q), 196), -torch.inf, device="cuda")
        for k in np.unique(slots[nonpanel]):
            support[:, k] = sim[:, ref_slots == int(k)].max(1).values
        idx = torch.arange(len(q), device="cuda")
        own = support[idx, source[q]].clamp_min(-1)
        support[idx, source[q]] = -torch.inf
        value, dest = support.max(1)
        assert torch.isfinite(value).all()
        candidates.extend(
            zip(
                ids[q].tolist(),
                slots[q].tolist(),
                dest.cpu().tolist(),
                (value - own).cpu().tolist(),
                strict=True,
            )
        )
    candidates.sort(key=lambda x: (-x[3], x[0]))
    return candidates, len(candidates) // 20, ids[nonpanel].tolist()


def options(prob, keys, legal, pool, *, strength=oracle.MASS):
    src = prob.mean(0).argmax(-1)
    mass = (
        (prob.gather(2, src[None, :, None].expand(2, -1, 1)) >= strength)
        .all(0)
        .flatten()
        .tolist()
    )
    live = src.tolist()
    result, counts = [], Counter()
    for i, (key, label) in enumerate(zip(keys, legal, strict=True)):
        if label >= 0 or key not in pool:
            continue
        source, target, rank = pool[key]
        counts["encounters"] += 1
        if live[i] != source:
            counts["stale"] += 1
        elif not mass[i]:
            counts["low_mass"] += 1
        else:
            result.append((i, key, source, target, rank))
    return sorted(result, key=lambda x: (x[-1], x[0])), counts


@torch.no_grad()
def live_pool(model, batch, pool, *, strength=oracle.MASS):
    model.train()
    h = oracle.encode(model[0], batch["tokens"])
    logits = (
        model[1].last_layer(F.normalize(h, dim=1)).reshape(2, len(batch["ids"]), -1)
    )
    return options(
        (logits / 0.04).softmax(-1),
        batch["ids"].tolist(),
        batch["labels"].tolist(),
        pool,
        strength=strength,
    )


def load_batch(model, ds, rows, seed):
    seed_all(seed)
    model.train()
    views, labels, keys, mask = next(
        iter(DataLoader(Subset(ds, rows.tolist()), batch_size=len(rows)))
    )
    assert torch.equal(mask.flatten().bool(), labels >= 0)
    return {
        "tokens": oracle.prefix(model[0], torch.cat(views)),
        "labels": labels,
        "ids": keys,
    }


def choose(moves, pool):
    def select(prob, ids, legal):
        if moves:
            permitted = set(options(prob, ids, legal, pool)[0])
            if not set(moves).issubset(permitted):
                raise ValueError(
                    "Live teacher eligibility diverged between preview and update"
                )
        actions = [
            oracle.Action(row, "move", src, dst) for row, _, src, dst, _ in moves
        ]
        return actions, [v[:4] for v in moves], {}

    return select


def paired_step(models, batch, targeted, random_pool, seeds, epoch, criterion):
    a, opt_a, _ = models["T1"]
    b, opt_b, _ = models["T2"]
    seed_all(seeds[1])
    direct, stats_a = live_pool(a, batch, targeted)
    seed_all(seeds[1])
    random_rows, stats_b = live_pool(b, batch, random_pool)
    count = min(len(direct), len(random_rows))
    stats_a["match_shortfall"] = len(direct) - count
    rng = np.random.default_rng(int(seeds[2]))
    matched = [random_rows[i] for i in rng.permutation(len(random_rows))[:count]]
    details = {}
    for name, model, opt, moves, pool, stats in (
        ("T0", *models["T0"][:2], [], {}, {}),
        ("T1", a, opt_a, direct[:count], targeted, stats_a),
        ("T2", b, opt_b, matched, random_pool, stats_b),
    ):
        seed_all(seeds[1])
        entries, train = oracle.training_step(
            model, opt, batch, criterion, choose(moves, pool), epoch=epoch - 1
        )
        assert len(entries) == (0 if name == "T0" else count)
        details[name] = {"moves": entries, "counts": dict(stats), **train}
    return details


def load_inputs():
    model, loader, _, _, _, protocol = oracle.build()
    ds = loader.dataset
    blind = ds.unlabelled_dataset.ids
    ids = np.r_[ds.labelled_dataset.uq_idxs, blind]
    ckpt = g1.WORK / "input/epoch-150.pt"
    digest = oracle.file_digest(ckpt)
    g1_report = json.loads(g6.G1_REPORT.read_text())
    assert digest == g1_report["snapshots"]["150"]["checkpoint_sha256"]
    archived = oracle.load_checkpoint(ckpt)
    assert archived["epoch"] == archived["scheduler"]["last_epoch"] == START
    assert archived["scheduler"]["T_max"] == protocol["epochs"] == 200
    for k, v in model[0].state_dict().items():
        if not k.startswith("blocks.11."):
            assert torch.equal(v.cpu(), archived["model"][f"0.{k}"]), k
    current = g4.snapshot(150, g1_report)
    np.testing.assert_array_equal(current["ids"], blind)
    slots = current["slots"]
    receipt = json.loads(g6.OUT.read_text())
    clip = confirmation.checked_cache(
        g6.LOCAL / "clip-image.npz", receipt["clip"]["cache_sha256"], blind
    )
    path = g1.WORK / "features.npy"
    feat_sha = oracle.file_digest(path)
    assert feat_sha == receipt["dinov1_cache_sha256"]
    feature = np.load(path, mmap_mode="r")
    assert feature.shape == (3, 3, len(blind), 768)
    aug = F.normalize(torch.from_numpy(feature[2, 1:].copy()), dim=2).cuda()
    old = json.loads(HISTORY.read_text())
    panel = np.asarray(old["panel_ids"])
    assert len(np.unique(panel)) == 588 and set(panel) <= set(blind)
    prior = np.load(g1.WORK / "actual_draws.npy", mmap_mode="r")[:START, panel]
    prior_counts = np.maximum(prior, 0).sum(
        0
    )  # -1 marks epochs without verified replay.
    assert (prior_counts > 0).all()
    bank = torch.load(
        g1.WORK / "oracle-move/step0-panel-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert (
        bank["panel_ids"] == panel.tolist()
        and bank["checkpoint_sha256"] == old["checkpoint_sha256"]
    )
    model.load_state_dict(archived["model"])
    model.eval()
    probe_sample = np.array([np.flatnonzero(blind == i).item() for i in panel[:8]])
    with torch.no_grad():
        torch.testing.assert_close(
            oracle.encode(model[0], bank["tokens"][0][:1]).cpu(),
            torch.from_numpy(feature[2, 0, probe_sample].copy()),
            atol=6e-5,
            rtol=6e-5,
        )
        center = F.normalize(torch.from_numpy(feature[2, 0].copy()), dim=1).cuda()
        np.testing.assert_array_equal(
            model[1].last_layer(center).argmax(1).cpu().numpy(), slots
        )
    proposals, cap, reference = lock_policies(model, blind, slots, clip, aug, panel)
    assert cap > 0 and not set(reference) & set(panel)
    assert not set(panel) & {p[0] for p in proposals}
    pool = {k: (a, b, i) for i, (k, a, b, _) in enumerate(proposals)}
    targeted = {k: p for k, p in pool.items() if p[-1] < cap}
    plans = schedule(loader, ids, panel, archived)
    sig = {
        "base": "f5688ef",
        "checkpoint_sha256": digest,
        "clip_cache_sha256": receipt["clip"]["cache_sha256"],
        "r2_cache_sha256": feat_sha,
        "panel_ids": panel.tolist(),
        "prior_panel_ids_sampled": int((prior_counts > 0).sum()),
        "prior_panel_min_verified_draws": int(prior_counts.min()),
        "reference_ids": reference,
        "pool_n": len(pool),
        "locked_n": cap,
        "move_mass": oracle.MASS,
        "rows_per_epoch": len(loader) * 128,
        "sample_conditioning": "original weighted replacement with panel weights zeroed",
        "base_objective": "full SimGCD without the R2 relational term",
        "draws": {
            str(e): {"rows": digest_array(r), "seeds": digest_array(s)}
            for e, (r, s) in plans.items()
        },
    }
    return (
        model,
        ds,
        ids,
        blind,
        slots,
        archived,
        panel,
        bank["tokens"],
        proposals,
        targeted,
        pool,
        plans,
        sig,
    )


def make_arms(model, state, *, names=ARMS):
    models = {}
    for name in names:
        network = (
            model
            if name == names[0]
            else torch.nn.Sequential(copy.deepcopy(model[0]), DINOHead(768, 196)).cuda()
        )
        opt, sched = oracle.restore(network, state[name])
        assert sched.get_last_lr() == [g["lr"] for g in opt.param_groups]
        assert sched.T_max == 200
        assert sched.last_epoch == state[name]["scheduler"]["last_epoch"]
        assert opt.state and all(
            entry["momentum_buffer"].shape == p.shape for p, entry in opt.state.items()
        )
        models[name] = (network, opt, sched)
    return models


def save(models, path, sig, epoch, records, rows):
    state = {}
    for name, (m, opt, sched) in models.items():
        state[name] = {
            "model": m.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": rng_state(),
        }
    oracle.atomic_checkpoint(
        {
            "signature": sig,
            "epoch": epoch,
            "state": state,
            "records": records,
            "batch_ledger": rows,
        },
        path,
    )


def report(sig, results, elapsed):
    # Full IDs and row ledgers remain in the verified local run artifacts.
    meta = {k: v for k, v in sig.items() if k not in ("panel_ids", "reference_ids")}
    paired = {}
    for epoch, arms in results.items():
        if "T1" not in arms:
            continue
        paired[epoch] = {}
        for left, right in (("T1", "T0"), ("T2", "T0"), ("T1", "T2")):
            a, b = (arms[name]["panel"] for name in (left, right))
            paired[epoch][f"{left}-{right}"] = {
                g: {
                    "fixed_accuracy_pp": 100
                    * (a["fixed"][g]["accuracy"] - b["fixed"][g]["accuracy"]),
                    "rematched_accuracy_pp": 100
                    * (a["rematched"][g]["accuracy"] - b["rematched"][g]["accuracy"]),
                    "purity5": a["geometry"][g]["purity5"]
                    - b["geometry"][g]["purity5"],
                    "centroid_margin": a["geometry"][g]["centroid_margin"]
                    - b["geometry"][g]["centroid_margin"],
                }
                for g in ("all", "old", "new")
            }
    oracle.write_json(
        OUT,
        oracle.json_native(
            {
                "protocol": meta,
                "panel_rows": len(sig["panel_ids"]),
                "reference_rows": len(sig["reference_ids"]),
                "epochs": results,
                "paired_differences": paired,
                "elapsed_seconds": elapsed,
            }
        ),
    )
    lines = [
        "# Five-epoch CLIP direct supervision repair on Stanford Cars",
        "",
        (
            "All arms resume R2-150 with the saved SGD momentum, cosine scheduler and RNG. "
            "Training keeps the full SimGCD objective, and the R2 relational term is off in all arms "
            "to isolate direct supervision repair. The 200-epoch scheduler continues through epoch 155. "
            "T0 keeps supervision, T1 applies locked CLIP MOVE and T2 samples eligible source occurrences "
            "uniformly from each same batch with the same frozen CLIP destination rule."
        ),
        "",
        (
            f"The top 5% of the initial teacher-eligible pool are {sig['locked_n']} of "
            f"{sig['pool_n']} IDs, rounded down. CLIP risk is best alternative minus current-slot "
            "maximum cosine, with self excluded. The initial source needs agreement with the average "
            "cached teacher distribution and at least .2 mass in both views. These checks repeat on "
            "each live augmented occurrence. T1 is capped by T2's live supply per batch; any shortfall "
            "is logged. The 5% ID budget does not set the training exposure, which counts applied "
            "MOVE occurrences divided by all sampled rows, including repeated IDs."
        ),
        "",
        (
            "The original checkpoint has already seen every image on the fixed 588-ID panel during "
            f"transductive training (at least {sig['prior_panel_min_verified_draws']} replay-verified "
            "draws per ID through R2-150). During continuation its IDs have zero sampler weight and are excluded from "
            "proposal selection and the CLIP reference bank. The remaining relative sampler weights, "
            "draw length and drop-last behavior are preserved, changing the labeled fraction slightly. "
            "All arms share each pair of augmented crops. Sampler and augmentation streams derive from "
            "the checkpoint CPU RNG; single-process crop generation differs from the historical loader."
        ),
        "",
        (
            "Ground truth is used solely in panel evaluation and fitting the fixed initial slot "
            "alignment on non-panel IDs. Panel rematching is descriptive. kNN purity uses five "
            "neighbors; centroid margin compares the leave-one-out own centroid with the nearest "
            "other class. Results are paired continuations of one trained checkpoint."
        ),
        "",
        "| Epoch / arm | Fixed All / Old / New % | Rematched All / Old / New % | kNN-5 | Centroid margin | Moves / exposure | Next LR |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for e, arms in results.items():
        for name, row in arms.items():
            p = row["panel"]
            fixed, rematch = [
                " / ".join(
                    f"{100 * p[m][g]['accuracy']:.2f}" for g in ("all", "old", "new")
                )
                for m in ("fixed", "rematched")
            ]
            geom = p["geometry"]["all"]
            lines.append(
                f"| {e} / {name} | {fixed} | {rematch} | {geom['purity5']:.4f} | "
                f"{geom['centroid_margin']:+.4f} | {row['applied']} / {100 * row['exposure']:.3f}% | {row['lr']:.6f} |"
            )
    lines += [
        "",
        f"Elapsed {elapsed / 60:.1f} minutes. Paired epoch checkpoints and per-step ledgers are local.",
        "",
        "[Subgroup geometry, checkpoint metrics and matched exposure](r2_clip_repair_training.json)",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--smoke", action="store_true")
    smoke = arg.parse_args().smoke
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    (
        model,
        ds,
        ids,
        blind,
        slots,
        archived,
        panel,
        tokens,
        proposals,
        targeted,
        pool,
        plans,
        sig,
    ) = load_inputs()
    lock = {"signature": sig, "proposals": proposals}
    if not smoke:
        frozen = LOCAL / "frozen-policy.json"
        if frozen.is_file():
            assert json.loads(frozen.read_text()) == oracle.json_native(lock)
        else:
            oracle.write_json(frozen, oracle.json_native(lock))
        sig = {**sig, "policy_sha256": oracle.file_digest(frozen)}
    epochs = (
        [e for e in plans if (LOCAL / f"epoch-{e}.pt").exists()] if not smoke else []
    )
    prev = max(epochs, default=START)
    results = {}
    if prev != START:
        saved = oracle.load_checkpoint(LOCAL / f"epoch-{prev}.pt")
        assert (saved["epoch"], saved["signature"]) == (prev, sig)
        models = make_arms(model, saved["state"])
        results = saved["records"]
        del saved
    else:
        models = make_arms(model, dict.fromkeys(ARMS, archived))
    lr0 = archived["scheduler"]["_last_lr"][0]
    del archived
    print(
        "Locked",
        len(targeted),
        "of",
        len(pool),
        "initially eligible IDs; starting after",
        prev,
        flush=True,
    )
    criterion = oracle.DistillLoss(30, 200)
    if smoke:
        rows, seeds = plans[START + 1]
        for i in range(5):
            batch = load_batch(model, ds, rows[i], seeds[i][0])
            detail = paired_step(
                models, batch, targeted, pool, seeds[i], START + 1, criterion
            )
            if detail["T1"]["moves"]:
                assert len(detail["T1"]["moves"]) == len(detail["T2"]["moves"])
                assert (
                    detail["T1"]["cls_edit_norm"] > 0
                    and detail["T2"]["cls_edit_norm"] > 0
                )
                print(
                    "Paired actual MOVE smoke",
                    len(detail["T1"]["moves"]),
                    "peak bytes",
                    torch.cuda.max_memory_allocated(),
                    flush=True,
                )
                return
        raise ValueError("No actual MOVE in the first five smoke batches")

    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    aligned, _ = oracle.align(
        sig["reference_ids"],
        dict(zip(blind.tolist(), slots.tolist(), strict=True)),
        truth,
    )
    if not results:
        point, _ = oracle.assess(model, tokens, truth[panel], aligned)
        results[str(START)] = {
            "shared": {"panel": point, "applied": 0, "exposure": 0.0, "lr": lr0}
        }
    for e, (rows, seeds) in plans.items():
        if e <= prev:
            continue
        t0 = time.monotonic()
        ledger = []
        for step, (rr, ss) in enumerate(zip(rows, seeds, strict=True), 1):
            batch = load_batch(model, ds, rr, ss[0])
            np.testing.assert_array_equal(batch["ids"].numpy(), ids[rr])
            ledger.append(paired_step(models, batch, targeted, pool, ss, e, criterion))
            if step % 10 == 0:
                print(
                    "Epoch",
                    e,
                    "steps",
                    step,
                    "matched moves",
                    sum(len(row["T1"]["moves"]) for row in ledger),
                    flush=True,
                )
        results[str(e)] = {}
        for name, (m, _, sched) in models.items():
            sched.step()
            assert sched.last_epoch == e
            panel_result, _ = oracle.assess(m, tokens, truth[panel], aligned)
            steps = [row[name] for row in ledger]
            count = sum(len(s["moves"]) for s in steps)
            counts = Counter()
            for row in steps:
                counts.update(row["counts"])
            results[str(e)][name] = {
                "panel": panel_result,
                "applied": count,
                "exposure": count / rows.size,
                "lr": sched.get_last_lr()[0],
                "mean_loss": float(np.mean([s["loss"] for s in steps])),
                "eligibility": dict(counts),
                "unique_applied_ids": len({a[1] for s in steps for a in s["moves"]}),
                "peak_vram_bytes": max(s["peak_vram"] for s in steps),
            }
        assert results[str(e)]["T1"]["applied"] == results[str(e)]["T2"]["applied"]
        path = LOCAL / f"epoch-{e}.pt"
        save(models, path, sig, e, results, ledger)
        oracle.write_json(LOCAL / "progress.json", oracle.json_native(results))
        print(
            "Checkpoint", path, "duration", round(time.monotonic() - t0, 1), flush=True
        )
    report(sig, results, time.monotonic() - start)


if __name__ == "__main__":
    main()
