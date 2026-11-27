"""One-epoch privileged MOVE dose diagnostic. No production integration."""

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from crish.experiment import (
    atomic_checkpoint,
    file_digest,
    load_checkpoint,
    restore_rng,
    rng_state,
)
from crish.repair import Action, corrected_distill
from model import DistillLoss
from relational.sensitivity_metrics import cls, prefix
from relational.state import write_json
from relational.train import baseline, build
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_r2_verifier import checkpoint_optimizer
from scripts.audit_r2_virtual import encode

ARMS = (
    ("O0", 0, None),
    ("O2", 0.001, None),
    ("O3", 0.01, None),
    ("O1", 0.01, "O3"),
    ("O4", 0.05, None),
    ("O5", 0.10, None),
    ("O6", 0.10, "O5"),
)
MASS = 0.2
EPOCH = 100


def save_epoch(path, model, opt, sched, progress):
    atomic_checkpoint(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": rng_state(),
            "progress": progress,
        },
        path,
    )


def batch_plan(loader, state):
    rng = torch.Generator().set_state(state["rng"][2])
    torch.empty((), dtype=torch.int64).random_(
        generator=rng
    )  # official iterator base seed
    order = torch.multinomial(
        loader.sampler.weights, len(loader.dataset), True, generator=rng
    )
    return order[: len(loader) * 128].reshape(-1, 128).tolist()


def load_batch(model, ds, rows, step):
    seed = 54301 + step
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    views, legal, ids, mask = next(
        iter(DataLoader(Subset(ds, rows), batch_size=len(rows)))
    )
    assert (legal[~mask.flatten().bool()] == -1).all()
    return {"tokens": prefix(model[0], torch.cat(views)), "labels": legal, "ids": ids}


def training_step(model, opt, batch, criterion, select, *, epoch=EPOCH, mass=MASS):
    model.train()
    backbone, head = model
    tokens = batch["tokens"]
    opt.zero_grad(set_to_none=True)
    h = encode(backbone, tokens).detach().requires_grad_()
    proj, logits = head(h)
    legal = batch["labels"].to(device=h.device, dtype=torch.long)
    n = len(legal)
    with torch.no_grad():
        probs = (logits.reshape(2, n, -1) / 0.04).softmax(-1)
    actions, entries, stats = select(probs, batch["ids"].tolist(), legal.tolist())
    base = baseline(proj, logits, legal, legal >= 0, criterion, epoch)
    loss = base
    if actions:
        delta = 0.65 * (
            corrected_distill(logits, logits.detach(), criterion, epoch, actions, mass)
            - criterion(logits, logits.detach(), epoch)
        )
        cls_delta, w_delta = torch.autograd.grad(
            delta, (h, head.last_layer.weight_v), retain_graph=True
        )
        stats.update(
            cls_edit_norm=cls_delta.norm().item(),
            classifier_edit_norm=w_delta.norm().item(),
            distill_delta=delta.item(),
        )
        loss = base + delta
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite diagnostic training loss")
    loss.backward()
    for tok, incoming in zip(tokens, h.grad.split(8), strict=True):
        cls(backbone, tok).backward(incoming)
    opt.step()
    stats.update(
        loss=loss.item(),
        original_loss=base.item(),
        peak_vram=torch.cuda.max_memory_allocated(),
    )
    return entries, stats


def no_edit(q, ids, legal):
    return [], [], {}


def select_moves(
    q,
    ids,
    legal,
    truth,
    mapping,
    inverse,
    frozen_slots,
    panel,
    quota,
    seed,
    match,
    random_slots,
):
    """Select eligible row occurrences, with random matching fixed by the oracle ledger."""
    stats = Counter()
    options = {}
    sources = q.mean(0).argmax(-1)
    mass_ok = (
        (q.gather(2, sources.reshape(1, -1, 1).expand(2, -1, 1)) >= MASS)
        .all(0)
        .flatten()
        .tolist()
    )
    sources = sources.tolist()
    for row, (key, label) in enumerate(zip(ids, legal, strict=True)):
        if label >= 0 or key in panel:
            continue
        source = sources[row]
        allowed = mass_ok[row]
        stats["unlabeled_non_panel"] += 1
        stats["live_mass_eligible"] += allowed
        if match is None:
            target = inverse.get(int(truth[key]))
            expected = frozen_slots[key]
            if mapping.get(expected, -1) == truth[key]:
                continue
            stats["wrong_source_occurrences"] += 1
            if target is None:
                stats["unmapped_true_slot"] += 1
                continue
            if source != expected:
                stats["stale_source"] += 1
                continue
            if not allowed:
                stats["low_mass"] += 1
                continue
            options[row] = (key, source, target)
    rng = np.random.default_rng(seed)
    if match is None:
        keys = list(options)
        rows = rng.permutation(keys)[:quota].tolist()
        stats["oracle_eligible"] = len(keys)
    else:
        rows = []
        for event in match:
            row, key, source, _ = event
            assert ids[row] == key and legal[row] == -1 and key not in panel
            live = sources[row]
            if live != source:
                stats["matched_stale_source"] += 1
                continue
            if not mass_ok[row]:
                stats["matched_low_mass"] += 1
                continue
            choices = [s for s in random_slots if s != source]
            if not choices:
                stats["matched_no_destination"] += 1
                continue
            options[row] = (key, source, int(rng.choice(choices)))
            rows.append(row)
    entries = [[int(row), *options[row]] for row in rows]
    stats["planned"] = quota if match is None else len(match)
    stats["applied"] = len(entries)
    return [Action(row, "move", s, t) for row, _, s, t in entries], entries, dict(stats)


def accuracy(pred, truth, mapping):
    correct = np.array(
        [mapping.get(int(k), -1) == v for k, v in zip(pred, truth, strict=True)]
    )
    return {
        name: {
            "n": int(m.sum()),
            "accuracy": float(correct[m].mean()) if m.any() else None,
        }
        for name, m in (("all", truth >= 0), ("old", truth < 98), ("new", truth >= 98))
    }


@torch.no_grad()
def assess(model, center_tokens, labels, fixed):
    model.eval()
    h = F.normalize(torch.cat([encode(model[0], tok) for tok in center_tokens]), dim=-1)
    pred = model[1].last_layer(h).argmax(-1).cpu().numpy()
    return assess_features(h, pred, labels, fixed)


@torch.no_grad()
def assess_features(h, pred, labels, fixed):
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Expected a nonempty vector of integer Cars class indices")
    if y.min() < 0 or y.max() >= 196:
        raise ValueError("Cars class indices must be in [0, 195]")
    idx = list(range(len(y)))
    remap, _ = align(idx, dict(zip(idx, pred, strict=True)), y)
    sim = h @ h.T
    sim.fill_diagonal_(-torch.inf)
    yt = torch.as_tensor(y, device=h.device, dtype=torch.long)
    knn = (yt[sim.topk(5).indices] == yt[:, None]).float().mean(1)
    assignment = F.one_hot(yt, 196).float()
    sums = assignment.T @ h
    own = (h * F.normalize(sums[yt] - h, dim=-1)).sum(1)
    cs = h @ F.normalize(sums, dim=-1).T
    cs[:, assignment.sum(0) == 0] = -torch.inf
    cs.scatter_(1, yt[:, None], -torch.inf)
    margin = own - cs.max(1).values
    groups = {}
    eligible = assignment.sum(0)[yt] > 1
    for key, m in (("all", yt >= 0), ("old", yt < 98), ("new", yt >= 98)):
        covered = m & eligible
        groups[key] = {
            "purity5": knn[m].mean().item() if m.any() else None,
            "centroid_margin": margin[covered].mean().item() if covered.any() else None,
            "margin_rows": int(covered.sum()),
        }
    return {
        "fixed": accuracy(pred, y, fixed),
        "rematched": accuracy(pred, y, remap),
        "geometry": groups,
    }, pred


def restore(model, state):
    model.load_state_dict(state["model"])
    opt = checkpoint_optimizer(model, state)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=0.0001)
    sched.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return opt, sched


def prep(model, loader, center, work, state):
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    frozen = json.loads((work / "e3-frozen-proposals.json").read_text())
    row = frozen["snapshots"]["100"]
    slots = {int(k): v for k, v in row["predicted_slots"].items()}
    ids = np.load(work / "ids.npy").tolist()
    mapping, info = align(ids, slots, truth)
    plan = batch_plan(loader, state)
    ds = loader.dataset
    iid = np.r_[ds.labelled_dataset.uq_idxs, ds.unlabelled_dataset.ids]
    sampled = set(iid[np.asarray(plan).ravel()].tolist())
    remaining = np.asarray(sorted(set(ids) - sampled))
    rng = np.random.default_rng(5293)
    panel = []
    for k in range(196):
        choices = remaining[truth[remaining] == k]
        panel.extend(rng.choice(choices, min(3, len(choices)), replace=False).tolist())
    assert not set(panel) & sampled and len(panel) > 300
    c = copy.copy(ds.unlabelled_dataset)
    c.transform = center
    lookup = {int(i): j for j, i in enumerate(c.ids)}
    return truth, slots, mapping, info, plan, sampled, panel, c, lookup


@torch.no_grad()
def center_cache(model, ds, rows):
    parts = []
    for images, _, _ in DataLoader(Subset(ds, rows), batch_size=32):
        parts.append(prefix(model[0], images))
    return parts


def run(work, scratch, out, *, step0_only=False):
    started = time.monotonic()
    model, loader, _, center, _, _ = build()
    ckpt = work / "input/epoch-100.pt"
    state = load_checkpoint(ckpt)
    restore(model, state)
    prep_data = prep(model, loader, center, work, state)
    truth, slots, mapping, info, plan, sampled, panel, c, lookup = prep_data
    inverse = {v: k for k, v in mapping.items()}
    random_slots = sorted({slots[k] for k in lookup})
    protected = set(panel)
    dl = loader.dataset
    checkpoint_sha = file_digest(ckpt)
    panel_cache = scratch / "step0-panel-tokens.pt"
    if panel_cache.is_file():
        cached = torch.load(panel_cache, map_location="cpu", weights_only=True)
        if (
            cached["panel_ids"] != panel
            or cached["checkpoint_sha256"] != checkpoint_sha
        ):
            raise ValueError(
                "Cached panel does not match the checkpoint and ordered panel IDs"
            )
        panel_tokens = cached["tokens"]
    else:
        panel_tokens = center_cache(model, c, [lookup[k] for k in panel])
        atomic_checkpoint(
            {
                "panel_ids": panel,
                "checkpoint_sha256": checkpoint_sha,
                "tokens": panel_tokens,
            },
            panel_cache,
        )
    rng = np.random.default_rng(8201)
    monitor = rng.permutation(sorted(sampled.intersection(lookup))).tolist()
    # Check for the full disjoint official test split before allocating feature storage.
    from types import SimpleNamespace

    from data.get_datasets import get_class_splits
    from data.stanford_cars import CarsDataset

    test = CarsDataset(train=False, transform=center)
    test_status = {
        "available_images": sum(Path(p).is_file() for p in test.data),
        "expected": len(test),
    }
    test_tokens = None
    if test_status["available_images"] == len(test):
        split = get_class_splits(
            SimpleNamespace(dataset_name="scars", use_ssb_splits=True)
        )
        class_order = list(split.train_classes) + list(split.unlabeled_classes)
        test_truth = np.asarray([class_order.index(y - 1) for y in test.target])
        test_rows = [
            int(k)
            for cls_id in range(196)
            for k in rng.choice(
                np.flatnonzero(test_truth == cls_id),
                min(3, int((test_truth == cls_id).sum())),
                replace=False,
            )
        ]
        test_tokens = center_cache(model, test, test_rows)
        test_y = test_truth[test_rows]
        test_status["panel_n"] = len(test_rows)
    initial, _ = assess(model, panel_tokens, truth[panel], mapping)
    test_initial = (
        assess(model, test_tokens, test_y, mapping)[0] if test_tokens else None
    )
    if step0_only:
        receipt = {
            "checkpoint_sha256": checkpoint_sha,
            "panel_n": len(panel),
            "input_label_dtype": str(truth[panel].dtype),
            "panel": initial,
            "test_availability": test_status,
            "test": test_initial,
        }
        write_json(scratch / "step0-smoke.json", json_native(receipt))
        print(
            json.dumps(
                {
                    "panel_n": len(panel),
                    "input_label_dtype": receipt["input_label_dtype"],
                    "fixed_all": initial["fixed"]["all"]["accuracy"],
                    "test_availability": test_status,
                },
                allow_nan=False,
            ),
            flush=True,
        )
        return
    criterion = DistillLoss(30, 200)
    receipts = {}
    total = len(plan) * 128
    result = {
        "checkpoint_sha256": checkpoint_sha,
        "epoch_budget": len(plan),
        "denominator": total,
        "move_mass": MASS,
        "panel_ids": panel,
        "alignment": info,
        "frozen_slot_to_class": mapping,
        "test": test_status,
        "arms": receipts,
        "smoke": json.loads((scratch / "smoke.json").read_text()),
    }
    steps = {20, 50, len(plan)}
    for arm, rate, matched in ARMS:
        opt, sched = restore(model, state)
        receipt = {
            "target_rate": rate,
            "target_moves": int(np.floor(rate * total + 0.5)),
            "matched_to": matched,
            "steps": [],
            "milestones": {
                "0": {
                    "panel": initial,
                    "test": test_initial,
                    "occurrences": 0,
                    "exposure": 0.0,
                    "correction_comparison": {
                        "corrected": {"ids": []},
                        "untouched": {
                            "ids": monitor[:64],
                            "fixed": accuracy(
                                [slots[k] for k in monitor[:64]],
                                truth[monitor[:64]],
                                mapping,
                            ),
                        },
                    },
                }
            },
        }
        receipts[arm] = receipt
        seen = set()
        applied = 0
        torch.cuda.reset_peak_memory_stats()
        for index, rows in enumerate(plan, 1):
            f = scratch / f"batch-{index:02d}.pt"
            if not f.exists():
                data = load_batch(model, dl, rows, index)
                atomic_checkpoint(data, f)
            else:
                data = torch.load(f, map_location="cpu", weights_only=True)
            assert data["ids"].tolist() == [
                int(k)
                for k in np.r_[dl.labelled_dataset.uq_idxs, dl.unlabelled_dataset.ids][
                    rows
                ]
            ]
            quota = max(0, int(np.floor(rate * 128 * index + 0.5)) - applied)
            desired = (
                receipts[matched]["steps"][index - 1]["edits"] if matched else None
            )

            def choose(q, ids, legal, index=index, quota=quota, desired=desired):
                return select_moves(
                    q,
                    ids,
                    legal,
                    truth,
                    mapping,
                    inverse,
                    slots,
                    protected,
                    quota,
                    3893 + index,
                    desired,
                    random_slots,
                )

            random.seed(69401 + index)
            np.random.seed(69401 + index)
            torch.manual_seed(69401 + index)
            edits, train = training_step(model, opt, data, criterion, choose)
            train["sealed_correct_destination"] = sum(
                bool(mapping.get(target, -1) == truth[key])
                for _, key, _, target in edits
            )
            applied += len(edits)
            seen.update(row[1] for row in edits)
            receipt["steps"].append({"step": index, "edits": edits, "stats": train})
            if index == len(plan):
                sched.step()
            if index in steps:
                point, _ = assess(model, panel_tokens, truth[panel], mapping)
                test_point = (
                    assess(model, test_tokens, test_y, mapping)[0]
                    if test_tokens
                    else None
                )
                touched = [k for k in monitor if k in seen][:64]
                untouched = [k for k in monitor if k not in seen][:64]
                comparison = {}
                for name, chosen in (("corrected", touched), ("untouched", untouched)):
                    if not chosen:
                        comparison[name] = {"ids": []}
                        continue
                    bank = center_cache(model, c, [lookup[k] for k in chosen])
                    with torch.no_grad():
                        feats = F.normalize(
                            torch.cat([encode(model[0], t) for t in bank]), dim=-1
                        )
                        preds = model[1].last_layer(feats).argmax(-1).cpu().numpy()
                    comparison[name] = {
                        "ids": chosen,
                        "fixed": accuracy(preds, truth[chosen], mapping),
                        "step0_on_same_ids": accuracy(
                            [slots[k] for k in chosen], truth[chosen], mapping
                        ),
                    }
                progress = {
                    "panel": point,
                    "test": test_point,
                    "occurrences": applied,
                    "exposure": applied / (128 * index),
                    "correction_comparison": comparison,
                }
                receipt["milestones"][str(index)] = progress
                save_epoch(
                    scratch / f"{arm}-recovery.pt",
                    model,
                    opt,
                    sched,
                    {"arm": arm, "step": index, "receipt": receipt},
                )
                print(arm, index, applied, point["fixed"]["all"], flush=True)
        receipt.update(
            realized_moves=applied,
            realized_exposure=applied / total,
            nominal_shortfall=receipt["target_moves"] - applied,
            matched_shortfall=receipts[matched]["realized_moves"] - applied
            if matched
            else None,
        )
        save_epoch(
            scratch / f"{arm}-final.pt",
            model,
            opt,
            sched,
            {"arm": arm, "step": len(plan), "receipt": receipt},
        )
        result["seconds"] = time.monotonic() - started
        write_json(scratch / "partial.json", json_native(result))
    paired_differences(result)
    write_json(out, json_native(result))
    report(result, out.with_name("CRISH_ORACLE_MOVE_TRAJECTORY.md"))


def paired_differences(result):
    def compare(left, right):
        d = {}
        for key in ("fixed", "rematched", "geometry"):
            d[key] = {}
            for group in ("all", "old", "new"):
                metrics = (
                    ("accuracy",)
                    if key != "geometry"
                    else ("purity5", "centroid_margin")
                )
                d[key][group] = {
                    m: (
                        left[key][group][m] - right[key][group][m]
                        if left[key][group][m] is not None
                        and right[key][group][m] is not None
                        else None
                    )
                    for m in metrics
                }
        return d

    comparisons = {}
    for arm, r in result["arms"].items():
        controls = ["O0"] if arm != "O0" else []
        controls += ["O1"] if arm == "O3" else ["O6"] if arm == "O5" else []
        comparisons[arm] = {
            k: {
                s: compare(q["panel"], result["arms"][k]["milestones"][s]["panel"])
                for s, q in r["milestones"].items()
            }
            for k in controls
        }
    result["paired_differences"] = comparisons


def report(result, path):
    def acc(p, mode):
        return " / ".join(
            f"{100 * p[mode][g]['accuracy']:.2f}"
            if p[mode][g]["accuracy"] is not None
            else "—"
            for g in ("all", "old", "new")
        )

    n = result["denominator"]
    end = str(result["epoch_budget"])
    rows = [
        "# Oracle MOVE one-epoch trajectory diagnostic",
        "",
        (
            f"Seven arms from R2 epoch 100 with saved SGD and cosine state, {result['epoch_budget']} steps "
            f"and {n:,} scoring rows per arm. We replay the archived sampler RNG and apply a common, "
            "explicit per-batch augmentation seed with a single loader process. Frozen-prefix tokens "
            "are reused across arms. Projector, classifier and block 11 all update through original "
            "SimGCD, including the contrastive losses, with R2 disabled. Only the diagnostic writes "
            "privileged MOVE targets. All moves have mass .2 and preserve the full loss denominator."
        ),
        "",
        (
            "Exposure is applied edits divided by all mixed training row occurrences, including labeled "
            "and repeated draws. Each oracle arm carries forward a quota deficit and stops at its "
            "prespecified cumulative target. An ID needs a wrong frozen source, an aligned true slot, "
            "agreement with its live teacher source and at least .2 teacher mass in both crops. "
            "Random controls replay the .01 and .10 oracle ledgers by exact step and row, drawing "
            "another supported predicted slot uniformly. They skip stale sources or low mass; there "
            "is no replacement or forced edit. The table gives actual count and shortfall."
        ),
        "",
        (
            f"The primary panel has {len(result['panel_ids'])} unlabeled train IDs that never occur in the "
            "shared training draw, sampled up to three per class. Its Hungarian rematch uses the same "
            "panel and is descriptive. Fixed alignment is the sealed initial full-split mapping. "
            "Probe geometry reports kNN-5 and leave-one-out own-centroid versus nearest-other margin. "
            "The corrected/untouched subsets sample up to 64 training IDs in each observed exposure "
            "group at each milestone and report fixed accuracy and step-0 accuracy on the same IDs. "
            "This comparison is conditional on selection and supports no causal claim by itself."
        ),
        "",
        (
            f"Disjoint official test images present: {result['test']['available_images']}/{result['test']['expected']}. "
            "When the split is complete, a fixed class-stratified test panel is reported separately in JSON."
        ),
        "",
        "| Arm | Exposure target | Applied / target | Nominal shortfall | Matched shortfall | Endpoint fixed All / Old / New (%) | Δ fixed All vs KEEP (pp) |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for arm, r in sorted(result["arms"].items()):
        p = r["milestones"][end]["panel"]
        control = result["arms"]["O0"]["milestones"][end]["panel"]
        delta = 100 * (
            p["fixed"]["all"]["accuracy"] - control["fixed"]["all"]["accuracy"]
        )
        rows.append(
            f"| {arm} | {100 * r['target_rate']:g}% | {r['realized_moves']} / {r['target_moves']} | "
            f"{r['nominal_shortfall']} | {r['matched_shortfall'] if r['matched_shortfall'] is not None else '—'} | "
            f"{acc(p, 'fixed')} | {delta:+.3f} |"
        )
    rows += [
        "",
        "| Arm / step | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 All | Centroid margin All |",
        "|---|---|---|---:|---:|",
    ]
    for arm, r in sorted(result["arms"].items()):
        for step, q in r["milestones"].items():
            if step == "0" and arm != "O0":
                continue
            p = q["panel"]
            g = p["geometry"]["all"]
            rows.append(
                f"| {arm} / {step} | {acc(p, 'fixed')} | {acc(p, 'rematched')} | {g['purity5']:.4f} | {g['centroid_margin']:+.4f} |"
            )
    rows += [
        "",
        (
            "Edit dose is logged at the backbone CLS features and classifier weights as "
            "the L2 norm of the differential gradient, alongside per-step eligibility, stale sources and "
            "matched-dose shortfall. These are single shared-seed trajectories. A gap on one epoch or "
            "one split does not establish a general dose-response or long-run effect."
        ),
        "",
        (
            f"Elapsed {result['seconds'] / 60:.1f} min. Smoke peak allocated VRAM "
            f"{result['smoke']['batch128_peak_bytes'] / 2**20:.0f} MiB. "
            "Completed arms and milestone recovery checkpoints are local artifacts."
        ),
        "",
        "[Per-step ledgers, geometry and paired differences](r2_oracle_move_trajectory.json)",
    ]
    for arm in ("O1", "O6"):
        stats = Counter()
        for step in result["arms"][arm]["steps"]:
            stats.update(
                {k: v for k, v in step["stats"].items() if k.startswith("matched_")}
            )
        rows.append(f"- {arm} eligibility divergence — {dict(stats)}.")
    path.write_text("\n".join(rows) + "\n", encoding="utf8")


def smoke(work, scratch):
    model, loader, _, _, _, _ = build()
    state = load_checkpoint(work / "input/epoch-100.pt")
    plan = batch_plan(loader, state)
    opt, _ = restore(model, state)
    ds = loader.dataset
    known_n = len(ds.labelled_dataset)
    selected = [k for k in plan[0] if k < known_n][:4] + [
        k for k in plan[0] if k >= known_n
    ][:4]
    criterion = DistillLoss(30, 200)
    torch.manual_seed(832)
    random.seed(832)
    np.random.seed(832)
    views, legal, ids, _ = next(iter(DataLoader(Subset(ds, selected), batch_size=8)))
    images = torch.cat(views)
    model.train()
    legal_gpu = legal.to(device="cuda", dtype=torch.long)
    baseline(
        *model(images.cuda()), legal_gpu, legal_gpu >= 0, criterion, EPOCH
    ).backward()
    opt.step()
    named = {
        k: p.detach().cpu().clone()
        for k, p in model.named_parameters()
        if p.requires_grad
    }
    opt, _ = restore(model, state)
    training_step(
        model,
        opt,
        {"tokens": prefix(model[0], images), "labels": legal, "ids": ids},
        criterion,
        no_edit,
    )
    diffs = []
    for k, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.detach().cpu(), named[k], atol=2e-6, rtol=2e-5)
            diffs.append((p.detach().cpu() - named[k]).abs().max().item())
    opt, _ = restore(model, state)
    torch.cuda.reset_peak_memory_stats()
    batch = load_batch(model, ds, plan[0], 1)

    def smoke_move(q, ids, labels):
        for row, label in enumerate(labels):
            source = int(q[:, row].mean(0).argmax())
            if label == -1 and bool((q[:, row, source] >= MASS).all()):
                target = (source + 1) % q.shape[-1]
                return (
                    [Action(row, "move", source, target)],
                    [[row, ids[row], source, target]],
                    {},
                )
        raise ValueError("Smoke batch has no valid unlabeled MOVE")

    edits, dose = training_step(model, opt, batch, criterion, smoke_move)
    assert (
        len(edits) == 1
        and dose["cls_edit_norm"] > 0
        and dose["classifier_edit_norm"] > 0
    )
    result = {
        "small_direct_update_max_abs_error": max(diffs),
        "batch128_peak_bytes": torch.cuda.max_memory_allocated(),
        "batch128_nonfinite": False,
        "batch128_move_cls_grad_norm": dose["cls_edit_norm"],
    }
    write_json(scratch / "smoke.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "scratch", "output"):
        p.add_argument(name, type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--step0-smoke", action="store_true")
    a = p.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        smoke(a.work, a.scratch)
    else:
        run(a.work, a.scratch, a.output, step0_only=a.step0_smoke)
