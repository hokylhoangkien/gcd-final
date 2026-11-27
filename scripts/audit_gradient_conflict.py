"""Paired per-loss and SGD gradient conflict probes for actual CLIP MOVE rows."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F

from crish.repair import Action, corrected_distill
from model import DistillLoss, SupConLoss, info_nce_logits
from relational.sensitivity_metrics import block_gradient
from relational.train import baseline
from scripts import audit_clip_propagation as propagation
from scripts import audit_clip_repair as repair
from scripts import audit_oracle_move as audit
from scripts.audit_e3_proposals import align, json_native

LOCAL = repair.g1.WORK / "gradient-conflict"
OUT = Path("reports/r2_gradient_conflict.json")
REPORT = Path("reports/CRISH_GRADIENT_CONFLICT.md")
MASS = (0.1, 0.2, 0.4, 0.8)
TAGS = ("R150", "T1-155")
GROUPS = ("classifier", "projector", "block11")
SEED = 262620
STUDENT_TEMP = 0.1


def groups(model):
    return (
        (model[1].last_layer.weight_v,),
        tuple(model[1].mlp.parameters()),
        tuple(model[0].blocks[-1].parameters()),
    )


def load(model, state):
    model[0].blocks[-1].requires_grad_(True)
    model[1].mlp.requires_grad_(True)
    model[1].last_layer.weight_v.requires_grad_(True)
    model.eval()
    opt, _ = audit.restore(model, state)
    return opt


def losses(proj, logits, legal, epoch):
    criterion = DistillLoss(30, 200)
    known = legal >= 0
    p = (logits / 0.1).softmax(1).mean(0)
    mixed = torch.cat([v[known] for v in (logits / 0.1).chunk(2)])
    z, target = info_nce_logits(proj, device=proj.device)
    f = torch.stack([v[known] for v in proj.chunk(2)], dim=1)
    terms = {
        "distill": 0.65 * criterion(logits, logits.detach(), epoch),
        "memax": 0.65 * (-(p ** (-p)).log().sum() + np.log(len(p))),
        "known_ce": 0.35 * F.cross_entropy(mixed, legal[known].repeat(2)),
        "infonce": 0.65 * F.cross_entropy(z, target),
        "supcon": 0.35 * SupConLoss()(F.normalize(f, dim=-1), labels=legal[known]),
    }
    total = baseline(proj, logits, legal, known, criterion, epoch)
    torch.testing.assert_close(sum(terms.values()), total)
    return terms, total


class Replay:
    def __init__(self, model, opt, tokens, labels, probe, epoch):
        self.model, self.opt, self.tokens, self.probe, self.epoch = (
            model,
            opt,
            tokens,
            probe,
            epoch,
        )
        self.param = groups(model)
        self.h = audit.encode(model[0], tokens).detach().requires_grad_()
        proj, flat = model[1](self.h)
        self.terms, self.total = losses(proj, flat, labels.cuda().long(), epoch)
        self.logits = flat.reshape(2, len(labels), -1)
        self.source_p = (self.logits.detach() / 0.04).softmax(-1)
        self.group_for = {p: g for g in opt.param_groups for p in g["params"]}

    def gradient(self, objective, backbone_tokens=None, features=None):
        h = self.h if features is None else features
        tokens = self.tokens if backbone_tokens is None else backbone_tokens
        head = sum(self.param[:2], ())
        raw = torch.autograd.grad(objective, head, allow_unused=True, retain_graph=True)
        head = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(raw, head, strict=True)
        ]
        tail = block_gradient(objective, h, self.model[0], tokens, self.param[-1])
        return head[0].flatten(), torch.cat([v.flatten() for v in head[1:]]), tail

    def factual(self, detailed):
        by = {k: self.gradient(v) for k, v in self.terms.items()} if detailed else {}
        return by, self.gradient(self.total)

    def direction(self, row, src, dst):
        return self.gradient(
            (self.logits[:, row, dst] - self.logits[:, row, src]).mean()
        )

    def verify(self, occurrence, factual):
        row, _, src, dst = occurrence
        flat = self.logits.flatten(0, 1)
        criterion = DistillLoss(30, 200)
        reference = 0.65 * (
            corrected_distill(
                flat,
                flat.detach(),
                criterion,
                self.epoch,
                [Action(row, "move", src, dst)],
                0.1,
            )
            - criterion(flat, flat.detach(), self.epoch)
        )
        for actual, d in zip(
            self.gradient(reference), self.direction(row, src, dst), strict=True
        ):
            torch.testing.assert_close(
                actual, -(0.65 * 0.1 / (128 * STUDENT_TEMP)) * d, atol=2e-8, rtol=2e-4
            )
        # Compare functional SGD with PyTorch SGD on independent copies of the actual tensors.
        with torch.no_grad():
            computed, _ = self.step(factual)
            clones = {}
            for params, grad in zip(self.param, factual, strict=True):
                for p, v in zip(
                    params, grad.split([p.numel() for p in params]), strict=True
                ):
                    q = torch.nn.Parameter(p.detach().clone())
                    q.grad = v.view_as(q).clone()
                    clones[p] = q
            state = copy.deepcopy(self.opt.state_dict())
            clone_groups = [
                {**g, "params": [clones[p] for p in g["params"]]}
                for g in self.opt.param_groups
            ]
            actual = torch.optim.SGD(clone_groups)
            actual.load_state_dict(state)
            actual.step()
            for p, q in clones.items():
                torch.testing.assert_close(computed[p], q, atol=1e-7, rtol=2e-6)

    @torch.no_grad()
    def step(self, grads):
        """Functional saved SGD. Parameters and optimizer buffers are read-only."""
        points, delta = {}, []
        for params, flat in zip(self.param, grads, strict=True):
            local = []
            for p, grad in zip(
                params, flat.split([p.numel() for p in params]), strict=True
            ):
                g = self.group_for[p]
                u = grad.view_as(p)
                if g["maximize"]:
                    u = -u
                u = u.add(p, alpha=g["weight_decay"])
                if g["momentum"]:
                    saved = self.opt.state[p].get("momentum_buffer")
                    buf = (
                        u
                        if saved is None
                        else saved.mul(g["momentum"]).add(u, alpha=1 - g["dampening"])
                    )
                    u = u.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                points[p] = p.add(u, alpha=-g["lr"])
                local.append((points[p] - p).flatten())
            delta.append(torch.cat(local))
        return points, tuple(delta)

    @torch.no_grad()
    def outputs(self, points):
        b, head = self.model
        block = b.blocks[-1]
        parameters = {k: points[p] for k, p in block.named_parameters()}
        layer = head.last_layer
        weight = layer.weight_g * F.normalize(points[layer.weight_v], dim=1)

        def encode_logits(tokens):
            h = torch.cat(
                [
                    b.norm(functional_call(block, parameters, (t.cuda(),)))[:, 0]
                    for t in tokens
                ]
            )
            return F.linear(F.normalize(h, dim=1), weight).cpu().numpy()

        scores = encode_logits(self.tokens).reshape(self.logits.shape)
        return scores, encode_logits(self.probe)


def vector_stats(vector, direction, other, sign=-1):
    norm = float(vector.norm())
    d = float(direction.norm())
    scale = float(other.norm())
    return {
        "norm": norm,
        "cosine_to_edit": float(torch.dot(vector, other)) / (norm * scale)
        if norm and scale
        else None,
        "directional_projection": sign * float(torch.dot(vector, direction)) / d
        if d
        else None,
    }


def dynamics(before, keep, moved, row, src, dst, ref_src, ref_dst, neighbors):
    def margin(a, s, d):
        return a[..., d] - a[..., s]

    a, b, c = (margin(v[0][:, row], src, dst) for v in (before, keep, moved))
    u, v = (margin(z[1], src, dst) for z in (keep, moved))
    return {
        "keep_train_change": float((b - a).mean()),
        "move_train_change": float((c - a).mean()),
        "paired_train_gain": float((c - b).mean()),
        "paired_original_proposal_gain": float(
            (
                margin(moved[0][:, row], ref_src, ref_dst)
                - margin(keep[0][:, row], ref_src, ref_dst)
            ).mean()
        ),
        "paired_probe_gain": float((v - u).mean()),
        "paired_neighbor_probe_gain": float((v - u)[neighbors].mean()),
    }


def intervention(
    replay, by, factual, keep_step, before, keep, occurrence, target, masses, neighbors
):
    row, _key, src, dst = occurrence
    assert src != target and int(replay.source_p[:, row].mean(0).argmax()) == src
    assert (replay.source_p[:, row, src] >= max(masses)).all()
    direction = replay.direction(row, src, target)
    outputs, predictions = {}, {}
    for m in masses:
        delta = tuple(
            (-0.65 * m / (replay.logits.shape[1] * STUDENT_TEMP)) * v for v in direction
        )
        moved = tuple(g + d for g, d in zip(factual, delta, strict=True))
        new_points, update = replay.step(moved)
        moved_logits = replay.outputs(new_points)
        summary = {}
        for i, name in enumerate(GROUPS):
            base, alt, edit, anchor = factual[i], moved[i], delta[i], direction[i]
            per = {
                k: {
                    "keep": vector_stats(g[i], anchor, edit),
                    "move": vector_stats(
                        g[i] + edit if k == "distill" else g[i], anchor, edit
                    ),
                }
                for k, g in by.items()
            }
            if name == "projector":
                assert not float(anchor.norm()) and not float(edit.norm())
            per["total"] = {
                "keep": vector_stats(base, anchor, edit),
                "move": vector_stats(alt, anchor, edit),
            }
            lr = replay.group_for[replay.param[i][0]]["lr"]
            per["move_delta"] = vector_stats(edit, anchor, edit)
            per["margin_gradient_norm"] = float(anchor.norm())
            per["sgd_keep"] = vector_stats(keep_step[i], anchor, edit, sign=1)
            per["sgd_move"] = vector_stats(update[i], anchor, edit, sign=1)
            per["sgd_paired"] = vector_stats(
                update[i] - keep_step[i], anchor, edit, sign=1
            )
            per["sgd_keep_minus_naive"] = float((keep_step[i] + lr * base).norm())
            per["cosine_keep_move_total"] = (
                float(F.cosine_similarity(base, alt, dim=0))
                if base.norm() and alt.norm()
                else None
            )
            summary[name] = per
        outputs[str(m)] = {
            "groups": summary,
            "response": dynamics(
                before, keep, moved_logits, row, src, target, src, dst, neighbors
            ),
        }
        predictions[str(m)] = moved_logits[1]
    return outputs, predictions


def choose_batches(events):
    rng = np.random.default_rng(SEED)
    out = []
    for e in (151, 152, 154, 155):
        subset = [v for v in events if v[0] == e and v[2] == "T1"]
        batches = sorted({v[1] for v in subset})
        step = int(rng.choice(batches))
        out.append((e, step, [list(v[3:]) for v in subset if v[1] == step]))
    return out


@torch.no_grad()
def support(model, state, tokens, occurrences):
    model.load_state_dict(state["model"])
    h = audit.encode(model[0], tokens)
    p = (model[1].last_layer(F.normalize(h, dim=1)).reshape(2, 128, -1) / 0.04).softmax(
        -1
    )
    live = p.mean(0).argmax(-1)
    return np.asarray(
        [
            float(p[:, r, s].min()) if int(live[r]) == s else -1.0
            for r, _, s, _ in occurrences
        ]
    )


def prepare(smoke):
    (
        model,
        ds,
        _,
        ids,
        panel,
        panel_chunks,
        _,
        final,
        events,
        signature,
        provenance,
        _,
    ) = propagation.load_inputs()
    checkpoint = audit.load_checkpoint(repair.g1.WORK / "input/epoch-150.pt")
    states = dict(zip(TAGS, (checkpoint, final["T1"]), strict=True))
    for tag in TAGS:
        for k, v in model[0].state_dict().items():
            if not k.startswith("blocks.11."):
                assert torch.equal(v.cpu(), states[tag]["model"][f"0.{k}"]), k
    from relational.train import sampler_for

    loader = torch.utils.data.DataLoader(
        ds, batch_size=128, sampler=sampler_for(ds), drop_last=True
    )
    plans = repair.schedule(loader, ids, panel, checkpoint)
    for e, (rows, seeds) in plans.items():
        assert repair.digest_array(rows) == signature["draws"][str(e)]["rows"]
        assert repair.digest_array(seeds) == signature["draws"][str(e)]["seeds"]
    probe = [t for part in panel_chunks[:2] for t in part]
    probe_ids = panel[: sum(len(t) for t in probe)]
    assert len(probe_ids) == 64
    g1_info = json.loads(repair.g6.G1_REPORT.read_text())
    ref = repair.g4.snapshot(150, g1_info)
    represented = np.unique(
        ref["slots"][np.isin(ref["ids"], signature["reference_ids"])]
    ).tolist()
    batches = []
    for e, s, occurrences in choose_batches(events)[: 1 if smoke else 4]:
        rows, seeds = plans[e]
        path = LOCAL / f"tokens-{e}-{s}.pt"
        stamp = {
            "rows_sha256": repair.digest_array(rows[s - 1]),
            "crop_seed": int(seeds[s - 1, 0]),
            "checkpoint_sha256": provenance["checkpoint_sha256"]["150"],
        }
        if path.exists():
            batch = torch.load(path, map_location="cpu", weights_only=True)
            assert batch["stamp"] == stamp
        else:
            batch = repair.load_batch(model, ds, rows[s - 1], seeds[s - 1, 0])
            batch["stamp"] = stamp
            audit.atomic_checkpoint(batch, path)
        np.testing.assert_array_equal(batch["ids"].numpy(), ids[rows[s - 1]])
        assert not set(batch["ids"].tolist()) & set(probe_ids)
        source_mass = np.stack(
            [support(model, st, batch["tokens"], occurrences) for st in states.values()]
        )
        common = source_mass.min(0)
        order = sorted(
            np.flatnonzero(common >= min(MASS)),
            key=lambda i: (-common[i], occurrences[i][0]),
        )
        selected, keys = [], set()
        for i in order:
            key = occurrences[i][1]
            if key not in keys:
                keys.add(key)
                selected.append(i)
            if len(selected) == (1 if smoke else 2):
                break
        rng = np.random.default_rng(SEED + e * 315 + s)
        items = []
        for i in selected:
            occurrence = occurrences[i]
            targets = sorted(set(represented) - set(occurrence[2:]))
            items.append(
                {
                    "action": occurrence,
                    "random": int(rng.choice(targets)),
                    "masses": [m for m in MASS if common[i] >= m],
                    "source_mass": source_mass[:, i].tolist(),
                }
            )
        batches.append(
            {
                "epoch": e,
                "step": s,
                "ledger_rows": occurrences,
                "both_view_source_mass": source_mass.tolist(),
                "eligible_by_mass": {str(m): int((common >= m).sum()) for m in MASS},
                "items": items,
                "cache": path,
            }
        )
    return model, states, probe, probe_ids, batches, signature, ref, provenance


def context(model, state, batch, probe, detailed):
    opt = load(model, state)
    replay = Replay(
        model,
        opt,
        batch["tokens"],
        batch["labels"],
        probe,
        state["scheduler"]["last_epoch"],
    )
    p = {p: p for pg in groups(model) for p in pg}
    before = replay.outputs(p)
    np.testing.assert_allclose(
        before[0], replay.logits.detach().cpu().numpy(), atol=3e-6, rtol=3e-6
    )
    by, factual = replay.factual(detailed)
    point, step = replay.step(factual)
    keep = replay.outputs(point)
    return replay, by, factual, step, before, keep


def snapshot_batch(
    model, states, probe, spec, probes, oracle_targets=None, *, verify=False
):
    batch = torch.load(spec["cache"], map_location="cpu", weights_only=True)
    rows = []
    for tag, state in states.items():
        replay, by, factual, step, before, keep = context(
            model, state, batch, probe, oracle_targets is None
        )
        if verify and spec["items"]:
            replay.verify(spec["items"][0]["action"], factual)
        cases = []
        for i, item in enumerate(spec["items"]):
            occurrence = item["action"]
            occurrence[0]
            neighbors = probes[i]
            variants = (
                {"MOVE": occurrence[-1], "Random": item["random"]}
                if oracle_targets is None
                else (
                    {} if oracle_targets[i] is None else {"Oracle": oracle_targets[i]}
                )
            )
            response = {}
            for variant, target in variants.items():
                summary, pred = intervention(
                    replay,
                    by,
                    factual,
                    step,
                    before,
                    keep,
                    occurrence,
                    target,
                    item["masses"],
                    neighbors,
                )
                response[variant] = {
                    "target": target,
                    "scores": summary,
                    "probe_logits": pred,
                }
            cases.append({"candidate": i, "variants": response})
        rows.append(
            {
                "checkpoint": tag,
                "cases": cases,
                "probe_before": before[1],
                "probe_keep": keep[1],
            }
        )
    return rows


def neighbor_ids(model, state, spec, probe):
    model.load_state_dict(state["model"])
    batch = torch.load(spec["cache"], map_location="cpu", weights_only=True)
    h = (
        F.normalize(audit.encode(model[0], batch["tokens"]), dim=1)
        .reshape(2, 128, -1)
        .mean(0)
    )
    ref = F.normalize(audit.encode(model[0], probe), dim=1)
    return [
        (h[item["action"][0]] @ ref.T).topk(10).indices.cpu().numpy()
        for item in spec["items"]
    ]


def semantic_map(ids, sig, ref, truth):
    mp, receipt = align(
        sig["reference_ids"],
        dict(zip(ids.tolist(), ref["slots"].tolist(), strict=True)),
        truth,
    )
    table = np.array([mp.get(k, -1) for k in range(196)])
    inverse = {v: k for k, v in mp.items()}
    return table, inverse, receipt


def transfer(logits, keep, targets, labels):
    valid = targets >= 0
    rows = np.flatnonzero(valid)

    def lp(a):
        v = a / 0.1
        return v[rows, targets[rows]] - np.logaddexp.reduce(v[rows], axis=1)

    gain = lp(logits) - lp(keep)
    before = keep[rows].argmax(1) == targets[rows]
    after = logits[rows].argmax(1) == targets[rows]
    results = {}
    for key, mask in propagation.masks(labels[rows]).items():
        results[key] = {
            "n": int(mask.sum()),
            "logp_gain": float(gain[mask].mean()) if mask.any() else None,
            "accuracy_gain": float(after[mask].mean() - before[mask].mean())
            if mask.any()
            else None,
        }
    return results


def seal(model, states, batches, probes, probe, probe_ids, ordinary, signature, ref):
    from scripts.audit_r2_evidence import sealed_labels

    truth = sealed_labels()
    table, inverse, alignment = semantic_map(ref["ids"], signature, ref, truth)
    target = np.asarray([inverse.get(int(y), -1) for y in truth[probe_ids]])
    result = []
    for spec, nbr, rows in zip(batches, probes, ordinary, strict=True):
        labels, destinations = [], []
        for item in spec["items"]:
            _, key, src, dst = item["action"]
            y = truth[key]
            a, b = table[src], table[dst]
            name = (
                "unresolved"
                if min(a, b) < 0
                else "correct"
                if b == y
                else "harmful"
                if a == y
                else "wrong_to_wrong"
            )
            labels.append({"semantic": name, "group": "Old" if y < 98 else "New"})
            slot = inverse.get(int(y))
            destinations.append(slot if slot is not None and slot != src else None)
        oracle = (
            snapshot_batch(model, states, probe, spec, nbr, destinations)
            if any(d is not None for d in destinations)
            else None
        )
        public = []
        for i, r in enumerate(rows):
            moved = []
            for case in r["cases"]:
                idx = case["candidate"]
                if oracle is not None:
                    np.testing.assert_array_equal(
                        r["probe_keep"], oracle[i]["probe_keep"]
                    )
                    case["variants"].update(oracle[i]["cases"][idx]["variants"])
                for v in case["variants"].values():
                    for m, logits in v.pop("probe_logits").items():
                        v["scores"][m]["semantic_probe"] = transfer(
                            logits, r["probe_keep"], target, truth[probe_ids]
                        )
                moved.append({**case, **labels[idx], **spec["items"][idx]})
            public.append({"checkpoint": r["checkpoint"], "cases": moved})
        result.append(
            {k: v for k, v in spec.items() if k not in ("items", "cache")}
            | {"states": public}
        )
    return result, alignment


def aggregate(batches):
    out = {}
    for tag in TAGS:
        candidates = [
            c
            for b in batches
            for r in b["states"]
            if r["checkpoint"] == tag
            for c in r["cases"]
        ]
        groups = {}
        for stratum, pool in (
            ("common_all_masses", [c for c in candidates if len(c["masses"]) == 4]),
            ("available_per_mass", candidates),
            ("oracle_matched", [c for c in candidates if "Oracle" in c["variants"]]),
        ):
            groups[stratum] = {}
            for variant in ("MOVE", "Random", "Oracle"):
                groups[stratum][variant] = {}
                for mass in MASS:
                    m = str(mass)
                    rows = [
                        c
                        for c in pool
                        if variant in c["variants"]
                        and m in c["variants"][variant]["scores"]
                    ]
                    report_rows = {}
                    for group in (
                        "All",
                        "Old",
                        "New",
                        "correct",
                        "harmful",
                        "wrong_to_wrong",
                        "unresolved",
                    ):
                        take = [
                            c["variants"][variant]["scores"][m]
                            for c in rows
                            if group == "All" or group in (c["group"], c["semantic"])
                        ]
                        report_rows[group] = {
                            "n": len(take),
                            "mean_response": {
                                k: float(np.mean([v["response"][k] for v in take]))
                                for k in take[0]["response"]
                            }
                            if take
                            else {},
                        }
                    groups[stratum][variant][m] = report_rows
        out[tag] = groups
    return out


def render(result):
    lines = [
        "# Supervision gradient conflict on Stanford Cars",
        "",
        (
            "Four label-blind draws from batches with actual T1 MOVE ledger entries are replayed at "
            "both R2-150 and T1-155. Each replay uses full batch size 128 and the original paired crops "
            "from the saved continuation seeds. Up to two distinct source IDs per batch are ranked "
            "by the minimum source mass across both views and both endpoints. The probe is disjoint "
            "from all replay batches and reuses 64 frozen-prefix center-crop tokens from the excluded panel."
        ),
        "",
        (
            "Losses are the coefficient-weighted SimGCD distillation, MeMax, known CE, InfoNCE and "
            "SupCon. R2 loss is disabled. Positive projection means the direction of descent raises "
            "the proposed raw destination-minus-source margin to first order, or the logged SGD "
            "displacement raises it when applying the same projection. Cosines compare raw gradients "
            "to the raw edit delta. MOVE only changes distillation. Its differential gradient is "
            "the raw margin gradient times −0.65 × mass / (128 × student temperature 0.1). "
            "Gradients are reused across masses exactly. "
            "The projector does not affect the classifier margin directly."
        ),
        "",
        (
            "Functional SGD reads saved learning rate, momentum and weight decay. Every branch starts at "
            "its checkpoint with the same optimizer state. Nonlinear post-step margin changes are measured "
            "after recomputing block-11 features and logits. Random selects a fixed represented destination "
            "other than source or proposed target for the same ID, dose and replay batch. Oracle uses a "
            "different target only when the initial bank alignment supplies the true slot and the source "
            "is wrong. The oracle branch runs after non-oracle outputs are frozen."
        ),
        "",
        (
            "Tables condition on common four-mass support. JSON separately reports available support "
            "and invalid rows at each strength. Proposal correctness is joined with held-out labels "
            "after freezing non-oracle scores. The replays are endpoint-conditioned and do not recover historical training gradients. "
            "Independent probe transfer includes candidate-slot margin "
            "and true-slot log probability gains. These one-step effects do not identify training effects."
        ),
        "",
        "| Snapshot / arm / mass | Occurrences | Train margin gain vs KEEP | Probe margin gain vs KEEP | Nearest-probe gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, groups in result["summary"].items():
        for variant, by_mass in groups["common_all_masses"].items():
            for m, info in by_mass.items():
                a = info["All"]
                g = a["mean_response"]
                nums = [
                    f"{g[k]:+.3e}" if g else "—"
                    for k in (
                        "paired_train_gain",
                        "paired_probe_gain",
                        "paired_neighbor_probe_gain",
                    )
                ]
                lines.append(
                    f"| {tag} / {variant} / {m} | {a['n']} | " + " | ".join(nums) + " |"
                )
    lines += [
        "",
        "## Gradient contribution at MOVE mass 0.2",
        "",
        "| Snapshot / group / weighted loss | N | Mean factual norm | Mean cosine to edit | Mean factual descent projection |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in TAGS:
        terms = [
            c["variants"]["MOVE"]["scores"]["0.2"]["groups"]
            for b in result["batches"]
            for r in b["states"]
            if r["checkpoint"] == tag
            for c in r["cases"]
            if "0.2" in c["variants"]["MOVE"]["scores"]
        ]
        for g in GROUPS:
            for loss in ("distill", "memax", "known_ce", "infonce", "supcon", "total"):
                entries = [v[g][loss]["keep"] for v in terms]
                nums = [
                    [v[k] for v in entries if v[k] is not None]
                    for k in ("norm", "cosine_to_edit", "directional_projection")
                ]
                cells = [f"{np.mean(v):+.3e}" if v else "—" for v in nums]
                lines.append(
                    f"| {tag} / {g} / {loss} | {len(entries)} | "
                    + " | ".join(cells)
                    + " |"
                )
    lines += [
        "",
        f"Runtime {result['seconds']:.1f} seconds.",
        "",
        "[Per-loss KEEP/MOVE vectors summarized by group, validity, raw/SGD differences and sealed transfer](r2_gradient_conflict.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--smoke", action="store_true")
    smoke = arg.parse_args().smoke
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    model, states, probe, probe_ids, batches, signature, ref, provenance = prepare(
        smoke
    )
    nearest = [neighbor_ids(model, states["R150"], spec, probe) for spec in batches]
    ordinary = []
    for spec, nbr in zip(batches, nearest, strict=True):
        ordinary.append(snapshot_batch(model, states, probe, spec, nbr, verify=smoke))
        print(
            "Non-oracle replay",
            spec["epoch"],
            spec["step"],
            "occurrences",
            len(spec["items"]),
            flush=True,
        )
    if smoke:
        assert ordinary[0][0]["cases"]
        print("Actual full-objective paired SGD smoke passed", flush=True)
        return
    assert any(spec["items"] for spec in batches)
    path = LOCAL / "non-oracle-frozen.pt"
    audit.atomic_checkpoint(
        {"batches": batches, "nearest": nearest, "ordinary": ordinary}, path
    )
    result = {
        "base": "262e620",
        "checkpoint_sha256": {
            str(e): provenance["checkpoint_sha256"][str(e)] for e in (150, 155)
        },
        "clip_policy_sha256": provenance["source_policy_sha256"],
        "seed": SEED,
        "non_oracle_sha256": audit.file_digest(path),
        "probe_ids": probe_ids.tolist(),
        "masses": MASS,
        "learning_rates": {
            t: [pg["lr"] for pg in st["optimizer"]["param_groups"]]
            for t, st in states.items()
        },
    }
    audit.write_json(LOCAL / "frozen-provenance.json", json_native(result))
    result["batches"], result["alignment"] = seal(
        model, states, batches, nearest, probe, probe_ids, ordinary, signature, ref
    )
    result["summary"] = aggregate(result["batches"])
    result["seconds"] = time.monotonic() - start
    render(result)
    audit.write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
