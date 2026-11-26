"""Exact gradient bases and one-step feature responses for a fixed Cars batch."""

import torch
import torch.nn.functional as F

from relational.train import baseline

COEFFICIENTS = (0.1, 0.5, 1.0, 2.0, 5.0)
MAGNITUDES = (0.25, 0.5)


@torch.no_grad()
def prefix(backbone, images):
    chunks = []
    for micro in images.split(8):
        x = backbone.prepare_tokens(micro.cuda())
        for layer in list(backbone.blocks)[:-1]:
            x = layer(x)
        chunks.append(x.cpu())
    return chunks


def cls(backbone, chunk):
    return backbone.norm(backbone.blocks[-1](chunk.cuda()))[:, 0]


def block_gradient(objective, features, backbone, tokens, params):
    (incoming,) = torch.autograd.grad(objective, features, retain_graph=True)
    total = [torch.zeros_like(p) for p in params]
    for chunk, grad in zip(tokens, incoming.split(8), strict=True):
        local = torch.autograd.grad(cls(backbone, chunk), params, grad)
        for dst, value in zip(total, local, strict=True):
            dst.add_(value)
    return torch.cat([g.flatten() for g in total])


def gradient_metrics(vector, base):
    return {
        "norm": vector.norm().item(),
        "ratio": (vector.norm() / base.norm()).item(),
        "cos_base": F.cosine_similarity(vector, base, dim=0).item(),
    }


def health(unit_features):
    return {
        "variance": unit_features.var(dim=0, unbiased=False).mean().item(),
        "resultant": unit_features.mean(dim=0).norm().item(),
    }


def rms(x):
    return x.square().mean().sqrt().item()


def effect(x):
    return (
        {
            "count": x.numel(),
            "mean": x.mean().item(),
            "rms": rms(x),
            "fraction_positive": (x > 0).double().mean().item(),
        }
        if x.numel()
        else {"count": 0}
    )


def measure_batch(
    backbone, heads, graph, criterion, images, legal, ids, mask, probe, probe_ids
):
    params = list(backbone.blocks[-1].parameters())
    sizes = [p.numel() for p in params]
    initial = torch.cat([p.detach().flatten() for p in params])
    decay = torch.cat(
        [
            (
                5e-5 * p.detach()
                if p.ndim > 1 and not name.endswith(".bias")
                else torch.zeros_like(p)
            ).flatten()
            for name, p in backbone.blocks[-1].named_parameters()
        ]
    )
    tokens, panel_tokens = prefix(backbone, images), prefix(backbone, probe)

    @torch.no_grad()
    def encode(chunks):
        return torch.cat([cls(backbone, x) for x in chunks])

    @torch.no_grad()
    def install(vector):
        for param, part in zip(params, vector.split(sizes), strict=True):
            param.copy_(part.view_as(param))

    @torch.no_grad()
    def poststep(gradient):
        install(initial - 0.1 * (gradient + decay))
        # FP32 model forward; accumulate cosine differences in FP64.
        return tuple(
            F.normalize(encode(group).double(), dim=1)
            for group in (tokens, panel_tokens)
        )

    h = encode(tokens).requires_grad_()
    panel_before = F.normalize(encode(panel_tokens).double(), dim=1)
    initial_health = {
        "batch": health(F.normalize(h.detach().double(), dim=1)),
        "panel": health(panel_before),
    }
    edited, plain, (slots, _) = graph.loss(h, ids)
    keep_grad = block_gradient(plain, h, backbone, tokens, params)
    signed_grad = block_gradient(
        (edited - plain) / graph.settings.magnitude, h, backbone, tokens, params
    )
    # Both heads use exactly the same unchanged backbone; cache all gradients before virtual updates.
    bases = []
    for head in heads:
        loss = baseline(*head(h), legal, mask, criterion, 10)
        bases.append(block_gradient(loss, h, backbone, tokens, params))

    rows = [graph.lookup.get(key, len(graph.ids)) for key in ids.tolist()]
    peers = graph.keys[graph.peer_slots[rows].cuda()].double()
    sign = (graph.weights - graph.base)[slots.cuda()].sign()
    counts = {
        key: int(cond.sum())
        for key, cond in (
            ("strengthen", sign > 0),
            ("weaken", sign < 0),
            ("keep", (sign == 0) & (slots.cuda() != 0)),
        )
    }
    panel_rows = [graph.lookup[key] for key in probe_ids.tolist()]
    pslots = graph.edge_slots[panel_rows].cuda() != 0
    panel_peers = graph.keys[graph.peer_slots[panel_rows].cuda()].double()
    # Train-independent panel top-5 on the original center features; keep pair memberships frozen.
    sim = panel_before @ panel_before.T
    sim.fill_diagonal_(-torch.inf)
    nearest = torch.zeros_like(sim, dtype=torch.bool)
    nearest.scatter_(1, sim.topk(5, dim=1).indices, True)
    distant = ~(nearest | torch.eye(len(probe), device=sim.device, dtype=torch.bool))
    result = {
        "unlabeled_draws": int((~mask).sum()),
        "edge_occurrences": counts,
        "health_before": initial_health,
        "snapshots": {},
    }
    try:
        for name, base in zip(("fresh", "learned_head"), bases, strict=True):
            original_post, panel_original_post = poststep(base)
            data = {
                "base_norm": base.norm().item(),
                "r0_post": health(original_post),
                "r0_panel_post": health(panel_original_post),
                "settings": [],
            }
            for lam in COEFFICIENTS:
                weighted_keep = lam * keep_grad
                kept, panel_kept = poststep(base + weighted_keep)
                if lam == COEFFICIENTS[0]:
                    repeat, panel_repeat = poststep(base + weighted_keep)
                    if not (
                        torch.equal(repeat, kept)
                        and torch.equal(panel_repeat, panel_kept)
                    ):
                        raise ValueError(
                            "Identical virtual updates produced different features"
                        )
                    data["repeated_keep_max_abs_difference"] = 0.0
                for delta in MAGNITUDES:
                    edit = (lam * delta) * signed_grad
                    rel = weighted_keep + edit
                    after, panel_after = poststep(base + rel)
                    diff = after - kept
                    shift = torch.einsum(
                        "vbd,bkd->vbk", diff.reshape(2, len(ids), -1), peers
                    )
                    panel_diff = panel_after - panel_kept
                    pair_change = (
                        panel_after @ panel_after.T - panel_kept @ panel_kept.T
                    )
                    abs_mass = delta * (counts["strengthen"] + counts["weaken"])
                    row = {
                        "lambda": lam,
                        "delta": delta,
                        "r1": gradient_metrics(weighted_keep, base),
                        "r2": gradient_metrics(rel, base),
                        "edit": gradient_metrics(edit, base),
                        "edit_cos_keep": F.cosine_similarity(
                            edit, weighted_keep, dim=0
                        ).item(),
                        "total_r2_norm": (base + rel).norm().item(),
                        "total_r1_norm": (base + weighted_keep).norm().item(),
                        "cls_l2_rms": rms(diff.norm(dim=1)),
                        "pairwise_cos_rms": rms(after @ after.T - kept @ kept.T),
                        "strengthen": effect(shift[:, sign > 0]),
                        "weaken": effect(shift[:, sign < 0]),
                        "unchanged_edge": effect(
                            shift[:, (sign == 0) & (slots.cuda() != 0)]
                        ),
                        "panel_cls_l2_rms": rms(panel_diff.norm(dim=1)),
                        "panel_reciprocal": effect(
                            torch.einsum("bd,bkd->bk", panel_diff, panel_peers)[pslots]
                        ),
                        "panel_near": effect(pair_change[nearest]),
                        "panel_other": effect(pair_change[distant]),
                        "keep_health": health(kept),
                        "r2_health": health(after),
                        "panel_keep_health": health(panel_kept),
                        "panel_r2_health": health(panel_after),
                        "absolute_mass": abs_mass,
                        "effective_mass": lam
                        * abs_mass
                        / (len(ids) * graph.settings.k),
                    }
                    data["settings"].append(row)
            result["snapshots"][name] = data
    finally:
        install(initial)
    if not torch.equal(initial, torch.cat([p.detach().flatten() for p in params])):
        raise ValueError("Diagnostic block restoration failed")
    return result
