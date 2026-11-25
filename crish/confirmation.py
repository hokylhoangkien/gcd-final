"""Fresh-fold confirmation of a fixed discovery ranking."""

import math

import torch
import torch.nn.functional as F

MAX_ATTEMPTS = 12


@torch.no_grad()
def sign_audit(virtual, factual, trial, fold, family_size):
    """One vote per known-class block, with both views of each ID kept together."""
    x, labels, anchor = fold
    known = labels >= 0
    groups = labels[known].unique(sorted=True)
    blocks = len(groups)
    if not blocks or int((~known).sum()) % blocks:
        return {"status": "unequal_blocks"}

    def block_losses(v):
        logp = (virtual.logits(x, v) / 0.1).log_softmax(-1)
        y = labels[known].repeat(2)
        known_ce = (
            F.nll_loss(logp[:, known].flatten(0, 1), y, reduction="none")
            .reshape(2, -1)
            .mean(0)
        )
        known_means = torch.stack([known_ce[labels[known] == k].mean() for k in groups])
        unlabeled_ce = -(anchor.flip(0)[:, ~known] * logp[:, ~known]).sum(-1).mean(0)
        return known_means + unlabeled_ce.reshape(blocks, -1).mean(1)

    delta = block_losses(factual) - block_losses(trial)
    if not torch.isfinite(delta).all():
        return {"status": "nonfinite"}
    positive = int((delta > 0).sum())
    p = sum(math.comb(blocks, k) for k in range(positive, blocks + 1)) / 2**blocks
    return {
        "blocks": blocks,
        "positive": positive,
        "p_one_sided": p,
        "p_bonferroni": min(1.0, family_size * p),
    }


def confirm_ranked(
    ranked,
    *,
    fallback,
    limit,
    rejected,
    rec,
    clock,
    get_probe,
    probe,
    step,
    accepted,
    current,
    original,
    threshold,
    round_index,
    audit,
):
    """Never use confirmation scores to reorder the remaining alternatives."""
    for rank, item in enumerate(ranked, 1):
        if rec["confirmations"] >= limit:
            rec["stop"] = "confirmation_budget"
            return None
        action, edit, trial, new_base, trace = item
        fold_index = rec["confirmations"] + 1
        if fallback:
            attempt = {
                "attempt": fold_index,
                "fold": fold_index,
                "round": round_index,
                "rank": rank,
                "id": action.sample_id,
                "kind": action.kind,
                "source": action.source,
                "target": action.target,
                "gain": trace["gain"],
                "marginal": None,
                "group": None,
                "known_delta": None,
                "group_known_delta": None,
                "result": "started",
            }
            rec["confirmation_attempts"].append(attempt)
            rec["confirmations"] += 1
        if trial is None:
            with clock.stage("discovery"):
                trial = step(accepted + [edit])
        fresh = get_probe(fold_index)
        with clock.stage("confirmation"):
            b, bk = probe(current, *fresh)
            o, ok = probe(original, *fresh) if accepted else (b, bk)
            c, ck = probe(trial, *fresh)
            if not fallback:
                rec["confirmations"] += 1
            reason = "nonfinite_confirmation"
            if all(math.isfinite(v) for v in (b, bk, o, ok, c, ck)):
                trace.update(
                    confirmation=b - c,
                    group_confirmation=o - c,
                    confirmation_known_delta=ck - bk,
                )
                reason = None
                if min(b - c, o - c) <= threshold:
                    reason = "gain_veto"
                elif ck > min(bk, ok):
                    reason = "known_veto"
                if fallback:
                    attempt.update(
                        marginal=b - c,
                        group=o - c,
                        known_delta=ck - bk,
                        group_known_delta=ck - ok,
                        sign_audit=audit(current, trial, fresh, limit),
                    )
            if reason is None:
                clock.check()
                if fallback:
                    attempt["result"] = "accepted"
                return action, edit, trial, new_base, trace
            if not fallback:
                rec["stop"] = (
                    "nonfinite_confirmation"
                    if reason == "nonfinite_confirmation"
                    else "confirmation"
                )
                return None
            attempt["result"] = reason
            rejected.add(action)
    rec["stop"] = (
        "confirmation_budget"
        if rec["confirmations"] >= limit
        else "confirmation_ranking_exhausted"
    )
    return None
