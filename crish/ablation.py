"""Opt-in local policy controls; no evaluation targets or global RNG access."""
import random

from .candidates import propose_actions
from .policy import RepairBudgetExhausted
from .rebuild import rebuild

from .config import ABLATION_VARIANTS as VARIANTS


def order_boundaries(boundaries, variant, seed):
    ordered = sorted(boundaries, key=lambda item: (-item[0], item[1], item[2]))
    if variant == 'random_proposal':
        random.Random(seed).shuffle(ordered)
    return ordered


def filter_actions(actions, variant):
    excluded = {'no_abstain': 'abstain', 'no_move': 'move'}.get(variant)
    return [a for a in actions if a.kind != excluded]


def supported_actions(state, action, evidence, settings):
    return propose_actions(state, action.sample_id, action.level, evidence,
        max_targets=settings.max_targets, min_support=settings.min_support,
        min_edges=settings.min_edges, min_reliability=settings.min_reliability)


def evidence_repair(state, actions, evidence, settings, check_budget, guard):
    """Priority order, supported Move else Abstain; sequential structural guards.

    Failure of a chosen Move guard leaves Keep; it does not introduce an
    unplanned Abstain fallback. Recheck evidence after each committed rebuild.
    """
    current, trace, used = state, [], set()
    stop = 'candidates_exhausted'
    for action in actions:
        if action.sample_id in used:
            continue
        if current.version - state.version >= settings.max_accepted:
            stop = 'acceptance_cap'
            break
        used.add(action.sample_id)
        try:
            check_budget()
            legal = supported_actions(current, action, evidence, settings)
            # Revalidation may discover other destinations after earlier edits;
            # keep the original candidate pool shared with the CF variants.
            original = {(a.kind, a.level, a.target) for a in actions if a.sample_id == action.sample_id}
            legal = [a for a in legal if (a.kind, a.level, a.target) in original]
            chosen = next((a for a in legal if a.kind == 'move'),
                          next((a for a in legal if a.kind == 'abstain'), None))
            if chosen is None:
                raise ValueError('No eligible evidence action')
            trial, record = rebuild(current, chosen, min_support=settings.min_support)
            guard(current, trial, chosen)
            trial.validate()
            check_budget()
            current = trial
            record.update(status='accepted', policy='evidence_only')
        except RepairBudgetExhausted as exc:
            stop = exc.reason
            break
        except (ValueError, RuntimeError) as exc:
            record = dict(kind=action.kind, sample_id=action.sample_id, level=action.level,
                          source=action.source, target=action.target, version=current.version,
                          status='rejected', reason=str(exc))
        trace.append(record)
    count = current.version-state.version
    if count == settings.max_accepted:
        stop = 'acceptance_cap'
    return current, trace, dict(stop_reason=stop, accepted=count,
        cap_reached=stop == 'acceptance_cap', budget_exhausted=stop in ('time_budget', 'probe_budget'))
