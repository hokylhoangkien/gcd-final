"""Sequential frozen-snapshot repair with atomic commits and complete re-scoring.

The scorer is a trusted adapter of rebuilt training targets to LegalProbe scores.
This package supplies no oracle/data loader. External scorer provenance must be
audited at integration; Python cannot prove a callback's closure is label-free.
"""
from dataclasses import dataclass, replace
import math
from typing import Callable, Sequence, Tuple

from .candidates import Action
from .rebuild import rebuild
from .state import SupervisionState


class RepairBudgetExhausted(RuntimeError):
    """A bounded stop, distinct from an invalid counterfactual or scorer error."""

    def __init__(self, message, *, reason='time_budget'):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class DecisionEvidence:
    fold_gains: Tuple[float, ...]
    confirmation_gain: float
    known_increase: float
    changed_mass: float

    def passes(self, threshold: float, agreement: float, known_tolerance: float) -> bool:
        """Discovery screen only; confirmation cannot influence candidate ranking."""
        values = self.fold_gains + (self.known_increase, self.changed_mass)
        if not self.fold_gains or not all(math.isfinite(x) for x in values) or self.changed_mass <= 0:
            return False
        return (sum(self.fold_gains) / len(self.fold_gains) > threshold and
                sum(x > 0 for x in self.fold_gains) / len(self.fold_gains) >= agreement and
                self.known_increase <= known_tolerance)


def sequential_repair(initial: SupervisionState, actions: Sequence[Action],
                      scorer: Callable[[SupervisionState, SupervisionState, Action], DecisionEvidence], *,
                      max_accepted: int, threshold: float, agreement: float,
                      known_tolerance: float, min_support: int = 1, enabled: bool = True,
                      confirmer=None, budget_check=None, diagnostics=None, require_confirmation=True):
    if type(max_accepted) is not int or max_accepted < 0 or not all(math.isfinite(x) for x in (threshold, agreement, known_tolerance)) or not 0 <= agreement <= 1 or known_tolerance < 0:
        raise ValueError('Invalid predeclared acceptance/budget settings')
    current, pending, trace = initial, list(actions), []
    summary = diagnostics if diagnostics is not None else {}
    summary.update(stop_reason=None, rounds=0, scoring_attempts=0, confirmation_attempts=0,
                   accepted=0, cap_reached=False, budget_exhausted=False)

    def finish(reason, detail=None):
        summary.update(stop_reason=reason, accepted=current.version-initial.version,
                       cap_reached=reason == 'acceptance_cap',
                       budget_exhausted=reason in ('time_budget', 'probe_budget'))
        if detail is not None:
            summary['stop_detail'] = str(detail)
        return current, trace

    def check_budget():
        if budget_check is not None:
            budget_check()

    if not enabled or max_accepted == 0:
        return finish('disabled')
    # Each scoring round starts from current, never independently scored updates.
    for round_index in range(max_accepted):
        if not any(a.kind != 'keep' for a in pending):
            return finish('no_candidates' if not trace else 'candidates_exhausted')
        try:
            check_budget()
        except RepairBudgetExhausted as exc:
            return finish(exc.reason, exc)
        summary['rounds'] += 1
        ranked = []
        for action in pending:
            if action.kind == 'keep':
                continue
            if action.version != initial.version:
                trace.append(dict(kind=action.kind, sample_id=action.sample_id,
                                  level=action.level, version=action.version,
                                  status='failed', reason='Stale initial action snapshot'))
                continue
            # A prior accepted edit may make a source invalid. Only the snapshot
            # version is refreshed; validate_action still checks source/support.
            fresh = replace(action, version=current.version)
            try:
                check_budget()
                trial, record = rebuild(current, fresh, min_support=min_support)
                # Scorer receives disposable copies; failures cannot mutate state.
                summary['scoring_attempts'] += 1
                evidence = scorer(current.copy(), trial.copy(), fresh)
                check_budget()
                accepted = evidence.passes(threshold, agreement, known_tolerance)
                finite_or_none = lambda value: float(value) if math.isfinite(value) else None
                record.update(status='eligible' if accepted else 'rejected',
                              fold_gains=[finite_or_none(x) for x in evidence.fold_gains],
                              confirmation_gain=finite_or_none(evidence.confirmation_gain),
                              known_increase=finite_or_none(evidence.known_increase),
                              changed_mass=finite_or_none(evidence.changed_mass))
                if accepted:
                    ranked.append((sum(evidence.fold_gains) / len(evidence.fold_gains), fresh, trial, record))
            except RepairBudgetExhausted as exc:
                return finish(exc.reason, exc)
            except (ValueError, RuntimeError) as exc:
                record = dict(kind=fresh.kind, sample_id=fresh.sample_id, level=fresh.level,
                              version=current.version, status='failed', reason=str(exc))
            record['round'] = round_index
            trace.append(record)
        if not ranked:
            return finish('no_eligible_candidates')
        _, chosen, trial, record = max(ranked, key=lambda item: item[0])
        if confirmer is not None:
            try:
                check_budget()
                summary['confirmation_attempts'] += 1
                confirmation = confirmer(current.copy(), trial.copy(), chosen)
                gain, known = confirmation['gain'], confirmation['known_increase']
                record['confirmation_gain'] = float(gain) if math.isfinite(gain) else None
                record['confirmation_known_increase'] = float(known) if math.isfinite(known) else None
                if not math.isfinite(known) or known > known_tolerance:
                    record.update(status='rejected', reason='Independent confirmation known guard failed')
                    return finish('confirmation_known_guard')
                check_budget()
            except RepairBudgetExhausted as exc:
                record.update(status='uncommitted', reason=str(exc))
                return finish(exc.reason, exc)
            except (ValueError, RuntimeError) as exc:
                record.update(status='failed', reason=str(exc))
                return finish('confirmation_error', exc)
        # Inspect the independent confirmation value only for the top discovery
        # candidate. If it fails, stop; never search runners-up on that fold.
        if require_confirmation and (record['confirmation_gain'] is None or record['confirmation_gain'] <= threshold):
            record.update(status='rejected', reason='Independent confirmation failed; refresh stopped')
            return finish('confirmation_failed')
        if not require_confirmation:
            record['confirmation_gain'] = None
        try:
            trial.validate()
        except (ValueError, RuntimeError) as exc:
            record.update(status='failed', reason=str(exc))
            return finish('invalid_trial', exc)
        current = trial  # atomic commit; initial is untouched
        record['status'] = 'accepted'
        pending = [a for a in pending if a.sample_id != chosen.sample_id]
        # All remaining actions are rebuilt and rescored, including global USE
        # dependencies. No cached independently accepted gains are composed.
    return finish('acceptance_cap')
