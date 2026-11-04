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
                      confirmer=None):
    if type(max_accepted) is not int or max_accepted < 0 or not all(math.isfinite(x) for x in (threshold, agreement, known_tolerance)) or not 0 <= agreement <= 1 or known_tolerance < 0:
        raise ValueError('Invalid predeclared acceptance/budget settings')
    if not enabled or max_accepted == 0:
        return initial, []
    current, pending, trace = initial, list(actions), []
    # Each scoring round starts from current, never independently scored updates.
    for _ in range(max_accepted):
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
                trial, record = rebuild(current, fresh, min_support=min_support)
                # Scorer receives disposable copies; failures cannot mutate state.
                evidence = scorer(current.copy(), trial.copy(), fresh)
                accepted = evidence.passes(threshold, agreement, known_tolerance)
                finite_or_none = lambda value: float(value) if math.isfinite(value) else None
                record.update(status='eligible' if accepted else 'rejected',
                              fold_gains=[finite_or_none(x) for x in evidence.fold_gains],
                              confirmation_gain=finite_or_none(evidence.confirmation_gain),
                              known_increase=finite_or_none(evidence.known_increase),
                              changed_mass=finite_or_none(evidence.changed_mass))
                if accepted:
                    ranked.append((sum(evidence.fold_gains) / len(evidence.fold_gains), fresh, trial, record))
            except (ValueError, RuntimeError) as exc:
                record = dict(kind=fresh.kind, sample_id=fresh.sample_id, level=fresh.level,
                              version=current.version, status='failed', reason=str(exc))
            trace.append(record)
        if not ranked:
            break
        _, chosen, trial, record = max(ranked, key=lambda item: item[0])
        if confirmer is not None:
            try:
                confirmation = confirmer(current.copy(), trial.copy(), chosen)
                gain, known = confirmation['gain'], confirmation['known_increase']
                record['confirmation_gain'] = float(gain) if math.isfinite(gain) else None
                record['confirmation_known_increase'] = float(known) if math.isfinite(known) else None
                if not math.isfinite(known) or known > known_tolerance:
                    record.update(status='rejected', reason='Independent confirmation known guard failed')
                    break
            except (ValueError, RuntimeError) as exc:
                record.update(status='failed', reason=str(exc))
                break
        # Inspect the independent confirmation value only for the top discovery
        # candidate. If it fails, stop; never search runners-up on that fold.
        if record['confirmation_gain'] is None or record['confirmation_gain'] <= threshold:
            record.update(status='rejected', reason='Independent confirmation failed; refresh stopped')
            break
        trial.validate()
        current = trial  # atomic commit; initial is untouched
        record['status'] = 'accepted'
        pending = [a for a in pending if a.sample_id != chosen.sample_id]
        # All remaining actions are rebuilt and rescored, including global USE
        # dependencies. No cached independently accepted gains are composed.
    return current, trace
