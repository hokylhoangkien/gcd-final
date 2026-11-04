"""Bounded label-free candidates and conservative action dependency graph."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .ids import rows_for_ids
from .reliability import EdgeEvidence, edge_score
from .state import SupervisionState


@dataclass(frozen=True)
class Action:
    kind: str
    sample_id: Optional[int] = None
    level: int = 0
    source: Optional[int] = None
    target: Optional[int] = None
    version: int = 0

    def __post_init__(self):
        if self.kind not in ('keep', 'abstain', 'move'):
            raise ValueError('Unknown action')


def validate_action(state: SupervisionState, action: Action, min_support: int = 1) -> Optional[int]:
    if type(min_support) is not int or min_support < 1:
        raise ValueError('Positive minimum cluster support required')
    if action.version != state.version:
        raise ValueError('Stale action snapshot')
    if action.kind == 'keep':
        return None
    if type(action.level) is not int or not 0 <= action.level < len(state.labels):
        raise ValueError('Invalid level')
    row = int(rows_for_ids(state.ids, [action.sample_id])[0])
    if state.labeled[row]:
        raise ValueError('Legally labeled assignments are protected at every level')
    labels = state.labels[action.level]
    source = int(labels[row])
    if action.source != source:
        raise ValueError('Stale source cluster')
    if action.kind == 'abstain' and state.abstained[action.level][row]:
        raise ValueError('Already abstained; no target change')
    if action.kind == 'move':
        if type(action.target) is not int or not 0 <= action.target < len(state.prototypes[action.level]) or action.target == source:
            raise ValueError('Move requires a distinct existing destination')
        if int((labels == source).sum()) - 1 < min_support or int((labels == action.target).sum()) < min_support:
            raise ValueError('Move would empty or under-support a cluster')
    return row


def propose_actions(state: SupervisionState, sample_id: int, level: int,
                    evidence: Dict[Tuple[int, int], EdgeEvidence], *,
                    max_targets: int, min_support: int, min_edges: int,
                    min_reliability: float) -> List[Action]:
    if type(max_targets) is not int or max_targets < 0 or type(min_edges) is not int or min_edges < 1:
        raise ValueError('Bounded target budget and positive edge support required')
    if not 0 < min_reliability <= 1:
        raise ValueError('Move needs positive measured reciprocal evidence')
    if type(level) is not int or not 0 <= level < len(state.labels):
        raise ValueError('Invalid level')
    row = int(rows_for_ids(state.ids, [sample_id])[0])
    keep = Action('keep', version=state.version)
    if state.labeled[row]:
        return [keep]
    source = int(state.labels[level][row])
    output = [keep, Action('abstain', sample_id, level, source, version=state.version)]
    distances = (state.prototypes[level] - state.features[row]).square().sum(1)
    for target in distances.argsort().tolist():
        if len(output) - 2 >= max_targets:
            break
        action = Action('move', sample_id, level, source, target, state.version)
        try:
            validate_action(state, action, min_support)
        except ValueError:
            continue
        supporters = [int(state.ids[j]) for j in (state.labels[level] == target).nonzero().flatten()]
        support = sum((edge_score(evidence, sample_id, j) or 0) >= min_reliability for j in supporters)
        if support >= min_edges:
            output.append(action)
    return output


def action_dependencies(action: Action) -> set:
    if action.kind == 'keep':
        return set()
    deps = {('sample', action.sample_id), ('sse_level_rows', action.level)}
    if action.kind == 'move':
        deps.update({('prototype', action.level, action.source), ('prototype', action.level, action.target),
                     ('all_use_normalization',)})
    return deps


def conflicts(first: Action, second: Action) -> bool:
    return bool(action_dependencies(first) & action_dependencies(second))
