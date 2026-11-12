"""Bounded scoring context with pseudo-cluster peers, excluding all probes."""
import torch

from .ids import rows_for_ids


def scoring_ids(state, actions, adjacency, pool, reserved, budget):
    """Reserve action rows, then source/destination peers before cache fill.

    Peer selection consults frozen features and existing pseudo-assignments
    only. It does not add measured evidence edges or relax Move legality.
    Two peers for Move reduce tiny-batch denominator artifacts; actual drift
    is still checked by the unchanged protected-coefficient guard.
    """
    excluded = set(state.ids[reserved].tolist())
    available = [int(x) for x in pool.tolist() if int(state.ids[x]) not in excluded]
    chosen = list(dict.fromkeys(a.sample_id for a in actions))
    if len(chosen) > budget or any(x in excluded for x in chosen):
        raise ValueError('Scoring actions exceed budget or overlap reserved probes')
    rows_for_ids(state.ids, chosen)
    cached = set(available)
    requirements = []
    for action in actions:
        row = int(rows_for_ids(state.ids, [action.sample_id])[0])
        labels = state.labels[action.level]
        groups = [action.source] + ([action.target] if action.kind == 'move' else [])
        needed = 2 if action.kind == 'move' else 1
        for group in groups:
            candidates = [i for i in (labels == group).nonzero().flatten().tolist()
                          if i != row and int(state.ids[i]) not in excluded
                          and not bool(state.abstained[action.level][i])]
            if candidates:
                distances = (state.features[candidates]-state.features[row]).square().sum(1)
                candidates = sorted(zip(candidates, distances.tolist()),
                                    key=lambda pair: (pair[0] not in cached, pair[1], int(state.ids[pair[0]])))
                for peer, _ in candidates[:needed]:
                    requirements.append(int(state.ids[peer]))
    for sample in requirements:
        if sample not in chosen and len(chosen) < budget:
            chosen.append(sample)
    for action in actions:
        for sample, _ in sorted(adjacency.get(action.sample_id, []), key=lambda item: -item[1]):
            if sample not in chosen and sample not in excluded and len(chosen) < budget:
                chosen.append(sample)
    for row in available:
        sample = int(state.ids[row])
        if sample not in chosen and len(chosen) < budget:
            chosen.append(sample)
    ids = torch.tensor(chosen, dtype=torch.long)
    selected_rows = rows_for_ids(state.ids, ids)
    support = []
    for action in actions:
        labels = state.labels[action.level][selected_rows]
        other = ids != action.sample_id
        support.append(dict(sample_id=action.sample_id, level=action.level, kind=action.kind,
                            source_peers=int(((labels == action.source) & other).sum()),
                            target_peers=int(((labels == action.target) & other).sum())
                            if action.target is not None else None))
    return ids, dict(selection='action_rows_then_pseudo_cluster_peers_then_evidence',
                     rows=len(ids), added_support_ids=[x for x in chosen if x not in set(state.ids[pool].tolist())],
                     action_support=support)
