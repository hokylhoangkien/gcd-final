"""Bounded evidence/probe sampling from training-safe frozen features only."""
import torch
import torch.nn.functional as F

from .reliability import blockwise_query_knn


def balanced_known_folds(rows, targets, count, generator):
    """Two ID-disjoint round-robin samples using legally labeled targets only.

    Classes are randomized separately per fold, then cycled while rows remain.
    Rare classes are never duplicated or invented to fill a requested quota.
    """
    if (rows.ndim != 1 or targets.shape != rows.shape or
            rows.dtype != torch.long or targets.dtype != torch.long or
            len(rows.unique()) != len(rows) or type(count) is not int or count < 1
            or len(rows) < 2*count):
        raise ValueError('Insufficient/invalid labeled rows for two balanced folds')
    groups = targets.unique(sorted=True)
    queues = []
    for group in groups:
        members = rows[targets == group]
        queues.append(members[torch.randperm(len(members), generator=generator)].tolist())
    folds = []
    for _ in range(2):
        order = torch.randperm(len(groups), generator=generator).tolist()
        chosen = []
        while len(chosen) < count:
            for group in order:
                if queues[group]:
                    chosen.append(queues[group].pop())
                    if len(chosen) == count:
                        break
        folds.append(torch.tensor(chosen, dtype=torch.long))
    return folds


def global_neighbor_pool(state, available, budget, k, generator, *,
                         reference_block=1024, deadline=None):
    """Cache-budgeted neighborhoods shortlisted against all non-probe rows.

    Unlabeled anchors are sampled without hidden class identities. Each anchor's
    k nearest frozen-backbone neighbors is considered before random fill. The
    later augmented-view graph is local to this cache, not global-view kNN.
    """
    if (available.ndim != 1 or available.dtype != torch.long or not len(available)
            or len(available.unique()) != len(available)
            or ((available < 0) | (available >= len(state.ids))).any()
            or type(budget) is not int or budget < 1
            or type(k) is not int or not 1 <= k < len(available)):
        raise ValueError('Invalid available evidence rows or neighbor/cache budget')
    budget = min(budget, len(available))
    if budget <= k:
        raise ValueError('Evidence cache must exceed k')
    unlabeled = (~state.labeled[available]).nonzero().flatten()
    if not len(unlabeled):
        raise ValueError('No unlabeled evidence anchors remain after reserving probes')
    query_count = min(len(unlabeled), max(1, budget // (k+1)))
    anchors = unlabeled[torch.randperm(len(unlabeled), generator=generator)[:query_count]]
    features = F.normalize(state.features[available], dim=-1)
    neighbors = blockwise_query_knn(features, anchors, k, reference_block=reference_block, deadline=deadline)
    # Reserve every query first; only then include reference neighbors.
    chosen = anchors.tolist()
    used = set(chosen)
    for group in neighbors.tolist():
        for row in group:
            if row not in used and len(chosen) < budget:
                chosen.append(row)
                used.add(row)
    if len(chosen) < budget:
        for row in torch.randperm(len(available), generator=generator).tolist():
            if row not in used:
                chosen.append(row)
                used.add(row)
                if len(chosen) == budget:
                    break
    return available[torch.tensor(chosen, dtype=torch.long)]
