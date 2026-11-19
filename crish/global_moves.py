"""CRiSH-G Move: base-level reassignment from global neighbor shares, propagated to coarse levels.

A low-agreement unlabeled row i (r^0 < tau) moves to the other base cluster b
holding the largest neighbor weight when share(b) >= move_min_share, b and the
source keep min_support, b receives at most max(1, floor(cap*support_b)) rows,
and departures (Move + Abstain) respect the per-cluster Abstain quota. Coarse
labels follow b's non-moved members, so the hierarchy stays a function of L0.

Protected coefficients (same-ID other view; labeled-labeled pairs) equal
1/(2 n_c - 1) for population cluster support n_c. A Move is skipped when it
would shift any of them, at any level, by more than max_protected_shift. This
is the exact population quantity; scoring-batch shifts are only logged.
"""
from dataclasses import replace

import torch

from .rebuild import prototype_radii


def neighbor_shares(state, evidence, rows):
    """Best other base cluster and its share of the row's total neighbor weight."""
    labels = state.labels[0]
    clusters = len(state.prototypes[0])
    best, share = [], []
    for row in rows:
        neighbors, weights = evidence.neighbors[row], evidence.weights[row]
        tally = torch.zeros(clusters, dtype=torch.float64).index_add_(0, labels[neighbors], weights)
        tally[int(labels[row])] = -1.
        target = int(tally.argmax())  # first maximum: ties go to the smaller cluster ID
        best.append(target)
        share.append(float(tally[target]/weights.sum()) if tally[target] > 0 else 0.)
    return best, share


def protected(support):
    return 1./(2.*support-1.) if support > 0 else 0.


def apply_moves(state, evidence, tau, settings, *, min_support=2, max_protected_shift=.25):
    """Return (moved_state, moved_mask, move_target, record); labeled rows never move."""
    if type(min_support) is not int or min_support < 1:
        raise ValueError('Positive minimum cluster support required')
    n = len(state.ids)
    labels0 = state.labels[0]
    clusters = len(state.prototypes[0])
    unlabeled = ~state.labeled
    candidates = (unlabeled & (evidence.reliability[0] < tau)).nonzero().flatten().tolist()
    best, share = neighbor_shares(state, evidence, candidates)
    r0 = evidence.reliability[0]
    order = sorted(range(len(candidates)), key=lambda i: (-share[i], float(r0[candidates[i]]),
                                                          int(state.ids[candidates[i]])))
    support = torch.bincount(labels0, minlength=clusters)
    current = support.clone()
    received = torch.zeros(clusters, dtype=torch.long)
    target_cap = torch.clamp((support.double()*settings.move_target_cap).floor().long(), min=1)
    # Departure quotas at every level, identical to the Abstain quota.
    quotas, departures = [], []
    for level, labels in enumerate(state.labels):
        u = torch.bincount(labels[unlabeled], minlength=len(state.prototypes[level]))
        quotas.append(torch.minimum((u.double()*settings.max_cluster_fraction).floor().long(),
                                    (u-settings.min_cluster_kept).clamp_min(0)))
        departures.append(torch.zeros_like(quotas[-1]))
    # Coarse level of each base cluster, from its members (a function of L0).
    parents = []
    for labels in state.labels:
        parent = torch.full((clusters,), -1, dtype=torch.long)
        parent[labels0] = labels
        if not torch.equal(parent[labels0], labels):
            raise ValueError('CRiSH-G Move requires coarse labels to be a function of L0')
        parents.append(parent)
    original = [torch.bincount(labels, minlength=len(state.prototypes[level])).tolist()
                for level, labels in enumerate(state.labels)]
    running = [list(x) for x in original]
    moved = torch.zeros(n, dtype=torch.bool)
    move_target = torch.full((n,), -1, dtype=torch.long)
    counts = dict(move_candidates=len(candidates), move_rejected_share=0, move_rejected_support=0,
                  move_rejected_target_cap=0, move_rejected_source_quota=0, move_rejected_protected_shift=0)
    for i in order:
        row, source, target = candidates[i], int(labels0[candidates[i]]), best[i]
        if share[i] < settings.move_min_share or target == source:
            counts['move_rejected_share'] += 1; continue
        if int(support[target]) < min_support or int(current[source])-1 < min_support:
            counts['move_rejected_support'] += 1; continue
        if received[target] >= target_cap[target]:
            counts['move_rejected_target_cap'] += 1; continue
        leaving = [(level, int(state.labels[level][row])) for level in range(len(state.labels))
                   if int(parents[level][target]) != int(state.labels[level][row])]
        if any(departures[level][group] >= quotas[level][group] for level, group in leaving):
            counts['move_rejected_source_quota'] += 1; continue
        shift = 0.
        for level, group in leaving:
            new_group = int(parents[level][target])
            before_src, before_dst = original[level][group], original[level][new_group]
            after_src, after_dst = running[level][group]-1, running[level][new_group]+1
            shift = max(shift, abs(protected(before_src)-protected(after_src)),
                        abs(protected(before_dst)-protected(after_dst)),
                        abs(protected(before_src)-protected(after_dst)))  # the moved row itself
        if shift > max_protected_shift:
            counts['move_rejected_protected_shift'] += 1; continue
        for level, group in leaving:
            departures[level][group] += 1
            running[level][group] -= 1; running[level][int(parents[level][target])] += 1
        moved[row] = True; move_target[row] = target
        received[target] += 1; current[source] -= 1; current[target] += 1
    if not moved.any():
        return state, moved, move_target, dict(counts, moved=0)
    labels, centers, radii = list(state.labels), list(state.prototypes), list(state.radii)
    rows = moved.nonzero().flatten()
    for level in range(len(state.labels)):
        # b's level-l label, looked up from b's non-moved members.
        carrier = torch.full((clusters,), -1, dtype=torch.long)
        carrier[labels0[~moved]] = labels[level][~moved]
        new = labels[level].clone()
        new[rows] = carrier[move_target[rows]]
        if (new[rows] < 0).any():
            raise ValueError('Move target has no non-moved member to carry its coarse label')
        diff = new[rows] != labels[level][rows]
        changed = torch.cat([labels[level][rows][diff], new[rows][diff]]).unique()
        if len(changed):
            proto = centers[level].clone()
            for group in changed.tolist():
                members = new == group
                if members.any():
                    proto[group] = state.features[members].mean(0)
            centers[level] = proto
            radii[level] = prototype_radii(proto)
        labels[level] = new
    proposed = replace(state, labels=tuple(labels), prototypes=tuple(centers), radii=tuple(radii),
                       version=state.version+1)
    proposed.validate()
    if not torch.equal(proposed.labels[0][state.labeled], state.known_targets):
        raise ValueError('Move changed legally labeled anchors')
    for level in range(len(labels)):
        if not torch.equal(proposed.labels[level][~moved], state.labels[level][~moved]):
            raise ValueError('Move changed a non-moved row')
    return proposed, moved, move_target, dict(counts, moved=int(moved.sum()),
                                              per_target_received=received[received > 0].tolist())
