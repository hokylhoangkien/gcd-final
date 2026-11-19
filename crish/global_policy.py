"""CRiSH-G: global reciprocal-kNN agreement and thresholded multi-level abstention.

Evidence uses only frozen normalized backbone features and the supervision
snapshot's own labels (legal L0 anchors for labeled rows, pseudo-labels for
unlabeled rows). No hidden unlabeled or evaluation labels are read. Every
function here is deterministic and never consumes the global RNG streams.
"""
from dataclasses import dataclass, replace
import time
from typing import Optional, Tuple

import torch
from torch import Tensor

from .reliability import blockwise_query_knn

HISTOGRAM_BINS = 10


@dataclass(frozen=True)
class GlobalEvidence:
    neighbors: Tensor          # [N, k] row indices into the state, self excluded
    weights: Tensor            # [N, k] float64: 1 mutual, nonreciprocal_weight otherwise
    reliability: Tuple[Tensor, ...]  # per active level, [N] float64 in [0, 1]
    mutual_fraction: float
    seconds: float


def agreement(labels: Tensor, neighbors: Tensor, weights: Tensor) -> Tensor:
    """r_i = sum_j w_ij 1[L(j)=L(i)] / sum_j w_ij, exactly, in float64."""
    same = (labels[neighbors] == labels[:, None]).double()
    return (weights*same).sum(1)/weights.sum(1)


@torch.no_grad()
def knn_evidence(state, *, k: int, nonreciprocal_weight: float, device='cpu', query_block: int = 256,
                 reference_block: int = 4096, deadline: Optional[float] = None) -> GlobalEvidence:
    """Exact blockwise kNN over ALL N train rows; no dense N-by-N matrix."""
    started = time.monotonic()
    n = len(state.ids)
    if type(k) is not int or not 1 <= k < n:
        raise ValueError('CRiSH-G needs 1 <= knn_k < N')
    if not 0 < nonreciprocal_weight <= 1:
        raise ValueError('Nonreciprocal edge weight must be in (0,1]')
    features = state.features.to(device).float()
    neighbors = blockwise_query_knn(features, torch.arange(n), k, query_block=query_block,
                                    reference_block=reference_block, deadline=deadline).long()
    if neighbors.shape != (n, k) or (neighbors == torch.arange(n)[:, None]).any():
        raise ValueError('Invalid global neighbor table')
    mutual = (neighbors[neighbors] == torch.arange(n)[:, None, None]).any(-1)
    weights = torch.where(mutual, torch.ones((), dtype=torch.float64),
                          torch.full((), float(nonreciprocal_weight), dtype=torch.float64))
    reliability = tuple(agreement(labels, neighbors, weights) for labels in state.labels)
    for r in reliability:
        if not torch.isfinite(r).all() or ((r < 0) | (r > 1)).any():
            raise ValueError('Nonfinite or out-of-range CRiSH-G reliability')
    return GlobalEvidence(neighbors, weights, reliability, float(mutual.double().mean()),
                          time.monotonic()-started)


def histogram(values: Tensor):
    return torch.histc(values.double(), bins=HISTOGRAM_BINS, min=0., max=1.).long().tolist()


def _capacity(labels, unlabeled, clusters, settings):
    support = torch.bincount(labels[unlabeled], minlength=clusters)
    return torch.minimum((support.double()*settings.max_cluster_fraction).floor().long(),
                         (support-settings.min_cluster_kept).clamp_min(0))


def _mass_decrements(labels, clusters, n_rows):
    """Exact decrement of base_target_mass when one more active row of c abstains."""
    support = torch.bincount(labels, minlength=clusters).double()
    def decrement(cluster, active):
        return 4.*(active-1)/(max(2.*float(support[cluster])-1., 1.)*n_rows)
    return support, decrement


def _greedy(order, labels, capacity, used, limit=None):
    selected, capped = [], 0
    for row in order:
        group = int(labels[row])
        if used[group] < capacity[group]:
            selected.append(row); used[group] += 1
            if limit is not None and len(selected) == limit:
                break
        else:
            capped += 1
    return selected, capped


def exact_mass(labels, abstained, clusters):
    """Same arithmetic as diagnostics.base_target_mass, without building a state."""
    n = torch.bincount(labels, minlength=clusters).double()
    m = torch.bincount(labels[~abstained], minlength=clusters).double()
    return float(((m*(2*m-1)+(n-m))/(2*n-1).clamp_min(1)).sum()/len(labels))


def _mass_floor(state, level, labels, selected, floor):
    """Keep the longest r-ordered prefix that keeps base_target_mass >= floor.

    One more abstention in cluster c (support n_c, active m_c) lowers the mass
    by exactly 4(m_c-1)/((2n_c-1)N); the prefix is then re-verified exactly.
    """
    clusters = len(state.prototypes[level])
    support, decrement = _mass_decrements(labels, clusters, len(labels))
    active = support.clone()
    mass, kept = 1., []
    for row in selected:
        group = int(labels[row])
        step = decrement(group, float(active[group]))
        if mass-step < floor:
            break
        mass -= step; active[group] -= 1; kept.append(row)
    while kept:
        mask = torch.zeros(len(labels), dtype=torch.bool); mask[kept] = True
        if exact_mass(labels, mask, clusters) >= floor:
            break
        kept.pop()
    return kept, len(selected)-len(kept)


def scoped_levels(state, settings):
    return [0] if settings.levels == 'base' else list(range(len(state.labels)))


def random_generator(seed, epoch, level):
    return torch.Generator().manual_seed((int(seed)*1000003+int(epoch)*103+int(level)*7919+29) % (2**32))


def apply_threshold(state, evidence: GlobalEvidence, tau: float, settings, *, min_support: int = 2,
                    max_protected_shift: float = .25, random_seed=None):
    """pi_tau: Abstain unlabeled rows with r^l < tau, per scoped level, under guards.

    Returns (proposed_state, coverage). tau == 0 returns the identical state.
    Labeled rows are never touched; per-cluster quotas and the exact mass
    floor are enforced at every level; ties order by sample ID.
    """
    if not 0 <= tau <= 1:
        raise ValueError('Threshold must be in [0,1]')
    if len(evidence.reliability) != len(state.labels):
        raise ValueError('Evidence levels differ from the supervision snapshot')
    if any(a.any() for a in state.abstained):
        raise ValueError('CRiSH-G requires a fresh unmodified supervision snapshot')
    unlabeled = ~state.labeled
    n_unlabeled = int(unlabeled.sum())
    moved = torch.zeros(len(state.ids), dtype=torch.bool)
    move_target = torch.full((len(state.ids),), -1, dtype=torch.long)
    move_record = dict(moved=0)
    current = state
    if tau > 0 and settings.actions == 'abstain_move':
        from .global_moves import apply_moves
        current, moved, move_target, move_record = apply_moves(state, evidence, tau, settings,
                                                               min_support=min_support,
                                                               max_protected_shift=max_protected_shift)
    scope = scoped_levels(state, settings) if tau > 0 else []
    masks = [torch.zeros(len(state.ids), dtype=torch.bool) for _ in state.labels]
    levels = []
    id_order = state.ids.argsort(stable=True)
    for level in range(len(state.labels)):
        r = evidence.reliability[level]
        entry = dict(level=level, scoped=level in scope, candidates=0, abstained_unlabeled=0, fraction=0.,
                     cluster_capped=0, truncated_by_mass_floor=0, retained_mass=1.,
                     reliability_histogram=histogram(r[unlabeled]),
                     reliability_mean_unlabeled=float(r[unlabeled].mean()) if n_unlabeled else None)
        levels.append(entry)
        if level not in scope:
            continue
        labels = current.labels[level]
        clusters = len(state.prototypes[level])
        capacity = _capacity(state.labels[level], unlabeled, clusters, settings)
        used = torch.zeros_like(capacity)
        # Rows that left a cluster by Move consume that cluster's quota.
        for row in moved.nonzero().flatten().tolist():
            source = int(state.labels[level][row])
            if int(labels[row]) != source:
                used[source] += 1
        eligible = unlabeled & ~moved & (r < tau)
        entry['candidates'] = int(eligible.sum())
        rows = id_order[eligible[id_order]]
        rows = rows[r[rows].argsort(stable=True)].tolist()
        # Capacity is indexed by the pre-move source labels, as in the quota.
        selected, capped = _greedy(rows, labels, capacity, used)
        kept, truncated = _mass_floor(current, level, labels, selected, settings.min_retained_mass)
        entry.update(cluster_capped=capped, truncated_by_mass_floor=truncated, evidence_selected=len(kept))
        if settings.sample_choice == 'random':
            if random_seed is None:
                raise ValueError('Random same-dose control needs an explicit (seed, epoch)')
            generator = random_generator(*random_seed, level)
            pool = (unlabeled & ~moved) if level == 0 else masks[0].clone()
            pool_rows = id_order[pool[id_order]]
            pool_rows = pool_rows[torch.randperm(len(pool_rows), generator=generator)].tolist()
            chosen = _greedy(pool_rows, labels, capacity, torch.zeros_like(capacity), limit=len(kept))[0] if kept else []
            chosen, extra = _mass_floor(current, level, labels, chosen, settings.min_retained_mass)
            entry.update(random_shortfall=len(kept)-len(chosen), truncated_by_mass_floor=extra)
            kept = chosen
        masks[level][kept] = True
        entry.update(abstained_unlabeled=len(kept), fraction=len(kept)/max(1, n_unlabeled))
    if not any(m.any() for m in masks) and not moved.any():
        coverage = dict(move_record, threshold=float(tau), levels=levels, changed=False, moved=0)
        return state, coverage
    proposed = replace(current, abstained=tuple(masks), version=state.version+1)
    proposed.validate()
    from .diagnostics import base_target_mass
    for entry in levels:
        entry['retained_mass'] = base_target_mass(proposed, entry['level'])
        if entry['retained_mass'] < settings.min_retained_mass:
            raise ValueError('CRiSH-G policy violated the retained-mass floor')
    coverage = dict(move_record, threshold=float(tau), levels=levels, changed=True, moved=int(moved.sum()))
    return proposed, coverage


def epoch_arrays(before, after, evidence: GlobalEvidence):
    """Post-hoc arrays; no hidden labels. Labels are the pre-intervention partition.

    Only Move changes labels, so moved rows and their level-0 targets are exact.
    """
    n = len(before.ids)
    moved = (after.labels[0] != before.labels[0]) & ~before.labeled
    move_target = torch.where(moved, after.labels[0], torch.full((n,), -1, dtype=torch.long))
    arrays = dict(ids=before.ids.numpy(), labeled=before.labeled.numpy(),
                  moved=moved.numpy(), move_target=move_target.numpy())
    for level in range(len(before.labels)):
        arrays[f'label_{level}'] = before.labels[level].numpy()
        arrays[f'reliability_{level}'] = evidence.reliability[level].to(torch.float16).numpy()
        arrays[f'abstained_{level}'] = after.abstained[level].numpy()
    return arrays
