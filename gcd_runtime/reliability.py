"""Experimental hierarchy-level reliability; never accepts evaluation targets.

Neighbors use the pinned upstream Poincare distance. Scores are detached and
symmetric. Repeated sample IDs share a deterministic representative for scoring.
"""
from typing import Dict, Tuple
import torch
from torch import Tensor
from .method_config import ReliabilitySettings



def _validate(features: Tensor, sample_ids: Tensor) -> None:
    if features.ndim != 3 or features.shape[1] != 2 or features.shape[2] == 0 or len(features) == 0:
        raise ValueError('Expected nonempty-dimensional [batch,2,dim] features')
    if sample_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Sample IDs must be integer tensors')
    if sample_ids.ndim != 1 or len(sample_ids) != len(features):
        raise ValueError('Sample IDs must be [batch]')
    if not torch.isfinite(features).all():
        raise ValueError('Non-finite reliability input')


@torch.no_grad()
def neighborhood_scores(features: Tensor, sample_ids: Tensor, curvature: float,
                        settings: ReliabilitySettings) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Feature-only score in [0,1]; no class labels or probabilities are used."""
    _validate(features, sample_ids)
    if curvature <= 0:
        raise ValueError('Positive Poincare curvature required')
    from hyptorch.pmath import dist_matrix
    features = features.detach().float()
    if not (features.norm(dim=-1) < curvature ** -0.5).all():
        raise ValueError('Features outside the Poincare ball')
    ids = sample_ids.to(features.device)
    unique, inverse = torch.unique(ids, return_inverse=True)
    count = len(unique)
    first = torch.full((count,), len(ids), dtype=torch.long, device=features.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(ids), device=features.device),
                          reduce='amin', include_self=True)
    representatives = features[first]
    neighbors = []
    k = min(settings.k, max(0, count - 1))
    for view in range(2):
        distance = dist_matrix(representatives[:, view], representatives[:, view], c=curvature)
        if not torch.isfinite(distance).all():
            raise ValueError('Non-finite hyperbolic neighborhood distance')
        distance.fill_diagonal_(float('inf'))
        adjacency = torch.zeros((count, count), dtype=torch.bool, device=features.device)
        if k:
            # Stable tie breaking by sorted unique sample ID, without RNG consumption.
            indices = torch.argsort(distance, dim=1, stable=True)[:, :k]
            adjacency.scatter_(1, indices, True)
        neighbors.append(adjacency)
    a, b = neighbors
    reciprocal = ((a & a.T).float() + (b & b.T).float()) * 0.5
    overlap = (a & b).sum(1).float()
    union = (a | b).sum(1).float()
    agreement = overlap / union.clamp_min(1)
    stable_edge = (a & b) | (a & b).T
    cross_view = stable_edge.float() * torch.sqrt(agreement[:, None] * agreement[None, :])
    score = torch.ones_like(reciprocal)
    if settings.reciprocal:
        score *= reciprocal
    if settings.cross_view:
        score *= cross_view
    score.fill_diagonal_(1.0)  # same-ID views are protected, never mined as neighbors
    expanded = score[inverse][:, inverse].clamp(0, 1)
    return expanded, {'agreement': agreement[inverse], 'unique_samples': torch.tensor(count),
                      'reciprocal': reciprocal[inverse][:, inverse],
                      'cross_view': cross_view[inverse][:, inverse]}


@torch.no_grad()
def pair_weights(score: Tensor, pseudo_labels: Tensor, labeled: Tensor,
                 sample_ids: Tensor, settings: ReliabilitySettings, epoch: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Modify only existing pseudo-positive pairs in the selected membership scope.

    Rows with no supported eligible pair fall back to baseline. Self/duplicate-ID
    pairs and labeled-labeled pairs always have weight 1. No true labels are used.
    """
    n = len(pseudo_labels)
    device = score.device
    if score.shape != (n, n) or labeled.shape != (n,) or sample_ids.shape != (n,):
        raise ValueError('Inconsistent pair shapes')
    if labeled.dtype != torch.bool:
        raise ValueError('labeled must be a boolean membership mask')
    if not torch.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise ValueError('Invalid reliability score')
    pseudo_labels, labeled, sample_ids = (t.to(device) for t in (pseudo_labels, labeled, sample_ids))
    positive = pseudo_labels[:, None].eq(pseudo_labels[None, :])
    different_id = ~sample_ids[:, None].eq(sample_ids[None, :])
    if settings.pair_scope == 'either_unlabeled':
        scope = ~(labeled[:, None] & labeled[None, :])
    else:
        scope = ~labeled[:, None] & ~labeled[None, :]
    eligible = positive & different_id & scope
    support = (score * eligible).sum(1) > 0
    # Symmetric fallback: if either anchor lacks supported eligible pairs, protect
    # its incident pairs too. This prevents one row's fallback breaking symmetry.
    active = eligible & support[:, None] & support[None, :]
    strength = settings.strength(epoch)
    target = settings.floor + (1 - settings.floor) * score
    weights = torch.where(active, 1 - strength + strength * target, torch.ones_like(score))
    return weights.detach(), eligible, ~support


def weighted_supcon(features: Tensor, pseudo_labels: Tensor, weights: Tensor,
                    criterion, curvature: float) -> Tensor:
    """Same positive set, logits and temperature as upstream; weighted anchor mean.

    All-one weights delegate to upstream for exact baseline value/gradient parity.
    Changed weights use logsumexp to avoid exp underflow. The denominator over
    contrast examples is unchanged; only positive averaging is reweighted.
    """
    n = len(features)
    if weights.shape != (n,n) or not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError('Invalid pair weights')
    if bool(torch.all(weights == 1)):
        return criterion(features, labels=pseudo_labels)
    if features.ndim != 3 or features.shape[1] != 2 or len(features) == 0:
        raise ValueError('Weighted SupCon expects nonempty [batch,2,dim]')
    if criterion.contrast_mode != 'all':
        raise ValueError('Only audited contrast_mode=all is supported')
    from hyptorch.pmath import dist_matrix
    n = len(features)
    flat = torch.cat(torch.unbind(features, dim=1), dim=0)
    distance = torch.cdist(flat, flat) if criterion.hyper_c == 0 else dist_matrix(flat, flat, c=curvature)
    logits = -distance / criterion.temperature
    diagonal = torch.eye(2 * n, dtype=torch.bool, device=features.device)
    log_prob = logits - torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    positive = pseudo_labels[:, None].eq(pseudo_labels[None, :]).to(features.device).repeat(2, 2)
    positive &= ~diagonal
    weighted = weights.to(features.device).repeat(2, 2).detach() * positive
    mass = weighted.sum(1)
    if not (mass > 0).all():
        raise ValueError('Positive mass vanished; protected self-view invariant broken')
    per_anchor = -(weighted * log_prob).sum(1) / mass
    return (criterion.temperature / criterion.base_temperature) * per_anchor.mean()
