"""Detached Euclidean two-view evidence on distinct current minibatch IDs.

Only measured edges may be downweighted. Non-neighbors are unknown, not
negatives. No class probabilities, hidden labels or persistent cluster IDs.
"""
import torch
import torch.nn.functional as F

from .config import PairReliabilitySettings
from .ids import integer_ids


@torch.no_grad()
def online_pair_weights(backbone_views, batch_ids, labeled, settings: PairReliabilitySettings, epoch):
    strength = settings.strength(epoch)
    diagnostics = dict(enabled=settings.enabled, strength=strength, status='off' if not settings.enabled else 'warmup')
    if strength == 0 or settings.floor == 1:
        return None, diagnostics
    ids = integer_ids(batch_ids).to(backbone_views.device)
    n = len(ids)
    if (not n or backbone_views.ndim != 2 or len(backbone_views) != 2*n
            or not backbone_views.is_floating_point() or not torch.isfinite(backbone_views).all()):
        raise ValueError('Online reliability requires finite [2B,D] backbone views')
    labeled = torch.as_tensor(labeled, device=ids.device)
    if labeled.shape != (n,) or labeled.dtype != torch.bool:
        raise ValueError('Online reliability requires boolean legal membership')
    unique, inverse, counts = ids.unique(sorted=True, return_inverse=True, return_counts=True)
    known_count = torch.zeros(len(unique), device=ids.device).index_add_(0, inverse, labeled.float())
    if ((known_count > 0) & (known_count < counts)).any():
        raise ValueError('Duplicate ID has conflicting labeled membership')
    diagnostics.update(unique_ids=len(unique), repeated_rows=n-len(unique))
    if len(unique) < 3:
        diagnostics['status'] = 'insufficient_unique_ids_keep'
        return None, diagnostics
    views, valid = [], torch.ones(len(unique), device=ids.device, dtype=torch.bool)
    for features in backbone_views.detach().float().chunk(2):
        valid &= torch.zeros(len(unique), device=ids.device).index_add_(
            0, inverse, (features.norm(dim=1) > 1e-8).float()) == counts
        encoded = F.normalize(features, dim=-1)
        means = encoded.new_zeros(len(unique), encoded.shape[1]).index_add_(0, inverse, encoded)
        means /= counts[:, None]
        if means.var(dim=0, unbiased=False).sum() <= 1e-8:
            diagnostics['status'] = 'collapsed_view_keep'
            return None, diagnostics
        valid &= means.norm(dim=1) > 1e-8
        views.append(F.normalize(means, dim=-1))
    # Invalid/cancelled vectors provide no evidence, including as references.
    active = valid.nonzero().flatten()
    if len(active) < 3:
        diagnostics['status'] = 'insufficient_valid_ids_keep'
        return None, diagnostics
    k = min(settings.k, len(active)-1)
    adjacency = []
    for features in views:
        similarity = features[active] @ features[active].T
        similarity.fill_diagonal_(-torch.inf)
        neighbor = similarity.topk(k, dim=1).indices
        graph = torch.zeros((len(unique), len(unique)), dtype=torch.bool, device=ids.device)
        graph[active[:, None], active[neighbor]] = True
        adjacency.append(graph)
    a, b = adjacency
    measured = a | a.T | b | b.T
    rn = ((a & a.T).float() + (b & b.T).float())/2
    cvs = ((a & b) | (a.T & b.T)).float()
    score = torch.ones_like(rn)
    if settings.reciprocal:
        score *= rn
    if settings.cross_view:
        score *= cvs
    unique_weights = torch.where(measured, 1-strength*(1-settings.floor)*(1-score), 1.)
    weights = unique_weights[inverse[:, None], inverse[None, :]]
    protected = (ids[:, None] == ids[None, :]) | (labeled[:, None] & labeled[None, :])
    weights.masked_fill_(protected, 1.)
    diagnostics.update(status='active', k=k, measured_edges=int(measured.sum())//2,
                       invalid_unique_ids=int((~valid).sum()))
    return weights.detach(), diagnostics


@torch.no_grad()
def pair_diagnostics(weights, labels, ids, labeled, diagnostics, *, retained=None):
    """Summarize only eligible pseudo-positives; no unknown truth required."""
    result = dict(diagnostics)
    labels, ids, labeled = (x.detach().cpu() for x in (labels, integer_ids(ids), labeled))
    eligible = ((labels[:, None] == labels[None, :]) & (ids[:, None] != ids[None, :])
                & ~(labeled[:, None] & labeled[None, :]))
    original_count = int(eligible.sum())
    if retained is not None:
        if retained.shape != eligible.shape or retained.dtype != torch.bool:
            raise ValueError('Expected boolean retained pseudo-positive mask')
        eligible &= retained.detach().cpu()
    values = torch.ones(int(eligible.sum())) if weights is None else weights.detach().cpu()[eligible]
    result.update(original_eligible_pairs=original_count,
                  excluded_by_abstention=original_count-int(eligible.sum()),
                  eligible_pairs=len(values), changed_pairs=int((values < 1-1e-7).sum()),
                  weight_sum=float(values.sum()),
                  mean_weight=float(values.mean()) if len(values) else None,
                  weight_histogram=torch.histc(values, bins=5, min=0, max=1).long().tolist())
    for name, membership in (('labeled_anchor', labeled), ('unlabeled_anchor', ~labeled)):
        selected = eligible & membership[:, None]
        values = torch.ones(int(selected.sum())) if weights is None else weights.detach().cpu()[selected]
        result[name+'_mean_weight'] = float(values.mean()) if len(values) else None
    return result
