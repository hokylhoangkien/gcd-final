"""Faithful Euclidean USE rebuild and mass-preserving protected SSE abstention.

Derived from HypCD train_HypSelEx.py (MIT), pinned in audit/baseline_trace.py.
Move semantics are explicit in presentation/CRISH_GCD.md; this module never reclusters or uses labels
of D_U. Keep preserves supplied k-means centers, even if not assignment means.
"""
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor

from .candidates import Action, validate_action
from .ids import rows_for_ids
from .state import SupervisionState


@torch.no_grad()
def prototype_radii(centers: Tensor) -> Tensor:
    if centers.ndim != 2 or len(centers) < 2 or centers.shape[1] == 0 or not centers.is_floating_point():
        raise ValueError(f'Invalid prototypes for radii: expected floating [K>=2,D>0], got {tuple(centers.shape)}')
    if not torch.isfinite(centers).all():
        invalid = (~torch.isfinite(centers).all(dim=1)).nonzero().flatten().tolist()
        raise ValueError(f'Invalid prototypes for radii: nonfinite cluster IDs={invalid}')
    distance = torch.cdist(centers, centers)
    return (distance + torch.eye(len(centers), device=centers.device) * distance.max()).min(dim=1)[0] / 2


def rebuild(state: SupervisionState, action: Action, *, min_support: int = 1,
            cluster_momentum: float = 1.0):
    row = validate_action(state, action, min_support)
    if action.kind == 'keep':
        return state, dict(kind='keep', version=state.version, potential_changed_rows=0)
    if cluster_momentum != 1:
        raise ValueError('Verified repair contract requires cluster_momentum=1')
    labels, centers, radii, abstained = [list(x) for x in (state.labels, state.prototypes, state.radii, state.abstained)]
    ell = action.level
    before = torch.bincount(labels[ell], minlength=len(centers[ell]))
    record = dict(kind=action.kind, sample_id=action.sample_id, level=ell, version=state.version,
                  source=action.source, target=action.target,
                  source_support=int(before[action.source]),
                  origin_prototype=centers[ell][action.source].tolist())
    if action.kind == 'abstain':
        abstained[ell] = abstained[ell].clone()
        abstained[ell][row] = True
        active_partners = (labels[ell] == action.source) & ~state.abstained[ell]
        record['changed_global_pseudo_pairs'] = 2 * (int(active_partners.sum()) - 1)
    else:
        record.update(target_support=int(before[action.target]), target_prototype=centers[ell][action.target].tolist())
        record['changed_global_pseudo_pairs'] = 2 * (int(before[action.source]) - 1 + int(before[action.target]))
        labels[ell], centers[ell] = labels[ell].clone(), centers[ell].clone()
        labels[ell][row] = action.target
        for group in (action.source, action.target):
            centers[ell][group] = state.features[labels[ell] == group].mean(0)
        radii[ell] = prototype_radii(centers[ell])
        record.update(source_support_after=int((labels[ell] == action.source).sum()),
                      target_support_after=int((labels[ell] == action.target).sum()),
                      source_prototype_after=centers[ell][action.source].tolist(),
                      target_prototype_after=centers[ell][action.target].tolist())
    trial = replace(state, labels=tuple(labels), prototypes=tuple(centers), radii=tuple(radii),
                    abstained=tuple(abstained), version=state.version + 1)
    # USE row normalization can depend globally on the changed level.
    record['potential_changed_rows'] = len(state.ids) if action.kind == 'move' else 1
    return trial, record


@torch.no_grad()
def confusion_factor(state: SupervisionState, batch_ids: Tensor, backbone_views: Tensor,
                     strategy: str = 'zero_one', *, normalization: str = 'legacy') -> Tensor:
    """Default source geometry; two-view order [view0 batch, view1 batch].

    Legacy preserves the source's column broadcasting and zero-row rejection.
    Row mode excludes self before normalization; empty support falls back only
    to the other augmented view of the same image, without consulting labels.
    """
    if strategy not in ('zero_one', 'pair_dist', 'pair_cluster'):
        raise ValueError('Unsupported USE strategy')
    if normalization not in ('legacy', 'row', 'selex_legacy'):
        raise ValueError('Unsupported USE target normalization')
    rows = rows_for_ids(state.ids, batch_ids)
    n = len(rows)
    if not n or backbone_views.ndim != 2 or len(backbone_views) != 2 * n or backbone_views.shape[1] != state.features.shape[1]:
        raise ValueError('Expected two ID-aligned backbone views')
    if not torch.isfinite(backbone_views).all():
        raise ValueError('Nonfinite backbone views')
    pair_dist = torch.cdist(backbone_views, backbone_views)
    factor = 0
    for ell, (labels, proto, radius) in enumerate(zip(state.labels, state.prototypes, state.radii)):
        repeated = labels[rows].to(backbone_views.device).repeat(2)
        if strategy == 'zero_one':
            term = (pair_dist > 2 * radius.to(backbone_views.device)[repeated]).float()
        elif strategy == 'pair_dist':
            term = pair_dist
        else:
            term = torch.cdist(backbone_views, proto.to(backbone_views.device).float())[:, repeated]
        factor = factor + term / 2 ** ell
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 0.0000001)
    if normalization == 'row':
        if not torch.isfinite(factor).all() or (factor < 0).any():
            raise ValueError('Nonfinite/negative USE smoothing support')
        factor.fill_diagonal_(0)
        total = factor.sum(dim=1, keepdim=True)
        empty = total.squeeze(1) == 0
        factor = factor / torch.where(total > 0, total, torch.ones_like(total))
        empty_rows = empty.nonzero().flatten()
        factor[empty_rows, (empty_rows + n) % (2*n)] = 1
        return factor
    total = factor.sum(dim=1)
    if normalization == 'selex_legacy':
        if not torch.isfinite(factor).all():
            raise ValueError('Nonfinite SelEx USE factor')
        # Updated SelEx retains 1D column broadcasting, with a zero-sum clamp.
        return factor / total.clamp_min(1e-12)
    if not torch.isfinite(factor).all() or (total <= 0).any():
        raise ValueError('Baseline USE has a zero/nonfinite row; shared numerical policy required')
    # Deliberately preserve the original 1D broadcasting (row sum broadcasts by
    # COLUMN in PyTorch). This is a source defect, not silently corrected here.
    return factor / total


def pseudo_masks(state: SupervisionState, batch_ids: Tensor, level: int):
    rows = rows_for_ids(state.ids, batch_ids)
    labels, ids, labeled, abstained = (state.labels[level][rows], state.ids[rows],
                                      state.labeled[rows], state.abstained[level][rows])
    original = labels[:, None] == labels[None, :]
    protected = (ids[:, None] == ids[None, :]) | (labeled[:, None] & labeled[None, :])
    removed = original & ~protected & (abstained[:, None] | abstained[None, :])
    return original, original & ~removed


def checked_pair_weights(weights, state, batch_ids, device):
    """Validate detached symmetric coefficients and exact protected weights."""
    if weights is None:
        return None
    rows = rows_for_ids(state.ids, batch_ids)
    n = len(rows)
    if (weights.shape != (n, n) or not weights.is_floating_point()
            or weights.requires_grad or not torch.isfinite(weights).all()
            or ((weights < 0) | (weights > 1)).any()
            or not torch.equal(weights, weights.T)):
        raise ValueError('Pair weights must be detached finite symmetric [B,B] in [0,1]')
    ids, labeled = state.ids[rows].to(weights.device), state.labeled[rows].to(weights.device)
    protected = (ids[:, None] == ids[None, :]) | (labeled[:, None] & labeled[None, :])
    if (weights[protected] != 1).any():
        raise ValueError('Pair weights may not change protected coefficients')
    return weights.to(device)


@torch.no_grad()
def target_drift(before: SupervisionState, after: SupervisionState, batch_ids: Tensor,
                 backbone_views: Tensor, *, smoothing: float, strategy: str = 'zero_one',
                 normalization: str = 'legacy', pair_weights=None) -> dict:
    """Actual batch target-coefficient changes, not semantic error estimates.

    Legacy USE targets need not be probabilities; row mode has unit mass.
    Report half-L1 consistently and keep SSE and USE masses separate.
    """
    if not 0 <= smoothing <= 1 or not torch.equal(before.ids, after.ids):
        raise ValueError('Need matched snapshots and smoothing in [0,1]')
    n = len(batch_ids)
    offdiag = ~torch.eye(2 * n, dtype=torch.bool)
    cf0 = confusion_factor(before, batch_ids, backbone_views, strategy, normalization=normalization).cpu()
    cf1 = confusion_factor(after, batch_ids, backbone_views, strategy, normalization=normalization).cpu()
    delta = smoothing * (cf1 - cf0) * offdiag
    same_id = before.ids[rows_for_ids(before.ids, batch_ids)].repeat(2)
    identity = (same_id[:, None] == same_id[None, :]) & offdiag
    levels = []
    for ell in range(len(before.labels)):
        coefficients = []
        for state in (before, after):
            original, retained = pseudo_masks(state, batch_ids, ell)
            denominator = (original.repeat(2, 2) & offdiag).sum(1, keepdim=True)
            if (denominator <= 0).any():
                raise ValueError('Missing protected positives')
            numerator = (retained.repeat(2, 2) & offdiag).float()
            if ell == 0 and pair_weights is not None:
                numerator *= checked_pair_weights(pair_weights, state, batch_ids, 'cpu').repeat(2, 2)
            coefficients.append(numerator / denominator)
        change = coefficients[1] - coefficients[0]
        rows = rows_for_ids(before.ids, batch_ids)
        labeled = before.labeled[rows].repeat(2)
        protected = identity | (labeled[:, None] & labeled[None, :])
        levels.append(dict(level=ell, coefficient_half_l1_mean=float(change.abs().sum(1).mean() / 2),
                           protected_coefficient_shift_max=float(change[protected].abs().max()),
                           remaining_mass_mean=float(coefficients[1].sum(1).mean())))
    return dict(use_target_half_l1_mean=float(delta.abs().sum(1).mean() / 2),
                use_identity_smoothing_shift_max=float(delta[identity].abs().max()), sse_levels=levels)


def pseudo_sse_loss(criterion, features: Tensor, state: SupervisionState, batch_ids: Tensor, level: int,
                    *, pair_weights=None):
    """Abstain removes eligible numerators, retains baseline positive counts.

    No-action delegates directly to original SupConLoss. Only its verified
    contrast_mode='all', two-view contract is supported for active abstention.
    """
    original, kept = pseudo_masks(state, batch_ids, level)
    rows = rows_for_ids(state.ids, batch_ids)
    if features.ndim != 3 or features.shape[0] != len(rows) or features.shape[1] != 2 or features.shape[2] == 0 or not torch.isfinite(features).all():
        raise ValueError('Expected finite [B,2,d] features')
    weights = checked_pair_weights(pair_weights, state, batch_ids, features.device)
    if torch.equal(original, kept) and (weights is None or (weights == 1).all()):
        return criterion(features, labels=state.labels[level][rows].to(features.device))
    if criterion.contrast_mode != 'all':
        raise ValueError('Only contrast_mode=all is verified for abstention')
    flat = torch.cat(torch.unbind(features, dim=1), dim=0)
    if criterion.hyper_c == 0:
        distance = -torch.cdist(flat, flat)
    else:
        # Reuse the original manifold operations; no Poincare reimplementation.
        from hyptorch.pmath import dist_matrix
        distance = -dist_matrix(flat, flat, c=criterion.hyper_c)
    logits = distance / criterion.temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    offdiag = ~torch.eye(len(flat), dtype=torch.bool, device=flat.device)
    logprob = logits - torch.log((torch.exp(logits) * offdiag).sum(1, keepdim=True))
    p = original.to(flat.device).repeat(2, 2) & offdiag
    retained = (kept.to(flat.device).repeat(2, 2) & offdiag).float()
    if weights is not None:
        retained *= weights.repeat(2, 2)
    count = p.sum(1)
    if (count <= 0).any() or not torch.isfinite(logprob).all():
        raise ValueError('Invalid positive count or contrastive logits')
    return -(criterion.temperature / criterion.base_temperature) * ((retained * logprob).sum(1) / count).mean()
