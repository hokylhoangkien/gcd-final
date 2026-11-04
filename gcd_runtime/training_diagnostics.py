"""Optional loss-geometry diagnostics at projected features, never parameters.

Reads gradients without accumulating .grad, updating weights, consuming RNG,
or accepting class targets. Feature-space conflict is not parameter-space conflict.
"""
import math
from typing import Dict

import torch
from torch import Tensor


def _gradient_stats(angle: Tensor, distance: Tensor) -> Dict:
    a, d = angle.detach().double().reshape(-1), distance.detach().double().reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(d).all():
        raise FloatingPointError('Non-finite projected-feature diagnostic gradients')
    a_norm, d_norm = float(a.norm()), float(d.norm())
    denominator = a_norm * d_norm
    cosine = float(torch.dot(a, d).double() / denominator) if denominator > 0 else None
    return dict(angle_norm=a_norm, distance_norm=d_norm,
                cosine=max(-1., min(1., cosine)) if cosine is not None else None)


def loss_geometry(angle_loss: Tensor, distance_loss: Tensor, features: Tensor,
                  labeled: Tensor, distance_weight: float) -> Dict:
    """Two-view rows are [all view 1, all view 2], matching the actual trainer."""
    if not math.isfinite(distance_weight) or not 0 <= distance_weight <= 1:
        raise ValueError('Distance weight must be finite and bounded')
    if features.ndim != 2 or labeled.ndim != 1 or len(features) != 2 * len(labeled):
        raise ValueError('Expected two-view features and per-sample membership')
    if labeled.dtype != torch.bool or not features.requires_grad:
        raise ValueError('Boolean membership and differentiable features required')
    if not torch.isfinite(angle_loss) or not torch.isfinite(distance_loss):
        raise FloatingPointError('Non-finite component loss')
    angle_grad = torch.autograd.grad(angle_loss, features, retain_graph=True)[0]
    distance_grad = torch.autograd.grad(distance_loss, features, retain_graph=True)[0]
    membership = labeled.to(features.device).repeat(2)
    groups = {'all': torch.ones(len(features), dtype=torch.bool, device=features.device),
              'labeled': membership, 'unlabeled': ~membership}
    summary = {}
    for name, mask in groups.items():
        stats = _gradient_stats(angle_grad[mask], distance_grad[mask])
        angle_mass = (1 - distance_weight) * stats['angle_norm']
        distance_mass = distance_weight * stats['distance_norm']
        summary[name] = dict(rows=int(mask.sum()), **stats,
                             weighted_angle_norm=angle_mass,
                             weighted_distance_norm=distance_mass,
                             distance_norm_share=distance_mass/(angle_mass + distance_mass)
                             if angle_mass + distance_mass > 0 else None)
    return dict(space='projected_features_not_parameters',
                angle_loss=float(angle_loss.detach()), distance_loss=float(distance_loss.detach()),
                distance_weight=distance_weight, groups=summary)
