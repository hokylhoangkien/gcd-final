"""A narrow, label-safe embedding probe; no loaders or objective callbacks.

This is a research proxy, not a classifier head or an accuracy oracle. Legal-label
provenance is an explicit caller obligation, checked against a labeled registry.
"""
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ids import integer_ids, rows_for_ids


def trusted_weight_norm_hooks(model):
    """Allow only PyTorch's original WeightNorm reparameterization hook.

    Official DINOHead uses this hook; arbitrary callbacks remain prohibited.
    """
    from torch.nn.utils.weight_norm import WeightNorm
    hooks = []
    for module in model.modules():
        if module._forward_hooks or module._backward_hooks:
            raise ValueError('Probe models with callback hooks require a separate leakage audit')
        for hook in module._forward_pre_hooks.values():
            if type(hook) is not WeightNorm:
                raise ValueError('Probe models with callback hooks require a separate leakage audit')
            hooks.append((module, hook))
    return hooks


@dataclass(frozen=True)
class ProbeWeights:
    known: float
    view: float
    variance: float
    std_floor: float

    def __post_init__(self):
        values = torch.tensor([self.known, self.view, self.variance, self.std_floor])
        if not torch.isfinite(values).all() or (values < 0).any() or self.known + self.view + self.variance <= 0:
            raise ValueError('Finite nonnegative probe weights and a nonzero objective required')


class LegalProbe:
    __slots__ = ('known_ids', 'known_views', 'known_targets', 'unlabeled_ids',
                 'unlabeled_views', 'teacher_views', 'weights')

    def __init__(self, *, known_ids: Tensor, known_views: Tensor,
                 known_targets: Tensor, unlabeled_ids: Tensor, unlabeled_views: Tensor,
                 teacher_views: Tensor, proposal_ids: Tensor,
                 legal_labels: Mapping[int, int], weights: ProbeWeights):
        self.known_ids, self.unlabeled_ids = integer_ids(known_ids), integer_ids(unlabeled_ids)
        ids = torch.cat([self.known_ids, self.unlabeled_ids])
        rows_for_ids(ids, ids)
        if set(ids.tolist()) & set(integer_ids(proposal_ids).tolist()):
            raise ValueError('Probe and proposal/train-gradient IDs must be disjoint')
        if set(self.unlabeled_ids.tolist()) & set(legal_labels):
            raise ValueError('Unlabeled probe membership conflicts with the legal registry')
        targets = integer_ids(known_targets)
        if len(targets) != len(self.known_ids) or any(legal_labels.get(i) != int(y) for i, y in zip(self.known_ids.tolist(), targets)):
            raise ValueError('Known probe labels must match the legal labeled-ID registry')
        if not len(self.known_ids) or len(self.unlabeled_ids) < 2:
            raise ValueError('Reference probe requires known examples and >=2 unlabeled examples')
        for views, count in ((known_views, len(self.known_ids)), (unlabeled_views, len(self.unlabeled_ids)), (teacher_views, len(self.unlabeled_ids))):
            if views.ndim < 3 or views.shape[:2] != (count, 2) or not torch.isfinite(views).all():
                raise ValueError('Probe inputs/teacher must contain finite two-view rows')
        self.known_views, self.unlabeled_views, self.teacher_views = (
            x.detach().clone() for x in (known_views, unlabeled_views, teacher_views))
        self.known_targets, self.weights = targets.clone(), weights

    def assert_disjoint(self, other: 'LegalProbe'):
        if set(torch.cat([self.known_ids, self.unlabeled_ids]).tolist()) & set(torch.cat([other.known_ids, other.unlabeled_ids]).tolist()):
            raise ValueError('Confirmation probe folds must be ID-disjoint')

    def components(self, model: nn.Module) -> dict:
        # Probe is evaluated on a clone in eval mode by lookahead. No train/eval
        # mode switch is applied to a live caller-owned model here.
        trusted_weight_norm_hooks(model)
        known = model(self.known_views.flatten(0, 1))
        student = model(self.unlabeled_views.flatten(0, 1))
        if known.ndim != 2 or student.ndim != 2 or not torch.isfinite(known).all() or not torch.isfinite(student).all():
            raise ValueError('Probe model must return finite [rows, embedding_dim]')
        teacher = self.teacher_views.to(student).reshape(len(self.unlabeled_ids), 2, -1).detach()
        if student.shape[1] != teacher.shape[-1]:
            raise ValueError('Teacher/student embedding dimensions differ')
        z = F.normalize(known, dim=1)
        logits = -torch.cdist(z, z)
        offdiag = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        target = self.known_targets.to(z.device).repeat_interleave(2)
        positives = (target[:, None] == target[None, :]) & offdiag
        logprob = logits - torch.logsumexp(logits.masked_fill(~offdiag, -torch.inf), dim=1, keepdim=True)
        item = -(logprob * positives).sum(1) / positives.sum(1)
        known_loss = torch.stack([item[target == group].mean() for group in target.unique()]).mean()
        u = F.normalize(student, dim=1).reshape(len(self.unlabeled_ids), 2, -1)
        t = F.normalize(teacher, dim=-1)
        view_loss = (u - t.flip(1)).square().mean()
        std = (student.var(dim=0, unbiased=False) + 1e-4).sqrt()
        variance_loss = torch.relu(self.weights.std_floor - std).mean()
        return dict(known=known_loss, view=view_loss, variance=variance_loss)

    def loss(self, model: nn.Module) -> Tensor:
        c = self.components(model)
        return self.weights.known * c['known'] + self.weights.view * c['view'] + self.weights.variance * c['variance']
