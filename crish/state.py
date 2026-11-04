"""Frozen-feature supervision snapshots contain legal labeled targets only."""
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .ids import integer_ids, rows_for_ids


@dataclass(frozen=True)
class SupervisionState:
    ids: Tensor
    features: Tensor
    labeled: Tensor
    known_targets: Tensor  # compressed: exactly labeled.sum() values, no D_U labels
    labels: Tuple[Tensor, ...]
    prototypes: Tuple[Tensor, ...]
    radii: Tuple[Tensor, ...]
    abstained: Tuple[Tensor, ...]
    version: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'ids', integer_ids(self.ids))
        for key in ('features', 'labeled', 'known_targets'):
            object.__setattr__(self, key, getattr(self, key).detach().cpu().clone())
        for key in ('labels', 'prototypes', 'radii', 'abstained'):
            object.__setattr__(self, key, tuple(x.detach().cpu().clone() for x in getattr(self, key)))
        self.validate()

    def validate(self):
        n = len(self.ids)
        rows_for_ids(self.ids, self.ids)
        if not n or self.features.ndim != 2 or len(self.features) != n or not torch.isfinite(self.features).all():
            raise ValueError('Invalid frozen feature rows')
        if self.labeled.shape != (n,) or self.labeled.dtype != torch.bool:
            raise ValueError('Expected labeled membership mask')
        if self.known_targets.shape != (int(self.labeled.sum()),) or self.known_targets.dtype != torch.long:
            raise ValueError('Only compressed legally labeled targets may enter the state')
        if not self.labels or not (len(self.labels) == len(self.prototypes) == len(self.radii) == len(self.abstained)):
            raise ValueError('Incomplete level state')
        if type(self.version) is not int or self.version < 0:
            raise ValueError('Invalid snapshot version')
        for labels, centers, radius, abstained in zip(self.labels, self.prototypes, self.radii, self.abstained):
            if labels.shape != (n,) or labels.dtype != torch.long:
                raise ValueError('Invalid pseudo-label rows')
            if centers.ndim != 2 or centers.shape[1] != self.features.shape[1] or len(centers) < 2:
                raise ValueError('Expected >=2 prototypes per level')
            if not centers.is_floating_point() or not torch.isfinite(centers).all():
                raise ValueError('Nonfinite prototypes')
            if ((labels < 0) | (labels >= len(centers))).any() or (torch.bincount(labels, minlength=len(centers)) == 0).any():
                raise ValueError('Invalid assignment or empty cluster')
            if radius.shape != (len(centers),) or not torch.isfinite(radius).all() or (radius < 0).any():
                raise ValueError('Invalid prototype radii')
            if abstained.shape != (n,) or abstained.dtype != torch.bool or abstained[self.labeled].any():
                raise ValueError('Abstain can only affect unlabeled pseudo-supervision')
        if not torch.equal(self.labels[0][self.labeled], self.known_targets):
            raise ValueError('Base known anchors disagree with legally available true labels')

    def copy(self):
        return SupervisionState(self.ids, self.features, self.labeled, self.known_targets,
                                self.labels, self.prototypes, self.radii, self.abstained, self.version)
