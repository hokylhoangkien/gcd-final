"""Explicit settings for the executable CRiSH research path (no tensor imports)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CrishSettings:
    lookahead_scope: str = 'projector'
    evidence_space: str = 'normalized_backbone_euclidean'
    enabled: bool = True
    warmup_epochs: int = 10
    refresh_every: int = 5
    evidence_samples: int = 128
    evidence_trials: int = 2
    k: int = 5
    boundaries: int = 2
    max_targets: int = 1
    max_accepted: int = 1
    min_support: int = 2
    min_edges: int = 1
    min_reliability: float = 0.5
    train_batch: int = 32
    probe_known: int = 4
    probe_unlabeled: int = 8
    probe_known_weight: float = 1.0
    probe_view_weight: float = 1.0
    probe_variance_weight: float = 0.1
    probe_std_floor: float = 0.1
    acceptance_gain: float = 0.0
    known_tolerance: float = 0.0
    max_protected_shift: float = 0.25
    max_refresh_seconds: int = 120
    stability_trials: int = 0
    stability_noise: float = 0.005
    save_partitions: bool = True

    def __post_init__(self):
        if self.lookahead_scope != 'projector' or self.evidence_space != 'normalized_backbone_euclidean':
            raise ValueError('Only the explicitly tested projector lookahead/backbone evidence path is implemented')
        for name in ('enabled', 'save_partitions'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'crish.{name} must be bool')
        for name in ('warmup_epochs', 'stability_trials'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'crish.{name} must be a nonnegative integer')
        for name in ('refresh_every', 'evidence_samples', 'evidence_trials', 'k', 'boundaries',
                     'max_targets', 'max_accepted', 'min_support', 'min_edges', 'train_batch',
                     'probe_known', 'probe_unlabeled', 'max_refresh_seconds'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'crish.{name} must be a positive integer')
        if self.probe_unlabeled < 2 or self.train_batch < 2 or self.evidence_samples <= self.k:
            raise ValueError('Probe needs >=2 unlabeled rows; evidence_samples must exceed k')
        if self.boundaries > self.train_batch:
            raise ValueError('Boundary budget must fit the scoring train_batch')
        for name in ('min_reliability', 'probe_known_weight', 'probe_view_weight',
                     'probe_variance_weight', 'probe_std_floor', 'acceptance_gain',
                     'known_tolerance', 'max_protected_shift', 'stability_noise'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid crish.{name}')
        if not 0 < self.min_reliability <= 1 or self.max_protected_shift > 1:
            raise ValueError('Reliability/protected-shift limits must be in [0,1]')
        if self.probe_known_weight + self.probe_view_weight + self.probe_variance_weight <= 0:
            raise ValueError('At least one legal probe component is required')

    def due(self, epoch):
        return self.enabled and epoch >= self.warmup_epochs and (epoch-self.warmup_epochs) % self.refresh_every == 0
