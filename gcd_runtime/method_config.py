"""Dependency-free settings for the experimental method."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilitySettings:
    k: int = 5
    floor: float = 0.1
    warmup_epochs: int = 10
    ramp_epochs: int = 20
    reciprocal: bool = True
    cross_view: bool = True
    pair_scope: str = 'either_unlabeled'
    diagnostics_every: int = 50

    def __post_init__(self):
        for name in ('k', 'warmup_epochs', 'ramp_epochs', 'diagnostics_every'):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or (name == 'diagnostics_every' and value == 0):
                raise ValueError(f'Invalid reliability {name}')
        if not 0 <= self.floor <= 1:
            raise ValueError('Reliability floor must be in [0,1]')
        if type(self.reciprocal) is not bool or type(self.cross_view) is not bool:
            raise ValueError('Component switches must be bool')
        if self.pair_scope not in ('either_unlabeled', 'both_unlabeled'):
            raise ValueError('Invalid pair scope')

    def strength(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs == 0:
            return 1.0
        return min(1.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs)
