"""Tensor-free settings for bounded population-policy experiments."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class PopulationSettings:
    enabled: bool = False
    mode: str = 'counterfactual'
    fractions: tuple = (0., .05, .1, .2)
    fixed_fraction: float = .1
    warmup_epochs: int = 10
    refresh_every: int = 5
    schedule_epochs: int = 30
    train_batch: int = 128
    rank_block: int = 256
    max_cluster_fraction: float = .5
    min_cluster_kept: int = 2
    min_retained_mass: float = .7
    max_refresh_seconds: int = 180

    def __post_init__(self):
        if type(self.enabled) is not bool or self.mode not in ('counterfactual', 'fixed', 'scheduled', 'random'):
            raise ValueError('Invalid population_policy enabled/mode')
        object.__setattr__(self, 'fractions', tuple(self.fractions))
        if (len(self.fractions) < 2 or self.fractions[0] != 0
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x < 1 for x in self.fractions)
                or tuple(sorted(set(self.fractions))) != self.fractions):
            raise ValueError('Population fractions must be increasing, include Keep=0 and stay below 1')
        for name in ('fixed_fraction', 'max_cluster_fraction', 'min_retained_mass'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'population_policy.{name} must be in (0,1)')
        for name in ('refresh_every', 'schedule_epochs', 'train_batch', 'rank_block',
                     'min_cluster_kept', 'max_refresh_seconds'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'population_policy.{name} must be positive integer')
        if self.train_batch < 4 or type(self.warmup_epochs) is not int or self.warmup_epochs < 0:
            raise ValueError('Invalid population batch/warmup')

    def to_dict(self):
        values = asdict(self)
        values['fractions'] = list(self.fractions)
        return values

    def due(self, epoch):
        return self.enabled and epoch >= self.warmup_epochs and (epoch-self.warmup_epochs) % self.refresh_every == 0

    def candidates(self, epoch):
        if self.mode == 'counterfactual':
            return self.fractions
        q = self.fixed_fraction
        if self.mode == 'scheduled':
            q *= min(1., max(0., (epoch-self.warmup_epochs+1)/self.schedule_epochs))
        return (0., q)
