"""Tensor-free settings for bounded population-policy experiments.

Historical CRiSH-P (margin ranking, fraction selection) is the default. The
CRiSH-G fields below only enter a serialized config when one of them differs
from its default, so historical config snapshots and resume checks are stable.
"""
from dataclasses import dataclass, asdict, fields
import math

# CRiSH-G fields, all defaulting to historical CRiSH-P behavior.
GLOBAL_FIELDS = ('ranking', 'selection', 'thresholds', 'fixed_threshold', 'gate', 'sample_choice',
                 'persist', 'levels', 'actions', 'knn_k', 'nonreciprocal_weight', 'move_min_share',
                 'move_target_cap', 'scoring_batches', 'scoring_sampling', 'probe_float64',
                 'save_arrays', 'compose_local')


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
    # CRiSH-G: global reciprocal-kNN agreement, thresholded, persistent, multi-level.
    ranking: str = 'margin'
    selection: str = 'fraction'
    thresholds: tuple = (0., .3, .5, .7)
    fixed_threshold: float = .5
    gate: str = 'counterfactual'
    sample_choice: str = 'evidence'
    persist: bool = False
    levels: str = 'base'
    actions: str = 'abstain'
    knn_k: int = 10
    nonreciprocal_weight: float = .5
    move_min_share: float = .6
    move_target_cap: float = .25
    scoring_batches: int = 1
    scoring_sampling: str = 'uniform'
    probe_float64: bool = False
    save_arrays: bool = False
    compose_local: bool = False

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
        self._validate_global()

    def _validate_global(self):
        choices = dict(ranking=('margin', 'knn_agreement'), selection=('fraction', 'threshold'),
                       gate=('counterfactual', 'none'), sample_choice=('evidence', 'random'),
                       levels=('base', 'active'), actions=('abstain', 'abstain_move'),
                       scoring_sampling=('uniform', 'train_sampler'))
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f'population_policy.{name} must be one of {allowed}')
        for name in ('persist', 'probe_float64', 'save_arrays', 'compose_local'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'population_policy.{name} must be bool')
        object.__setattr__(self, 'thresholds', tuple(self.thresholds))
        if (len(self.thresholds) < 2 or self.thresholds[0] != 0
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in self.thresholds)
                or tuple(sorted(set(self.thresholds))) != self.thresholds):
            raise ValueError('population_policy.thresholds must be increasing, include Keep=0 and stay in [0,1]')
        for name in ('fixed_threshold', 'nonreciprocal_weight', 'move_min_share', 'move_target_cap'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'population_policy.{name} must be in (0,1]')
        for name in ('knn_k', 'scoring_batches'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'population_policy.{name} must be positive integer')
        # Explicit combinations; every message names the offending pair.
        if self.selection == 'threshold' and self.ranking != 'knn_agreement':
            raise ValueError("population_policy.selection='threshold' requires ranking='knn_agreement'")
        if self.ranking == 'knn_agreement' and self.selection != 'threshold':
            raise ValueError("population_policy.ranking='knn_agreement' requires selection='threshold'")
        if self.gate == 'none' and self.mode != 'fixed':
            raise ValueError("population_policy.gate='none' requires mode='fixed'")
        if self.sample_choice == 'random' and self.gate != 'none':
            raise ValueError("population_policy.sample_choice='random' requires gate='none'")
        if self.selection == 'threshold' and self.mode not in ('counterfactual', 'fixed'):
            raise ValueError("population_policy.selection='threshold' supports mode counterfactual or fixed")
        threshold_only = dict(gate=self.gate != 'counterfactual', sample_choice=self.sample_choice != 'evidence',
                              persist=self.persist, levels=self.levels != 'base',
                              actions=self.actions != 'abstain', compose_local=self.compose_local,
                              scoring_sampling=self.scoring_sampling != 'uniform',
                              scoring_batches=self.scoring_batches != 1, probe_float64=self.probe_float64,
                              save_arrays=self.save_arrays)
        if self.selection != 'threshold':
            used = [name for name, active in threshold_only.items() if active]
            if used:
                raise ValueError(f"population_policy.{used[0]} requires selection='threshold'")
        if self.actions == 'abstain_move' and self.sample_choice != 'evidence':
            raise ValueError("population_policy.actions='abstain_move' requires sample_choice='evidence'")
        if self.selection == 'threshold' and self.mode == 'counterfactual' and self.gate != 'counterfactual':
            raise ValueError("population_policy.mode='counterfactual' requires gate='counterfactual'")

    @property
    def is_global(self):
        return self.selection == 'threshold'

    def to_dict(self):
        values = asdict(self)
        values['fractions'] = list(self.fractions)
        values['thresholds'] = list(self.thresholds)
        defaults = {f.name: f.default for f in fields(self)}
        if all(values[name] == (list(defaults[name]) if isinstance(defaults[name], tuple) else defaults[name])
               for name in GLOBAL_FIELDS):
            # Historical CRiSH-P snapshot: keep its exact original key set.
            for name in GLOBAL_FIELDS:
                values.pop(name)
        return values

    def full_dict(self):
        """Every field, including defaults; for dry-run display only."""
        values = asdict(self)
        values['fractions'] = list(self.fractions)
        values['thresholds'] = list(self.thresholds)
        return values

    def due(self, epoch):
        return self.enabled and epoch >= self.warmup_epochs and (epoch-self.warmup_epochs) % self.refresh_every == 0

    def active(self, epoch):
        """Epochs where a CRiSH-G policy is (re)applied: due, or persisted after warmup."""
        if not self.enabled or epoch < self.warmup_epochs:
            return False
        return self.due(epoch) or (self.is_global and self.persist)

    def candidates(self, epoch):
        if self.mode == 'counterfactual':
            return self.fractions
        q = self.fixed_fraction
        if self.mode == 'scheduled':
            q *= min(1., max(0., (epoch-self.warmup_epochs+1)/self.schedule_epochs))
        return (0., q)

    def threshold_candidates(self):
        if self.mode == 'counterfactual':
            return self.thresholds
        return (0., self.fixed_threshold)
