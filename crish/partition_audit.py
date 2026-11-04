"""Level-wise partition diagnostics; no tree assumption or oracle labels."""
from typing import Sequence

import torch
from torch import Tensor

from .ids import integer_ids


def nesting_violation(fine: Tensor, coarse: Tensor) -> float:
    fine, coarse = integer_ids(fine), integer_ids(coarse)
    if fine.shape != coarse.shape or fine.numel() == 0:
        raise ValueError('Partitions must cover the same nonempty, ID-aligned rows')
    retained = sum(int(coarse[fine == group].unique(return_counts=True)[1].max())
                   for group in fine.unique())
    return 1.0 - retained / fine.numel()


def audit_stack(partitions: Sequence[Tensor]) -> dict:
    if not partitions:
        raise ValueError('Empty partition stack')
    n = len(partitions[0])
    if n == 0 or any(len(p) != n for p in partitions):
        raise ValueError('Mismatched partition coverage')
    violations = [nesting_violation(a, b) for a, b in zip(partitions, partitions[1:])]
    return dict(samples=n, clusters=[integer_ids(p).unique().numel() for p in partitions],
                adjacent_violation=violations, nested=all(v == 0 for v in violations))
