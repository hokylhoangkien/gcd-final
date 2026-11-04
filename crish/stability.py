"""Permutation-invariant partition matching with explicit unmatched coverage."""
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .ids import integer_ids


def match_partition(reference: Tensor, perturbed: Tensor) -> dict:
    a, b = integer_ids(reference).numpy(), integer_ids(perturbed).numpy()
    if a.shape != b.shape or not a.size:
        raise ValueError('Align common sample IDs before partition matching')
    groups, other = np.unique(a), np.unique(b)
    size = len(groups) + len(other)  # real rows/columns plus explicit dummy nodes
    score = np.zeros((size, size))
    for i, group in enumerate(groups):
        for j, candidate in enumerate(other):
            x, y = a == group, b == candidate
            score[i, j] = (x & y).sum() / (x | y).sum()
    rows, cols = linear_sum_assignment(-score)
    mapping, iou = {}, {}
    for row, col in zip(rows, cols):
        if row < len(groups):
            matched = col < len(other) and score[row, col] > 0
            mapping[int(groups[row])] = int(other[col]) if matched else None
            iou[int(groups[row])] = float(score[row, col]) if matched else 0.0
    covered = sum((a == g).sum() for g in groups if mapping[int(g)] is not None)
    return dict(mapping=mapping, iou=iou,
                cluster_coverage=sum(v is not None for v in mapping.values()) / len(groups),
                sample_coverage=float(covered / len(a)), observed_samples=len(a))


def partition_stability(reference: Tensor, perturbations: Sequence[Tensor]) -> dict:
    if not perturbations:
        raise ValueError('No perturbations supplied; stability is unknown')
    matches = [match_partition(reference, p) for p in perturbations]
    return dict(stability={g: float(np.mean([m['iou'][g] for m in matches]))
                           for g in matches[0]['iou']},
                mean_cluster_coverage=float(np.mean([m['cluster_coverage'] for m in matches])),
                mean_sample_coverage=float(np.mean([m['sample_coverage'] for m in matches])))
