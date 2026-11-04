"""Sparse M1 from measured kNN rows; absent edges are unknown, never negatives."""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .ids import integer_ids, rows_for_ids


@dataclass(frozen=True)
class EdgeEvidence:
    reciprocal: float
    cross_view: float
    joint: float
    trials: int


def sparse_evidence(ids: Tensor, trials: Sequence[Tuple[Tensor, Tensor]]) -> Dict[Tuple[int, int], EdgeEvidence]:
    """Neighbor tensors contain row indices, [N,k], for the same IDs in all trials.

    RN averages reciprocal status across individual views. CVS averages directed
    edge persistence across a pair of views, symmetrized by either orientation.
    Joint is reciprocity in both views. Full coverage in every trial is required.
    """
    ids = integer_ids(ids)
    rows_for_ids(ids, ids)
    if not trials:
        raise ValueError('At least one independent two-view trial is required')
    measured, records = set(), []
    for pair in trials:
        if len(pair) != 2:
            raise ValueError('Each trial requires exactly two views')
        edges = []
        for neighbors in pair:
            if neighbors.ndim != 2 or neighbors.shape[0] != len(ids) or neighbors.dtype != torch.long:
                raise ValueError('Expected integer [N,k] neighbor row indices')
            if ((neighbors < 0) | (neighbors >= len(ids))).any():
                raise ValueError('Invalid neighbor row index')
            if any(len(set(row)) != len(row) for row in neighbors.tolist()):
                raise ValueError('Duplicate neighbors')
            directed = {(i, j) for i, row in enumerate(neighbors.tolist()) for j in row}
            if any(i == j for i, j in directed):
                raise ValueError('Self neighbors are forbidden')
            measured.update(tuple(sorted((i, j))) for i, j in directed)
            edges.append(directed)
        records.append(edges)
    output = {}
    for i, j in sorted(measured):
        rn = cvs = joint = 0.0
        for a, b in records:
            ra = (i, j) in a and (j, i) in a
            rb = (i, j) in b and (j, i) in b
            rn += (ra + rb) / 2
            cvs += ((i, j) in a and (i, j) in b) or ((j, i) in a and (j, i) in b)
            joint += ra and rb
        key = tuple(sorted((int(ids[i]), int(ids[j]))))
        output[key] = EdgeEvidence(rn / len(trials), cvs / len(trials), joint / len(trials), len(trials))
    return output


def edge_score(evidence: Dict[Tuple[int, int], EdgeEvidence], first: int, second: int) -> Optional[float]:
    record = evidence.get(tuple(sorted((first, second))))
    return None if record is None else record.joint


@torch.no_grad()
def blockwise_euclidean_knn(features: Tensor, k: int, block_size: int = 64) -> Tensor:
    """Exact reference: O(N^2 d) work, O(block*N+Nk) storage. No dense NxN output.

    Tie order follows torch.topk and may differ across hardware; record device/seed.
    Hyperbolic evidence must use upstream distances externally, not this helper.
    """
    if features.ndim != 2 or not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError('Expected finite [N,d] floating features')
    n = len(features)
    if type(k) is not int or not 0 <= k < n or type(block_size) is not int or block_size < 1:
        raise ValueError('Require 0 <= k < N and positive block size')
    chunks = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        distances = torch.cdist(features[start:stop], features)
        distances[torch.arange(stop-start, device=features.device), torch.arange(start, stop, device=features.device)] = torch.inf
        chunks.append(distances.topk(k, largest=False).indices)
    return torch.cat(chunks).cpu()
