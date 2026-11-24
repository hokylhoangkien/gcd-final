"""Explicit sample ID alignment; batch queries may repeat, references may not."""
from typing import Sequence

import torch
from torch import Tensor


def integer_ids(values: Sequence[int]) -> Tensor:
    ids = torch.as_tensor(values).detach().cpu()
    if ids.ndim != 1 or ids.dtype == torch.bool or ids.is_complex():
        raise ValueError('IDs must be a one-dimensional integer-valued sequence')
    if ids.is_floating_point():
        # Upstream np.append starts with a float array; accept only exact small IDs.
        if not torch.isfinite(ids).all() or (ids.abs() >= 2 ** 53).any() or not (ids == ids.trunc()).all():
            raise ValueError('Floating IDs must be finite, integral and exactly representable')
    return ids.to(torch.int64).clone()


def rows_for_ids(reference_ids: Sequence[int], query_ids: Sequence[int]) -> Tensor:
    reference, query = integer_ids(reference_ids), integer_ids(query_ids)
    if reference.numel() != reference.unique().numel():
        raise ValueError('Duplicate reference sample IDs')
    lookup = {value: row for row, value in enumerate(reference.tolist())}
    missing = sorted(set(query.tolist()) - lookup.keys())
    if missing:
        raise ValueError(f'Unknown sample IDs: {missing[:5]}')
    return torch.tensor([lookup[value] for value in query.tolist()], dtype=torch.long)
