"""Restore fit_mix outputs from [anchor rows, other rows] to input row order.

The mask is training membership or predicted membership, never hidden labels.
Cluster centers are indexed by cluster ID and must NOT be reordered here.
"""
import torch
from torch import Tensor


def restore_input_order(grouped_labels: Tensor, anchor_mask: Tensor) -> Tensor:
    mask = torch.as_tensor(anchor_mask, device=grouped_labels.device)
    if mask.dtype != torch.bool or mask.ndim != 1 or grouped_labels.ndim != 1:
        raise ValueError('Expected 1D labels and a boolean anchor mask')
    if mask.numel() != grouped_labels.numel():
        raise ValueError('Clustering output and membership mask lengths differ')
    order = torch.cat((mask.nonzero().flatten(), (~mask).nonzero().flatten()))
    output = torch.empty_like(grouped_labels)
    output[order] = grouped_labels
    return output
