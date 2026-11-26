"""Epoch-local relational objective. Inputs are unlabeled image IDs and embeddings only."""

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crish.global_repair import frozen
from crish.ids import integer_ids
from crish.reliability import blockwise_euclidean_knn


@dataclass(frozen=True)
class Settings:
    k: int = 5
    baseline_weight: float = 0.5
    magnitude: float = 0.25
    coefficient: float = 1.0
    warmup: int = 10


class EdgePolicy(Protocol):
    """Future policies return one signed delta per canonical edge, in the supplied order."""

    def __call__(
        self, edge_ids: torch.Tensor, votes: torch.Tensor, settings: Settings
    ) -> torch.Tensor: ...


def keep(edge_ids, votes, settings):
    return torch.zeros_like(votes, dtype=torch.float32)


def evidence(edge_ids, votes, settings):
    return (votes.float() - 1) * settings.magnitude


POLICIES = {"R1": keep, "R2": evidence}
LOCKED_SETTINGS = Settings()


def reciprocal_edges(features, k):
    neighbors = blockwise_euclidean_knn(features, k)
    directed = [set(row) for row in neighbors.tolist()]
    return {
        (i, j)
        for i, row in enumerate(directed)
        for j in row
        if i < j and i in directed[j]
    }


class Graph:
    def __init__(
        self, ids, center, first, second, policy: EdgePolicy, settings=LOCKED_SETTINGS
    ):
        self.settings = settings
        ids = integer_ids(ids)
        if ids.unique().numel() != len(ids):
            raise ValueError("Expected unique integer image IDs")
        if not (
            center.shape == first.shape == second.shape and center.shape[0] == len(ids)
        ):
            raise ValueError("Views must share shape and ID order")
        self.keys = F.normalize(center.detach().float(), dim=1)
        ids = ids.cpu()
        self.ids = ids
        self.lookup = {key: i for i, key in enumerate(ids.tolist())}
        baseline = sorted(reciprocal_edges(self.keys, settings.k))
        a, b = (
            reciprocal_edges(F.normalize(view.detach().float(), dim=1), settings.k)
            for view in (first, second)
        )
        pairs = torch.tensor(baseline, dtype=torch.long).reshape(-1, 2)
        self.edge_ids = ids[pairs].sort(dim=1).values
        votes = torch.tensor(
            [int(edge in a) + int(edge in b) for edge in baseline], dtype=torch.long
        )
        delta = policy(self.edge_ids.clone(), votes.clone(), settings).detach().cpu()
        weights = settings.baseline_weight + delta
        if (
            delta.shape != votes.shape
            or not torch.isfinite(weights).all()
            or not ((weights >= 0) & (weights <= 1)).all()
        ):
            raise ValueError("Policy must return finite bounded edge edits")
        self.delta = delta
        self.delta_list = delta.tolist()
        # One sentinel row for labeled IDs, one zero-weight padding slot per absent neighbor.
        self.edge_slots = torch.zeros(len(ids) + 1, settings.k, dtype=torch.long)
        self.peer_slots = torch.zeros_like(self.edge_slots)
        degree = Counter()
        for e, (i, j) in enumerate(baseline, 1):
            for anchor, peer in ((i, j), (j, i)):
                col = degree[anchor]
                self.edge_slots[anchor, col] = e
                self.peer_slots[anchor, col] = peer
                degree[anchor] += 1
        self.weights = F.pad(weights, (1, 0)).to(self.keys.device)
        self.base = F.pad(
            torch.full_like(weights, settings.baseline_weight), (1, 0)
        ).to(self.keys.device)
        self.votes = votes
        variance = self.keys.var(dim=0, unbiased=False)
        self.health = {
            "nodes": len(ids),
            "edges": len(baseline),
            "isolated": len(ids) - len(degree),
            "crossview_edge_jaccard": len(a & b) / max(1, len(a | b)),
            "votes": torch.bincount(votes, minlength=3).tolist(),
            "feature_variance": float(variance.mean()),
            "mean_resultant_length": float(self.keys.mean(0).norm()),
        }
        self.exposure = Counter()
        self.anchors = set()
        self.rows = 0

    def record(self):
        return self.health | {
            "edge_count": self.health["edges"],
            "base": self.settings.baseline_weight,
            "edges": [
                [u, v, vote, float(d)]
                for (u, v), vote, d in zip(
                    self.edge_ids.tolist(), self.votes.tolist(), self.delta.tolist()
                )
            ],
            "proposed_strengthen": int((self.delta > 0).sum()),
            "proposed_weaken": int((self.delta < 0).sum()),
        }

    def loss(self, features, ids):
        """Return edited and baseline losses on the same graph and pending optimizer exposures."""
        batch = len(ids)
        if features.ndim != 2 or len(features) != 2 * batch:
            raise ValueError(
                "Expected view-major features over the original mixed minibatch"
            )
        rows = [self.lookup.get(key, len(self.ids)) for key in ids.tolist()]
        slots = self.edge_slots[rows]
        peers = self.keys[self.peer_slots[rows].to(features.device)]
        anchors = F.normalize(features.float(), dim=-1).reshape(2, batch, -1)
        cost = 1 - torch.einsum("vbd,bkd->vbk", anchors, peers)
        slot_ids = slots.to(features.device)
        total = (cost * self.weights[slot_ids]).mean()
        baseline = (cost * self.base[slot_ids]).mean()
        return total, baseline, (slots, ids)

    def applied(self, pending):
        slots, ids = pending
        self.rows += len(ids)
        for row, key in zip(slots.tolist(), ids.tolist()):
            for slot in row:
                if slot:
                    self.exposure[slot - 1] += 1
                    if self.delta_list[slot - 1]:
                        self.anchors.add(key)

    def exposure_record(self, batch_size):
        applied = [
            [*self.edge_ids[e].tolist(), count]
            for e, count in sorted(self.exposure.items())
            if self.delta_list[e]
        ]
        mass = sum(
            abs(self.delta_list[e]) * count for e, count in self.exposure.items()
        )
        return {
            "rows": self.rows,
            "graph_occurrences": sum(self.exposure.values()),
            "applied_edges": applied,
            "unique_edited_edges": len(applied),
            "unique_edited_anchors": len(self.anchors),
            "edited_anchor_ids": sorted(self.anchors),
            "strengthen": sum(
                count for e, count in self.exposure.items() if self.delta_list[e] > 0
            ),
            "weaken": sum(
                count for e, count in self.exposure.items() if self.delta_list[e] < 0
            ),
            "intervention_mass": mass,
            "effective_mass": self.settings.coefficient
            * mass
            / (batch_size * self.settings.k),
        }


@torch.no_grad()
def refresh(
    backbone,
    image_only,
    center_transform,
    variant,
    epoch,
    device,
    workers=0,
    micro=16,
    *,
    policy: EdgePolicy | None = None,
):
    """ImageOnly never calls the underlying annotated dataset's __getitem__."""
    center_data = copy.copy(image_only)
    center_data.transform = center_transform
    with frozen(backbone, 1979 + epoch, device):

        def encode(data):
            chunks, keys = [], []
            for images, sentinel, ids in DataLoader(
                data, batch_size=micro, num_workers=workers
            ):
                if not (sentinel == -1).all():
                    raise ValueError("Graph input must be image-only")
                images = images if isinstance(images, (tuple, list)) else [images]
                chunks.append(
                    torch.stack([backbone(view.to(device)).detach() for view in images])
                )
                keys.append(ids)
            return torch.cat(chunks, dim=1), torch.cat(keys)

        center, ids = encode(center_data)
        crops, crop_ids = encode(image_only)
        if not torch.equal(ids, crop_ids) or len(center) != 1 or len(crops) != 2:
            raise ValueError(
                "Center and independent paired views must align by stable ID"
            )
        return Graph(
            ids,
            center[0],
            crops[0],
            crops[1],
            POLICIES[variant] if policy is None else policy,
        )
