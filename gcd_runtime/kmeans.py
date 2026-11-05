"""Bounded, empty-cluster-safe HypCD semi-supervised k-means.

Derived from Visual-AI/HypCD (MIT), ffc895b7ff38be8502bb7e3bd851f1d2a55dff73,
methods/clustering/faster_mix_k_means_pytorch.py. See NOTICE and LICENSES/.
Preserves the two source optimization phases, balance heuristic, best-iterate
selection and nondegenerate initialization. Both B0 and CRiSH use this policy.
"""
import numpy as np
import torch
from sklearn.utils import check_random_state
from torch import Tensor

from methods.clustering.faster_mix_k_means_pytorch import (
    K_Means as SourceKMeans, pairwise_distance,
)


class SemiSupKMeans(SourceKMeans):
    @torch.no_grad()
    def kpp(self, X, pre_centers=None, k=10, random_state=None):
        rng = check_random_state(random_state)
        centers = pre_centers if pre_centers is not None else X[rng.randint(0, len(X))]
        centers = centers.view(-1, X.shape[1])
        initial_count = len(centers)
        # Exactly k - initial_count insertions: collapsed features cannot loop.
        for _ in range(k - initial_count):
            distance = pairwise_distance(X, centers, self.pairwise_batch_size)
            closest = distance.view(-1, len(centers)).min(dim=1)[0]
            total = closest.sum()
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite k-means++ distances')
            if total <= 0:
                index = (len(centers) - initial_count) % len(X)
                self.degenerate_initializations_ += 1
            else:
                cumulative = torch.cumsum(closest / total, dim=0)
                candidates = (cumulative >= rng.rand()).nonzero().flatten()
                # Source can reference an uninitialized/stale index on rounding.
                index = int(candidates[0]) if len(candidates) else len(X) - 1
            centers = torch.cat((centers, X[index].view(1, -1)), dim=0)
        return centers

    @torch.no_grad()
    def _repair_empty(self, features: Tensor, labels: Tensor, anchor_count: int,
                      centers: Tensor) -> None:
        counts = torch.bincount(labels, minlength=self.k)
        empty = (counts == 0).nonzero().flatten()
        if not len(empty):
            return
        # Only non-anchor rows may move. Stable argmax breaks ties by row order.
        residual = (features[anchor_count:] - centers[labels[anchor_count:]]).square().sum(1)
        for cluster in empty.tolist():
            donors = counts[labels[anchor_count:]] > 1
            if not donors.any():
                raise ValueError('Cannot fill empty clusters without moving anchors or emptying a donor')
            offset = int(residual.masked_fill(~donors, -torch.inf).argmax())
            row = anchor_count + offset
            source = int(labels[row])
            labels[row] = cluster
            counts[source] -= 1
            counts[cluster] += 1
            centers[cluster] = features[row]
            self.empty_cluster_repairs_ += 1

    @torch.no_grad()
    def fit_mix(self, u_feats, l_feats, l_targets):
        if (u_feats.ndim != 2 or l_feats.ndim != 2 or u_feats.shape[1] == 0
                or u_feats.shape[1] != l_feats.shape[1] or not len(l_feats)
                or u_feats.device != l_feats.device or u_feats.dtype != l_feats.dtype
                or not u_feats.is_floating_point() or not torch.isfinite(u_feats).all()
                or not torch.isfinite(l_feats).all()):
            raise ValueError('Expected finite, compatible labeled/unlabeled feature matrices')
        targets = torch.as_tensor(l_targets, device=l_feats.device)
        if (targets.shape != (len(l_feats),) or not torch.isfinite(targets).all()
                or not torch.equal(targets, targets.long().to(targets.dtype))):
            raise ValueError('Expected one integral legal target per anchor')
        classes = targets.unique(sorted=True)
        if not torch.equal(classes, torch.arange(len(classes), device=classes.device)):
            raise ValueError('Anchor classes must be contiguous from zero')
        if (type(self.k) is not int or not len(classes) <= self.k <= len(l_feats) + len(u_feats)
                or (self.k > len(classes) and not len(u_feats))
                or self.max_iterations < 1 or self.n_init < 1 or self.protos is not None):
            raise ValueError('Invalid cluster budget, iterations or unsupported supplied prototypes')
        self.empty_cluster_repairs_ = self.degenerate_initializations_ = 0
        super().fit_mix(u_feats, l_feats, targets)
        counts = torch.bincount(self.labels_, minlength=self.k)
        if (not torch.isfinite(self.cluster_centers_).all() or (counts == 0).any()
                or not torch.equal(self.labels_[:len(l_feats)], targets.long())):
            raise FloatingPointError('Invalid fitted clusters or changed legal anchors')
        if self.empty_cluster_repairs_ or self.degenerate_initializations_:
            print(f'KMEANS k={self.k} empty_cluster_repairs={self.empty_cluster_repairs_} '
                  f'degenerate_initializations={self.degenerate_initializations_}', flush=True)

    @torch.no_grad()
    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):
        device = l_feats.device
        classes = torch.unique(l_targets)
        l_centers = torch.stack([l_feats[l_targets.eq(c)].mean(0) for c in classes])
        features = torch.cat((l_feats, u_feats))
        anchor_count = len(l_feats)
        labels = -torch.ones(len(features)).type_as(features).long()
        labels[:anchor_count] = l_targets.long()
        centers = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)
        centers[:len(classes)] = l_centers
        anchor_counts = torch.zeros(self.k).type_as(features).cpu()
        cluster_free = torch.zeros_like(labels[anchor_count:]).bool()
        sparse = torch.zeros_like(anchor_counts).bool()
        best_labels = best_inertia = best_centers = None
        iterations = 0

        # Phase 0 freezes anchor prototypes and applies the source balance
        # heuristic. Phase 1 updates every prototype without that heuristic.
        for phase in range(2):
            for iteration in range(self.max_iterations):
                iterations += 1
                for cluster in range(self.k):
                    anchor_counts[cluster] = (labels[:anchor_count] == cluster).sum()
                old_centers = centers.clone()
                distance = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
                closest, u_labels = torch.min(distance, dim=1)
                l_distance = torch.sum((l_feats - centers[labels[:anchor_count]]) ** 2, dim=1)
                inertia = closest.sum() + l_distance.sum()
                if phase == 0 and self.cluster_size is not None:
                    cluster_free[:] = False
                    sparse[:] = True
                    for _ in range(iteration):
                        for cluster in range(self.k):
                            selected = (u_labels == cluster).nonzero().flatten()
                            capacity = self.cluster_size - anchor_counts[cluster].int()
                            # Match the source's singleton treatment and slicing.
                            if len(selected) <= 1 or len(selected) < capacity:
                                sparse[cluster] = True
                            elif len(selected) > capacity:
                                indices = torch.argsort(distance[selected, cluster], dim=0)[capacity:]
                                cluster_free[selected[indices].to(device)] = True
                                sparse[cluster] = False
                            else:
                                sparse[cluster] = False
                        rank = torch.argsort(distance[cluster_free.cpu(), :][:, sparse], dim=1)
                        mid_labels = torch.from_numpy(np.arange(self.k))[sparse][rank]
                        if mid_labels.shape[1] != 0:
                            u_labels[cluster_free.cpu()] = mid_labels[:, 0]
                            cluster_free[:] = False
                labels[anchor_count:] = u_labels.to(device)
                # Never take mean(empty). Relocate an unanchored row from a
                # nonsingleton donor, preserving k and every legal anchor.
                self._repair_empty(features, labels, anchor_count, centers)
                first = len(classes) if phase == 0 else 0
                for cluster in range(first, self.k):
                    selected = (labels == cluster).nonzero().flatten()
                    centers[cluster] = torch.index_select(features, 0, selected).mean(0)
                if not torch.isfinite(centers).all():
                    raise FloatingPointError('Nonfinite prototype after k-means update')
                if best_inertia is None or inertia < best_inertia:
                    best_labels, best_centers, best_inertia = labels.clone(), centers.clone(), inertia
                shift = torch.sum(torch.sqrt(torch.sum((centers - old_centers) ** 2, dim=1)))
                if shift ** 2 < self.tolerance:
                    break
        return best_labels, best_inertia, best_centers, iterations
