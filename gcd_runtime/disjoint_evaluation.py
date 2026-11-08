"""Released Hyp-SelEx's plain k-means / global Hungarian-v2 test selector."""
import hashlib
import inspect
import json
from importlib.metadata import distribution
import time

import numpy as np
import torch
from torch.nn import functional as F
from .config import PLAIN_KMEANS_SHA


def plain_kmeans(features, clusters, device):
    from kmeans_pytorch import kmeans
    device = torch.device(device)
    package = distribution('kmeans-pytorch')
    origin = json.loads(package.read_text('direct_url.json') or '{}')
    commit = origin.get('vcs_info', {}).get('commit_id')
    if device.type == 'cuda' and (commit != PLAIN_KMEANS_SHA or 'tqdm_flag' not in inspect.signature(kmeans).parameters):
        raise RuntimeError('CUDA selector needs pinned Git kmeans_pytorch; install environments/requirements-selector-cuda.txt')
    if (features.ndim != 2 or len(features) < clusters or clusters < 2
            or not torch.isfinite(features).all()):
        raise ValueError('Invalid disjoint-test features/cluster count')
    keywords = dict(X=features.to(device), num_clusters=clusters, distance='euclidean', device=device)
    # Pinned PyPI 0.3 lacks the Git version's display-only tqdm_flag. Never pass
    # a nonexistent option or silently substitute sklearn/SS-k-means.
    if 'tqdm_flag' in inspect.signature(kmeans).parameters:
        keywords['tqdm_flag'] = False
    predictions, centers = kmeans(**keywords)
    if not torch.isfinite(centers).all() or predictions.shape != (len(features),):
        raise FloatingPointError('Invalid plain k-means result')
    return predictions.cpu().numpy(), dict(backend='kmeans_pytorch', backend_version=package.version,
        backend_git_commit=commit, legacy_cpu_fixture_backend=commit != PLAIN_KMEANS_SHA,
        backend_function_sha256=hashlib.sha256(inspect.getsource(kmeans).encode()).hexdigest())


@torch.no_grad()
def evaluate_disjoint(model, loader, args, runtime, device):
    from project_utils.cluster_and_log_utils import split_cluster_acc_v2
    started = time.monotonic()
    model.eval()
    features, targets = [], []
    for images, labels, _ids in loader:
        features.append(F.normalize(model(images.to(device)), dim=-1).cpu())
        targets.append(labels.cpu())
    features, targets = torch.cat(features), torch.cat(targets).numpy().astype(int)
    # Predictions depend on image features only; test labels enter only matching
    # and this explicitly authorized checkpoint-selection evaluator.
    predictions, backend = plain_kmeans(features, args.num_labeled_classes+args.num_unlabeled_classes, device)
    known = np.isin(targets, np.arange(len(args.train_classes)))
    if not known.any() or known.all():
        raise ValueError('Disjoint test must contain both known and novel examples')
    all_acc, old_acc, new_acc = split_cluster_acc_v2(targets, predictions, known)
    metrics = dict(all_acc=float(all_acc), old_acc=float(old_acc), new_acc=float(new_acc),
                   n_samples=len(targets), n_old=int(known.sum()), n_new=int((~known).sum()),
                   seconds=time.monotonic()-started, **backend)
    runtime.evaluation(metrics)
    return metrics


def disjoint_probe(device):
    """Bounded by the calling preflight worker; exercises the actual backend."""
    points = torch.eye(2, device=device)
    predictions, backend = plain_kmeans(points, 2, device)
    if len(set(predictions.tolist())) != 2:
        raise RuntimeError('Plain k-means readiness probe failed')
    return dict(device=str(device), **backend)
