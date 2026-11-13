"""Updated SelEx joint coarse grouping, with legal-label-only capture."""
import math
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from crish.state import SupervisionState
from crish.rebuild import prototype_radii
from gcd_runtime.cluster_alignment import restore_input_order


@torch.no_grad()
def finite_base_centers(labels, centers, known_classes):
    """Apply pinned SelEx's side-mean fallback only to empty nonfinite slots.

    Empty slots are legal source output: never move images or compact IDs to
    make them occupied. A nonfinite *occupied* center is a genuine fitter
    failure, not a case where inventing a finite center is justified.
    """
    k = len(centers)
    if (labels.ndim != 1 or labels.dtype != torch.long
            or ((labels < 0) | (labels >= k)).any()):
        raise ValueError('Invalid assignment at hierarchy level=0')
    if centers.ndim != 2 or not centers.is_floating_point() or not 0 < known_classes <= k:
        raise ValueError('Invalid base prototype layout')
    counts = torch.bincount(labels, minlength=k)
    bad = ~torch.isfinite(centers).all(dim=1)
    invalid = (bad & (counts > 0)).nonzero().flatten().tolist()
    if invalid:
        raise ValueError(f'Nonfinite occupied prototypes at hierarchy level=0: cluster IDs={invalid}')
    centers = centers.clone()
    for low, high in ((0, known_classes), (known_classes, k)):
        good = ~bad[low:high]
        if (~good).any():
            centers[low:high][~good] = centers[low:high][good].mean(0) if good.any() else 0.
    return centers, bad.nonzero().flatten().tolist()


def log_hierarchy(runtime, features, mask, labels, prototypes, active, fallback_ids):
    """Record source support/degeneracy without feeding it into supervision."""
    levels = []
    for level, (assignment, centers) in enumerate(zip(labels, prototypes)):
        k = len(centers)
        counts = torch.bincount(assignment, minlength=k)
        empty = (counts == 0).nonzero().flatten().tolist()
        entry = dict(level=level, active=level < active, prototype_slots=k,
                     occupied_clusters=int((counts > 0).sum()), empty_ids=empty,
                     support=counts.tolist(),
                     labeled_support=torch.bincount(assignment[mask], minlength=k).tolist(),
                     unlabeled_support=torch.bincount(assignment[~mask], minlength=k).tolist(),
                     duplicate_prototype_rows=k-len(torch.unique(centers, dim=0)))
        levels.append(entry)
        print(f'HIERARCHY epoch={runtime.epoch} level={level} '
              f'occupied={entry["occupied_clusters"]}/{k} empty_ids={empty} '
              f'duplicate_prototypes={entry["duplicate_prototype_rows"]} '
              'policy=preserve_source_assignments', flush=True)
    record = dict(epoch=runtime.epoch, policy='preserve_source_assignments',
                  empty_base_center_fallback_ids=fallback_ids,
                  feature_total_variance=float(features.var(dim=0, unbiased=False).sum()),
                  levels=levels)
    runtime.last_hierarchy_diagnostics = record
    if hasattr(runtime, 'run_dir'):
        path = Path(runtime.run_dir)/'hierarchy.jsonl'
        with path.open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')


def coarse_levels(features, base_labels, base_centers, known_classes, novel_classes):
    """Every coarse partition groups level-0 prototypes directly, as upstream.

    A CRiSH Move subsequently edits one level independently. This deliberate
    supervision intervention is not represented as a propagated tree repair.
    """
    labels, centers = [base_labels.long()], [base_centers.float()]
    known, novel = known_classes, novel_classes
    while known > 1:
        known, novel = max(known//2, 1), max(novel//2, 1)
        count = known+novel
        parent = torch.from_numpy(KMeans(n_clusters=count, random_state=0).fit(
            base_centers.cpu().numpy()).labels_).to(base_labels.device)
        assignment = parent[base_labels]
        support = torch.bincount(assignment, minlength=count).float()
        sums = features.new_zeros(count, features.shape[1]).index_add_(0, assignment, features)
        fallback = base_centers.new_zeros(count, features.shape[1]).index_add_(0, parent, base_centers)
        fallback /= torch.bincount(parent, minlength=count).clamp_min(1)[:, None]
        centers.append(torch.where(support[:, None] > 0, sums/support.clamp_min(1)[:, None], fallback))
        labels.append(assignment.long())
    return tuple(labels), tuple(centers)


@torch.no_grad()
def extract_state(model, loader, args, runtime):
    from methods.clustering.faster_mix_k_means_pytorch import K_Means
    model.eval()
    device = next(model.parameters()).device
    features, ids, memberships, legal = [], [], [], []
    for views, targets, sample_ids, mask in loader:
        images = views[0] if isinstance(views, (tuple, list)) else views
        features.append(F.normalize(model(images.to(device)), dim=-1).cpu())
        ids.append(torch.as_tensor(sample_ids))
        mask = mask[:, 0].bool()
        memberships.append(mask)
        legal.append(targets[mask].long())  # no unlabeled target read
    features, ids, mask = torch.cat(features), torch.cat(ids), torch.cat(memberships)
    legal = torch.cat(legal)
    k = args.num_labeled_classes+args.num_unlabeled_classes
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite extracted backbone features before SelEx clustering')
    if not torch.equal(legal.unique(sorted=True), torch.arange(args.num_labeled_classes)):
        raise ValueError('SelEx requires legal anchors for every contiguous known class')
    fit = K_Means(k=k, tolerance=1e-4, max_iterations=10, init='k-means++',
                  n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                  mode=None, protos=None,
                  cluster_size=None if getattr(args, 'unbalanced', False) else math.ceil(len(ids)/k))
    fit.fit_mix(features[~mask].to(device), features[mask].to(device), legal.to(device))
    base = restore_input_order(fit.labels_, mask).cpu()
    centers = fit.cluster_centers_.cpu().float().clone()
    if centers.shape != (k, features.shape[1]):
        raise ValueError('SelEx fitter returned an unexpected prototype shape')
    if not torch.equal(base[mask], legal):
        raise ValueError('SelEx fitter changed legally labeled anchors')
    centers, fallback_ids = finite_base_centers(base, centers, args.num_labeled_classes)
    labels, prototypes = coarse_levels(features, base, centers, args.num_labeled_classes,
                                       args.num_unlabeled_classes)
    active = min(len(labels), 1 + runtime.epoch//args.level_interval)
    args.hierarchy_levels = len(labels)
    log_hierarchy(runtime, features, mask, labels, prototypes, active, fallback_ids)
    labels, prototypes = labels[:active], prototypes[:active]
    state = SupervisionState(ids, features, mask, legal, labels, prototypes,
        tuple(prototype_radii(p) for p in prototypes),
        tuple(torch.zeros(len(ids), dtype=torch.bool) for _ in labels),
        allow_empty_clusters=True)
    runtime.hierarchy_state = state
    return state
