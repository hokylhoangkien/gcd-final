"""Updated SelEx joint coarse grouping, with legal-label-only capture."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from crish.state import SupervisionState
from crish.rebuild import prototype_radii
from gcd_runtime.cluster_alignment import restore_input_order


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
    fit = K_Means(k=k, tolerance=1e-4, max_iterations=10, init='k-means++',
                  n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                  mode=None, protos=None, cluster_size=math.ceil(len(ids)/k))
    fit.fit_mix(features[~mask].to(device), features[mask].to(device), legal.to(device))
    base = restore_input_order(fit.labels_, mask).cpu()
    centers = fit.cluster_centers_.cpu().float().clone()
    # Match updated source's fallback before coarse grouping. A genuinely empty
    # level is rejected by the CRiSH registry rather than silently inventing rows.
    bad = ~torch.isfinite(centers).all(dim=1)
    for low, high in ((0, args.num_labeled_classes), (args.num_labeled_classes, k)):
        good = ~bad[low:high]
        if (~good).any():
            centers[low:high][~good] = centers[low:high][good].mean(0) if good.any() else 0.
    labels, prototypes = coarse_levels(features, base, centers, args.num_labeled_classes,
                                       args.num_unlabeled_classes)
    active = min(len(labels), 1 + runtime.epoch//args.level_interval)
    args.hierarchy_levels = len(labels)
    labels, prototypes = labels[:active], prototypes[:active]
    state = SupervisionState(ids, features, mask, legal, labels, prototypes,
        tuple(prototype_radii(p) for p in prototypes),
        tuple(torch.zeros(len(ids), dtype=torch.bool) for _ in labels))
    runtime.hierarchy_state = state
    return state
