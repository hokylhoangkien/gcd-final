"""Population supervision dose; no ground-truth accuracy or dense NxN graph."""
import torch


def base_target_mass(state, level=0):
    """Exact mean retained SupCon coefficient mass before online weighting.

    Every image contributes its protected other-view positive. An active
    image also retains both views of every other active image in its cluster.
    Denominators use original cluster support, as in pseudo_sse_loss.
    """
    labels = state.labels[level]
    n = torch.bincount(labels, minlength=len(state.prototypes[level])).double()
    m = torch.bincount(labels[~state.abstained[level]], minlength=len(n)).double()
    return float(((m*(2*m-1)+(n-m))/(2*n-1).clamp_min(1)).sum()/len(labels))


def _pairs(counts):
    return int((counts*(counts-1)).sum())//2


def supervision_dose(before, after):
    if not torch.equal(before.ids, after.ids) or len(before.labels) != len(after.labels):
        raise ValueError('Dose requires aligned supervision snapshots')
    edited = torch.zeros(len(before.ids), dtype=torch.bool)
    levels = []
    for level, (a, b) in enumerate(zip(before.labels, after.labels)):
        active_a, active_b = ~before.abstained[level], ~after.abstained[level]
        edited |= (a != b) | (active_a != active_b)
        na = torch.bincount(a[active_a], minlength=len(before.prototypes[level]))
        nb = torch.bincount(b[active_b], minlength=len(after.prototypes[level]))
        both = active_a & active_b
        common = torch.stack([a[both], b[both]], dim=1).unique(dim=0, return_counts=True)[1]
        # Count relation topology changes, distinct from coefficient mass/drift.
        changed = 2*(_pairs(na)+_pairs(nb)-2*_pairs(common))
        levels.append(dict(level=level, moved_rows=int((a != b).sum()),
            abstained_unlabeled=int((~after.labeled & ~active_b).sum()),
            changed_directed_image_positive_pairs=changed,
            unweighted_target_mass_before=base_target_mass(before, level),
            unweighted_target_mass_after=base_target_mass(after, level)))
    return dict(changed_ids=int(edited.sum()), changed_unlabeled_ids=int((edited & ~after.labeled).sum()),
                changed_labeled_ids=int((edited & after.labeled).sum()), levels=levels)


@torch.no_grad()
def batch_exposure(before,after,ids):
    """Actual directed view-pair removal in this sampled batch (duplicates legal)."""
    from .rebuild import pseudo_masks
    offdiag=~torch.eye(2*len(ids),dtype=torch.bool)
    levels=[]
    for level in range(len(before.labels)):
        _,original=pseudo_masks(before,ids,level)
        _,retained=pseudo_masks(after,ids,level)
        p=original.repeat(2,2)&offdiag
        removed=p&~retained.repeat(2,2)
        denominator=p.sum(1)
        if (denominator<=0).any():raise ValueError('Exposure lacks protected positives')
        levels.append(dict(level=level,removed_directed_view_pairs=int(removed.sum()),
            original_directed_view_pairs=int(p.sum()),exposure=float((removed.sum(1).double()/denominator).mean())))
    return levels


@torch.no_grad()
def use_saturation(state,ids,raw):
    """Exact zero_one indicator BEFORE confusion-factor min/max normalization."""
    from .ids import rows_for_ids
    rows=rows_for_ids(state.ids,ids)
    distance=torch.cdist(raw.detach(),raw.detach())
    offdiag=~torch.eye(len(raw),dtype=torch.bool,device=raw.device)
    result=[]
    for level in range(len(state.labels)):
        labels=state.labels[level][rows].to(raw.device).repeat(2)
        diameter=2*state.radii[level].to(raw.device)[labels]
        result.append(dict(level=level,frac_F1=float((distance>diameter)[offdiag].double().mean()),
                           mean_raw_feature_norm=float(raw.norm(dim=1).mean()),
                           mean_pair_distance=float(distance[offdiag].mean()),mean_2R=float(diameter.mean())))
    return result
