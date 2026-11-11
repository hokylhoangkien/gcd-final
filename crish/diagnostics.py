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
