"""Full Hyp-SelEx USE/SSE objective with explicit ID-aligned supervision.

The original loss classes/logit routine remain in trainer.py, unchanged. Both
strict B0 and CRiSH use this function and the same fail-on-zero USE contract.
"""
import torch

from crish.ids import rows_for_ids
from crish.rebuild import confusion_factor, pseudo_sse_loss


def distance_weight(args, epoch):
    denominator = args.hyper_end_epoch-args.hyper_start_epoch
    if denominator <= 0:
        raise ValueError('Invalid baseline hyperbolic ramp interval')
    return max(0., min(1., (epoch-(args.hyper_start_epoch-1))/denominator)) * args.hyper_max_weight


def hypselex_loss(head, backbone_views, batch_ids, labeled, legal_targets, state, args, epoch):
    from gcd_runtime import trainer
    n = len(batch_ids)
    rows = rows_for_ids(state.ids, batch_ids)
    membership = state.labeled[rows]
    if not torch.equal(labeled.detach().cpu().bool(), membership):
        raise ValueError('Batch labeled membership disagrees with frozen registry')
    expected = state.labels[0][rows][membership]
    if not torch.equal(legal_targets.detach().cpu().long(), expected):
        raise ValueError('Only registered legally labeled targets may enter the objective')
    if len(backbone_views) != 2*n or args.n_views != 2 or args.contrast_unlabel_only:
        raise ValueError('Verified CRiSH objective requires two views and full-batch USE')
    flat = head(backbone_views)
    paired = torch.stack(flat.chunk(2), dim=1)
    cf = confusion_factor(state, batch_ids, backbone_views.detach(), args.strategy)
    mask = membership.to(flat.device)
    branches, protected = [], []
    for curvature in (0., args.c):
        criterion = trainer.SupConLoss(hyper_c=curvature)
        logits, targets, similarity = trainer.info_nce_logits(flat, cf, args, hyper_c=curvature)
        use = trainer.LabelSmoothingLoss()(logits, targets, similarity, args.unsupervised_smoothing)
        true_loss = (criterion(paired[mask], labels=legal_targets.to(flat.device)) if mask.any()
                     else flat.sum()*0)
        pseudo = flat.sum()*0
        for level in range(len(state.labels)):
            width = flat.shape[-1] // 2**(level+1)
            if width < 1:
                raise ValueError('Projection dimension too small for this HSSK stack')
            pseudo = pseudo + pseudo_sse_loss(criterion, paired[:, :, :width], state, batch_ids, level) / 2**(level+1)
        branches.append((1-args.sup_con_weight)*use + args.sup_con_weight*(true_loss+pseudo)/2)
        protected.append(true_loss)
    weight = distance_weight(args, epoch)
    loss = (1-weight)*branches[0] + weight*branches[1]
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite strict Hyp-SelEx loss')
    return dict(loss=loss, angle=branches[0], distance=branches[1],
                supervised=protected, features=flat, distance_weight=weight)
