"""Updated SelEx arithmetic with CRiSH's protected pseudo-SSE abstention."""
from functools import lru_cache
import torch
import torch.nn.functional as F

from crish.ids import rows_for_ids
from crish.rebuild import confusion_factor, pseudo_sse_loss
from .reference import definitions


@lru_cache(maxsize=4)
def source_losses(device):
    return definitions(device)


def selex_loss(head, backbone_views, batch_ids, labeled, legal_targets, state, args, epoch):
    rows = rows_for_ids(state.ids, batch_ids)
    membership = state.labeled[rows]
    if not torch.equal(labeled.detach().cpu().bool(), membership):
        raise ValueError('SelEx membership mismatch')
    if not torch.equal(legal_targets.detach().cpu().long(), state.labels[0][rows][membership]):
        raise ValueError('SelEx legal targets mismatch')
    expected = min(args.hierarchy_levels, 1 + epoch // args.level_interval)
    if len(state.labels) != expected:
        raise ValueError('Loss, repair and drift must see exactly the active levels')
    n = len(batch_ids)
    if len(backbone_views) != 2*n or args.n_views != 2:
        raise ValueError('SelEx requires two complete views')
    scope = source_losses(str(backbone_views.device))
    # Match source normalization before both USE and prefix-wise SSE. The
    # original info_nce_logits normalizes again, deliberately preserved.
    flat = F.normalize(head(backbone_views), dim=-1)
    paired = torch.stack(flat.chunk(2), dim=1)
    cf = confusion_factor(state, batch_ids, backbone_views.detach(), args.strategy,
                          normalization='selex_legacy')
    logits, targets, smoothing = scope['info_nce_logits'](flat, cf, args)
    use = scope['LabelSmoothingLoss']()(logits, targets, smoothing, args.unsupervised_smoothing)
    criterion = scope['SupConLoss']()
    criterion.hyper_c = 0  # shared Abstain adapter uses original Euclidean logits
    mask = membership.to(flat.device)
    true = criterion(paired[mask], labels=legal_targets.to(flat.device)) if mask.any() else flat.sum()*0
    pseudo = flat.sum()*0
    for level in range(len(state.labels)):
        width = flat.shape[-1] // 2**(level+1)
        if width < 1:
            raise ValueError('SelEx head too small for active hierarchy')
        pseudo = pseudo + pseudo_sse_loss(criterion, paired[:, :, :width], state, batch_ids, level) / 2**(level+1)
    loss = (1-args.sup_con_weight)*use + args.sup_con_weight*(true+pseudo)/2
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite SelEx loss')
    return dict(loss=loss, angle=loss, distance=loss*0, features=flat,
                distance_weight=0., supervised=[true], use=use, pseudo=pseudo)
