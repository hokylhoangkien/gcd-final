"""Updated SelEx arithmetic with CRiSH's protected pseudo-SSE abstention."""
from functools import lru_cache
import torch
import torch.nn.functional as F

from crish.ids import rows_for_ids
from crish.rebuild import confusion_factor, pseudo_sse_loss, pseudo_masks
from crish.config import PairReliabilitySettings
from crish.online_pairs import online_pair_weights, pair_diagnostics
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
    weights, diagnostics = None, None
    settings = PairReliabilitySettings(**(getattr(args, 'pair_reliability', None) or {}))
    if getattr(args, 'method', 'baseline') == 'crish' and settings.enabled:
        weights, evidence = online_pair_weights(backbone_views, batch_ids, membership, settings, epoch)
        diagnostics = pair_diagnostics(weights, state.labels[0][rows], batch_ids, membership, evidence,
                                       retained=pseudo_masks(state, batch_ids, 0)[1])
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
        pseudo = pseudo + pseudo_sse_loss(criterion, paired[:, :, :width], state, batch_ids, level,
                            pair_weights=weights if level == 0 else None) / 2**(level+1)
    loss = (1-args.sup_con_weight)*use + args.sup_con_weight*(true+pseudo)/2
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite SelEx loss')
    return dict(loss=loss, angle=loss, distance=loss*0, features=flat,
                distance_weight=0., supervised=[true], use=use, pseudo=pseudo,
                pseudo_pair_weights=weights, pair_reliability=diagnostics)


def abstain_delta_loss(head, chunk, state_0, state_a, args):
    """Direct L(a)-L(Keep) from removed numerators; no subtraction of full losses.

    Chunk keys follow PopulationEngine: flat=[view0,view1], ids=stable IDs.
    Only pure Abstain with unchanged assignments/prototypes is admissible.
    """
    if (getattr(args, 'pair_reliability', None) or {}).get('enabled', False):
        raise ValueError('abstain_delta_loss requires pair_reliability.enabled=false')
    if not torch.equal(state_0.ids, state_a.ids) or len(state_0.labels) != len(state_a.labels):
        raise ValueError('Abstain requires aligned states')
    if any(not torch.equal(a, b) for a, b in zip(state_0.labels+state_0.prototypes+state_0.radii,
                                                  state_a.labels+state_a.prototypes+state_a.radii)):
        raise ValueError('Direct Abstain delta cannot score Move')
    flat = F.normalize(head(chunk['flat']), dim=-1)
    criterion = source_losses(str(flat.device))['SupConLoss']()
    loss = flat.sum()*0
    offdiag = ~torch.eye(len(flat), dtype=torch.bool, device=flat.device)
    for level in range(len(state_0.labels)):
        original, before = pseudo_masks(state_0, chunk['ids'], level)
        _, after = pseudo_masks(state_a, chunk['ids'], level)
        if (after & ~before).any():
            raise ValueError('Abstain delta may only remove positives')
        removed = (before & ~after).to(flat.device).repeat(2, 2) & offdiag
        width = flat.shape[1] // 2**(level+1)
        if width < 1:
            raise ValueError('Head too small for active levels')
        prefix = flat[:, :width]
        logits = -torch.cdist(prefix, prefix)/criterion.temperature
        logits = logits-logits.max(1, keepdim=True).values.detach()
        logprob = logits-torch.log((logits.exp()*offdiag).sum(1, keepdim=True))
        count = (original.to(flat.device).repeat(2, 2) & offdiag).sum(1)
        if (count <= 0).any() or not torch.isfinite(logprob).all():
            raise ValueError('Invalid positive count/logits')
        loss = loss + (criterion.temperature/criterion.base_temperature) * (
            (removed*logprob).sum(1)/count).mean()/2**(level+1)
    return args.sup_con_weight*loss/2
