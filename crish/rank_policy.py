"""CRiSH-G v2 rank dose, partition-invariant EMA, deterministic hashed ties."""
from dataclasses import replace
import hashlib
import math
import torch
from .global_policy import _capacity, _mass_floor, histogram
from .diagnostics import base_target_mass


def hash_key(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(str(x) for x in parts).encode()).digest()[:8], 'big')


@torch.no_grad()
def ema_scores(state, evidence, previous, settings):
    """Stable sample IDs, not cluster IDs, define persistence across epochs."""
    previous = previous or {}
    enabled = (settings.reliability_ema or {}).get('enabled', False)
    beta = (settings.reliability_ema or {}).get('beta', .7)
    threshold = (settings.reliability_ema or {}).get('reset_jaccard', .5)
    aligned = previous.get('ids') is not None and torch.equal(previous['ids'], state.ids)
    scores, labels, resets = [], [], []
    for level, (current, r) in enumerate(zip(state.labels, evidence.reliability)):
        reset = torch.ones(len(current), dtype=torch.bool)
        smooth = r.clone()
        if enabled and aligned and level < len(previous.get('labels', [])):
            old = previous['labels'][level]
            pairs = torch.stack([current, old], 1)
            _, inverse, intersection = torch.unique(pairs, dim=0, return_inverse=True, return_counts=True)
            na = torch.bincount(current)[current]; nb = torch.bincount(old)[old]
            common = intersection[inverse]
            denominator = na+nb-common-1
            jac = torch.where(denominator>0,(common-1).double()/denominator.clamp_min(1),torch.ones_like(r))
            reset = jac < threshold
            smooth = torch.where(reset,r,beta*previous['scores'][level]+(1-beta)*r)
        scores.append(smooth if enabled else r)
        labels.append(current.clone())
        resets.append(int(reset.sum()) if enabled else 0)
    saved = dict(ids=state.ids.clone(), labels=labels, scores=[x.clone() for x in scores]) if enabled else {}
    return tuple(scores), saved, resets


def rank_order(state, rows, scores, seed, epoch, level, *, random=False):
    if random:
        return sorted(rows, key=lambda i: hash_key(seed,epoch,level,int(state.ids[i])))
    return sorted(rows, key=lambda i: (float(scores[i]),hash_key(seed,int(state.ids[i]))))


@torch.no_grad()
def apply_rank_dose(state, evidence, q, settings, *, seed, epoch, scores=None,
                    min_support=2, max_protected_shift=.25):
    if not math.isfinite(q) or not 0 <= q < 1:
        raise ValueError('rank dose must be finite in [0,1)')
    if any(a.any() for a in state.abstained):
        raise ValueError('Rank dose needs a fresh snapshot')
    scores = evidence.reliability if scores is None else scores
    if len(scores)!=len(state.labels):
        raise ValueError('Rank score levels differ')
    moved = torch.zeros(len(state.ids),dtype=torch.bool)
    current, move_record = state, dict(moved=0)
    if q and settings.actions=='abstain_move':
        from .rank_moves import apply_rank_moves
        preview, _ = apply_rank_dose(state,evidence,q,replace(settings,actions='abstain'),
                                    seed=seed,epoch=epoch,scores=scores)
        candidates=preview.abstained[0].nonzero().flatten().tolist()
        candidates=rank_order(state,candidates,scores[0],seed,epoch,0)
        current,moved,move_record=apply_rank_moves(state,evidence,candidates,settings,seed=seed,
                                                   min_support=min_support,max_protected_shift=max_protected_shift)
    masks=[torch.zeros(len(state.ids),dtype=torch.bool) for _ in state.labels]
    levels=[];unlabeled=~state.labeled
    oracle=None
    if settings.diagnostic_oracle:
        # Deliberately lazy: normal training never imports the annotation reader.
        from .oracle import load_oracle_errors
        oracle=load_oracle_errors(state,settings)
    for level,labels in enumerate(current.labels):
        r,s=evidence.reliability[level],scores[level]
        if (r.shape!=state.ids.shape or s.shape!=state.ids.shape or not torch.isfinite(r).all()
                or not torch.isfinite(s).all() or ((r<0)|(r>1)).any() or ((s<0)|(s>1)).any()):
            raise ValueError('Invalid rank reliability')
        # With two views, unsupported rows carry r=1 and are outside the boundary.
        boundary=unlabeled & (r<1)
        count=int(math.floor(q*int(boundary.sum()))) if int(boundary.sum())>=settings.min_boundary_size else 0
        pool=unlabeled & ~moved
        if settings.random_control!='count_matched' and oracle is None:
            pool &= boundary & (s<settings.abstain_ceiling)
        if oracle is not None:pool &= oracle[level]
        if level and settings.hierarchical_abstain:pool &= masks[level-1]
        order=rank_order(state,pool.nonzero().flatten().tolist(),s,seed,epoch,level,
                          random=settings.random_control=='count_matched' or oracle is not None)
        capacity=_capacity(state.labels[level],unlabeled,len(state.prototypes[level]),settings)
        support=torch.bincount(state.labels[level][unlabeled],minlength=len(capacity))
        used=torch.zeros_like(capacity)
        # Every Move departure consumes source quota, even when coarse parent agrees.
        for row in moved.nonzero().flatten().tolist():used[int(state.labels[level][row])]+=1
        selected=[];capblocked=keptblocked=0
        if count:
            for row in order:
                group=int(state.labels[level][row])
                if used[group]>=int(support[group]*settings.max_cluster_fraction):capblocked+=1;continue
                if support[group]-used[group]<=settings.min_cluster_kept:keptblocked+=1;continue
                used[group]+=1;selected.append(row)
                if len(selected)==count:break
        kept,truncated=_mass_floor(current,level,labels,selected,settings.min_retained_mass)
        masks[level][kept]=True
        anchors=torch.zeros(len(state.prototypes[level]),dtype=torch.bool)
        anchors[labels[state.labeled]]=True
        levels.append(dict(level=level,scoped=True,boundary_size=int(boundary.sum()),requested_count=count,
            candidates=int(pool.sum()),abstained_unlabeled=len(kept),fraction=len(kept)/max(1,int(unlabeled.sum())),
            cluster_capped=capblocked,min_kept_blocked=keptblocked,truncated_by_mass_floor=truncated,
            retained_mass=1.,shortfall=count-len(kept),anchored_fraction=float(anchors[labels[kept]].double().mean()) if kept else None,
            reliability_histogram=histogram(r[unlabeled]),reliability_mean_unlabeled=float(r[unlabeled].mean())))
    changed=any(m.any() for m in masks) or bool(moved.any())
    outcome=replace(current,abstained=tuple(masks),version=state.version+1) if changed else state
    for level,record in enumerate(levels):
        record['retained_mass']=base_target_mass(outcome,level)
        if record['retained_mass']+1e-12<settings.min_retained_mass:
            raise ValueError('Rank mass guard failed')
    return outcome,dict(move_record,q=float(q),changed=changed,levels=levels,moved=int(moved.sum()))
