"""Offline-only image bootstrap and semantic diagnostics; never imported by train."""
from collections import Counter
import numpy as np
from scripts.audit_global_policy import auroc,fraction,unique_majority,pair_counts


def errors(clusters,truth,unlabeled):
    groups={}
    for c,y in zip(clusters,truth):groups.setdefault(int(c),Counter())[int(y)]+=1
    valid=np.zeros(len(truth),dtype=bool);wrong=valid.copy()
    for i in np.flatnonzero(unlabeled):
        counts=groups[int(clusters[i])].copy();counts[int(truth[i])]-=1
        predicted=unique_majority(counts)
        if predicted is not None:valid[i]=True;wrong[i]=predicted!=truth[i]
    return valid,wrong,groups


def extended_epoch(arrays,truth_map):
    truth=np.array([truth_map[int(i)] for i in arrays['ids']]);labeled=arrays['labeled'].astype(bool)
    old=np.isin(truth,np.unique(truth[labeled]));out=[]
    for level in sorted(int(k.split('_')[1]) for k in arrays if k.startswith('label_')):
        labels=arrays[f'label_{level}'];a=arrays[f'abstained_{level}'].astype(bool)
        valid,wrong,_=errors(labels,truth,~labeled)
        count=pair_counts(labels,truth,a);allpairs=count['removed']+count['retained']
        wrongpairs=count['removed_wrong']+count['retained_wrong']
        base=wrongpairs/allpairs if allpairs else None
        removed=count['removed_wrong']/count['removed'] if count['removed'] else None
        scores={}
        for prefix in ('reliability','ema_reliability','single_view_reliability'):
            key=f'{prefix}_{level}'
            if key not in arrays:continue
            scores[prefix]={name:auroc(1-arrays[key][valid&mask],wrong[valid&mask]) for name,mask in
                [('All',np.ones(len(truth),dtype=bool)),('Old',old),('New',~old)]}
        out.append(dict(level=level,auroc=scores,baseline_pair_wrong_rate=fraction(wrongpairs,allpairs),
            removed_pair_error_enrichment=removed/base if base and removed is not None else None,
            abstained_old=fraction((a&~labeled&old).sum(),(~labeled&old).sum()),
            abstained_new=fraction((a&~labeled&~old).sum(),(~labeled&~old).sum())))
    labels=arrays['label_0'];_,_,groups=errors(labels,truth,~labeled)
    move={name:[0,0] for name in ('anchored','unanchored')}
    anchored=set(labels[labeled].tolist())
    for i in np.flatnonzero(arrays['moved']):
        target=int(arrays['move_target'][i]);pred=unique_majority(groups.get(target,Counter()))
        if pred is not None:
            row=move['anchored' if target in anchored else 'unanchored'];row[0]+=int(pred==truth[i]);row[1]+=1
    return dict(levels=out,move_target_accuracy={k:fraction(*v) for k,v in move.items()})


def weighted_enrichment(clusters,truth,abstained,weights):
    # Weighted distinct-original-image pairs; repeated bootstrap copies never
    # introduce same-ID pairs that cannot be removed by pseudo_masks.
    def counts(mask):
        c,y,w=clusters[mask],truth[mask],weights[mask]
        _,ci=np.unique(c,return_inverse=True)
        _,cy=np.unique(np.stack([c,y],axis=1),axis=0,return_inverse=True)
        total=(np.bincount(ci,weights=w)**2-np.bincount(ci,weights=w*w)).sum()/2
        correct=(np.bincount(cy,weights=w)**2-np.bincount(cy,weights=w*w)).sum()/2
        return total,total-correct
    all_n,all_bad=counts(np.ones(len(truth),dtype=bool));kept_n,kept_bad=counts(~abstained)
    n,bad=all_n-kept_n,all_bad-kept_bad
    return (bad/n)/(all_bad/all_n) if n>0 and all_bad>0 and all_n>0 else None


def enrichment_ratio(first,second,truth_map,level=0,repetitions=2000,seed=0):
    if not np.array_equal(first['ids'],second['ids']):raise ValueError('Bootstrap requires aligned identical sample IDs')
    if not np.array_equal(first['labeled'],second['labeled']):raise ValueError('Bootstrap memberships differ')
    n=len(first['ids']);truth=np.array([truth_map[int(i)] for i in first['ids']]);rng=np.random.default_rng(seed)
    def measure(a,w):return weighted_enrichment(a[f'label_{level}'],truth,a[f'abstained_{level}'].astype(bool),w)
    a,b=measure(first,np.ones(n)),measure(second,np.ones(n))
    ratios=[]
    # Resample images with the same multiplicities for both runs. Legal anchors
    # remain in the pre-intervention relation graph as in the pair definition.
    for _ in range(repetitions):
        w=np.bincount(rng.integers(n,size=n),minlength=n).astype(float)
        x,y=measure(first,w),measure(second,w)
        if x is not None and y:ratios.append(x/y)
    return dict(level=level,ratio=a/b if a is not None and b else None,bootstrap_unit='image, paired across runs',
        repetitions=repetitions,valid_repetitions=len(ratios),seed=seed,
        ci95=np.quantile(ratios,[.025,.975]).tolist() if ratios else None)


def persistence(snapshots,truth_map):
    out=[]
    for epoch,a in sorted(snapshots.items()):
        if epoch+5 not in snapshots:continue
        b=snapshots[epoch+5]
        index={int(i):j for j,i in enumerate(b['ids'])}
        if set(index)!=set(a['ids'].tolist()):raise ValueError('Temporal snapshot population changed')
        order=np.array([index[int(i)] for i in a['ids']]);truth=np.array([truth_map[int(i)] for i in a['ids']])
        va,wa,_=errors(a['label_0'],truth,~a['labeled'].astype(bool))
        vb,wb,_=errors(b['label_0'][order],truth,~b['labeled'][order].astype(bool))
        mask=va&vb
        out.append(dict(epoch=epoch,next_epoch=epoch+5,
            wrong_after_given_wrong=fraction((wa&wb&mask).sum(),(wa&mask).sum()),
            marginal_wrong_after=fraction((wb&mask).sum(),mask.sum())))
    return out


def replay_ema(snapshots,beta=.7,reset_jaccard=.5):
    """Counterfactual score smoothing ONLY, with no annotations or model training.

    Gaps reset rather than invent unobserved epochs. Existing raw reliabilities
    and partitions are preserved; the derived scores are not observed runs.
    """
    previous=None;previous_epoch=None;gap_resets=[]
    for epoch,a in sorted(snapshots.items()):
        contiguous=previous is not None and epoch==previous_epoch+1
        if previous is not None and not contiguous:gap_resets.append(epoch)
        aligned=contiguous and np.array_equal(a['ids'],previous['ids'])
        for key in sorted(k for k in a if k.startswith('label_')):
            level=key.split('_')[-1];r=a[f'reliability_{level}'].astype(float)
            score=r.copy()
            if aligned and key in previous:
                x,y=a[key],previous[key]
                _,ix,nx=np.unique(x,return_inverse=True,return_counts=True)
                _,iy,ny=np.unique(y,return_inverse=True,return_counts=True)
                _,joint,nxy=np.unique(np.stack([x,y],1),axis=0,return_inverse=True,return_counts=True)
                common=nxy[joint];den=nx[ix]+ny[iy]-common-1
                jac=np.divide(common-1,den,out=np.ones(len(x),dtype=float),where=den>0)
                score=np.where(jac<reset_jaccard,r,beta*previous[f'ema_reliability_{level}']+(1-beta)*r)
            a[f'ema_reliability_{level}']=score
        previous=a;previous_epoch=epoch
    return dict(kind='offline_counterfactual_score_ema_not_training',beta=beta,reset_jaccard=reset_jaccard,
                gap_reset_epochs=gap_resets,complete_epoch_sequence=not gap_resets)
