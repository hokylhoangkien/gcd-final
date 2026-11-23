"""Frozen single/two-view evidence; augmentation never advances training RNG."""
from dataclasses import replace
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from .engine import fixed_modes
from .ids import rows_for_ids
from .lookahead import preserve_rng
from .global_policy import GlobalEvidence,knn_evidence
from .rank_policy import hash_key


@torch.no_grad()
def augmented_features(model,dataset,state,*,device,seed,epoch,batch_size=128,num_workers=0):
    indices=rows_for_ids(dataset.sample_ids,state.ids).tolist()
    generator=torch.Generator().manual_seed(hash_key(seed,epoch)%(2**63))
    result=[]
    with preserve_rng(),fixed_modes(model):
        # Workers (if any) are seeded from this loader's private generator.
        loader=DataLoader(Subset(dataset,indices),batch_size=batch_size,shuffle=False,
                          num_workers=num_workers,generator=generator)
        offset=0
        for views,_,ids,_membership in loader:
            if not torch.equal(torch.as_tensor(ids),state.ids[offset:offset+len(ids)]):
                raise ValueError('Augmented evidence IDs changed')
            result.append(F.normalize(model(views[0].to(device)),dim=-1).detach().cpu())
            offset+=len(ids)
    if offset!=len(state.ids):raise ValueError('Incomplete augmented evidence')
    return torch.cat(result)


@torch.no_grad()
def evidence_for_epoch(state,model,dataset,settings,*,device,seed,epoch,batch_size,num_workers,deadline=None):
    started=time.monotonic()
    normalized=replace(state,features=F.normalize(state.features,dim=-1))
    kw=dict(k=settings.knn_k,nonreciprocal_weight=settings.nonreciprocal_weight,
            device=device,query_block=settings.rank_block,deadline=deadline)
    single=knn_evidence(normalized,**kw)
    if settings.evidence_views==1:return single,single
    features=augmented_features(model,dataset,state,device=device,seed=seed,epoch=epoch,
                                 batch_size=batch_size,num_workers=num_workers)
    other=knn_evidence(replace(normalized,features=features),**kw)
    n,k=single.neighbors.shape
    match=single.neighbors[:,:,None]==other.neighbors[:,None,:]
    weights=torch.minimum(single.weights,(match*other.weights[:,None,:]).sum(-1))
    total=weights.sum(1)
    r=tuple(torch.where(total>0,(weights*(y[single.neighbors]==y[:,None])).sum(1)/total.clamp_min(1e-12),
                        torch.ones(n,dtype=torch.float64)) for y in state.labels)
    out=GlobalEvidence(single.neighbors,weights,r,float((weights==1).double().mean()),time.monotonic()-started)
    return out,single
