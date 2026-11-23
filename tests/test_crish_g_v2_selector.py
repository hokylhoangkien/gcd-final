import pytest

from crish.normalized_selector import component_statistics,select_dose

def test_pooled_selector_one_se_and_scale_floor():
    chunks={q:[dict(known=0.,view=v,variance=1e-8) for v in vals] for q,vals in
            [(.1,[1.,2.,3.,2.]),(.25,[1.2,2.2,3.2,2.2]),(.4,[1.3,2.3,3.3,2.3]) ]}
    stats,scale=component_statistics(chunks)
    assert scale['view']['raw']==pytest.approx((2/3)**.5)
    q,highest=select_dose(stats)
    assert highest==.4 and q==.1
    assert all(x['t']['variance']==0 for x in stats.values())
    floored,sc=component_statistics(chunks,{'view':[10.,12.]})
    assert sc['view']['effective']==pytest.approx(2.75)
    assert select_dose(floored,t_min=100)==(0.,0.)

import torch
from dataclasses import replace
from tests.test_selex_runtime import fixture
from crish.global_policy import GlobalEvidence,knn_evidence
from crish.population_settings import PopulationSettings
from crish.rank_policy import apply_rank_dose,ema_scores

def v2_settings(**changes):
    v=dict(enabled=True,ranking='knn_agreement',selection='threshold',dose_mode='rank_quantile',
        scorer='first_order_fp64',selector='normalized_v1',scoring_batches=2,confirmation_batches=2,
        levels='active',hierarchical_abstain=True,tie_break='hash',min_retained_mass=.2,abstain_ceiling=.5)
    v.update(changes);return PopulationSettings(**v)

def test_rank_dose_nested_quotas_and_hash_row_permutation():
    _,s,_,_=fixture(epoch=10)
    r=torch.full((len(s.ids),),.2,dtype=torch.float64)
    ev=GlobalEvidence(torch.zeros(len(r),1,dtype=torch.long),torch.ones(len(r),1),(r,r),1.,0.)
    settings=v2_settings()
    a,cov=apply_rank_dose(s,ev,.25,settings,seed=3,epoch=10)
    assert all(int(m.sum())==8 for m in a.abstained)
    assert (a.abstained[1]<=a.abstained[0]).all()
    assert not a.abstained[0][s.labeled].any()
    perm=torch.arange(len(s.ids)-1,-1,-1)
    b=replace(s,ids=s.ids[perm],features=s.features[perm],labeled=s.labeled[perm],known_targets=s.labels[0][perm][s.labeled[perm]],
        labels=tuple(x[perm] for x in s.labels),abstained=tuple(x[perm] for x in s.abstained))
    out,_=apply_rank_dose(b,ev,.25,settings,seed=3,epoch=10)
    assert set(s.ids[a.abstained[0]].tolist())==set(b.ids[out.abstained[0]].tolist())

def test_ema_partition_label_permutation_and_reset():
    _,s,_,_=fixture(epoch=10);ev=knn_evidence(s,k=3,nonreciprocal_weight=.5)
    settings=v2_settings(reliability_ema=dict(enabled=True,beta=.7,reset_jaccard=.5))
    score,cache,resets=ema_scores(s,ev,None,settings)
    # Legal known anchors fixed; permute novel labels only.
    y=s.labels[0].clone();y[s.labels[0]==2]=3;y[s.labels[0]==3]=2
    swapped=replace(s,labels=(y,s.labels[1]))
    again,saved,resets=ema_scores(swapped,ev,cache,settings)
    assert resets==[0,0]
    assert all(torch.equal(a,b) for a,b in zip(score,again))
    assert saved['labels'][0].device.type=='cpu'

def test_rank_move_propagates_and_protects_anchors():
    from tests.test_global_policy import blob_state,blob_evidence
    from crish.rank_moves import apply_rank_moves
    s=blob_state();ev=blob_evidence(s)
    out,moved,record=apply_rank_moves(s,ev,[18,19],v2_settings(actions='abstain_move',move_target_cap=.5),seed=1)
    assert moved.sum()==2 and not moved[s.labeled].any()
    assert (out.labels[0][moved]==2).all() and (out.labels[1][moved]==1).all()
    assert torch.equal(out.labels[0][s.labeled],s.known_targets)
    assert all(torch.isfinite(p).all() for p in out.prototypes)

def test_two_view_evidence_preserves_rng_and_modes():
    from crish.evidence_v2 import evidence_for_epoch
    from tests.test_global_policy import rng_snapshot,same_rng
    ds,s,args,head=fixture(epoch=10)
    model=torch.nn.Identity().train()
    before=rng_snapshot()
    ev,one=evidence_for_epoch(s,model,ds,v2_settings(evidence_views=2,knn_k=3),device='cpu',seed=4,epoch=10,batch_size=8,num_workers=0)
    assert same_rng(before,rng_snapshot()) and model.training
    assert (ev.weights<=one.weights).all()
    assert all(((r>=0)&(r<=1)).all() for r in ev.reliability)


def test_ema_resets_after_partition_membership_change():
    _,s,_,_=fixture(epoch=10)
    r=torch.full((len(s.ids),),.2,dtype=torch.float64)
    ev=GlobalEvidence(torch.zeros(len(r),1,dtype=torch.long),torch.ones(len(r),1),(r,r),1.,0.)
    settings=v2_settings(reliability_ema=dict(enabled=True,beta=.7,reset_jaccard=.5))
    _,cache,_=ema_scores(s,ev,None,settings)
    y=s.labels[0].clone()
    rows=(~s.labeled).nonzero().flatten();y[rows]=torch.arange(len(rows))%4
    changed=replace(s,labels=(y,s.labels[1]))
    newr=torch.full_like(r,.8);new_ev=replace(ev,reliability=(newr,newr))
    scores,_,resets=ema_scores(changed,new_ev,cache,settings)
    assert resets[0]>0 and (scores[0]==.8).any()


def test_random_rank_count_determinism_and_score_independent_order():
    _,s,_,_=fixture(epoch=10)
    r=torch.full((len(s.ids),),.2,dtype=torch.float64)
    ev=GlobalEvidence(torch.zeros(len(r),1,dtype=torch.long),torch.ones(len(r),1),(r,r),1.,0.)
    settings=v2_settings(mode='fixed',gate='none')
    fixed,_=apply_rank_dose(s,ev,.25,settings,seed=2,epoch=10)
    random=replace(settings,random_control='count_matched')
    a,_=apply_rank_dose(s,ev,.25,random,seed=2,epoch=10)
    b,_=apply_rank_dose(s,replace(ev,reliability=(r+.1,r+.1)),.25,random,seed=2,epoch=10)
    assert all(int(x.sum())==int(y.sum()) for x,y in zip(a.abstained,fixed.abstained))
    assert all(torch.equal(x,y) for x,y in zip(a.abstained,b.abstained))


def test_oracle_explicit_permission_and_leave_one_out(tmp_path):
    from crish.oracle import load_oracle_errors
    _,s,_,_=fixture(epoch=10)
    truth=s.labels[0].clone();row=int((~s.labeled).nonzero()[0]);truth[row]=(truth[row]+1)%4
    labels=tmp_path/'labels.csv'
    labels.write_text('sample_id,true_label\n'+''.join(f'{int(i)},{int(y)}\n' for i,y in zip(s.ids,truth)))
    with pytest.raises(ValueError,match='diagnostic_oracle'):
        load_oracle_errors(s,v2_settings())
    settings=v2_settings(mode='fixed',gate='none',diagnostic_oracle=True,oracle_labels_csv=str(labels))
    masks=load_oracle_errors(s,settings)
    assert masks[0][row] and not masks[0][s.labeled].any()


def test_augmented_pass_seed_is_private_even_with_zero_workers():
    import random
    import numpy as np
    from crish.evidence_v2 import augmented_features
    from tests.test_global_policy import rng_snapshot,same_rng
    ds,state,_,_=fixture(epoch=10)
    class StochasticDataset:
        sample_ids=ds.sample_ids
        def __len__(self):return len(ds)
        def __getitem__(self,i):
            views,y,sid,m=ds[i]
            noise=torch.rand_like(views[0])+random.random()+float(np.random.rand())
            return [v+noise for v in views],y,sid,m
    model=torch.nn.Identity().train();wrapped=StochasticDataset()
    before=rng_snapshot()
    a=augmented_features(model,wrapped,state,device='cpu',seed=1,epoch=10,batch_size=8,num_workers=0)
    assert same_rng(before,rng_snapshot())
    torch.rand(17);random.random();np.random.rand(19)
    rng=rng_snapshot()
    b=augmented_features(model,wrapped,state,device='cpu',seed=1,epoch=10,batch_size=8,num_workers=0)
    assert same_rng(rng,rng_snapshot()) and torch.equal(a,b) and model.training
