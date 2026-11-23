"""CPU contracts for the opt-in CRiSH-G v2 path."""
from dataclasses import replace
import torch
import pytest
from crish.probe import LegalProbe, ProbeWeights

def probe(head, reduction='mean_all'):
    x = torch.tensor([[[1., 2.], [2., 1.]], [[-1., 2.], [2., -1.]]], dtype=next(head.parameters()).dtype)
    return LegalProbe(known_ids=torch.tensor([1,2]), known_views=x, known_targets=torch.tensor([0,1]),
        unlabeled_ids=torch.tensor([3,4]), unlabeled_views=x+.2, teacher_views=head(x.flatten(0,1)).reshape(2,2,-1),
        proposal_ids=torch.tensor([9]), legal_labels={1:0,2:1}, weights=ProbeWeights(1.,1.,.1,.1,reduction))

def test_view_reduction():
    torch.manual_seed(3)
    head=torch.nn.Linear(2,16)
    a,b=probe(head),probe(head,'per_row_sum')
    c,d=a.components(head),b.components(head)
    assert 0 <= d['view'] <= 4.000001
    assert torch.equal(c['known'],d['known']) and torch.equal(c['variance'],d['variance'])
    assert float(d['view']) == pytest.approx(float(c['view'])*16)
    u=torch.nn.functional.normalize(head(a.unlabeled_views.flatten(0,1)),dim=1).reshape(2,2,-1)
    t=torch.nn.functional.normalize(a.teacher_views,dim=-1)
    assert torch.equal(c['view'],(u-t.flip(1)).square().mean())

from crish.lookahead import assert_plain_sgd, first_order_component_gains

def test_optimizer_and_fp64_score():
    h=torch.nn.Linear(2,2)
    for kw in [dict(momentum=.9,nesterov=True),dict(momentum=.9,dampening=.1)]:
        with pytest.raises(ValueError):assert_plain_sgd(torch.optim.SGD(h.parameters(),lr=.1,**kw))
    assert_plain_sgd(torch.optim.SGD(h.parameters(),lr=.1,momentum=.9,weight_decay=.01))
    q=torch.tensor([1e8,1.,-1e8]); d=torch.ones(3)
    assert first_order_component_gains({'view':(q,)},(d,),.1)['view']==pytest.approx(.1)

def test_first_order_matches_real_sgd_quadratic_remainder():
    # An analytic smooth probe isolates O(eta^2), with nonzero momentum and wd.
    p=torch.nn.Parameter(torch.tensor([.5,-.3],dtype=torch.float64))
    g0=torch.tensor([.4,-.1],dtype=torch.float64); delta=torch.tensor([.2,-.3],dtype=torch.float64)
    residual=[]
    for lr in [1e-3,5e-4]:
        values=[]
        for g in [g0,g0+delta]:
            v=torch.nn.Parameter(p.detach().clone())
            opt=torch.optim.SGD([v],lr=lr,momentum=.9,weight_decay=.2)
            opt.state[v]['momentum_buffer']=torch.tensor([.23,-.51],dtype=torch.float64)
            v.grad=g.clone();opt.step();values.append(v.detach())
        q=2*values[0]
        approx=first_order_component_gains({'view':(q,)},(delta,),lr)['view']
        exact=float(values[0].square().sum()-values[1].square().sum())
        assert abs(approx-exact)/abs(exact)<1e-3
        residual.append(abs(approx-exact))
    assert 3 <= residual[0]/residual[1] <= 5

def test_abstain_delta_gradients_match_full_objective():
    from tests.test_selex_runtime import fixture
    from selex_runtime.objective import selex_loss,abstain_delta_loss
    ds,state,args,head=fixture(epoch=10)
    head=head.double();raw=torch.cat([ds.features,ds.features+.03]).double() if hasattr(ds,'features') else None
    # FeatureFixture exposes image tensors through __getitem__, not hidden labels.
    raw=torch.cat([torch.stack([ds[i][0][v] for i in range(len(ds))]) for v in range(2)]).double()
    masks=[m.clone() for m in state.abstained]
    for m in masks:m[(~state.labeled).nonzero().flatten()[:7]]=True
    proposed=replace(state,abstained=tuple(masks),version=1)
    ids=state.ids;membership=state.labeled;legal=state.known_targets
    a=selex_loss(head,raw,ids,membership,legal,proposed,args,10)['loss']
    b=selex_loss(head,raw,ids,membership,legal,state,args,10)['loss']
    expected=torch.autograd.grad(a-b,tuple(head.parameters()))
    actual=torch.autograd.grad(abstain_delta_loss(head,dict(flat=raw,ids=ids),state,proposed,args),tuple(head.parameters()))
    for x,y in zip(expected,actual):torch.testing.assert_close(x,y,rtol=1e-8,atol=1e-11)
    zero=torch.autograd.grad(abstain_delta_loss(head,dict(flat=raw,ids=ids),state,state,args),tuple(head.parameters()))
    assert all(torch.count_nonzero(x)==0 for x in zero)
