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
