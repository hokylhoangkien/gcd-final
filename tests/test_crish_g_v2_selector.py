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
