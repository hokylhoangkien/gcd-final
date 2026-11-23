import pytest
from gcd_runtime.monitoring import monitor_due
from scripts.evaluate_kmeans_seeds import summarize

def test_last_epoch_schedule_and_multiseed_summary():
    c=dict(epochs=200,ss_eval_every_epochs=5,ss_eval_last_epochs=10,final_ss_evaluation=True)
    assert monitor_due(c,189) and all(monitor_due(c,e) for e in range(190,199))
    assert not monitor_due(c,188) and not monitor_due(c,199)
    rows=[dict(all_acc=x,old_acc=x+.01,new_acc=x-.01) for x in [.7,.8,.9,.8,.8]]
    stats=summarize(rows)
    assert stats['all_acc']['median']==.8 and stats['all_acc']['mean']==pytest.approx(.8)
    assert stats['all_acc']['std']==pytest.approx((.02/4)**.5)
