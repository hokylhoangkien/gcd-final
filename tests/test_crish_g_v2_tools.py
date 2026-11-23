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


def test_offline_scale_audit_and_paired_tests(tmp_path):
    import json
    from scripts.audit_probe_scale import analyze
    from scripts.paired_statistics import analyze as paired
    log=tmp_path/'crish.jsonl'
    log.write_text(json.dumps(dict(gains=[dict(kind='move',proxy=dict(raw=.2),target_drift=dict(use_target_half_l1_mean=0.),
        exact_discovery=dict(gain=.2,keep=dict(known=2.,view=.3,variance=.1),action=dict(known=1.9,view=.2,variance=.1)))]))+'\n')
    out=analyze([log])
    assert out['same_sign_fraction']==1. and out['move_zero_use_drift']['fraction']==1.
    assert out['median_component_share']['known']==pytest.approx(.5)
    report=paired([1.,2.,3.])
    assert report['mean']==2 and report['std']==1 and report['ci95'][0]<2<report['ci95'][1]
    assert report['sign_test']['pvalue']==pytest.approx(.25)


def test_offline_image_bootstrap_and_temporal_errors():
    import numpy as np
    from scripts.global_policy_statistics import weighted_enrichment,enrichment_ratio,persistence
    arrays=dict(ids=np.arange(8),labeled=np.array([True,False,False,False,True,False,False,False]),
        label_0=np.array([0,0,0,0,1,1,1,1]),abstained_0=np.array([False,False,False,True,False,False,False,True]))
    truth=np.array([0,0,0,1,1,1,1,0]);labels=dict(enumerate(truth.tolist()))
    value=weighted_enrichment(arrays['label_0'],truth,arrays['abstained_0'],np.ones(8))
    assert value==pytest.approx(2.)
    stats=enrichment_ratio(arrays,arrays,labels,repetitions=30)
    assert stats['ratio']==1. and stats['ci95']==[1.,1.]
    temporal=persistence({0:arrays,5:arrays},labels)
    assert temporal[0]['wrong_after_given_wrong']['fraction']==1.


def test_phase3_profile_commands_and_smoke_cli(tmp_path):
    from scripts.prepare_global_policy import prepare
    from gcd_runtime.config import ROOT
    from tests.test_global_policy import cli_dry_run
    import json
    prepare(ROOT/'configs/selex_scars_crish_200.json',tmp_path/'plan',[1],profiles=('matched_selex','g2_cf'),phase='phase3')
    config=tmp_path/'plan/g2_cf_seed1.json'
    cfg=json.loads(config.read_text())
    assert cfg['ss_eval_every_epochs']==5 and cfg['ss_eval_last_epochs']==10
    assert '--eval-every 5' in (tmp_path/'plan/commands.txt').read_text()
    _,(smoke,)=cli_dry_run(['smoke','--config',str(config),'--variant','global','--smoke-epochs','2','--dry-run'])
    assert smoke['epochs']==2 and smoke['max_steps_per_epoch']==2
