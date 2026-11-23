"""Real-objective v2 controller integration and label/RNG isolation."""
import copy
import json
from unittest.mock import patch
import numpy as np
import torch
from crish.population import PopulationEngine
from gcd_runtime.checkpoint import EpochRuntime
from selex_runtime.objective import selex_loss
from tests.test_global_policy import small_config, rng_snapshot, same_rng
from tests.test_selex_runtime import fixture
from tests.test_crish_g_v2_selector import v2_settings


def run(directory,**changes):
    ds,state,args,head=fixture(epoch=10)
    cfg=small_config()
    cfg['crish']['probe_view_reduction']='per_row_sum'
    cfg['population_policy']=v2_settings(train_batch=4,knn_k=3,warmup_epochs=10,refresh_every=5,
        persist=True,confirm_fresh_chunks=True,**changes).to_dict()
    runtime=EpochRuntime(directory,cfg)
    engine=PopulationEngine(runtime,objective=selex_loss)
    opt=torch.optim.SGD(head.parameters(),lr=.01,momentum=.9,weight_decay=.01)
    before=copy.deepcopy(head.state_dict());rng=rng_snapshot()
    result=engine.prepare(state,torch.nn.Identity(),head,opt,ds,args,10)
    assert same_rng(rng,rng_snapshot())
    assert all(torch.equal(before[k],head.state_dict()[k]) for k in before)
    return ds,state,args,head,engine,result


def test_real_controller_scores_and_uses_disjoint_fresh_chunks(tmp_path):
    ds,state,args,head,engine,out=run(tmp_path)
    record=engine.runtime.last_population_record
    assert record['status']=='completed',record
    assert any('discovery' in c for c in record['candidates'])
    assert set(record['scoring_ids']).isdisjoint(record['confirmation_scoring_ids'])
    assert set(record['probe_ids'][0]).isdisjoint(record['probe_ids'][1])
    for ids in record['probe_ids']:
        assert set(ids).isdisjoint(record['scoring_ids']+record['confirmation_scoring_ids'])
    with patch('crish.population_v2.RankPopulationController.select',side_effect=AssertionError('probe')):
        before=rng_snapshot()
        again=engine.prepare(state,None,head,None,None,args,11)
        assert same_rng(before,rng_snapshot())
    assert 'population_policy_state' in engine.runtime.checkpoint_extra


def test_fixed_random_oracle_profiles_and_isolation(tmp_path):
    import subprocess,sys
    from scripts.prepare_global_policy import global_variants
    from gcd_runtime.config import load_config,ROOT
    base=load_config(ROOT/'configs/selex_scars_crish_200.json')
    profiles=global_variants(base,('matched_selex','g2_fixed','g2_random','g2_cf','g2_cf_noconfirm'))
    assert profiles['matched_selex']['method']=='baseline'
    assert profiles['g2_random']['population_policy']['random_control']=='count_matched'
    assert profiles['g2_cf_noconfirm']['population_policy']['confirm'] is False
    code='import sys;from scripts.prepare_global_policy import global_variants;import crish.training,crish.population_v2;assert "crish.oracle" not in sys.modules'
    p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    import pytest
    with pytest.raises(ValueError,match='oracle'):
        global_variants(base,('g2_oracle_dose',))
