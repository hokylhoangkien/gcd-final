"""Real-objective v2 controller integration and label/RNG isolation."""
import copy
import json
from unittest.mock import patch
import numpy as np
import pytest
import torch
from crish.population import PopulationEngine
from gcd_runtime.checkpoint import EpochRuntime
from selex_runtime.objective import selex_loss
from tests.test_global_policy import small_config, rng_snapshot, same_rng
from tests.test_selex_runtime import fixture
from tests.test_crish_g_v2_selector import v2_settings


def run(directory,hidden_poison=None,**changes):
    ds,state,args,head=fixture(epoch=10)
    if hidden_poison is not None:ds.targets[~state.labeled]=hidden_poison
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


def test_confirmation_failure_is_keep_and_no_runner_up(tmp_path):
    calls=[]
    def score(*unused):
        n=len(calls);calls.append(n)
        return dict(known=0.,variance=0.,view=(1.+.2*(n%2)) if n<6 else -1.-.2*(n%2))
    with patch('crish.population_v2.first_order_component_gains',side_effect=score):
        *_,engine,out=run(tmp_path)
    r=engine.runtime.last_population_record
    assert r['q_candidate']>0 and r['q']==0 and len(calls)==8
    assert r['confirmation']['statistics']['failures']==['view_gain']
    assert not any(x.any() for x in out.abstained)


def test_emergency_deadline_returns_keep(tmp_path):
    *_,engine,out=run(tmp_path,emergency_deadline_seconds=1e-12)
    assert engine.runtime.last_population_record['deadline_triggered'] is True
    assert engine.policy['q']==0 and not any(x.any() for x in out.abstained)


@pytest.mark.parametrize('counterfactual',[False,True])
def test_resume_reproduces_policy_masks_model_and_rng(tmp_path,counterfactual):
    from crish.training import train_strict
    from torch.utils.data import DataLoader
    import pytest
    def execute(directory,stop=False,resume=None):
        directory.mkdir()
        ds,state,args,head=fixture();cfg=small_config()
        cfg.update(epochs=4,max_steps_per_epoch=1)
        cfg['crish']['probe_view_reduction']='per_row_sum'
        cfg['population_policy']=v2_settings(mode='counterfactual' if counterfactual else 'fixed',
            gate='counterfactual' if counterfactual else 'none',train_batch=4,knn_k=3,warmup_epochs=0,
            refresh_every=2,persist=True,save_arrays=True,reliability_ema=dict(enabled=True,beta=.7,reset_jaccard=.5)).to_dict()
        args.epochs=4
        class Runtime(EpochRuntime):
            def monitor_epoch(self,*unused):pass
            def save(self,epoch,*a):
                super().save(epoch,*a)
                if stop and epoch==0:raise InterruptedError('bounded fixture stop')
        runtime=Runtime(directory,cfg,resume)
        model=torch.nn.Linear(4,4)
        loader=DataLoader(ds,batch_size=16)
        def extract(*unused):runtime.hierarchy_state=state.copy()
        train_strict(head,model,loader,loader,args,runtime,objective=selex_loss,extractor=extract)
        ckpt=torch.load(directory/'resume.pt',weights_only=False)
        return ckpt,[json.loads(x) for x in (directory/'crish_policy_epochs.jsonl').read_text().splitlines()]
    expected,epochs=execute(tmp_path/'continuous')
    with pytest.raises(InterruptedError):execute(tmp_path/'stopped',True)
    actual,again=execute(tmp_path/'resumed',resume=str(tmp_path/'stopped/resume.pt'))
    for key in ('model','head'):
        assert all(torch.equal(expected[key][k],actual[key][k]) for k in expected[key])
    assert torch.equal(expected['torch_rng'],actual['torch_rng'])
    a,b=expected['population_policy_state'],actual['population_policy_state']
    assert a['q']==b['q'] and a['sigma_history']==b['sigma_history']
    assert all(torch.equal(x,y) for x,y in zip(a['ema']['scores'],b['ema']['scores']))
    assert [(r['epoch'],r['source'],r['q'],r['levels']) for r in epochs[1:]]==[(r['epoch'],r['source'],r['q'],r['levels']) for r in again]


def test_diagnostics_preserve_loss_gradient_rng():
    from dataclasses import replace
    from crish.diagnostics import use_saturation,batch_exposure
    ds,state,args,head=fixture()
    raw=torch.cat([ds.features,ds.features+.02])
    m=state.abstained[0].clone();m[(~state.labeled).nonzero().flatten()[:5]]=True
    after=replace(state,abstained=(m,),version=1)
    before=rng_snapshot()
    loss=selex_loss(head,raw,state.ids,state.labeled,state.known_targets,after,args,0)['loss']
    expected=torch.autograd.grad(loss,tuple(head.parameters()))
    stats=use_saturation(after,state.ids,raw)
    exposure=batch_exposure(state,after,state.ids)
    actual=torch.autograd.grad(selex_loss(head,raw,state.ids,state.labeled,state.known_targets,after,args,0)['loss'],tuple(head.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(expected,actual)) and same_rng(before,rng_snapshot())
    assert exposure[0]['removed_directed_view_pairs']>0 and 0<=stats[0]['frac_F1']<=1


def test_legacy_json_matches_119d073_fixture(tmp_path):
    from pathlib import Path
    from tests.test_global_policy import run_select
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if 'seconds' not in k}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    _,out,record,engine=run_select(tmp_path)
    expected=json.loads((Path(__file__).parent/'fixtures/crish_g_119d073.json').read_text())
    assert dict(record=clean(record),policy=engine.policy,masks=[x.tolist() for x in out.abstained])==expected


@pytest.mark.parametrize('mode',['cf','fixed','random'])
def test_two_epoch_cpu_smoke_real_training(tmp_path,mode):
    from crish.training import train_strict
    from torch.utils.data import DataLoader
    ds,state,args,head=fixture();cfg=small_config()
    cfg['crish']['probe_view_reduction']='per_row_sum'
    changes={} if mode=='cf' else dict(mode='fixed',gate='none')
    if mode=='random':changes['random_control']='count_matched'
    cfg['population_policy']=v2_settings(train_batch=4,knn_k=3,warmup_epochs=0,refresh_every=1,
        persist=True,confirm_fresh_chunks=True,diagnostics=dict(use_saturation=True,exposure=True),**changes).to_dict()
    cfg['require_intervention_smoke']=True
    args.epochs=2
    class Runtime(EpochRuntime):
        def monitor_epoch(self,*unused):pass
    runtime=Runtime(tmp_path,cfg)
    model=torch.nn.Linear(4,4);loader=DataLoader(ds,batch_size=16)
    def extract(*unused):runtime.hierarchy_state=state.copy()
    train_strict(head,model,loader,loader,args,runtime,objective=selex_loss,extractor=extract)
    summary=runtime.last_training_summary
    assert summary['steps']==1 and summary['peak_cuda_allocated_bytes'] is None
    assert all(np.isfinite(summary[k]) for k in ['loss','seconds'])
    assert 'exposure' in summary and (tmp_path/'use_saturation.jsonl').is_file()
    records=[json.loads(x) for x in (tmp_path/'crish_p.jsonl').read_text().splitlines()]
    assert len(records)==2 and all(r['status']=='completed' and not r['deadline_triggered'] for r in records)


@pytest.mark.parametrize('mode',['cf','fixed','random'])
def test_hidden_training_labels_do_not_affect_v2(tmp_path,mode):
    changes={} if mode=='cf' else dict(mode='fixed',gate='none')
    if mode=='random':changes['random_control']='count_matched'
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if 'seconds' not in k}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    (tmp_path/'a').mkdir();(tmp_path/'b').mkdir()
    *_,a,first=run(tmp_path/'a',**changes)
    *_,b,second=run(tmp_path/'b',hidden_poison=98765,**changes)
    assert all(torch.equal(x,y) for x,y in zip(first.abstained,second.abstained))
    assert clean(a.runtime.last_population_record)==clean(b.runtime.last_population_record)


def test_no_confirmation_does_not_compute_confirmation_probe_gradients(tmp_path):
    from crish.lookahead import component_gradients_after_keep
    lengths=[]
    def wrapped(model,opt,keep,probes):
        lengths.append(len(probes));return component_gradients_after_keep(model,opt,keep,probes)
    with patch('crish.population_v2.component_gradients_after_keep',side_effect=wrapped):
        run(tmp_path,confirm=False)
    assert lengths==[1]
