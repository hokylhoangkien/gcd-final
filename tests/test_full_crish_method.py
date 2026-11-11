"""Structured legal probes, realized dose and population-policy contracts."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import time
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from crish.candidates import Action
from crish.config import CrishSettings
from crish.diagnostics import base_target_mass, supervision_dose
from crish.engine import _draw_context, CrishEngine
from crish.lookahead import preserve_rng
from crish.population import PopulationEngine, assignment_reliability, abstain_fraction
from crish.population_settings import PopulationSettings
from crish.rebuild import rebuild, pseudo_masks
from crish.sampling import balanced_known_folds, known_fold_structure
from crish.training import train_strict
from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.config import ROOT, load_config
from selex_runtime.config import validate_config
from selex_runtime.objective import selex_loss
from scripts.prepare_crish_experiments import prepare, variants
from scripts.summarize_crish_experiments import summarize, aggregate
from tests.crish_v21.fixtures import state_fixture
from tests.test_selex_runtime import fixture


def small_config():
    cfg = load_config(ROOT/'configs/selex_crish_smoke.json')
    cfg.update(device='cpu', synthetic_fixture=True, epochs=2, mlp_out_dim=16,
               max_steps_per_epoch=1, final_ss_evaluation=False)
    cfg['population_policy'] = PopulationSettings(enabled=True, warmup_epochs=0,
        train_batch=28, max_refresh_seconds=20).to_dict()
    cfg['crish'] = asdict(CrishSettings(enabled=False, warmup_epochs=0,
        evidence_samples=8, k=2, boundaries=1, train_batch=8, min_support=1,
        probe_known=4, probe_unlabeled=2, probe_sampling='class_balanced', probe_known_per_class=2,
        max_refresh_seconds=20, save_partitions=False))
    return cfg


def test_structured_folds_have_real_cross_image_positives_and_no_reused_ids():
    rows = torch.arange(64)*7+3
    targets = torch.arange(64)//16
    folds = balanced_known_folds(rows, targets, 8, torch.Generator().manual_seed(9), per_class=2, folds=4)
    labels_by_id = dict(zip(rows.tolist(), targets.tolist()))
    classes = []
    for fold in folds:
        y = torch.tensor([labels_by_id[i] for i in fold.tolist()])
        structure = known_fold_structure(y)
        assert structure['images_per_class'] == [2]*4
        assert structure['cross_image_positive_pairs'] == 8
        classes.append(set(y.tolist()))
    assert all(c == classes[0] for c in classes)
    assert len(torch.cat(folds).unique()) == 32
    for bad in (dict(probe_known_per_class=0), dict(probe_known_per_class=2, probe_known=3),
                dict(probe_known_per_class=2, probe_known=4, probe_sampling='random')):
        with pytest.raises(ValueError): CrishSettings(**bad)
    with pytest.raises(ValueError, match='Insufficient'):
        balanced_known_folds(torch.arange(32), torch.arange(32)//2, 8,
                             torch.Generator(), per_class=2, folds=2)


def test_multi_action_context_reserves_fresh_disjoint_confirmation_folds():
    # Many legal images/class are required to reserve exact multi-action quotas.
    dataset, state, _, _ = fixture()
    cfg = CrishSettings(max_accepted=2, boundaries=2, train_batch=4, evidence_samples=8, k=2,
        probe_known=2, probe_unlabeled=2, probe_sampling='class_balanced')
    folds, pool = _draw_context(state, cfg, torch.Generator().manual_seed(43))
    assert len(folds) == 3
    reserved = [torch.cat([k, u]) for k, u in folds]
    assert len(torch.cat(reserved).unique()) == sum(map(len, reserved))
    assert not set(pool.tolist()) & set(torch.cat(reserved).tolist())


def test_population_mass_and_relation_dose_match_dense_reference():
    state = state_fixture()
    changed, _ = rebuild(state, Action('abstain', 22, 0, 2))
    dose = supervision_dose(state, changed)
    assert dose['changed_ids'] == 1 and dose['changed_labeled_ids'] == 0
    for level in range(len(state.labels)):
        original, kept = pseudo_masks(changed, changed.ids, level)
        off = ~torch.eye(2*len(state.ids), dtype=torch.bool)
        coef = (kept.repeat(2, 2) & off).float()/(original.repeat(2, 2) & off).sum(1, keepdim=True)
        assert base_target_mass(changed, level) == pytest.approx(float(coef.sum(1).mean()))
    original, _ = pseudo_masks(state, state.ids, 0)
    _, retained = pseudo_masks(changed, changed.ids, 0)
    diff = (original != retained) & ~torch.eye(len(state.ids), dtype=torch.bool)
    assert dose['levels'][0]['changed_directed_image_positive_pairs'] == int(diff.sum())


@pytest.mark.parametrize('change', [dict(enabled=1), dict(fractions=[.1,.2]), dict(fractions=[0.,.2,.1]),
    dict(fractions=[0.,1.]), dict(max_cluster_fraction=1), dict(min_retained_mass=0),
    dict(train_batch=2), dict(rank_block=0), dict(warmup_epochs=-1), dict(mode='oracle')])
def test_population_config_rejects_invalid_policy(change):
    with pytest.raises(ValueError): PopulationSettings(**change)


def test_global_margin_matches_dense_formula_and_ignores_empty_slots():
    _, state, _, _ = fixture()
    empty = torch.full((1, state.features.shape[1]), 1e6)
    state = replace(state, prototypes=(torch.cat([state.prototypes[0], empty]),),
        radii=(torch.cat([state.radii[0], torch.tensor([0.])]),), allow_empty_clusters=True)
    d = torch.cdist(state.features, state.prototypes[0][:-1]).square()
    own = d.gather(1, state.labels[0][:, None]).squeeze(1)
    d.scatter_(1, state.labels[0][:, None], torch.inf)
    other = d.min(1).values
    expected = (.5+.5*(other-own)/(other+own+1e-8)).clamp(0,1)
    got = assignment_reliability(state, block_size=3)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    assert not got.requires_grad and len(got) == len(state.ids)
    with pytest.raises(RuntimeError, match='budget'):
        assignment_reliability(state, deadline=time.monotonic()-1)


def test_population_abstention_respects_quotas_protection_and_nested_doses():
    _, state, _, _ = fixture()
    settings = PopulationSettings(enabled=True)
    reliability = torch.linspace(0,1,len(state.ids))
    a, da = abstain_fraction(state, reliability, .1, settings)
    b, db = abstain_fraction(state, reliability, .2, settings)
    assert da['selected'] <= db['selected']
    assert torch.all(~a.abstained[0] | b.abstained[0])
    assert not b.abstained[0][state.labeled].any()
    assert torch.equal(state.labels[0], b.labels[0])
    assert torch.equal(state.prototypes[0], b.prototypes[0])
    for group in state.labels[0].unique():
        u = ~state.labeled & (state.labels[0] == group)
        assert int((u & ~b.abstained[0]).sum()) >= min(settings.min_cluster_kept, int(u.sum()))
    assert supervision_dose(state, b)['changed_labeled_ids'] == 0
    keep, _ = abstain_fraction(state, reliability, 0, settings)
    assert keep is state
    random_settings = replace(settings, mode='random')
    first, _ = abstain_fraction(state, reliability, .2, random_settings, generator=torch.Generator().manual_seed(3))
    again, _ = abstain_fraction(state, reliability, .2, random_settings, generator=torch.Generator().manual_seed(3))
    assert torch.equal(first.abstained[0], again.abstained[0])


def run_policy(directory, *, poison=-1, controlled=False, fail_confirmation=False, source_head=False):
    dataset, state, args, head = fixture()
    if source_head:
        import importlib.util
        spec = importlib.util.spec_from_file_location('policy_source_dino', ROOT/'upstream/SelEx/models/vision_transformer2.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        head = module.DINOHead(in_dim=4, out_dim=16, hidden_dim=8, bottleneck_dim=4)
    dataset.targets[~dataset.labeled] = poison
    cfg = small_config()
    runtime = EpochRuntime(directory, cfg)
    optimizer = torch.optim.SGD(head.parameters(), lr=.002, momentum=.9)
    before = copy.deepcopy((head.state_dict(), optimizer.state_dict()))
    calls = []
    def compare(model, opt, g0, ga, probe):
        probe_ids = tuple(probe.known_ids.tolist()+probe.unlabeled_ids.tolist())
        confirmation = bool(calls) and probe_ids != calls[0]
        calls.append(probe_ids)
        return dict(gain=-1. if confirmation and fail_confirmation else 1., known_increase=0.)
    with patch('crish.population.exact_sgd_comparison', side_effect=compare) if controlled else preserve_rng():
        rng = torch.get_rng_state().clone()
        outcome = PopulationEngine(runtime, objective=selex_loss).prepare(
            state, torch.nn.Identity(), head, optimizer, dataset, args, 0)
        assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(head.state_dict()[k], before[0][k]) for k in before[0])
    assert optimizer.state_dict() == before[1]
    record = json.loads((Path(directory)/'crish_p.jsonl').read_text())
    return state, outcome, record, calls


def test_real_policy_loss_scoring_is_hidden_target_invariant_and_mutation_free():
    outputs = []
    for poison in (-1, 98765):
        with tempfile.TemporaryDirectory() as directory:
            state, outcome, record, _ = run_policy(directory, poison=poison)
            assert record['status'] in ('keep', 'completed'), record
            assert any('discovery' in c for c in record['candidates']), record
            assert record['probe_folds'][0]['images_per_class'] == [2,2]
            outputs.append((outcome, record))
    assert outputs[0][1]['candidates'] == outputs[1][1]['candidates']
    assert torch.equal(outputs[0][0].abstained[0], outputs[1][0].abstained[0])


def test_source_dino_head_executes_population_scoring_without_live_mutation():
    with tempfile.TemporaryDirectory() as directory:
        _, _, record, _ = run_policy(directory, source_head=True)
    assert record['status'] in ('keep', 'completed'), record
    assert any('discovery' in c for c in record['candidates'])


def test_policy_baseline_disabled_and_warmup_bypass_all_ranking():
    _, state, args, _ = fixture()
    with tempfile.TemporaryDirectory() as directory:
        for method, enabled, warmup in (('baseline',True,0), ('crish',False,0), ('crish',True,1)):
            cfg = small_config(); cfg['method'] = method
            cfg['population_policy'].update(enabled=enabled, warmup_epochs=warmup)
            engine = PopulationEngine(EpochRuntime(directory,cfg), objective=selex_loss)
            with patch('crish.population.assignment_reliability', side_effect=AssertionError('Ranking reached bypass')):
                assert engine.prepare(state,None,None,None,None,args,0) is state


def test_policy_deadline_and_mass_guard_return_original_state():
    with tempfile.TemporaryDirectory() as directory:
        with patch('crish.population._check_deadline', side_effect=RuntimeError('bounded deadline')):
            state, actual, record, _ = run_policy(directory)
        assert state is actual and record['status'] == 'fallback_keep'
    with tempfile.TemporaryDirectory() as directory:
        with patch('crish.population.base_target_mass', return_value=0.), \
             patch('crish.population.exact_sgd_comparison', side_effect=AssertionError('Mass guard bypassed')):
            state, actual, record, _ = run_policy(directory)
        assert state is actual and all('discovery' not in c for c in record['candidates'])


def test_two_accepted_local_edits_use_different_confirmation_folds():
    dataset, state, args, head = fixture()
    cfg = small_config(); cfg['population_policy']['enabled'] = False
    cfg['crish'] = asdict(CrishSettings(warmup_epochs=0, max_accepted=2, boundaries=2,
        train_batch=4, evidence_samples=4, k=1, probe_known=2, probe_unlabeled=2,
        min_support=1, max_protected_shift=1., max_refresh_seconds=20))
    folds = [(torch.tensor([0,1]),torch.tensor([10,11])),
             (torch.tensor([4,5]),torch.tensor([14,15])),
             (torch.tensor([8,9]),torch.tensor([18,19]))]
    pool = torch.tensor([2,3,6,7])
    actions = [Action('abstain',int(state.ids[i]),0,int(state.labels[0][i])) for i in (2,3)]
    queried = []
    def compare(model,opt,g0,ga,probe):
        queried.append(set(probe.known_ids.tolist()+probe.unlabeled_ids.tolist()))
        return dict(gain=1.,known_increase=0.)
    with tempfile.TemporaryDirectory() as directory:
        runtime = EpochRuntime(directory,cfg)
        with patch('crish.engine._draw_context',return_value=(folds,pool)), \
             patch('crish.engine._shortlist',return_value=(actions,{a.sample_id:[] for a in actions})), \
             patch('crish.engine.exact_sgd_comparison',side_effect=compare):
            outcome = CrishEngine(runtime,objective=selex_loss).prepare(state,torch.nn.Identity(),head,
                torch.optim.SGD(head.parameters(),lr=.002),dataset,args,0)
        record = json.loads((Path(directory)/'crish.jsonl').read_text())
    assert outcome.version == 2 and record['accepted'] == 2, record
    confirmed = [entry for entry in record['gains'] if entry['confirmation'] is not None]
    assert [entry['confirmation_fold'] for entry in confirmed] == [1,2]
    assert len({tuple(sorted(q)) for q in queried}) == 3


def test_confirmation_only_checks_winner_and_failure_never_tries_runner_up():
    for fail in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            state, outcome, record, calls = run_policy(directory, controlled=True, fail_confirmation=fail)
        discoveries = sum('discovery' in candidate for candidate in record['candidates'])
        assert discoveries >= 2 and len(calls) == discoveries+1, record
        assert all(c == calls[0] for c in calls[:-1]) and set(calls[0]).isdisjoint(calls[-1])
        assert sum('confirmation' in c for c in record['candidates']) == 1
        assert (outcome is state) == fail
        assert record['accepted'] == int(not fail)


def test_population_training_consumes_mask_and_resume_repeats_exactly():
    def execute(directory, *, stop=False, resume=None, enabled=True):
        dataset, state, args, head = fixture()
        cfg = small_config(); cfg['population_policy']['enabled'] = enabled
        args.epochs = 2
        class Runtime(EpochRuntime):
            def monitor_epoch(self, *unused): pass
            def save(self, epoch, *values):
                super().save(epoch, *values)
                if stop and epoch == 0: raise InterruptedError('fixture stop')
        runtime = Runtime(directory, cfg, resume)
        model = torch.nn.Linear(4,4)
        loader = DataLoader(dataset, batch_size=16)
        def extract(*unused): runtime.hierarchy_state = state.copy()
        def accept(engine, current, *unused):
            scores = torch.linspace(0,1,len(current.ids))
            return abstain_fraction(current, scores, .2, engine.settings)[0] if enabled else current
        # Controlled acceptance isolates actual SGD wiring from noisy proxy gains.
        with patch.object(PopulationEngine, 'prepare', new=accept):
            train_strict(head, model, loader, loader, args, runtime, objective=selex_loss, extractor=extract)
        return model.state_dict(), head.state_dict()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
         tempfile.TemporaryDirectory() as c, tempfile.TemporaryDirectory() as d:
        expected = execute(a)
        with pytest.raises(InterruptedError): execute(b, stop=True)
        actual = execute(c, resume=str(Path(b)/'resume.pt'))
        plain = execute(d, enabled=False)
        for x,y in zip(expected,actual): assert all(torch.equal(x[k],y[k]) for k in x)
        assert any(not torch.equal(expected[0][k],plain[0][k]) for k in plain[0])
        events = [json.loads(s) for s in (Path(a)/'intervention_diagnostics.jsonl').read_text().splitlines()]
        assert len(events) == 2 and events[0]['changed_ids'] == events[1]['changed_ids'] > 0


def test_experiment_axes_are_isolated_and_preparer_does_not_execute_jobs():
    cfg = load_config(ROOT/'configs/selex_scars_crish_200.json')
    profiles = variants(cfg, 'all')
    reference = profiles['local_reference']
    for name, p in profiles.items():
        if name.startswith('local_') and name != 'local_reference':
            assert {k for k in reference['crish'] if reference['crish'][k] != p['crish'][k]} == {
                name[len('local_'):].rsplit('_',1)[0]}
            assert not p['pair_reliability']['enabled'] and not p['population_policy']['enabled']
    with tempfile.TemporaryDirectory() as directory:
        result = prepare('scars', [1,2], Path(directory)/'plan', 'dose')
        assert result['status'] == 'prepared_not_executed' and result['jobs'] == 20
        manifest = json.loads((Path(result['directory'])/'manifest.json').read_text())
        assert manifest['benchmark_metrics'] is None
        assert not list(Path(directory).rglob('*.pt'))
    with pytest.raises(ValueError):
        validate_config(dict(cfg, crish=dict(cfg['crish'], enabled=True)))


def test_summary_ignores_synthetic_best_and_incomplete_runs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, status, epoch in (('observed','observed',1), ('fake','synthetic_fixture',1), ('partial','observed',0)):
            run = root/name; run.mkdir()
            (run/'config.json').write_text(json.dumps(dict(seed=1, dataset_name='scars', method='crish', epochs=2)))
            (run/'evaluation_last.json').write_text(json.dumps(dict(status=status, unit='fraction', epoch=epoch,
                protocol='balanced_ss_kmeans_train_unlabeled', all_acc=.5, old_acc=.6, new_acc=.4)))
        rows = summarize(root)
        assert len(rows) == 1 and rows[0]['run_id'] == 'observed' and rows[0]['all_percent'] == 50.


def test_summary_aggregates_only_matched_distinct_seeds_with_input_provenance():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for seed,accuracy in ((1,.5),(2,.7)):
            run=root/f'seed_{seed}';run.mkdir()
            (run/'config.json').write_text(json.dumps(dict(seed=seed,dataset_name='scars',method='crish',epochs=2)))
            (run/'evaluation_last.json').write_text(json.dumps(dict(status='observed',unit='fraction',epoch=1,
                protocol='balanced_ss_kmeans_train_unlabeled',all_acc=accuracy,old_acc=.6,new_acc=.4)))
            (run/'metadata.json').write_text(json.dumps(dict(code_fingerprint={'test_fixture':'same_source'})))
            (run/'readiness.json').write_text(json.dumps(dict(weight_sha256='same_fixture_weights')))
            (run/'environment.txt').write_text('fixture_environment\n')
        groups=aggregate(summarize(root))
        assert len(groups)==1 and groups[0]['n_seeds']==2
        assert groups[0]['all_percent']['mean']==60.
        assert groups[0]['all_percent']['std']==pytest.approx(2**.5*10)
        rows=summarize(root)
        assert 'all_percent' not in aggregate(rows+[rows[0]])[0]  # duplicate seed is not extra evidence
        (root/'seed_2/environment.txt').write_text('different_libraries\n')
        assert len(aggregate(summarize(root)))==2


def test_cli_variants_and_smoke_gate_are_explicit_and_tensor_free():
    decoder = json.JSONDecoder()
    for variant, expected in (('local',(True,False,False)), ('online',(False,True,False)),
                              ('policy',(False,False,True)), ('hybrid',(False,True,True))):
        command = ['bash','scripts/train_selex.sh','--dry-run','--variant',variant]
        result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10)
        assert result.returncode == 0,result.stderr
        remaining, objects = result.stdout.strip(), []
        while remaining:
            value,end = decoder.raw_decode(remaining); objects.append(value); remaining=remaining[end:].lstrip()
        smoke, full = objects[1]['config'],objects[2]['config']
        actual = (full['crish']['enabled'],full['pair_reliability']['enabled'],full['population_policy']['enabled'])
        assert actual == expected
        assert smoke['require_intervention_smoke'] is True
        assert full['crish']['probe_known_per_class'] == 2
        if expected[2]: assert smoke['population_policy']['warmup_epochs'] == 0


def test_smoke_fallback_blocks_automatic_full_training_but_does_not_require_acceptance():
    dataset,state,args,head=fixture()
    cfg=small_config();cfg['require_intervention_smoke']=True;args.epochs=2
    class Runtime(EpochRuntime):
        def monitor_epoch(self,*unused): pass
    with tempfile.TemporaryDirectory() as directory:
        runtime=Runtime(directory,cfg)
        def extract(*unused): runtime.hierarchy_state=state
        def fallback(engine,current,*unused):
            runtime.last_population_record=dict(status='fallback_keep',reason='bounded budget',candidates=[])
            return current
        model=torch.nn.Linear(4,4)
        loader=DataLoader(dataset,batch_size=16)
        with patch.object(PopulationEngine,'prepare',new=fallback), pytest.raises(RuntimeError,match='smoke'):
            train_strict(head,model,loader,loader,args,runtime,objective=selex_loss,extractor=extract)
        assert not (Path(directory)/'resume.pt').exists()
