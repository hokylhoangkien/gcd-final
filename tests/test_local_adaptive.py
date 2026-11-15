"""Adaptive local counts, independent evidence, bounded stops and handoff profiles."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from crish.candidates import Action
from crish.config import CrishSettings
from crish.engine import CrishEngine, _draw_context
from crish.policy import DecisionEvidence, RepairBudgetExhausted, sequential_repair
from gcd_runtime.config import ROOT, load_config
from scripts.prepare_local_adaptive import local_variants, prepare
from selex_runtime.objective import selex_loss
from tests.crish_v21.fixtures import state_fixture
from tests.test_selex_runtime import fixture


def run_policy(*, scorer=None, confirmer=None, budget_check=None, cap=3):
    state = state_fixture()
    actions = [Action('abstain', 91, 0, 0), Action('abstain', 22, 0, 2), Action('abstain', 80, 0, 3)]
    summary = {}
    current, trace = sequential_repair(state, actions,
        scorer or (lambda *unused: DecisionEvidence((1.,), 1., 0., 1.)),
        max_accepted=cap, threshold=0., agreement=1., known_tolerance=0.,
        confirmer=confirmer, budget_check=budget_check, diagnostics=summary)
    return state, current, trace, summary


def test_budget_during_rescoring_keeps_only_previously_confirmed_commit():
    seen = []
    def scorer(current, trial, action):
        seen.append(current.version)
        if current.version == 1:
            raise RepairBudgetExhausted('fixture deadline during rescore')
        return DecisionEvidence((1.,), 1., 0., 1.)
    original, current, trace, summary = run_policy(scorer=scorer)
    assert current.version == 1 and int(current.abstained[0].sum()) == 1
    assert summary['stop_reason'] == 'time_budget' and summary['budget_exhausted']
    assert summary['accepted'] == 1 and not summary['cap_reached']
    assert sum(r['status'] == 'accepted' for r in trace) == 1
    assert seen == [0, 0, 0, 1] and not original.abstained[0].any()


def test_deadline_after_confirmation_does_not_commit_unfinished_round():
    expired = False
    def check():
        if expired:
            raise RepairBudgetExhausted('fixture expired in confirmation')
    def confirmer(current, trial, action):
        nonlocal expired
        expired = current.version == 1
        return dict(gain=1., known_increase=0.)
    _, current, trace, summary = run_policy(confirmer=confirmer, budget_check=check)
    assert current.version == 1 and summary['confirmation_attempts'] == 2
    assert sum(r['status'] == 'uncommitted' for r in trace) == 1
    assert summary['stop_reason'] == 'time_budget'


@pytest.mark.parametrize('gain,known,reason', [(-1., 0., 'confirmation_failed'),
    (float('nan'), 0., 'confirmation_failed'), (1., .01, 'confirmation_known_guard')])
def test_explicit_stop_reason_and_zero_for_failed_first_confirmation(gain, known, reason):
    original, current, trace, summary = run_policy(
        confirmer=lambda *unused: dict(gain=gain, known_increase=known))
    assert current is original and summary['accepted'] == 0
    assert summary['stop_reason'] == reason and summary['confirmation_attempts'] == 1
    json.dumps(trace, allow_nan=False)


def test_cap_is_a_ceiling_and_no_gain_is_a_valid_zero_acceptance():
    _, current, _, summary = run_policy(cap=2)
    assert current.version == 2 and summary['cap_reached']
    _, current, _, summary = run_policy(scorer=lambda *unused: DecisionEvidence((-1.,), 1., 0., 1.))
    assert current.version == 0 and summary['stop_reason'] == 'no_eligible_candidates'


@pytest.mark.parametrize('bad', [-1, True, 1.5])
def test_confirmation_budget_validation(bad):
    with pytest.raises(ValueError):
        CrishSettings(confirmation_budget=bad)
    with pytest.raises(ValueError, match='cover'):
        CrishSettings(max_accepted=3, boundaries=3, confirmation_budget=2)


def test_cap1_control_draws_identical_reserved_folds_and_evidence_pool():
    _, state, _, _ = fixture()
    settings = CrishSettings(max_accepted=2, boundaries=2, confirmation_budget=2,
        train_batch=4, evidence_samples=8, k=2, probe_known=2, probe_unlabeled=2,
        probe_sampling='class_balanced', evidence_sampling='global_neighbors')
    a, pool_a = _draw_context(state, settings, torch.Generator().manual_seed(43))
    b, pool_b = _draw_context(state, replace(settings, max_accepted=1), torch.Generator().manual_seed(43))
    assert len(a) == len(b) == 3
    for (ak, au), (bk, bu) in zip(a, b):
        assert torch.equal(ak, bk) and torch.equal(au, bu)
    reserved = torch.cat([torch.cat(pair) for pair in a])
    assert len(reserved.unique()) == len(reserved)
    assert torch.equal(pool_a, pool_b) and not set(reserved.tolist()) & set(pool_a.tolist())
    assert CrishSettings().probe_fold_count == 2  # historical auto reservation
    with pytest.raises(ValueError, match='Insufficient'):
        _draw_context(state, replace(settings, confirmation_budget=8), torch.Generator())


def engine_fixture(directory, *, timeout=False, reject=False, rollback=False, cap=2, actual=False):
    dataset, state, args, head = fixture()
    settings = CrishSettings(warmup_epochs=0, max_accepted=cap, confirmation_budget=2,
        boundaries=2, train_batch=4, evidence_samples=4, k=1,
        probe_known=2, probe_unlabeled=2, min_support=1, max_protected_shift=1., max_refresh_seconds=20)
    runtime = SimpleNamespace(run_dir=directory,
        config=dict(seed=1, method='crish', crish=asdict(settings)))
    folds = [(torch.tensor([0, 1]), torch.tensor([10, 11])),
             (torch.tensor([4, 5]), torch.tensor([14, 15])),
             (torch.tensor([8, 9]), torch.tensor([18, 19]))]
    pool = torch.tensor([2, 3, 6, 7])
    actions = [Action('abstain', int(state.ids[i]), 0, int(state.labels[0][i])) for i in (2, 3)]
    comparisons = []
    from crish.lookahead import exact_sgd_comparison
    def compare(model, optimizer, g0, ga, probe):
        comparisons.append(set(probe.known_ids.tolist()+probe.unlabeled_ids.tolist()))
        if timeout and len(comparisons) == 4:
            raise RepairBudgetExhausted('fixture rescore deadline')
        if actual:
            return exact_sgd_comparison(model, optimizer, g0, ga, probe)
        # Controlled acceptance isolates multi-commit wiring from proxy quality.
        return dict(gain=-1. if reject else 1., known_increase=0.)
    optimizer = torch.optim.SGD(head.parameters(), lr=.002, momentum=.9)
    weights_before = copy.deepcopy(head.state_dict())
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch('crish.engine._draw_context', return_value=(folds, pool)))
        stack.enter_context(patch('crish.engine._shortlist', return_value=(actions, {a.sample_id: [] for a in actions})))
        stack.enter_context(patch('crish.engine.exact_sgd_comparison', side_effect=compare))
        if rollback:
            stack.enter_context(patch('crish.diagnostics.supervision_dose', side_effect=ValueError('fixture invalid final dose')))
        result = CrishEngine(runtime, objective=selex_loss).prepare(
            state, torch.nn.Identity(), head, optimizer, dataset, args, 0)
    assert all(torch.equal(weights_before[k], v) for k, v in head.state_dict().items())
    assert not optimizer.state and all(p.grad is None for p in head.parameters())
    assert not state.abstained[0].any()
    record = json.loads((directory/'crish.jsonl').read_text())
    return state, result, record, comparisons


@pytest.mark.parametrize('cap,expected', [(1, 1), (2, 2)])
def test_real_selex_gradients_feed_multi_commit_and_match_cap1_context(tmp_path, cap, expected):
    state, result, record, comparisons = engine_fixture(tmp_path, cap=cap)
    assert result.version == record['accepted'] == record['applied_actions'] == expected
    assert record['accepted_proposals'] == expected and record['cap_reached']
    assert record['dose']['changed_ids'] == expected
    confirmed = [r for r in record['gains'] if r['confirmation'] is not None]
    assert [r['confirmation_fold'] for r in confirmed] == list(range(1, expected+1))
    assert record['confirmation_folds_reserved'] == 2 and record['confirmation_folds_used'] == expected
    assert comparisons[0].isdisjoint(comparisons[2])
    if expected == 2:
        assert comparisons[2].isdisjoint(comparisons[-1])
        assert {g['version'] for g in record['gains']} == {0, 1}
    assert torch.equal(result.labels[0][state.labeled], state.known_targets)


def test_engine_real_exact_lookahead_runs_without_forced_acceptance(tmp_path):
    _, result, record, _ = engine_fixture(tmp_path, actual=True)
    assert record['status'] == 'completed' and record['gains']
    assert record['applied_actions'] == result.version
    assert all(torch.isfinite(torch.tensor(g['exact_discovery']['gain'])) for g in record['gains'])


def test_engine_deadline_after_commit_reports_applied_state_and_not_fallback(tmp_path):
    _, result, record, _ = engine_fixture(tmp_path, timeout=True)
    assert record['status'] == 'completed' and result.version == 1
    assert record['budget_exhausted'] and record['stop_reason'] == 'time_budget'
    assert record['accepted'] == record['applied_actions'] == record['dose']['changed_ids'] == 1


def test_engine_rollback_log_does_not_claim_applied_actions(tmp_path):
    state, result, record, _ = engine_fixture(tmp_path, rollback=True)
    assert result is state and record['status'] == 'fallback_keep' and record['rolled_back']
    assert record['accepted_proposals'] == 2 and record['accepted'] == record['applied_actions'] == 0
    assert not any(r['status'] == 'accepted' for r in record['decisions'])
    assert 'dose' not in record


def test_eight_commits_use_nine_real_disjoint_folds_and_rescore_each_version(tmp_path):
    from crish.smoke import FeatureFixture
    from crish.state import SupervisionState
    from crish.rebuild import prototype_radii
    torch.manual_seed(450)
    features = torch.nn.functional.normalize(torch.randn(256, 4), dim=1)
    labels = torch.arange(256) % 4
    membership = torch.arange(256) < 80
    dataset = FeatureFixture(features, membership, labels[membership])
    centers = torch.stack([features[labels == i].mean(0) for i in range(4)])
    state = SupervisionState(torch.tensor(dataset.sample_ids), features, membership, labels[membership],
        (labels,), (centers,), (prototype_radii(centers),), (torch.zeros(256, dtype=torch.bool),))
    _, _, args, head = fixture()
    settings = CrishSettings(warmup_epochs=0, max_accepted=8, boundaries=8, confirmation_budget=8,
        train_batch=64, evidence_samples=64, k=2, probe_known=4, probe_unlabeled=4,
        probe_sampling='class_balanced', probe_known_per_class=2, max_refresh_seconds=20)
    folds, pool = _draw_context(state, settings, torch.Generator().manual_seed(45))
    assert len(folds) == 9
    reserved = torch.cat([torch.cat(pair) for pair in folds])
    assert len(reserved) == len(reserved.unique())
    actions = [Action('abstain', int(state.ids[i]), 0, int(labels[i]))
               for i in pool[~membership[pool]][:8]]
    assert len(actions) == 8
    runtime = SimpleNamespace(run_dir=tmp_path, config=dict(seed=1, method='crish', crish=asdict(settings)))
    with patch('crish.engine._draw_context', return_value=(folds, pool)), \
         patch('crish.engine._shortlist', return_value=(actions, {a.sample_id: [] for a in actions})), \
         patch('crish.engine.exact_sgd_comparison', return_value=dict(gain=1., known_increase=0.)):
        result = CrishEngine(runtime, objective=selex_loss).prepare(state, torch.nn.Identity(), head,
            torch.optim.SGD(head.parameters(), lr=.002), dataset, args, 0)
    record = json.loads((tmp_path/'crish.jsonl').read_text())
    assert result.version == record['applied_actions'] == 8, record
    assert {g['version'] for g in record['gains']} == set(range(8))
    assert [g['confirmation_fold'] for g in record['gains'] if g['confirmation']] == list(range(1, 9))
    assert record['confirmation_folds_used'] == 8 and record['dose']['changed_ids'] == 8


def test_multi_action_training_consumes_repairs_and_resumes_exactly(tmp_path):
    from torch.utils.data import DataLoader
    from crish.training import train_strict
    from gcd_runtime.checkpoint import EpochRuntime
    def execute(directory, *, stop=False, resume=None):
        directory.mkdir()
        dataset, state, args, head = fixture()
        cfg = load_config(ROOT/'configs/selex_crish_smoke.json')
        cfg.update(device='cpu', synthetic_fixture=True, epochs=2, max_steps_per_epoch=1,
                   mlp_out_dim=16, final_ss_evaluation=False)
        cfg['crish'] = asdict(CrishSettings(warmup_epochs=0, refresh_every=1, max_accepted=2,
            confirmation_budget=2, boundaries=2, train_batch=4, evidence_samples=4, k=1,
            probe_known=2, probe_unlabeled=2, min_support=1, max_protected_shift=1.,
            save_partitions=False, max_refresh_seconds=20))
        class Runtime(EpochRuntime):
            def monitor_epoch(self, *unused):
                pass
            def save(self, epoch, *values):
                super().save(epoch, *values)
                if stop and epoch == 0:
                    raise InterruptedError('bounded fixture interruption')
        runtime = Runtime(directory, cfg, resume)
        model = torch.nn.Linear(4, 4)
        args.epochs = 2
        loader = DataLoader(dataset, batch_size=16)
        def extract(*unused):
            runtime.hierarchy_state = state.copy()
        folds = [(torch.tensor([0, 1]), torch.tensor([10, 11])),
                 (torch.tensor([4, 5]), torch.tensor([14, 15])),
                 (torch.tensor([8, 9]), torch.tensor([18, 19]))]
        actions = [Action('abstain', int(state.ids[i]), 0, int(state.labels[0][i])) for i in (2, 3)]
        with patch('crish.engine._draw_context', return_value=(folds, torch.tensor([2, 3, 6, 7]))), \
             patch('crish.engine._shortlist', return_value=(actions, {a.sample_id: [] for a in actions})), \
             patch('crish.engine.exact_sgd_comparison', return_value=dict(gain=1., known_increase=0.)):
            train_strict(head, model, loader, loader, args, runtime, objective=selex_loss, extractor=extract)
        return model.state_dict(), head.state_dict()
    expected = execute(tmp_path/'full')
    with pytest.raises(InterruptedError):
        execute(tmp_path/'interrupted', stop=True)
    actual = execute(tmp_path/'resumed', resume=str(tmp_path/'interrupted'/'resume.pt'))
    for a, b in zip(expected, actual):
        assert all(torch.equal(a[k], b[k]) for k in a)
    records = [json.loads(line) for line in (tmp_path/'full'/'crish.jsonl').read_text().splitlines()]
    assert [r['applied_actions'] for r in records] == [2, 2]
    doses = [json.loads(line) for line in (tmp_path/'full'/'intervention_diagnostics.jsonl').read_text().splitlines()]
    assert [d['changed_ids'] for d in doses] == [2, 2]


@pytest.mark.parametrize('dataset', ['scars', 'cub', 'aircraft', 'cifar100', 'herbarium_19'])
def test_profile_builder_is_dataset_agnostic_and_cap_controls_differ_only_in_cap(dataset):
    template = load_config(ROOT/f'configs/selex_{dataset}_crish_200.json')
    before = copy.deepcopy(template)
    configs = local_variants(template)
    assert template == before
    adaptive, control = configs['adaptive'], configs['cap1_context']
    assert adaptive['crish']['max_accepted'] == 8 and control['crish']['max_accepted'] == 1
    assert {k for k in adaptive['crish'] if adaptive['crish'][k] != control['crish'][k]} == {'max_accepted'}
    assert {k for k in adaptive if adaptive[k] != configs['baseline'][k]} == {'method'}
    assert adaptive['crish']['confirmation_budget'] == control['crish']['confirmation_budget'] == 8
    assert not adaptive['population_policy']['enabled'] and not adaptive['pair_reliability']['enabled']
    assert all(adaptive[k] == template[k] for k in template if k not in ('crish', 'method', 'population_policy', 'pair_reliability'))


def test_preparer_writes_configs_and_provenance_without_launching(tmp_path):
    target = tmp_path/'plan'
    result = prepare(ROOT/'configs/selex_scars_crish_200.json', target)
    assert result['status'] == 'prepared_not_executed'
    manifest = json.loads((target/'manifest.json').read_text())
    assert manifest['benchmark_metrics'] is None and not manifest['cuda_verified']
    assert len(manifest['entries']) == 4 and not list(target.rglob('*.pt'))
    assert all(load_config(target/entry['config']) for entry in manifest['entries'])
    with pytest.raises(FileExistsError):
        prepare(ROOT/'configs/selex_scars_crish_200.json', target)


@pytest.mark.parametrize('profile', ['adaptive', 'cap1_context'])
def test_checked_in_profile_and_cli_dry_run_select_local_with_expected_cap(profile):
    path = ROOT/f'configs/selex_scars_local_{profile}_200.json'
    cfg = load_config(path)
    expected = local_variants(load_config(ROOT/'configs/selex_scars_crish_200.json'))[profile]
    assert cfg == expected
    result = subprocess.run(['bash', 'scripts/train_selex.sh', '--config', str(path),
        '--variant', 'local', '--dry-run'], cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '"discrete_repair": true' in result.stdout and '"epochs": 200' in result.stdout
