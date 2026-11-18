"""Controlled policy wiring, common context and independent RNG regression."""
import copy
from dataclasses import asdict
import json
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from crish.ablation import VARIANTS, order_boundaries, filter_actions, evidence_repair
from crish.candidates import Action
from crish.config import CrishSettings
from crish.engine import CrishEngine
from crish.policy import DecisionEvidence, sequential_repair
from gcd_runtime.config import ROOT, load_config
from scripts.prepare_paper_plan import prepare
from selex_runtime.objective import selex_loss
from tests.test_selex_runtime import fixture
from tests.crish_v21.fixtures import state_fixture


def test_independent_random_order_same_pool_and_reproducibility():
    boundaries = [(float(i), i, 0, []) for i in range(30)]
    before = random.getstate()
    torch_before = torch.get_rng_state().clone()
    ordered = order_boundaries(boundaries, 'random_proposal', 71)
    assert random.getstate() == before and torch.equal(torch.get_rng_state(), torch_before)
    assert ordered == order_boundaries(boundaries, 'random_proposal', 71)
    assert sorted(x[1] for x in ordered) == list(range(30))
    assert ordered != order_boundaries(boundaries, 'full', 71)


def test_no_confirmation_preserves_discovery_and_known_guard():
    state = state_fixture()
    actions = [Action('abstain', 91, 0, 0)]
    kwargs = dict(max_accepted=1, threshold=0., agreement=1., known_tolerance=0.)
    scorer = lambda *args: DecisionEvidence((1.,), 0., 0., 1.)
    assert sequential_repair(state, actions, scorer, **kwargs)[0] is state
    result, trace = sequential_repair(state, actions, scorer, require_confirmation=False, **kwargs)
    assert result.version == 1 and trace[0]['status'] == 'accepted'
    bad = lambda *args: DecisionEvidence((1.,), 0., 1., 1.)
    assert sequential_repair(state, actions, bad, require_confirmation=False, **kwargs)[0] is state


def test_actions_disabled_do_not_get_converted_or_spend_budget():
    actions = [Action('abstain', 91, 0, 0), Action('move', 91, 0, 0, 2)]
    assert [a.kind for a in filter_actions(actions, 'no_move')] == ['abstain']
    assert [a.kind for a in filter_actions(actions, 'no_abstain')] == ['move']
    assert filter_actions(actions, 'full') == actions


def test_prepared_variants_only_change_one_field_and_seed(tmp_path):
    base = ROOT/'configs/selex_scars_local_adaptive_200.json'
    manifest = prepare(base, tmp_path/'plan', [1, 2, 3])
    reference = load_config(base)
    assert len(manifest['entries']) == 18 and manifest['benchmark_metrics'] is None
    for entry in manifest['entries']:
        cfg = load_config(tmp_path/'plan'/entry['config'])
        assert cfg.pop('crish_ablation') == entry['variant']
        cfg['seed'] = reference['seed']
        assert cfg == reference
    assert 'crish_ablation' not in load_config(base)  # historical config identity
    with pytest.raises(FileExistsError):
        prepare(base, tmp_path/'plan')


@pytest.mark.parametrize('variant', VARIANTS)
def test_engine_variant_runs_bounded_without_hidden_labels(tmp_path, variant):
    dataset, state, args, head = fixture()
    settings = CrishSettings(warmup_epochs=0, max_accepted=1, boundaries=2,
        train_batch=4, evidence_samples=4, k=1, probe_known=2, probe_unlabeled=2,
        min_support=1, max_protected_shift=1.)
    runtime = SimpleNamespace(run_dir=tmp_path,
        config=dict(seed=1, method='crish', crish=asdict(settings), crish_ablation=variant))
    folds = [(torch.tensor([0, 1]), torch.tensor([10, 11])),
             (torch.tensor([4, 5]), torch.tensor([14, 15]))]
    actions = [Action('abstain', int(state.ids[2]), 0, int(state.labels[0][2]))]
    before = copy.deepcopy(head.state_dict())
    with patch('crish.engine._draw_context', return_value=(folds, torch.tensor([2, 3, 6, 7]))), \
         patch('crish.engine._shortlist', return_value=(actions, {a.sample_id: [] for a in actions})), \
         patch('crish.engine.exact_sgd_comparison', return_value=dict(gain=1., known_increase=0.)) as cf, \
         patch('crish.ablation.supported_actions', return_value=actions):
        result = CrishEngine(runtime, objective=selex_loss).prepare(state, torch.nn.Identity(), head,
            torch.optim.SGD(head.parameters(), lr=.002), dataset, args, 0)
    record = json.loads((tmp_path/'crish.jsonl').read_text())
    assert record['status'] != 'fallback_keep', record
    assert record['accepted'] == (0 if variant == 'no_abstain' else 1)
    assert torch.equal(result.labels[0][state.labeled], state.known_targets)
    assert all(torch.equal(before[k], v) for k, v in head.state_dict().items())
    if variant == 'evidence_only':
        cf.assert_not_called()
    elif variant == 'no_confirmation':
        assert cf.call_count == 1 and record['confirmation_folds_used'] == 0
    assert len(record['proposal_outcomes']) == record['candidate_count']


def test_evidence_only_prefers_move_and_preserves_protected_labels():
    state = state_fixture()
    abstain = Action('abstain', 91, 0, 0)
    move = Action('move', 91, 0, 0, 2)
    with patch('crish.ablation.supported_actions', return_value=[abstain, move]):
        result, trace, _ = evidence_repair(state, [abstain, move], {},
            CrishSettings(min_support=1), lambda: None, lambda *args: None)
    assert trace[0]['kind'] == 'move' and result.version == 1
    assert torch.equal(result.labels[0][state.labeled], state.labels[0][state.labeled])


def test_default_full_trace_parity_and_rng(tmp_path):
    from tests.test_local_adaptive import engine_fixture
    a = tmp_path/'default'; b = tmp_path/'explicit'
    a.mkdir(); b.mkdir()
    torch.manual_seed(741)
    _, default, log_a, _ = engine_fixture(a)
    original = CrishEngine.__init__
    def explicit(self, runtime, **kwargs):
        runtime.config['crish_ablation'] = 'full'
        original(self, runtime, **kwargs)
    torch.manual_seed(741)
    with patch.object(CrishEngine, '__init__', explicit):
        _, actual, log_b, _ = engine_fixture(b)
    assert all(torch.equal(x,y) for x,y in zip(default.labels,actual.labels))
    for key in ('decisions','gains','proposals','action_counts','dose','scoring_batch'):
        assert log_a[key] == log_b[key]


@pytest.mark.parametrize('variant,expected', [('full', 'move'), ('no_abstain','move'), ('no_move','abstain')])
def test_move_and_abstain_filter_with_actual_rebuild_and_cf(tmp_path, variant, expected):
    dataset, state, args, head = fixture()
    settings = CrishSettings(warmup_epochs=0, boundaries=2, train_batch=4,
        evidence_samples=4, k=1, probe_known=2, probe_unlabeled=2,
        min_support=1, max_protected_shift=1.)
    sample = int(state.ids[2])
    actions = [Action('abstain', sample, 0, 2), Action('move', sample, 0, 2, 3)]
    runtime = SimpleNamespace(run_dir=tmp_path, config=dict(seed=1,method='crish',
        crish=asdict(settings),crish_ablation=variant))
    folds = [(torch.tensor([0,1]),torch.tensor([10,11])),
             (torch.tensor([4,5]),torch.tensor([14,15]))]
    # Prefer Move by a controlled legal probe gain; real objectives/drift/rebuild.
    def comparison(*unused):
        return dict(gain=1.,known_increase=0.)
    with patch('crish.engine._draw_context',return_value=(folds,torch.tensor([2,3,6,7]))), \
         patch('crish.engine._shortlist',return_value=(list(reversed(actions)),{sample:[]})), \
         patch('crish.engine.propose_actions',return_value=actions), \
         patch('crish.engine.exact_sgd_comparison',side_effect=comparison):
        result=CrishEngine(runtime,objective=selex_loss).prepare(state,torch.nn.Identity(),head,
            torch.optim.SGD(head.parameters(),lr=.002),dataset,args,0)
    record=json.loads((tmp_path/'crish.jsonl').read_text())
    applied=[d for d in record['decisions'] if d['status']=='accepted']
    assert len(applied)==1 and applied[0]['kind']==expected, record
    assert result.version==1


def test_evidence_revalidation_does_not_invent_new_proposal_targets():
    state=state_fixture()
    abstain=Action('abstain',91,0,0)
    original=Action('move',91,0,0,2)
    different=Action('move',91,0,0,3)
    with patch('crish.ablation.supported_actions',return_value=[abstain,different]):
        result,trace,_=evidence_repair(state,[abstain,original],{},CrishSettings(min_support=1),
                                     lambda:None,lambda *args:None)
    assert result.version==1 and trace[0]['kind']=='abstain'
