"""CRiSH-G: global reciprocal-kNN reliability, thresholded multi-level policy, persistence."""
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import subprocess
from collections import Counter
from contextlib import nullcontext, redirect_stdout
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crish.candidates import Action
from crish.config import CrishSettings
from crish.diagnostics import base_target_mass
from crish.engine import CrishEngine
from crish.global_policy import GlobalEvidence, agreement, apply_threshold, knn_evidence
from crish.population import PopulationEngine
from crish.population_settings import GLOBAL_FIELDS, PopulationSettings
from crish.rebuild import prototype_radii, pseudo_masks
from crish.state import SupervisionState
from crish.training import train_strict
from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.config import ROOT, load_config
from scripts.prepare_global_policy import PROFILES, global_variants, prepare
from selex_runtime.config import validate_config
from selex_runtime.objective import selex_loss
from tests.test_selex_runtime import fixture

HISTORICAL_KEYS = ['enabled', 'mode', 'fractions', 'fixed_fraction', 'warmup_epochs', 'refresh_every',
                   'schedule_epochs', 'train_batch', 'rank_block', 'max_cluster_fraction', 'min_cluster_kept',
                   'min_retained_mass', 'max_refresh_seconds']


def global_policy(**changes):
    values = dict(enabled=True, warmup_epochs=0, refresh_every=2, train_batch=8, max_refresh_seconds=20,
                  ranking='knn_agreement', selection='threshold', levels='active', knn_k=3, persist=True,
                  scoring_batches=2, scoring_sampling='train_sampler')
    values.update(changes)
    return PopulationSettings(**values)


def small_config(**changes):
    cfg = load_config(ROOT/'configs/selex_crish_smoke.json')
    cfg.update(device='cpu', synthetic_fixture=True, epochs=2, mlp_out_dim=16,
               max_steps_per_epoch=1, final_ss_evaluation=False)
    cfg['population_policy'] = global_policy(**changes).to_dict()
    cfg['crish'] = asdict(CrishSettings(enabled=False, warmup_epochs=0,
        evidence_samples=8, k=2, boundaries=1, train_batch=8, min_support=1,
        probe_known=4, probe_unlabeled=2, probe_sampling='class_balanced', probe_known_per_class=2,
        max_refresh_seconds=20, save_partitions=False))
    return cfg


def line_state():
    """Hand-checked kNN graph (k=2) with one mutual and one nonreciprocal edge per mixed row."""
    x = torch.tensor([0., 1., 2., 10., 11., 13., 3.])
    features = torch.stack([x, torch.zeros_like(x)], 1)
    base = torch.tensor([0, 0, 1, 2, 2, 3, 1])
    coarse = torch.tensor([0, 1, 1, 1])[base]
    labeled = torch.tensor([True, False, False, True, False, False, False])
    labels = (base, coarse)
    protos = tuple(torch.stack([features[y == c].mean(0) for c in range(int(y.max())+1)]) for y in labels)
    return SupervisionState(torch.arange(7)*5+3, features, labeled, base[labeled], labels, protos,
        tuple(prototype_radii(p) for p in protos), tuple(torch.zeros(7, dtype=torch.bool) for _ in labels))


def dense_reference(state, k, w):
    distance = torch.cdist(state.features, state.features)
    distance.fill_diagonal_(torch.inf)
    neighbors = distance.topk(k, largest=False).indices
    adjacency = torch.zeros(len(state.ids), len(state.ids), dtype=torch.bool)
    adjacency[torch.arange(len(state.ids))[:, None], neighbors] = True
    out = []
    for labels in state.labels:
        r = []
        for i in range(len(state.ids)):
            weights = [1. if adjacency[j, i] else w for j in neighbors[i].tolist()]
            same = [float(labels[j] == labels[i]) for j in neighbors[i].tolist()]
            r.append(sum(a*b for a, b in zip(weights, same))/sum(weights))
        out.append(torch.tensor(r, dtype=torch.float64))
    return out


def rng_snapshot():
    return random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone()


def same_rng(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and torch.equal(a[2], b[2])


# 1. Reliability -------------------------------------------------------------
def test_reliability_matches_hand_computation_with_mutual_and_nonreciprocal_edges():
    state = line_state()
    evidence = knn_evidence(state, k=2, nonreciprocal_weight=.5)
    expected_neighbors = [{1, 2}, {0, 2}, {1, 6}, {4, 5}, {3, 5}, {3, 4}, {1, 2}]
    assert [set(row) for row in evidence.neighbors.tolist()] == expected_neighbors
    weights = {(i, j): float(w) for i, row in enumerate(evidence.neighbors.tolist())
               for j, w in zip(row, evidence.weights[i].tolist())}
    assert weights[(0, 2)] == .5 and weights[(6, 1)] == .5 and weights[(0, 1)] == 1. and weights[(6, 2)] == 1.
    r0 = torch.tensor([2/3, 1/2, 1/2, 1/2, 1/2, 0., 2/3], dtype=torch.float64)
    r1 = torch.tensor([2/3, 1/2, 1/2, 1., 1., 1., 2/3], dtype=torch.float64)
    torch.testing.assert_close(evidence.reliability[0], r0)
    torch.testing.assert_close(evidence.reliability[1], r1)
    # Self is never a neighbor; labeled rows (0 and 3) contribute their legal L0.
    assert not (evidence.neighbors == torch.arange(7)[:, None]).any()
    assert evidence.mutual_fraction == pytest.approx(12/14)


def test_reliability_matches_dense_reference_bounds_and_coarse_monotonicity():
    _, state, _, _ = fixture(epoch=10)
    assert len(state.labels) == 2
    evidence = knn_evidence(state, k=5, nonreciprocal_weight=.5, query_block=7, reference_block=9)
    for got, want in zip(evidence.reliability, dense_reference(state, 5, .5)):
        torch.testing.assert_close(got, want)
        assert ((got >= 0) & (got <= 1)).all()
    assert (evidence.reliability[1] >= evidence.reliability[0]-1e-12).all()
    with pytest.raises(RuntimeError, match='deadline'):
        knn_evidence(state, k=5, nonreciprocal_weight=.5, deadline=0.)


# 2. Threshold policy ----------------------------------------------------------
def reference_selection(state, r, tau, settings, level):
    """Independent plain-Python reference: (r, id) order, quotas, mass-floor prefix."""
    labels, unlabeled = state.labels[level], ~state.labeled
    clusters = len(state.prototypes[level])
    support = torch.bincount(labels[unlabeled], minlength=clusters).tolist()
    cap = [min(int(s*settings.max_cluster_fraction), max(0, s-settings.min_cluster_kept)) for s in support]
    rows = sorted((i for i in range(len(labels)) if unlabeled[i] and r[i] < tau),
                  key=lambda i: (float(r[i]), int(state.ids[i])))
    used, chosen = [0]*clusters, []
    for i in rows:
        if used[int(labels[i])] < cap[int(labels[i])]:
            used[int(labels[i])] += 1; chosen.append(i)
    kept = []
    for i in chosen:
        mask = torch.zeros(len(labels), dtype=torch.bool); mask[kept+[i]] = True
        abstained = list(state.abstained); abstained[level] = mask
        if base_target_mass(replace(state, abstained=tuple(abstained)), level) < settings.min_retained_mass:
            break
        kept.append(i)
    return set(kept), chosen


@pytest.mark.parametrize('floor', [.2, .7, .9])
def test_threshold_policy_matches_reference_quotas_floor_and_never_touches_labeled(floor):
    _, state, _, _ = fixture(epoch=10)
    settings = global_policy(min_retained_mass=floor)
    evidence = knn_evidence(state, k=3, nonreciprocal_weight=.5)
    keep, coverage = apply_threshold(state, evidence, 0., settings)
    assert keep is state and all(x['abstained_unlabeled'] == 0 for x in coverage['levels'])
    previous = [set() for _ in state.labels]
    for tau in (.2, .4, .6, .8, 1.):
        proposed, coverage = apply_threshold(state, evidence, tau, settings)
        for level in range(len(state.labels)):
            got = set(proposed.abstained[level].nonzero().flatten().tolist())
            want, chosen = reference_selection(state, evidence.reliability[level], tau, settings, level)
            assert got == want
            assert not proposed.abstained[level][state.labeled].any()
            assert previous[level] <= got  # monotone, nested in tau
            previous[level] = got
            entry = coverage['levels'][level]
            assert entry['truncated_by_mass_floor'] == len(chosen)-len(want)
            assert base_target_mass(proposed, level) >= floor
            if len(want) < len(chosen):
                # Cut exactly at the floor: the next r-ordered row would violate it.
                mask = proposed.abstained[level].clone(); mask[chosen[len(want)]] = True
                abstained = list(proposed.abstained); abstained[level] = mask
                assert base_target_mass(replace(proposed, abstained=tuple(abstained)), level) < floor
            labels = state.labels[level]
            for group in labels.unique().tolist():
                u = ~state.labeled & (labels == group)
                removed = int((u & proposed.abstained[level]).sum())
                assert removed <= int(int(u.sum())*settings.max_cluster_fraction)
                assert int(u.sum())-removed >= min(settings.min_cluster_kept, int(u.sum()))
        assert torch.equal(proposed.labels[0], state.labels[0]) and proposed.validate() is None
    assert sum(len(x) for x in previous) > 0


def test_threshold_ties_follow_sample_id():
    _, state, _, _ = fixture(epoch=10)
    flat = torch.full((len(state.ids),), .25, dtype=torch.float64)
    evidence = GlobalEvidence(torch.zeros(len(state.ids), 1, dtype=torch.long), torch.ones(len(state.ids), 1),
                              (flat, flat.clone()), 1., 0.)
    # Reverse ID order relative to rows: ties must follow IDs, not loader rows.
    shuffled = replace(state, ids=state.ids.flip(0))
    settings = global_policy(min_retained_mass=.9, levels='base')
    proposed, _ = apply_threshold(shuffled, evidence, .5, settings)
    want, _ = reference_selection(shuffled, flat, .5, settings, 0)
    assert set(proposed.abstained[0].nonzero().flatten().tolist()) == want
    assert want and want != reference_selection(state, flat, .5, settings, 0)[0]


# 3. Levels ------------------------------------------------------------------
def test_active_levels_use_their_own_reliability_and_base_only_edits_level_zero():
    _, state, _, _ = fixture(epoch=10)
    evidence = knn_evidence(state, k=3, nonreciprocal_weight=.5)
    active, _ = apply_threshold(state, evidence, .6, global_policy(levels='active', min_retained_mass=.2))
    base, coverage = apply_threshold(state, evidence, .6, global_policy(levels='base', min_retained_mass=.2))
    for level in range(2):
        got = active.abstained[level]
        assert (evidence.reliability[level][got] < .6).all()
        want, _ = reference_selection(state, evidence.reliability[level], .6,
                                      global_policy(min_retained_mass=.2), level)
        assert set(got.nonzero().flatten().tolist()) == want
    assert active.abstained[1].any() and not torch.equal(active.abstained[0], active.abstained[1])
    assert torch.equal(base.abstained[0], active.abstained[0]) and not base.abstained[1].any()
    assert not coverage['levels'][1]['scoped'] and coverage['levels'][1]['abstained_unlabeled'] == 0


# 4. Persistence ---------------------------------------------------------------
def engine_for(directory, **changes):
    return PopulationEngine(EpochRuntime(directory, small_config(**changes)), objective=selex_loss)


def perturbed(state, scale=.3, seed=5):
    generator = torch.Generator().manual_seed(seed)
    features = F.normalize(state.features+scale*torch.randn(state.features.shape, generator=generator), dim=-1)
    return replace(state, features=features)


def test_non_due_epoch_reapplies_tau_on_fresh_evidence_without_rng_or_probe():
    dataset, state, args, head = fixture(epoch=10)
    with tempfile.TemporaryDirectory() as directory:
        engine = engine_for(directory, mode='fixed', gate='none', warmup_epochs=10, refresh_every=5,
                            min_retained_mass=.3)
        optimizer = torch.optim.SGD(head.parameters(), lr=.01)
        due = engine.prepare(state, torch.nn.Identity(), head, optimizer, dataset, args, 10)
        assert engine.policy == dict(tau=.5, selected_epoch=10, source='due') and due is not state
        fresh = perturbed(state)
        before = rng_snapshot()
        with patch('crish.population.exact_sgd_comparison', side_effect=AssertionError('probe')), \
             patch('crish.population.cached_views', side_effect=AssertionError('forward')), \
             patch('crish.population._draw_context', side_effect=AssertionError('folds')):
            again = engine.prepare(fresh, None, head, None, None, args, 11)
        assert same_rng(before, rng_snapshot())
        expected, _ = apply_threshold(fresh, knn_evidence(fresh, k=3, nonreciprocal_weight=.5), .5, engine.settings)
        assert all(torch.equal(a, b) for a, b in zip(again.abstained, expected.abstained))
        assert not all(torch.equal(a, b) for a, b in zip(again.abstained, due.abstained))
        records = [json.loads(x) for x in (Path(directory)/'crish_policy_epochs.jsonl').read_text().splitlines()]
        assert [(r['epoch'], r['source'], r['tau']) for r in records] == [(10, 'due', .5), (11, 'persisted', .5)]
        assert records[1]['levels'][0]['abstained_unlabeled'] == int(again.abstained[0].sum())
        assert len(records[1]['levels'][0]['reliability_histogram']) == 10
        # Before warmup nothing is computed or logged.
        assert engine.prepare(state, None, None, None, None, args, 9) is state
        assert len((Path(directory)/'crish_policy_epochs.jsonl').read_text().splitlines()) == 2


def test_persist_false_applies_only_at_due_epochs():
    dataset, state, args, head = fixture(epoch=10)
    with tempfile.TemporaryDirectory() as directory:
        engine = engine_for(directory, mode='fixed', gate='none', warmup_epochs=10, refresh_every=5,
                            persist=False, min_retained_mass=.3)
        due = engine.prepare(state, torch.nn.Identity(), head, torch.optim.SGD(head.parameters(), lr=.01),
                             dataset, args, 10)
        assert due is not state
        with patch('crish.population.knn_evidence', side_effect=AssertionError('evidence')):
            for epoch in (11, 12, 13, 14):
                assert engine.prepare(state, None, None, None, None, args, epoch) is state
        assert len((Path(directory)/'crish_policy_epochs.jsonl').read_text().splitlines()) == 1


# 5. Counterfactual -------------------------------------------------------------
def run_select(directory, compare=None, *, lr=.002, **changes):
    dataset, state, args, head = fixture(epoch=10)
    cfg = small_config(**dict(dict(warmup_epochs=10, refresh_every=5, min_retained_mass=.2,
                                   scoring_batches=3), **changes))
    runtime = EpochRuntime(directory, cfg)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=.9)
    # One live step creates real momentum buffers for the cloned optimizer.
    loss = head(torch.randn(3, 4)).square().mean(); loss.backward(); optimizer.step(); optimizer.zero_grad()
    before = copy.deepcopy((head.state_dict(), optimizer.state_dict()))
    engine = PopulationEngine(runtime, objective=selex_loss)
    rng = rng_snapshot()
    context = patch('crish.population.exact_sgd_comparison', side_effect=compare) if compare else nullcontext()
    with context, patch('crish.population.apply_threshold', side_effect=recording_apply):
        outcome = engine.prepare(state, torch.nn.Identity(), head, optimizer, dataset, args, 10)
    assert same_rng(rng, rng_snapshot())
    assert all(torch.equal(head.state_dict()[k], before[0][k]) for k in before[0])
    after = optimizer.state_dict()
    assert after['param_groups'] == before[1]['param_groups']
    for key, slot in before[1]['state'].items():
        assert all(torch.equal(slot[name], after['state'][key][name]) for name in slot)
    record = json.loads((Path(directory)/'crish_p.jsonl').read_text().splitlines()[-1])
    return state, outcome, record, engine


LAST_TAU = []


def recording_apply(state, evidence, tau, *args, **kwargs):
    LAST_TAU.append(tau)
    return apply_threshold(state, evidence, tau, *args, **kwargs)


def scripted(gains, confirmation):
    """Gains keyed by the threshold whose proposal is being compared."""
    calls = []
    def compare(model, opt, g0, ga, probe):
        calls.append(tuple(probe.known_ids.tolist()))
        if len(calls) > 1 and calls[-1] != calls[0]:
            return dict(gain=confirmation, known_increase=0.)
        gain, known = gains[LAST_TAU[-1]]
        return dict(gain=gain, known_increase=known)
    return compare, calls


def test_counterfactual_keeps_keep_reference_and_picks_largest_eligible_gain():
    compare, calls = scripted({.3: (1., 0.), .5: (3., 1.), .7: (2., 0.)}, confirmation=.5)
    with tempfile.TemporaryDirectory() as directory:
        state, outcome, record, engine = run_select(directory, compare)
    first = record['candidates'][0]
    assert first['threshold'] == 0. and first['status'] == 'keep_reference' and 'discovery' not in first
    statuses = {c['threshold']: c['status'] for c in record['candidates']}
    assert statuses == {0.: 'keep_reference', .3: 'eligible', .5: 'rejected', .7: 'accepted'}
    assert record['chosen_threshold'] == .7 and engine.policy['tau'] == .7 and outcome is not state
    assert len(calls) == 4 and calls[-1] != calls[0]
    assert record['scoring_rows'] == 24 and record['scoring_chunks'] == 3


def test_failed_confirmation_returns_keep_without_runner_up():
    compare, calls = scripted({.3: (1., 0.), .5: (3., 0.), .7: (2., 0.)}, confirmation=-1.)
    with tempfile.TemporaryDirectory() as directory:
        state, outcome, record, engine = run_select(directory, compare)
    assert outcome is state and record['chosen_threshold'] == 0. and engine.policy['tau'] == 0.
    assert sum('confirmation' in c for c in record['candidates']) == 1 and len(calls) == 4
    assert 'no runner-up' in record['reason']


def test_deadline_or_error_falls_back_to_tau_zero_with_reason():
    with tempfile.TemporaryDirectory() as directory:
        with patch('crish.population._check_deadline', side_effect=RuntimeError('bounded deadline')):
            state, outcome, record, engine = run_select(directory)
    assert outcome is state and record['status'] == 'fallback_keep' and 'deadline' in record['reason']
    assert engine.policy['tau'] == 0.


def test_float64_probe_keeps_decision_when_gains_are_large_and_real_state_is_untouched():
    results = {}
    for precision in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            _, outcome, record, _ = run_select(directory, probe_float64=precision, lr=.5)
        results[precision] = record
    for a, b in zip(results[False]['candidates'], results[True]['candidates']):
        if 'discovery' in a:
            g32, g64 = a['discovery']['gain'], b['discovery']['gain']
            if abs(g32) > 1e-3:
                assert np.sign(g32) == np.sign(g64) and g64 == pytest.approx(g32, rel=.05)
    large = all(abs(c['discovery']['gain']) > 1e-3 for c in results[False]['candidates'] if 'discovery' in c)
    if large:
        assert results[False]['chosen_threshold'] == results[True]['chosen_threshold']
    assert results[True]['probe_float64'] is True


def test_train_sampler_scoring_rows_are_disjoint_from_probes_and_reproducible():
    dataset, state, args, head = fixture(epoch=10)
    with tempfile.TemporaryDirectory() as directory:
        engine = engine_for(directory)
        available = torch.arange(10, 40)
        a = engine._scoring_rows(state, available, 123, torch.Generator().manual_seed(0))
        b = engine._scoring_rows(state, available, 123, torch.Generator().manual_seed(99))
    assert torch.equal(a, b) and len(a.unique()) == len(a) == 16 and set(a.tolist()) <= set(available.tolist())


# 6. Random same-dose control ------------------------------------------------
def test_random_control_matches_dose_nested_deterministic_and_rng_free():
    _, state, _, _ = fixture(epoch=10)
    evidence = knn_evidence(state, k=3, nonreciprocal_weight=.5)
    common = dict(mode='fixed', gate='none', min_retained_mass=.2)
    reference, ref_cov = apply_threshold(state, evidence, .6, global_policy(**common))
    rng = rng_snapshot()
    first, cov = apply_threshold(state, evidence, .6, global_policy(sample_choice='random', **common),
                                 random_seed=(1, 10))
    assert same_rng(rng, rng_snapshot())
    again, _ = apply_threshold(state, evidence, .6, global_policy(sample_choice='random', **common),
                               random_seed=(1, 10))
    other, _ = apply_threshold(state, evidence, .6, global_policy(sample_choice='random', **common),
                               random_seed=(1, 11))
    for level in range(2):
        assert int(first.abstained[level].sum()) == int(reference.abstained[level].sum()) - \
            cov['levels'][level].get('random_shortfall', 0)
        assert not first.abstained[level][state.labeled].any()
        assert torch.equal(first.abstained[level], again.abstained[level])
    assert cov['levels'][0]['random_shortfall'] == 0
    assert (first.abstained[1] <= first.abstained[0]).all()  # nested in level 0
    assert not torch.equal(first.abstained[0], other.abstained[0])
    assert not torch.equal(first.abstained[0], reference.abstained[0])
    with pytest.raises(ValueError, match='seed'):
        apply_threshold(state, evidence, .6, global_policy(sample_choice='random', **common))


# 8. Resume ----------------------------------------------------------------
def test_resume_between_refreshes_reproduces_masks_and_weights_exactly():
    def execute(directory, *, stop=False, resume=None):
        dataset, state, args, head = fixture()
        cfg = small_config(warmup_epochs=0, refresh_every=2, min_retained_mass=.3, save_arrays=True)
        cfg['epochs'] = 4
        args.epochs = 4
        class Runtime(EpochRuntime):
            def monitor_epoch(self, *unused): pass
            def save(self, epoch, *values):
                super().save(epoch, *values)
                if stop and epoch == 0: raise InterruptedError('fixture stop')
        runtime = Runtime(directory, cfg, resume)
        model = torch.nn.Linear(4, 4)
        loader = DataLoader(dataset, batch_size=16)
        def extract(*unused): runtime.hierarchy_state = state.copy()
        # A fixed eligible comparison isolates persistence wiring from noisy gains.
        with patch('crish.population.exact_sgd_comparison', return_value=dict(gain=1., known_increase=0.)):
            train_strict(head, model, loader, loader, args, runtime, objective=selex_loss, extractor=extract)
        masks = {}
        for path in Path(directory).glob('crish_g_epoch_*.npz'):
            with np.load(path, allow_pickle=False) as stored:
                masks[int(path.stem.split('_')[-1])] = stored['abstained_0']
        epochs = [json.loads(x) for x in (Path(directory)/'crish_policy_epochs.jsonl').read_text().splitlines()]
        return model.state_dict(), head.state_dict(), masks, epochs
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
        expected = execute(a)
        with pytest.raises(InterruptedError): execute(b, stop=True)
        tau = expected[3][0]['tau']
        assert tau > 0 and expected[3][0]['source'] == 'due'
        checkpoint = torch.load(Path(b)/'resume.pt', weights_only=False)
        assert checkpoint['population_policy_state'] == dict(tau=tau, selected_epoch=0, source='due')
        actual = execute(c, resume=str(Path(b)/'resume.pt'))
    for x, y in zip(expected[:2], actual[:2]):
        assert all(torch.equal(x[k], y[k]) for k in x)
    assert set(actual[2]) == {1, 2, 3} and all(np.array_equal(expected[2][e], actual[2][e]) for e in actual[2])
    assert expected[2][1].any()
    assert [(r['epoch'], r['source'], r['tau']) for r in actual[3]] == [
        (1, 'persisted', tau), (2, 'due', tau), (3, 'persisted', tau)]


def test_selex_runtime_save_merges_checkpoint_extra():
    from selex_runtime.trainer import SelExRuntime
    cfg = small_config()
    head = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(head.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    with tempfile.TemporaryDirectory() as directory:
        runtime = SelExRuntime(directory, cfg)
        runtime.checkpoint_extra = dict(population_policy_state=dict(tau=.5, selected_epoch=3, source='due'))
        runtime.plain_rng_pending = True
        runtime.save(0, torch.nn.Linear(2, 2), head, optimizer, scheduler)
        state = torch.load(Path(directory)/'resume.pt', weights_only=False)
    assert state['population_policy_state']['tau'] == .5 and state['plain_rng_pending'] is True


# 9. Real SelEx loss -------------------------------------------------------
def test_selex_loss_changes_only_pairs_with_abstained_rows_per_level():
    dataset, state, args, head = fixture(epoch=10)
    evidence = knn_evidence(state, k=3, nonreciprocal_weight=.5)
    proposed, _ = apply_threshold(state, evidence, .6, global_policy(min_retained_mass=.2))
    assert all(mask.any() for mask in proposed.abstained)
    for level in range(2):
        _, kept_before = pseudo_masks(state, state.ids, level)
        original, kept_after = pseudo_masks(proposed, state.ids, level)
        changed = kept_before != kept_after
        mask = proposed.abstained[level]
        involved = mask[:, None] | mask[None, :]
        assert changed.any() and not (changed & ~involved).any()
        assert torch.equal(original, kept_before)
    raw = torch.cat([dataset.features, dataset.features+.01])
    ids, membership = state.ids, state.labeled
    targets = state.labels[0][membership]
    losses = []
    for current in (state, proposed):
        head.zero_grad()
        result = selex_loss(head, raw, ids, membership, targets, current, args, 10)
        result['loss'].backward()
        assert torch.isfinite(result['loss'])
        assert all(torch.isfinite(p.grad).all() for p in head.parameters())
        losses.append(float(result['pseudo'].detach()))
    assert losses[0] != losses[1]


# 10. Parity ----------------------------------------------------------------
def test_historical_population_snapshots_keep_their_exact_keys():
    assert list(PopulationSettings().to_dict()) == HISTORICAL_KEYS
    for path in sorted(ROOT.glob('configs/selex_*.json')):
        raw = json.loads(path.read_text())
        if 'population_policy' in raw:
            assert load_config(path)['population_policy'] == raw['population_policy'], path.name
    g = global_policy().to_dict()
    assert set(GLOBAL_FIELDS) <= set(g) and PopulationSettings(**g).to_dict() == g


def test_crish_p_margin_path_never_enters_global_code():
    dataset, state, args, head = fixture()
    cfg = small_config()
    cfg['population_policy'] = PopulationSettings(enabled=True, warmup_epochs=0, train_batch=28,
                                                  max_refresh_seconds=20).to_dict()
    with tempfile.TemporaryDirectory() as directory:
        engine = PopulationEngine(EpochRuntime(directory, cfg), objective=selex_loss)
        with patch.object(PopulationEngine, '_prepare_global', side_effect=AssertionError('global')), \
             patch('crish.population.knn_evidence', side_effect=AssertionError('knn')):
            engine.prepare(state, torch.nn.Identity(), head, torch.optim.SGD(head.parameters(), lr=.002),
                           dataset, args, 0)
            engine.prepare(state, None, None, None, None, args, 1)
        assert not (Path(directory)/'crish_policy_epochs.jsonl').exists()
        assert (Path(directory)/'crish_p.jsonl').exists()


def test_default_training_order_is_unchanged_and_compose_runs_global_first():
    for compose in (False, True):
        dataset, state, args, head = fixture()
        cfg = small_config(compose_local=compose)
        if compose:
            cfg['crish']['enabled'] = True
        calls = []
        class Runtime(EpochRuntime):
            def monitor_epoch(self, *unused): pass
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory, cfg)
            def extract(*unused): runtime.hierarchy_state = state.copy()
            def local(engine, current, *unused): calls.append('local'); return current
            def population(engine, current, *unused): calls.append('global'); return current
            args.epochs = 1
            with patch.object(CrishEngine, 'prepare', new=local), patch.object(PopulationEngine, 'prepare', new=population):
                train_strict(head, torch.nn.Linear(4, 4), DataLoader(dataset, batch_size=16),
                             DataLoader(dataset, batch_size=16), args, runtime, objective=selex_loss, extractor=extract)
        assert calls == (['global', 'local'] if compose else ['local', 'global'])


# 11. Config, CLI, preparer, smoke gate --------------------------------------
@pytest.mark.parametrize('change', [
    dict(selection='threshold'), dict(ranking='knn_agreement'),
    dict(ranking='knn_agreement', selection='threshold', gate='none'),
    dict(ranking='knn_agreement', selection='threshold', sample_choice='random'),
    dict(ranking='knn_agreement', selection='threshold', mode='scheduled'),
    dict(ranking='knn_agreement', selection='threshold', thresholds=[.3, .5]),
    dict(ranking='knn_agreement', selection='threshold', thresholds=[0., .7, .5]),
    dict(ranking='knn_agreement', selection='threshold', knn_k=0),
    dict(ranking='knn_agreement', selection='threshold', nonreciprocal_weight=0.),
    dict(ranking='knn_agreement', selection='threshold', levels='all'),
    dict(ranking='knn_agreement', selection='threshold', persist=1),
    dict(ranking='knn_agreement', selection='threshold', scoring_sampling='weighted'),
    dict(ranking='knn_agreement', selection='threshold', mode='fixed', gate='none', sample_choice='random',
         actions='abstain_move'),
    dict(actions='abstain_move'),
    dict(persist=True), dict(levels='active'), dict(save_arrays=True), dict(compose_local=True)])
def test_invalid_global_combinations_are_rejected_with_named_fields(change):
    with pytest.raises(ValueError, match='population_policy'):
        PopulationSettings(**change)


def test_hybrid_requires_explicit_compose_local_and_local_skips_abstained_rows():
    cfg = load_config(ROOT/'configs/selex_scars_crish_200.json')
    cfg['population_policy'] = global_policy().to_dict()
    cfg['crish']['enabled'] = True
    with pytest.raises(ValueError, match='compose_local'):
        validate_config(cfg)
    cfg['population_policy']['compose_local'] = True
    assert validate_config(cfg)['population_policy']['compose_local'] is True
    cfg = small_config()
    dataset, state, args, head = fixture()
    row = int((~state.labeled).nonzero()[0])
    abstained = state.abstained[0].clone(); abstained[row] = True
    edited = replace(state, abstained=(abstained,))
    action = Action('abstain', int(state.ids[row]), 0, int(state.labels[0][row]))
    local_cfg = copy.deepcopy(cfg)
    local_cfg['crish'] = asdict(CrishSettings(warmup_epochs=0, train_batch=4, evidence_samples=4, k=1,
        probe_known=2, probe_unlabeled=2, min_support=1, max_refresh_seconds=20, save_partitions=False))
    with tempfile.TemporaryDirectory() as directory:
        with patch('crish.engine._shortlist', return_value=([action], {action.sample_id: []})):
            out = CrishEngine(EpochRuntime(directory, local_cfg), objective=selex_loss).prepare(
                edited, torch.nn.Identity(), head, torch.optim.SGD(head.parameters(), lr=.002), dataset, args, 0)
        record = json.loads((Path(directory)/'crish.jsonl').read_text())
    assert out is edited and record['candidate_count'] == 0


def test_preparer_profiles_manifest_and_commands_without_launching(tmp_path):
    template = load_config(ROOT/'configs/selex_scars_crish_200.json')
    profiles = global_variants(template)
    assert tuple(profiles) == PROFILES
    for name, cfg in profiles.items():
        p = cfg['population_policy']
        assert cfg['method'] == 'crish' and not cfg['crish']['enabled'] and not cfg['pair_reliability']['enabled']
        assert (p['ranking'], p['selection'], p['levels'], p['warmup_epochs'], p['refresh_every']) == (
            'knn_agreement', 'threshold', 'active', 10, 5)
        assert (p['scoring_batches'], p['scoring_sampling'], p['probe_float64'], p['save_arrays']) == (
            4, 'train_sampler', True, True)
        assert p['persist'] is (name != 'g_cf_nopersist')
    assert profiles['g_fixed']['population_policy']['gate'] == 'none'
    assert profiles['g_fixed']['population_policy']['fixed_threshold'] == .5
    assert profiles['g_random']['population_policy']['sample_choice'] == 'random'
    assert profiles['g_cf']['population_policy']['thresholds'] == [0., .3, .5, .7]
    differ = lambda a, b: {k for k in a['population_policy'] if a['population_policy'][k] != b['population_policy'][k]}
    assert differ(profiles['g_fixed'], profiles['g_random']) == {'sample_choice'}
    assert differ(profiles['g_cf'], profiles['g_cf_nopersist']) == {'persist'}
    result = prepare(ROOT/'configs/selex_scars_crish_200.json', tmp_path/'plan', [1, 2])
    assert result['status'] == 'prepared_not_executed' and result['jobs'] == 2*len(PROFILES)
    manifest = json.loads((tmp_path/'plan/manifest.json').read_text())
    assert manifest['git_sha'] and manifest['code_fingerprint'] and manifest['benchmark_metrics'] is None
    assert all(len(e['sha256']) == 64 for e in manifest['entries'])
    commands = (tmp_path/'plan/commands.txt').read_text()
    assert commands.count('--variant global --eval-every 1') == 2*len(PROFILES)
    assert json.loads((tmp_path/'plan/g_cf_seed2.json').read_text())['seed'] == 2
    assert not list(tmp_path.rglob('*.pt'))
    hybrid = global_variants(template, ('g_cf_local',))['g_cf_local']
    assert hybrid['crish']['enabled'] and hybrid['population_policy']['compose_local']
    assert hybrid['crish']['max_accepted'] == 8 and hybrid['crish']['boundaries'] == 16


def cli_dry_run(argv):
    from selex_runtime import cli
    stages = []
    def fake(command, **unused):
        stages.append(load_config(command[command.index('--config')+1]))
    out = io.StringIO()
    with patch.object(sys, 'argv', ['cli']+argv), patch('subprocess.run', side_effect=fake), redirect_stdout(out):
        cli.main()
    return json.JSONDecoder().raw_decode(out.getvalue().strip())[0], stages


@pytest.mark.parametrize('profile', PROFILES+('g_cf_local',))
def test_cli_dry_run_prints_all_global_fields_for_each_profile(tmp_path, profile):
    template = load_config(ROOT/'configs/selex_scars_crish_200.json')
    path = tmp_path/f'{profile}.json'
    path.write_text(json.dumps(global_variants(template, (profile,))[profile]))
    variant = 'global_local' if profile == 'g_cf_local' else 'global'
    summary, (smoke, full) = cli_dry_run(['--config', str(path), '--variant', variant, '--dry-run'])
    shown = summary['population_policy']
    assert set(GLOBAL_FIELDS) <= set(shown) and shown['enabled'] and shown['ranking'] == 'knn_agreement'
    assert summary['discrete_repair'] is (profile == 'g_cf_local')
    assert not summary['pair_reliability']['enabled']
    assert smoke['require_intervention_smoke'] and smoke['population_policy']['warmup_epochs'] == 0
    assert full['population_policy']['compose_local'] is (profile == 'g_cf_local')
    assert full['crish']['enabled'] is (profile == 'g_cf_local')


def test_cli_global_variant_rejects_historical_population_config():
    with pytest.raises(SystemExit):
        cli_dry_run(['--variant', 'global', '--dry-run'])


def test_launcher_subprocess_dry_run_for_global_profile(tmp_path):
    template = load_config(ROOT/'configs/selex_scars_crish_200.json')
    path = tmp_path/'g_cf.json'
    path.write_text(json.dumps(global_variants(template, ('g_cf',))['g_cf']))
    result = subprocess.run(['bash', 'scripts/train_selex.sh', '--config', str(path), '--variant', 'global',
                             '--dry-run'], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert '"selection": "threshold"' in result.stdout and '"status": "dry_run"' in result.stdout


@pytest.mark.parametrize('gate', ['none', 'counterfactual'])
def test_smoke_gate_requires_exercised_global_policy(gate):
    for exercised in (True, False):
        dataset, state, args, head = fixture()
        # min_cluster_kept=100 leaves no quota, so a gate-none policy abstains nothing.
        quota = dict(min_cluster_kept=2 if exercised else 100)
        if gate == 'none':
            cfg = small_config(mode='fixed', gate='none', fixed_threshold=.6, min_retained_mass=.3, **quota)
        else:
            cfg = small_config(min_retained_mass=.3)
        cfg['require_intervention_smoke'] = True
        cfg['epochs'] = 1
        args.epochs = 1
        class Runtime(EpochRuntime):
            def monitor_epoch(self, *unused): pass
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory, cfg)
            def extract(*unused): runtime.hierarchy_state = state.copy()
            run = lambda: train_strict(head, torch.nn.Linear(4, 4), DataLoader(dataset, batch_size=16),
                                       DataLoader(dataset, batch_size=16), args, runtime,
                                       objective=selex_loss, extractor=extract)
            if exercised:
                with patch('crish.population.exact_sgd_comparison', return_value=dict(gain=1., known_increase=0.)):
                    run()
                assert (Path(directory)/'resume.pt').exists()
            else:
                failure = (patch('crish.population.exact_sgd_comparison', side_effect=ValueError('fixture failure'))
                           if gate == 'counterfactual' else nullcontext())
                with failure, pytest.raises(RuntimeError, match='smoke'):
                    run()
                assert not (Path(directory)/'resume.pt').exists()


# 7. Move ------------------------------------------------------------------
def blob_state():
    """Tight blobs A,B,D (8 rows) and C (10 rows). In C, labeled row 16 and unlabeled
    rows 18,19 carry cluster 1; with k=6 each of 18,19 has >=4 cluster-2 neighbors."""
    generator = torch.Generator().manual_seed(7)
    centers = torch.tensor([[0., 0.], [0., 10.], [10., 0.], [10., 10.]])
    sizes = [8, 8, 10, 8]
    x = torch.cat([centers[c]+.3*torch.randn(sizes[c], 2, generator=generator) for c in range(4)])
    base = torch.cat([torch.full((s,), c) for c, s in enumerate(sizes)])
    base[[16, 18, 19]] = 1
    labeled = torch.zeros(34, dtype=torch.bool)
    labeled[[0, 1, 8, 9, 16, 20, 26, 27]] = True
    coarse = torch.tensor([0, 0, 1, 1])[base]
    labels = (base, coarse)
    protos = tuple(torch.stack([x[y == c].mean(0) for c in range(int(y.max())+1)]) for y in labels)
    return SupervisionState(torch.arange(34)*3+1, x, labeled, base[labeled], labels, protos,
        tuple(prototype_radii(p) for p in protos), tuple(torch.zeros(34, dtype=torch.bool) for _ in labels),
        allow_empty_clusters=True)


def blob_evidence(state):
    # Unit edge weights make neighbor shares count-based and the fixture exact.
    return knn_evidence(state, k=6, nonreciprocal_weight=1.)


def move_policy(**changes):
    return global_policy(**dict(dict(actions='abstain_move', min_retained_mass=.2, knn_k=6,
                                     move_target_cap=.5), **changes))


def test_move_reassigns_only_unlabeled_rows_with_sufficient_share_and_quotas():
    state = blob_state()
    evidence = blob_evidence(state)
    from crish.global_moves import apply_moves, neighbor_shares
    settings = move_policy()
    moved_state, moved, target, record = apply_moves(state, evidence, .5, settings, min_support=2)
    assert set(moved.nonzero().flatten().tolist()) == {18, 19} and (target[moved] == 2).all()
    assert not moved[state.labeled].any() and int(moved_state.labels[0][16]) == 1  # labeled anchor kept
    _, shares = neighbor_shares(state, evidence, moved.nonzero().flatten().tolist())
    assert min(shares) >= settings.move_min_share
    # Target cap: max(1, floor(.25*support_b)) with support_b=7 admits one row only.
    _, capped, _, capped_record = apply_moves(state, evidence, .5, move_policy(move_target_cap=.25), min_support=2)
    assert int(capped.sum()) == 1 and capped_record['move_rejected_target_cap'] == 1
    # Share threshold above every observed share: no Move.
    _, none, _, _ = apply_moves(state, evidence, .5, move_policy(move_min_share=1.), min_support=2)
    assert not none.any() or all(s >= 1. for s in neighbor_shares(state, evidence, none.nonzero().flatten().tolist())[1])
    # Source quota: cluster 1 has 8 unlabeled rows; fraction .2 lets one leave.
    _, quota, _, quota_record = apply_moves(state, evidence, .5, move_policy(max_cluster_fraction=.2), min_support=2)
    assert int(quota.sum()) == 1 and quota_record['move_rejected_source_quota'] >= 1
    # Exact population protected-coefficient cap, cross-checked with full-population drift.
    from crish.rebuild import target_drift
    views = torch.cat([state.features, state.features])
    shift = max(x['protected_coefficient_shift_max'] for x in target_drift(
        state, moved_state, state.ids, views, smoothing=.5, normalization='selex_legacy')['sse_levels'])
    assert 0 < shift <= .25
    _, blocked, _, blocked_record = apply_moves(state, evidence, .5, settings, min_support=2,
                                                max_protected_shift=shift/10)
    assert not blocked.any() and blocked_record['move_rejected_protected_shift'] == 2


def test_move_propagates_to_coarse_levels_and_rebuilds_finite_prototypes():
    state = blob_state()
    evidence = blob_evidence(state)
    proposed, coverage = apply_threshold(state, evidence, .5, move_policy())
    assert coverage['moved'] == 2 and proposed.validate() is None
    moved = proposed.labels[0] != state.labels[0]
    assert set(moved.nonzero().flatten().tolist()) == {18, 19}
    for group in proposed.labels[0].unique().tolist():
        assert len(proposed.labels[1][proposed.labels[0] == group].unique()) == 1
    assert (proposed.labels[1][moved] == 1).all()
    for level in range(2):
        assert torch.isfinite(proposed.prototypes[level]).all() and torch.isfinite(proposed.radii[level]).all()
        for group in proposed.labels[level][moved].unique().tolist() + state.labels[level][moved].unique().tolist():
            torch.testing.assert_close(proposed.prototypes[level][group],
                                       state.features[proposed.labels[level] == group].mean(0))
        assert not proposed.abstained[level][moved].any()  # moved rows are never abstained
        assert not proposed.abstained[level][state.labeled].any()
    assert torch.equal(proposed.labels[0][state.labeled], state.known_targets)
    assert torch.equal(proposed.known_targets, state.known_targets)
    # Departures (Move + Abstain) from each source respect the quota.
    for group in state.labels[0].unique().tolist():
        u = ~state.labeled & (state.labels[0] == group)
        left = int((u & (moved | proposed.abstained[0])).sum())
        assert left <= int(int(u.sum())*.5) and int(u.sum())-left >= min(2, int(u.sum()))
    from crish.rebuild import target_drift
    views = torch.cat([state.features, state.features])
    drift = target_drift(state, proposed, state.ids, views, smoothing=.5, normalization='selex_legacy')
    assert 0 < max(x['protected_coefficient_shift_max'] for x in drift['sse_levels']) <= .25


def test_move_counterfactual_and_epoch_arrays_record_moves():
    compare, _ = scripted({.3: (1., 0.), .5: (3., 0.), .7: (2., 0.)}, confirmation=.5)
    with tempfile.TemporaryDirectory() as directory:
        state, outcome, record, engine = run_select(directory, compare, actions='abstain_move',
                                                    move_min_share=.5, save_arrays=True)
        epoch = json.loads((Path(directory)/'crish_policy_epochs.jsonl').read_text())
        with np.load(Path(directory)/'crish_g_epoch_0010.npz', allow_pickle=False) as stored:
            arrays = dict(stored)
    assert record['chosen_threshold'] == .5 and outcome.validate() is None
    moved = outcome.labels[0] != state.labels[0]
    assert epoch['moved'] == int(moved.sum()) == record['moved']
    assert np.array_equal(arrays['moved'], moved.numpy()) and arrays['moved'].any()
    assert np.array_equal(arrays['move_target'][arrays['moved']], outcome.labels[0][moved].numpy())
    assert (arrays['move_target'][~arrays['moved']] == -1).all()
    assert np.array_equal(arrays['label_0'], state.labels[0].numpy())


# Phase 3: offline-only audit tools ----------------------------------------
def test_offline_tools_are_not_reachable_from_training_code():
    for package in ('crish', 'selex_runtime', 'gcd_runtime'):
        for path in (ROOT/package).glob('*.py'):
            text = path.read_text(encoding='utf8')
            assert 'audit_global_policy' not in text and 'export_offline_labels' not in text, path
    code = ('import sys; import crish.training, crish.population, selex_runtime.trainer, selex_runtime.cli, '
            'gcd_runtime.launch, gcd_runtime.worker; '
            'assert not any(m.endswith(("audit_global_policy", "export_offline_labels")) for m in sys.modules)')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_offline_label_rows_apply_offset_and_transform_without_duplicates():
    from types import SimpleNamespace
    from scripts.export_offline_labels import label_rows
    remap = {3: 0, 1: 1, 0: 2, 2: 3}.get
    merged = SimpleNamespace(
        labelled_dataset=SimpleNamespace(uq_idxs=[10, 11], target=[4, 2], target_transform=remap),
        unlabelled_dataset=SimpleNamespace(uq_idxs=[12, 13], target=[1, 3], target_transform=remap))
    assert label_rows(merged, 'scars') == [(10, 0), (11, 1), (12, 2), (13, 3)]
    merged.unlabelled_dataset.uq_idxs = [12, 10]
    with pytest.raises(ValueError, match='Duplicate'):
        label_rows(merged, 'scars')


def test_audit_matches_brute_force_pairs_auroc_and_move_majority(tmp_path):
    from scripts.audit_global_policy import audit
    generator = np.random.default_rng(3)
    n = 60
    ids = np.arange(n)*7+2
    truth = generator.integers(0, 4, n)
    labeled = np.zeros(n, dtype=bool); labeled[:12] = True
    base = np.where(generator.random(n) < .7, truth, generator.integers(0, 4, n))
    base[labeled] = truth[labeled]
    coarse = np.array([0, 0, 1, 1])[base]
    r0 = generator.random(n).astype(np.float16)
    abstained = (r0 < .3) & ~labeled
    moved = np.zeros(n, dtype=bool); moved[[20, 21]] = True; abstained[moved] = False
    target = np.full(n, -1); target[[20, 21]] = (base[[20, 21]]+1) % 4
    np.savez_compressed(tmp_path/'crish_g_epoch_0012.npz', ids=ids, labeled=labeled, label_0=base, label_1=coarse,
                        reliability_0=r0, reliability_1=r0, abstained_0=abstained, abstained_1=abstained & (r0 < .1),
                        moved=moved, move_target=target)
    labels = tmp_path/'labels.csv'
    labels.write_text('sample_id,true_label\n'+''.join(f'{i},{y}\n' for i, y in zip(ids, truth)))
    report = audit(tmp_path, labels)
    assert report['uses_evaluation_labels'] is True and len(report['epochs']) == 1
    epoch = report['epochs'][0]
    assert epoch['epoch'] == 12 and epoch['labeled_anchor_agreement']['fraction'] == 1.
    for level, (clusters, mask) in enumerate(((base, abstained), (coarse, abstained & (r0 < .1)))):
        removed = retained = removed_wrong = retained_wrong = 0
        for i in range(n):
            for j in range(i+1, n):
                if clusters[i] != clusters[j]: continue
                gone = mask[i] or mask[j]
                bad = truth[i] != truth[j]
                removed += gone; retained += not gone
                removed_wrong += gone and bad; retained_wrong += (not gone) and bad
        got = epoch['levels'][level]
        assert (got['removed_pair_wrong_rate']['numerator'], got['removed_pair_wrong_rate']['denominator']) == (
            removed_wrong, removed)
        assert (got['retained_pair_wrong_rate']['numerator'], got['retained_pair_wrong_rate']['denominator']) == (
            retained_wrong, retained)
        assert got['abstained_unlabeled']['numerator'] == int(mask.sum())
    # Leave-one-out majority error and brute-force AUROC.
    wrong, scores = [], []
    for i in np.flatnonzero(~labeled):
        others = [truth[j] for j in range(n) if j != i and base[j] == base[i]]
        counts = Counter(others).most_common()
        if not counts or (len(counts) > 1 and counts[0][1] == counts[1][1]): continue
        wrong.append(counts[0][0] != truth[i]); scores.append(1-float(r0[i]))
    wrong, scores = np.array(wrong), np.array(scores)
    level0 = epoch['level0']
    assert level0['pseudo_label_error']['numerator'] == int(wrong.sum())
    assert level0['pseudo_label_error']['denominator'] == len(wrong)
    pairs = [(a > b)+.5*(a == b) for a in scores[wrong] for b in scores[~wrong]]
    assert level0['auroc_one_minus_r0']['value'] == pytest.approx(sum(pairs)/len(pairs))
    assert epoch['move']['moved'] == 2
    assert epoch['move']['target_majority_correct']['denominator'] <= 2
