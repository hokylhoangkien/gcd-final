"""Online reliability coefficient semantics, baseline parity and real SGD wiring."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crish.candidates import Action
from crish.config import CrishSettings, PairReliabilitySettings
from crish.engine import CrishEngine
from crish.online_pairs import online_pair_weights, pair_diagnostics
from crish.rebuild import (checked_pair_weights, prototype_radii, pseudo_masks,
                          pseudo_sse_loss, rebuild, target_drift)
from crish.state import SupervisionState
from crish.training import train_strict
from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.config import ROOT, load_config
from selex_runtime.objective import selex_loss, source_losses
from selex_runtime.trainer import make_args


def settings(**changes):
    return PairReliabilitySettings(**dict(dict(enabled=True, k=1, warmup_epochs=0, ramp_epochs=1), **changes))


def fixture():
    angles = torch.tensor([0., .06, 1., 1.04, 2., 2.04, 3., 3.04])
    first = torch.stack([angles.cos(), angles.sin()], dim=1)
    second = first.clone()
    second[[1, 4]] = second[[4, 1]]
    ids, labeled = torch.arange(8)*11+17, torch.tensor([True, False, True]+[False]*5)
    labels = torch.arange(8)//2
    centers = torch.stack([first[labels == group].mean(0) for group in range(4)])
    state = SupervisionState(ids, first, labeled, labels[labeled], (labels,), (centers,),
        (prototype_radii(centers),), (torch.zeros(8, dtype=torch.bool),))
    cfg = load_config(ROOT/'configs/selex_scars_crish_200.json')
    cfg['pair_reliability'] = asdict(settings())
    args = make_args(cfg)
    args.hierarchy_levels, args.num_labeled_classes, args.num_unlabeled_classes = 1, 2, 2
    return torch.cat([first, second]), state, args


@pytest.mark.parametrize('changes', [dict(enabled=1), dict(reciprocal='yes'), dict(k=True),
    dict(k=0), dict(warmup_epochs=-1), dict(ramp_epochs=0), dict(floor=0),
    dict(floor=float('nan')), dict(floor=1.1), dict(reciprocal=False, cross_view=False)])
def test_settings_reject_invalid_or_unprotected_options(changes):
    with pytest.raises(ValueError):
        settings(**changes)


def test_warmup_ramp_bounds_symmetry_protection_and_unknown_edges():
    raw, state, _ = fixture()
    cfg = settings(warmup_epochs=10, ramp_epochs=10)
    assert [cfg.strength(e) for e in (0, 9, 10, 14, 19, 99)] == [0., 0., .1, .5, 1., 1.]
    for epoch in (9, 10, 19):
        w, diag = online_pair_weights(raw.requires_grad_(), state.ids, state.labeled, cfg, epoch)
        if epoch == 9:
            assert w is None
            continue
        assert not w.requires_grad and torch.equal(w, w.T)
        assert w.min() >= cfg.floor and w.max() <= 1
        assert torch.equal(w.diag(), torch.ones(8))
        assert w[0, 2] == 1  # both legally labeled
        assert w[0, 6] == 1  # unobserved in both local graphs: unknown
        assert w[2, 3] == 1  # stable reciprocal edge in both views
        assert float(w[0, 1]) == pytest.approx(1-cfg.strength(epoch)*.75)
        assert diag['measured_edges'] > 0


def test_component_ablation_duplicate_ids_and_permutation():
    raw, state, _ = fixture()
    both, _ = online_pair_weights(raw, state.ids, state.labeled, settings(), 0)
    rn, _ = online_pair_weights(raw, state.ids, state.labeled, settings(cross_view=False), 0)
    cvs, _ = online_pair_weights(raw, state.ids, state.labeled, settings(reciprocal=False), 0)
    assert torch.all(both <= rn) and torch.all(both <= cvs)
    assert rn[0, 1] == .625 and cvs[0, 1] == .25
    order = torch.tensor([7, 3, 4, 1, 5, 2, 0, 6, 1])
    a, b = raw.chunk(2)
    actual, diag = online_pair_weights(torch.cat([a[order], b[order]]), state.ids[order],
        state.labeled[order], settings(), 0)
    torch.testing.assert_close(actual, both[order[:, None], order[None, :]], rtol=0, atol=0)
    assert diag['unique_ids'] == 8 and diag['repeated_rows'] == 1
    assert actual[3, 8] == 1  # duplicate occurrences of one identity
    invalid = state.labeled[order].clone(); invalid[-1] = ~invalid[-1]
    with pytest.raises(ValueError, match='conflicting'):
        online_pair_weights(torch.cat([a[order], b[order]]), state.ids[order], invalid, settings(), 0)


def test_collapse_small_pool_and_invalid_vectors_use_documented_fallbacks():
    raw, state, _ = fixture()
    for collapsed in (torch.ones_like(raw), torch.zeros_like(raw)):
        w, diag = online_pair_weights(collapsed, state.ids, state.labeled, settings(), 0)
        assert w is None and diag['status'] == 'collapsed_view_keep'
    a, b = raw.chunk(2)
    w, diag = online_pair_weights(torch.cat([a[:2], b[:2]]), state.ids[:2], state.labeled[:2], settings(), 0)
    assert w is None and diag['status'] == 'insufficient_unique_ids_keep'
    broken = raw.clone(); broken[1] = 0; broken[9] = 0
    w, diag = online_pair_weights(broken, state.ids, state.labeled, settings(), 0)
    assert torch.equal(w[1], torch.ones(8)) and diag['invalid_unique_ids'] == 1
    broken[0, 0] = torch.nan
    with pytest.raises(ValueError, match='finite'):
        online_pair_weights(broken, state.ids, state.labeled, settings(), 0)


def test_weighted_pseudo_loss_matches_independent_coefficients_and_backward():
    raw, state, _ = fixture()
    w, _ = online_pair_weights(raw, state.ids, state.labeled, settings(), 0)
    torch.manual_seed(19)
    projected = torch.randn(8, 2, 5, requires_grad=True)
    criterion = source_losses('cpu')['SupConLoss'](); criterion.hyper_c = 0
    got = pseudo_sse_loss(criterion, projected, state, state.ids, 0, pair_weights=w)
    flat = torch.cat(projected.unbind(1))
    logits = -torch.cdist(flat, flat)/criterion.temperature
    offdiag = ~torch.eye(16, dtype=torch.bool)
    logprob = logits-torch.logsumexp(logits.masked_fill(~offdiag, -torch.inf), dim=1, keepdim=True)
    positives = (state.labels[0][:, None] == state.labels[0][None, :]).repeat(2, 2) & offdiag
    original_coefficients = positives.float()/positives.sum(1, keepdim=True)
    coefficients = original_coefficients*w.repeat(2, 2)
    expected = -(criterion.temperature/criterion.base_temperature)*(coefficients*logprob).sum(1).mean()
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    actual_grad = torch.autograd.grad(got, projected, retain_graph=True)[0]
    oracle_grad = torch.autograd.grad(expected, projected)[0]
    torch.testing.assert_close(actual_grad, oracle_grad, rtol=1e-5, atol=2e-6)
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
    protected = torch.eye(8, dtype=torch.bool) | (state.labeled[:, None] & state.labeled[None, :])
    assert torch.equal(coefficients[protected.repeat(2, 2)], original_coefficients[protected.repeat(2, 2)])
    assert coefficients[0, 1] == original_coefficients[0, 1]*.25


def test_weight_validation_and_drift_use_actual_weighted_target_mass():
    raw, state, _ = fixture()
    w, _ = online_pair_weights(raw, state.ids, state.labeled, settings(), 0)
    for bad in (w.clone().fill_(torch.nan), w*0, w.clone().requires_grad_()):
        with pytest.raises(ValueError): checked_pair_weights(bad, state, state.ids, 'cpu')
    asymmetric = w.clone(); asymmetric[0, 1] = .8
    with pytest.raises(ValueError, match='symmetric'):
        checked_pair_weights(asymmetric, state, state.ids, 'cpu')
    changed, _ = rebuild(state, Action('abstain', int(state.ids[1]), 0, 0))
    kwargs = dict(smoothing=.75, normalization='selex_legacy')
    plain = target_drift(state, changed, state.ids, raw, **kwargs)
    weighted = target_drift(state, changed, state.ids, raw, pair_weights=w, **kwargs)
    assert weighted['sse_levels'][0]['coefficient_half_l1_mean'] == pytest.approx(
        .25*plain['sse_levels'][0]['coefficient_half_l1_mean'])
    assert weighted['sse_levels'][0]['protected_coefficient_shift_max'] == 0
    assert weighted['use_target_half_l1_mean'] == 0
    diag = pair_diagnostics(w, state.labels[0], state.ids, state.labeled, {})
    assert diag['eligible_pairs'] == 8 and diag['changed_pairs'] == 4
    assert sum(diag['weight_histogram']) == diag['eligible_pairs']
    retained_diag = pair_diagnostics(w, state.labels[0], state.ids, state.labeled, {},
                                     retained=pseudo_masks(changed, state.ids, 0)[1])
    assert retained_diag['excluded_by_abstention'] == 2
    assert retained_diag['changed_pairs'] == diag['changed_pairs']-2


def test_baseline_off_warmup_and_floor_one_have_exact_loss_gradient_parity():
    raw, state, args = fixture()
    torch.manual_seed(31)
    head = torch.nn.Linear(2, 16)
    variants = [('baseline', settings()), ('crish', settings(enabled=False)),
                ('crish', settings(warmup_epochs=1)), ('crish', settings(floor=1))]
    results = []
    for method, cfg in variants:
        current = copy.deepcopy(args)
        current.method, current.pair_reliability = method, asdict(cfg)
        model, features = copy.deepcopy(head), raw.detach().clone().requires_grad_()
        result = selex_loss(model, features, state.ids, state.labeled, state.known_targets, state, current, 0)
        gradients = torch.autograd.grad(result['loss'], tuple(model.parameters())+(features,))
        assert result['pseudo_pair_weights'] is None
        results.append((result['loss'].detach(), gradients))
    for loss, grads in results[1:]:
        torch.testing.assert_close(loss, results[0][0], rtol=0, atol=0)
        for got, expected in zip(grads, results[0][1]):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_online_weights_preserve_use_supervised_and_coarse_objectives():
    raw, state, args = fixture()
    coarse = state.labels[0]//2
    centers = torch.stack([state.features[coarse == g].mean(0) for g in range(2)])
    state = replace(state, labels=state.labels+(coarse,), prototypes=state.prototypes+(centers,),
        radii=state.radii+(prototype_radii(centers),), abstained=state.abstained+(torch.zeros(8, dtype=torch.bool),))
    args.hierarchy_levels = 2
    head = torch.nn.Linear(2, 16)
    weighted = selex_loss(head, raw, state.ids, state.labeled, state.known_targets, state, args, 10)
    args.method = 'baseline'
    baseline = selex_loss(head, raw, state.ids, state.labeled, state.known_targets, state, args, 10)
    assert weighted['pair_reliability']['changed_pairs'] > 0
    torch.testing.assert_close(weighted['use'], baseline['use'], rtol=0, atol=0)
    torch.testing.assert_close(weighted['supervised'][0], baseline['supervised'][0], rtol=0, atol=0)
    # Only the base prefix accounts for the difference of the full pseudo loss.
    criterion = source_losses('cpu')['SupConLoss'](); criterion.hyper_c = 0
    paired = torch.stack(weighted['features'].chunk(2), dim=1)[:, :, :8]
    before = pseudo_sse_loss(criterion, paired, state, state.ids, 0)
    after = pseudo_sse_loss(criterion, paired, state, state.ids, 0, pair_weights=weighted['pseudo_pair_weights'])
    torch.testing.assert_close(weighted['pseudo']-baseline['pseudo'], (after-before)/2, rtol=1e-5, atol=1e-6)


def test_hybrid_lookahead_uses_weights_in_drift_and_ignores_hidden_targets():
    from tests.test_selex_runtime import fixture as training_fixture
    outcomes = []
    for poison in (-1, 98765):
        dataset, state, args, head = training_fixture()
        dataset.targets[~dataset.labeled] = poison
        cfg = load_config(ROOT/'configs/selex_crish.json')
        cfg['pair_reliability'] = asdict(settings(k=2))
        args.method, args.pair_reliability = 'crish', cfg['pair_reliability']
        cfg['crish'] = asdict(CrishSettings(warmup_epochs=0, refresh_every=1, evidence_samples=16,
            evidence_trials=1, k=2, boundaries=1, min_support=1, train_batch=8,
            probe_known=2, probe_unlabeled=2, max_protected_shift=1., max_refresh_seconds=20))
        with tempfile.TemporaryDirectory() as directory:
            runtime = EpochRuntime(directory, cfg)
            opt = torch.optim.SGD(head.parameters(), lr=.002, momentum=.9)
            with patch('crish.engine.target_drift', wraps=target_drift) as audit:
                outcome = CrishEngine(runtime, objective=selex_loss).prepare(
                    state, torch.nn.Identity(), head, opt, dataset, args, 0)
                assert audit.call_count > 0
                assert all(call.kwargs['pair_weights'] is not None for call in audit.call_args_list)
            record = json.loads((Path(directory)/'crish.jsonl').read_text())
            assert record['status'] == 'completed' and record['gains']
            outcomes.append((outcome, record['decisions']))
    assert outcomes[0][1] == outcomes[1][1]
    for a, b in zip(outcomes[0][0].labels, outcomes[1][0].labels): assert torch.equal(a, b)


def test_online_sgd_changes_backbone_updates_and_resume_matches_uninterrupted():
    from tests.test_selex_runtime import fixture as training_fixture
    def execute(directory, *, enabled=True, resume=None, stop=False):
        dataset, state, args, head = training_fixture()
        args.epochs = 2
        cfg = load_config(ROOT/'configs/selex_crish_smoke.json')
        cfg.update(device='cpu', epochs=2, mlp_out_dim=16, synthetic_fixture=True,
                   max_steps_per_epoch=1, final_ss_evaluation=False)
        cfg['crish']['enabled'] = False
        cfg['pair_reliability'] = asdict(settings(enabled=enabled, reciprocal=True, cross_view=True))
        class Runtime(EpochRuntime):
            def monitor_epoch(self, *unused): pass
            def save(self, epoch, *values):
                super().save(epoch, *values)
                if stop and epoch == 0: raise InterruptedError('bounded fixture')
        runtime = Runtime(directory, cfg, resume)
        model = torch.nn.Linear(4, 4)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        def extract(*unused): runtime.hierarchy_state = state
        train_strict(head, model, loader, loader, args, runtime, objective=selex_loss, extractor=extract)
        return model.state_dict(), head.state_dict()
    with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as stopped, \
         tempfile.TemporaryDirectory() as resumed, tempfile.TemporaryDirectory() as reference:
        expected = execute(full)
        with pytest.raises(InterruptedError): execute(stopped, stop=True)
        actual = execute(resumed, resume=str(Path(stopped)/'resume.pt'))
        old = execute(reference, enabled=False)
        for a, b in zip(actual, expected):
            assert all(torch.equal(a[k], b[k]) for k in a)
        assert any(not torch.equal(expected[0][k], old[0][k]) for k in expected[0])
        history = [json.loads(line) for line in (Path(full)/'metrics.jsonl').read_text().splitlines()]
        assert len(history) == 2 and all(row['pair_reliability']['changed_pairs'] > 0 for row in history)


def test_cli_exposes_legacy_and_component_controls_and_smoke_forces_activation():
    decoder = json.JSONDecoder()
    for mode in ('off', 'rn', 'cross_view', 'both'):
        command = ['bash', 'scripts/train_selex.sh', '--dry-run', '--pair-reliability', mode, '--no-repair']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        remaining, objects = result.stdout.strip(), []
        while remaining:
            value, end = decoder.raw_decode(remaining); objects.append(value); remaining = remaining[end:].lstrip()
        full, smoke = objects[-1]['config'], objects[-2]['config']
        assert not full['crish']['enabled']
        assert full['pair_reliability']['enabled'] == (mode != 'off')
        if mode != 'off':
            assert smoke['pair_reliability']['warmup_epochs'] == 0 and smoke['pair_reliability']['ramp_epochs'] == 1
            assert full['pair_reliability']['warmup_epochs'] == 10 and full['pair_reliability']['ramp_epochs'] == 10
