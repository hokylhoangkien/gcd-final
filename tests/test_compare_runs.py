"""Seed-paired log comparison and the matched SelEx control profile; no training."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from gcd_runtime.config import ROOT, load_config
from scripts.compare_runs import compare, markdown
from scripts.prepare_global_policy import POLICY_KEYS, config_differences, global_variants, prepare

PROTOCOL = 'balanced_ss_kmeans_train_unlabeled'
REFERENCE = ROOT/'results/paper-plan-verified-20261118/curves-cars/data.json'


def curve(epoch_number, seed, offset):
    """Deterministic synthetic SS fractions; offset shifts the whole run."""
    base = .70+.001*epoch_number+offset+.0005*seed
    return dict(all_acc=min(base, .99), old_acc=min(base+.1, .99), new_acc=min(base-.05, .99))


def make_run(root, name, seed, *, offset=0., epochs=12, skip=(), method='crish', policy=False):
    run = Path(root)/name
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(dict(seed=seed, epochs=epochs, method=method, dataset_name='scars')))
    (run/'metadata.json').write_text(json.dumps(dict(run_id=f'id-{name}', git_sha='abc123')))
    rows = []
    for number in range(1, epochs):
        if number in skip:
            continue
        rows.append(dict(status='observed', unit='fraction', protocol=PROTOCOL, purpose='monitor_only',
                         epoch=number-1, epoch_number=number, n_unlabeled=100, **curve(number, seed, offset)))
    (run/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    final = dict(status='observed', unit='fraction', protocol=PROTOCOL, purpose='final', epoch=epochs-1,
                 epoch_number=epochs, n_unlabeled=100, **curve(epochs, seed, offset))
    (run/'evaluation_last.json').write_text(json.dumps(final))
    # A best-Old checkpoint report must never overwrite an epoch of the history.
    (run/'best_old_ss_metrics.jsonl').write_text(json.dumps(dict(final, purpose='best_old_test', epoch=2,
                                                                  all_acc=.01, old_acc=.01, new_acc=.01))+'\n')
    if policy:
        epochs_log = []
        for log_epoch in range(2, epochs):  # 0-based policy epochs
            epochs_log.append(dict(epoch=log_epoch, tau=.5, moved=2 if log_epoch % 2 else 0, levels=[
                dict(level=0, abstained_unlabeled=log_epoch, fraction=log_epoch/100),
                dict(level=1, abstained_unlabeled=1, fraction=.01)]))
        (run/'crish_policy_epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in epochs_log))
        due = [dict(epoch=2, status='completed', chosen_threshold=.5, candidates=[
                   dict(threshold=.3, status='rejected', reason='Discovery known guard failed'),
                   dict(threshold=.5, status='accepted')]),
               dict(epoch=7, status='keep', chosen_threshold=0., reason='No eligible population threshold',
                    candidates=[dict(threshold=.3, status='rejected', reason='Discovery known guard failed')])]
        (run/'crish_p.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in due))
    return run


def test_readings_window_missing_epochs_and_best_old_rows_are_ignored(tmp_path):
    treatment = make_run(tmp_path, 't1', 1, offset=.02, skip=(4,), policy=True)
    control = make_run(tmp_path, 'c1', 1, method='baseline')
    result = compare([treatment], [control], window=(3, 6))
    t = result['treatment'][0]
    assert t['ss']['missing_epochs'] == [4] and t['ss']['window_missing'] == [4] and t['ss']['window_epochs'] == 3
    all_t = t['ss']['metrics']['All']
    assert all_t['best_epoch'] == 12 and all_t['best'] == pytest.approx(100*curve(12, 1, .02)['all_acc'])
    assert all_t['last'] == pytest.approx(100*curve(12, 1, .02)['all_acc'])
    window = [100*curve(e, 1, .02)['all_acc'] for e in (3, 5, 6)]
    assert all_t['window_mean'] == pytest.approx(statistics.mean(window))
    assert all_t['window_std'] == pytest.approx(statistics.stdev(window))
    assert result['per_seed'][0]['delta']['All']['window_mean'] == pytest.approx(
        statistics.mean(window)-statistics.mean(100*curve(e, 1, 0.)['all_acc'] for e in (3, 4, 5, 6)))
    assert t['git_sha'] == 'abc123' and len(t['config_sha256']) == 64 and t['run_id'] == 'id-t1'


def test_seed_pairing_aggregates_and_rejects_unmatched_or_duplicate_seeds(tmp_path):
    treatments = [make_run(tmp_path, f't{s}', s, offset=o) for s, o in ((1, .01), (2, -.002), (3, .03))]
    controls = [make_run(tmp_path, f'c{s}', s, method='baseline') for s in (3, 1, 2)]
    result = compare(treatments, controls, window=(5, 8))
    assert [row['seed'] for row in result['per_seed']] == [1, 2, 3]
    deltas = [row['delta']['All']['last'] for row in result['per_seed']]
    agg = result['aggregate']['All']['last']
    assert agg['mean'] == pytest.approx(statistics.mean(deltas)) and agg['std'] == pytest.approx(statistics.stdev(deltas))
    assert agg['min'] == pytest.approx(min(deltas)) and agg['positive'] == 2 and agg['seeds'] == 3
    assert set(result['aggregate']) == {'All', 'Old', 'New'}
    assert set(result['aggregate']['New']) == {'best', 'last', 'window_mean'}
    with pytest.raises(ValueError, match='without a treatment/control pair'):
        compare(treatments, controls[:2], window=(5, 8))
    with pytest.raises(ValueError, match='Duplicate'):
        compare(treatments+[make_run(tmp_path, 't1b', 1)], controls, window=(5, 8))
    with pytest.raises(ValueError, match='either'):
        compare(treatments, [], window=(5, 8))
    text = markdown(result)
    assert '| 1 | All | best |' in text and '| All | window_mean |' in text and '2/3' in text


def test_policy_dose_uses_zero_based_logs_and_counts_tau_and_reasons(tmp_path):
    treatment = make_run(tmp_path, 't1', 1, policy=True)
    control = make_run(tmp_path, 'c1', 1, method='baseline')
    result = compare([treatment], [control], window=(4, 6))  # epoch_number 4..6 = log epochs 3..5
    policy = result['treatment'][0]['policy']
    assert policy['window_epochs_logged'] == 3 and policy['window_epochs_missing'] == []
    assert policy['abstained_fraction_by_level']['0']['mean'] == pytest.approx(statistics.mean([.03, .04, .05]))
    assert policy['abstained_fraction_by_level']['1']['mean'] == pytest.approx(.01)
    assert policy['moved_fraction']['mean'] == pytest.approx(statistics.mean([.02, 0., .02]))
    assert policy['level0_abstain_or_move_fraction']['mean'] == pytest.approx(statistics.mean([.05, .04, .07]))
    assert [r['epoch_number'] for r in policy['refreshes']] == [3, 8]
    assert policy['tau_positive'] == dict(numerator=1, denominator=2, fraction=.5)
    assert policy['rejection_reasons'] == {'candidate: Discovery known guard failed': 2,
                                           'refresh: No eligible population threshold': 1}
    assert 'policy' not in result['control'][0]


def test_reference_history_replaces_control_for_g1(tmp_path):
    treatment = make_run(tmp_path, 't1', 1, epochs=200, policy=True)
    result = compare([treatment], reference=REFERENCE, window=(61, 100))
    reference = result['control'][0]
    assert reference['run_id'] == '20261115T112441Z-0ec6f014' and len(reference['reference_history_sha256']) == 64
    metrics = reference['ss']['metrics']
    assert round(metrics['All']['window_mean'], 2) == 81.35 and round(metrics['Old']['window_mean'], 2) == 92.81
    assert round(metrics['New']['window_mean'], 2) == 75.82 and reference['ss']['window_epochs'] == 40
    assert reference['ss']['missing_epochs'] == list(range(1, 7))
    with pytest.raises(ValueError, match='same seed'):
        compare([make_run(tmp_path, 't2', 2, epochs=200)], reference=REFERENCE, window=(61, 100))
    with pytest.raises(ValueError, match='either'):
        compare([treatment], [make_run(tmp_path, 'c1', 1)], reference=REFERENCE)


def test_command_line_writes_json_and_prints_markdown(tmp_path):
    treatment = make_run(tmp_path, 't1', 1, policy=True)
    control = make_run(tmp_path, 'c1', 1, method='baseline')
    output = tmp_path/'compare.json'
    result = subprocess.run([sys.executable, 'scripts/compare_runs.py', '--treatment', str(treatment),
                             '--control', str(control), '--window', '3', '6', '--output', str(output)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert '| Seed | Metric | Reading |' in result.stdout
    data = json.loads(output.read_text())
    assert data['kind'] == 'seed_paired_log_comparison_not_reevaluated' and data['window'] == [3, 6]
    again = subprocess.run([sys.executable, 'scripts/compare_runs.py', '--treatment', str(treatment),
                            '--control', str(control), '--output', str(output)],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert again.returncode != 0  # never overwrites an existing report


def test_control_profile_matches_g_cf_except_method_and_policy(tmp_path):
    result = prepare(ROOT/'configs/selex_scars_crish_200.json', tmp_path/'plan', [2, 3], control=True)
    manifest = json.loads((tmp_path/'plan/manifest.json').read_text())
    controls = [e for e in manifest['entries'] if e['name'] == 'control']
    assert result['jobs'] == 2*5+2 and [e['seed'] for e in controls] == [2, 3]
    template = load_config(ROOT/'configs/selex_scars_crish_200.json')
    for entry in controls:
        cfg = load_config(tmp_path/'plan'/entry['config'])
        g_cf = global_variants(dict(template, seed=entry['seed']), ('g_cf',))['g_cf']
        assert cfg['method'] == 'baseline' and cfg['seed'] == entry['seed'] and cfg['epochs'] == 200
        differences = config_differences(cfg, g_cf)
        assert differences == entry['differs_from_g_cf'] and set(differences) <= set(POLICY_KEYS)
        assert set(differences) == {'method', 'population_policy'}
        assert entry['command'][-3:] == ['--baseline', '--eval-every', '1']
    commands = (tmp_path/'plan/commands.txt').read_text()
    assert commands.count('--baseline --eval-every 1') == 2


def test_control_profile_cli_dry_run_selects_baseline(tmp_path):
    from tests.test_global_policy import cli_dry_run
    prepare(ROOT/'configs/selex_scars_crish_200.json', tmp_path/'plan', [1], control=True)
    summary, (smoke, full) = cli_dry_run(['--config', str(tmp_path/'plan/control_seed1.json'), '--baseline',
                                          '--eval-every', '1', '--dry-run'])
    assert summary['method'] == 'baseline' and summary['epochs'] == 200 and summary['seed'] == 1
    assert full['method'] == 'baseline' and full['ss_eval_every_epochs'] == 1
    assert smoke['epochs'] == 1 and smoke['require_intervention_smoke']
