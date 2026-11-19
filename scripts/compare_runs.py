"""Seed-paired treatment/control comparison from logged SS histories; read-only.

Never trains or re-evaluates. Per-epoch SS rows come from epoch_metrics.jsonl
plus evaluation_last.json (fixed-final), filtered to the run's SS protocol and
to purpose monitor_only/final, so best-Old-test checkpoint reports never
overwrite an epoch. Epochs are epoch_number (1-based); policy logs
(crish_policy_epochs.jsonl, crish_p.jsonl) use 0-based epochs and are shifted.
Values are percentages. Std is the sample std (n-1); n<2 gives null.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.metrics import read_rows
from gcd_runtime.ss_best import ss_protocol

METRICS = (('All', 'all_acc'), ('Old', 'old_acc'), ('New', 'new_acc'))
READINGS = ('best', 'last', 'window_mean')
PURPOSES = ('monitor_only', 'final')


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(mean=None, std=None, n=0)
    mean = sum(values)/len(values)
    std = math.sqrt(sum((v-mean)**2 for v in values)/(len(values)-1)) if len(values) > 1 else None
    return dict(mean=mean, std=std, n=len(values))


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def history_stats(history, epochs, window):
    """history: {epoch_number: {metric: percent}}; returns per-metric readings and coverage."""
    first, last = window
    expected = set(range(1, epochs+1))
    in_window = [e for e in range(first, last+1) if e in history]
    out = dict(evaluated_epochs=len(history), missing_epochs=sorted(expected-set(history)),
               window=[first, last], window_epochs=len(in_window),
               window_missing=sorted(set(range(first, last+1))-set(history)), metrics={})
    for name, _ in METRICS:
        values = {e: row[name] for e, row in history.items()}
        best_epoch = min(values, key=lambda e: (-values[e], e)) if values else None
        window_values = mean_std([values[e] for e in in_window])
        out['metrics'][name] = dict(
            best=None if best_epoch is None else values[best_epoch], best_epoch=best_epoch,
            last=values.get(epochs), last_epoch=epochs if epochs in values else None,
            window_mean=window_values['mean'], window_std=window_values['std'])
    return out


def load_run(directory, window, *, require_observed=True):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    metadata = json.loads((directory/'metadata.json').read_text()) if (directory/'metadata.json').is_file() else {}
    protocol = ss_protocol(config)
    rows = list(read_rows(directory/'epoch_metrics.jsonl'))
    if (directory/'evaluation_last.json').is_file():
        rows.append(json.loads((directory/'evaluation_last.json').read_text()))
    history, n_unlabeled = {}, None
    for row in rows:
        if row.get('protocol') != protocol or row.get('purpose') not in PURPOSES:
            continue
        if require_observed and row.get('status') != 'observed':
            raise ValueError(f'{directory}: non-observed SS row at epoch {row.get("epoch")}')
        if row.get('unit') != 'fraction':
            raise ValueError(f'{directory}: SS rows must store fractions')
        number = int(row['epoch'])+1
        values = {name: 100*float(row[key]) for name, key in METRICS}
        if number in history and history[number] != values:
            raise ValueError(f'{directory}: conflicting SS rows for epoch {number}')
        history[number] = values
        n_unlabeled = row.get('n_unlabeled', n_unlabeled)
    run = dict(run_id=metadata.get('run_id', directory.name), directory=str(directory), seed=int(config['seed']),
               method=config.get('method'), dataset=config.get('dataset_name'), git_sha=metadata.get('git_sha'),
               config_sha256=config_hash(config), protocol=protocol, epochs=int(config['epochs']),
               n_unlabeled=n_unlabeled, ss=history_stats(history, int(config['epochs']), window))
    policy = policy_summary(directory, window, n_unlabeled)
    if policy is not None:
        run['policy'] = policy
    return run


def load_reference(path, window):
    data = json.loads(Path(path).read_text())
    meta, rows = data.get('run', {}), data['history']
    history = {}
    for row in rows:
        if row.get('purpose') not in PURPOSES or row.get('unit') != 'fraction':
            continue
        history[int(row['epoch_number'])] = {name: 100*float(row[key]) for name, key in METRICS}
    seeds = {int(row['seed']) for row in rows if 'seed' in row} | ({int(meta['seed'])} if 'seed' in meta else set())
    if len(seeds) != 1:
        raise ValueError('Reference history must contain exactly one seed')
    epochs = max(history)
    return dict(run_id=meta.get('run_id', Path(path).name), directory=str(path), seed=seeds.pop(),
                method=meta.get('method'), dataset=meta.get('dataset'), git_sha=None,
                config_sha256=None, reference_history_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                epochs=epochs, ss=history_stats(history, epochs, window))


def policy_summary(directory, window, n_unlabeled):
    epochs_path, due_path = directory/'crish_policy_epochs.jsonl', directory/'crish_p.jsonl'
    if not epochs_path.is_file() and not due_path.is_file():
        return None
    first, last = window
    records = {int(r['epoch'])+1: r for r in read_rows(epochs_path)}
    inside = {e: r for e, r in records.items() if first <= e <= last}
    levels = {}
    for record in inside.values():
        for level in record.get('levels', []):
            levels.setdefault(int(level['level']), []).append(float(level.get('fraction', 0.)))
    dose = {str(level): mean_std(values) for level, values in sorted(levels.items())}
    moved = [r.get('moved', 0)/n_unlabeled for r in inside.values()] if n_unlabeled else []
    combined = ([(r['levels'][0].get('abstained_unlabeled', 0)+r.get('moved', 0))/n_unlabeled
                 for r in inside.values() if r.get('levels')] if n_unlabeled else [])
    refreshes, reasons = [], Counter()
    for record in read_rows(due_path):
        refreshes.append(dict(epoch_number=int(record['epoch'])+1, status=record.get('status'),
                              tau=record.get('chosen_threshold', record.get('chosen_fraction')),
                              reason=record.get('reason')))
        if record.get('reason'):
            reasons['refresh: '+record['reason']] += 1
        for candidate in record.get('candidates', []):
            if candidate.get('status') == 'rejected' and candidate.get('reason'):
                reasons['candidate: '+candidate['reason']] += 1
    positive = sum(1 for r in refreshes if (r['tau'] or 0) > 0)
    return dict(window_epochs_logged=len(inside),
                window_epochs_missing=sorted(set(range(first, last+1))-set(inside)),
                abstained_fraction_by_level=dose, moved_fraction=mean_std(moved),
                level0_abstain_or_move_fraction=mean_std(combined),
                refreshes=refreshes,
                tau_positive=dict(numerator=positive, denominator=len(refreshes),
                                  fraction=positive/len(refreshes) if refreshes else None),
                rejection_reasons=dict(reasons.most_common()))


def pair(treatment, control):
    by_seed = {}
    for side, runs in (('treatment', treatment), ('control', control)):
        for run in runs:
            slot = by_seed.setdefault(run['seed'], {})
            if side in slot:
                raise ValueError(f'Duplicate {side} run for seed {run["seed"]}')
            slot[side] = run
    unpaired = {seed: sorted(slot) for seed, slot in by_seed.items() if len(slot) != 2}
    if unpaired:
        raise ValueError(f'Seeds without a treatment/control pair: {unpaired}')
    return dict(sorted(by_seed.items()))


def compare(treatment_dirs, control_dirs=(), window=(61, 100), reference=None, require_observed=True):
    if (reference is None) == (not control_dirs):
        raise ValueError('Give either --control runs or --reference-history, not both or neither')
    first, last = window
    if not 1 <= first <= last:
        raise ValueError('Window must satisfy 1 <= A <= B (epoch_number)')
    treatment = [load_run(d, window, require_observed=require_observed) for d in treatment_dirs]
    if reference is not None:
        ref = load_reference(reference, window)
        control = [ref]
        if {r['seed'] for r in treatment} != {ref['seed']} or len(treatment) != 1:
            raise ValueError('A reference history pairs with exactly one treatment run of the same seed')
    else:
        control = [load_run(d, window, require_observed=require_observed) for d in control_dirs]
    pairs = pair(treatment, control)
    per_seed, deltas = [], {(m, r): [] for m, _ in METRICS for r in READINGS}
    for seed, slot in pairs.items():
        row = dict(seed=seed, treatment=slot['treatment']['run_id'], control=slot['control']['run_id'], delta={})
        for name, _ in METRICS:
            a, b = slot['treatment']['ss']['metrics'][name], slot['control']['ss']['metrics'][name]
            row['delta'][name] = {}
            for reading in READINGS:
                value = None if a[reading] is None or b[reading] is None else a[reading]-b[reading]
                row['delta'][name][reading] = value
                deltas[(name, reading)].append(value)
        per_seed.append(row)
    aggregate = {}
    for (name, reading), values in deltas.items():
        present = [v for v in values if v is not None]
        stats = mean_std(present)
        aggregate.setdefault(name, {})[reading] = dict(stats, min=min(present) if present else None,
            positive=sum(v > 0 for v in present), seeds=len(values), missing=len(values)-len(present))
    return dict(kind='seed_paired_log_comparison_not_reevaluated', unit='percent', window=list(window),
                readings=dict(best='per-metric best over logged epochs', last='final epoch',
                              window_mean=f'mean SS over epoch_number {first}-{last}'),
                treatment=treatment, control=control, per_seed=per_seed, aggregate=aggregate)


def fmt(value, digits=2):
    return 'NA' if value is None else f'{value:.{digits}f}'


def markdown(result):
    lines = [f'Window: epoch_number {result["window"][0]}-{result["window"][1]} (percent)', '',
             '| Seed | Metric | Reading | Treatment | Control | Delta |', '|---|---|---|---|---|---|']
    runs = {('t', r['seed']): r for r in result['treatment']} | {('c', r['seed']): r for r in result['control']}
    for row in result['per_seed']:
        for name, _ in METRICS:
            t, c = runs[('t', row['seed'])]['ss']['metrics'][name], runs[('c', row['seed'])]['ss']['metrics'][name]
            for reading in READINGS:
                suffix = lambda x: (f' (ep{x["best_epoch"]})' if reading == 'best' and x['best_epoch'] else
                                    f' +/- {fmt(x["window_std"])}' if reading == 'window_mean' and x['window_std'] is not None else '')
                lines.append(f'| {row["seed"]} | {name} | {reading} | {fmt(t[reading])}{suffix(t)} | '
                             f'{fmt(c[reading])}{suffix(c)} | {fmt(row["delta"][name][reading])} |')
    lines += ['', '| Metric | Reading | mean Delta | std | min | seeds Delta>0 |', '|---|---|---|---|---|---|']
    for name, readings in result['aggregate'].items():
        for reading, s in readings.items():
            lines.append(f'| {name} | {reading} | {fmt(s["mean"])} | {fmt(s["std"])} | {fmt(s["min"])} | '
                         f'{s["positive"]}/{s["seeds"]} |')
    for run in result['treatment']:
        policy = run.get('policy')
        if policy:
            dose = policy['level0_abstain_or_move_fraction']
            lines += ['', f'Policy {run["run_id"]} (seed {run["seed"]}): level-0 abstain+move fraction '
                      f'{fmt(None if dose["mean"] is None else 100*dose["mean"])}% over {dose["n"]} epochs; '
                      f'tau*>0 in {policy["tau_positive"]["numerator"]}/{policy["tau_positive"]["denominator"]} refreshes']
            for level, s in policy['abstained_fraction_by_level'].items():
                lines.append(f'- level {level}: abstained {fmt(None if s["mean"] is None else 100*s["mean"])}% (n={s["n"]})')
            for reason, count in policy['rejection_reasons'].items():
                lines.append(f'- {count}x {reason}')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--treatment', type=Path, nargs='+', required=True)
    parser.add_argument('--control', type=Path, nargs='*', default=[])
    parser.add_argument('--reference-history', type=Path,
                        help='curves-*/data.json history used as the control (G1 reference)')
    parser.add_argument('--window', type=int, nargs=2, metavar=('A', 'B'), default=[61, 100])
    parser.add_argument('--output', type=Path, help='JSON output (must not exist)')
    args = parser.parse_args()
    result = compare(args.treatment, args.control, tuple(args.window), args.reference_history)
    if args.output:
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(markdown(result))
