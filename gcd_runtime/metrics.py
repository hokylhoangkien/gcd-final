"""Protocol-separated histories and paired All/Old/New peak summaries."""
import json
import math
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2)+'\n')
    os.replace(temporary, path)


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: incomplete/invalid JSON; retry after the write completes') from exc
    return rows


def write_epoch(path, record):
    """Idempotent, atomic history update; one completed report per epoch."""
    path = Path(path)
    rows = {row['epoch']: row for row in read_rows(path)}
    rows[record['epoch']] = record
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(rows[e], allow_nan=False)+'\n' for e in sorted(rows)))
    os.replace(temporary, path)


def peaks(rows):
    rows = sorted(rows, key=lambda r: r['epoch'])
    for row in rows:
        if row.get('unit') != 'fraction':
            raise ValueError('Peak summaries require fraction metrics')
        for metric in ('all_acc', 'old_acc', 'new_acc'):
            if not math.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError(f'Invalid metric for summary: {metric}')
    return {metric: max(rows, key=lambda r: r[metric]) if rows else None
            for metric in ('all_acc', 'old_acc', 'new_acc')}


def collect(run_dir):
    run_dir = Path(run_dir)
    histories = {}
    for name in ('test_metrics.jsonl', 'epoch_metrics.jsonl', 'train_plain_metrics.jsonl', 'best_old_ss_metrics.jsonl'):
        for row in read_rows(run_dir/name):
            histories.setdefault(row['protocol'], {})[row['epoch']] = row
    finals = {}
    for name in ('evaluation_last.json', 'evaluation_best_old_test.json', 'evaluation.json'):
        path = run_dir/name
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        if 'epoch' not in row and (run_dir/'config.json').is_file():
            row['epoch'] = json.loads((run_dir/'config.json').read_text())['epochs']-1
            row['epoch_number'] = row['epoch']+1
        finals[name] = row
        if 'epoch' in row:
            histories.setdefault(row['protocol'], {})[row['epoch']] = row
    return histories, finals


def summarize_run(run_dir, *, write=False):
    run_dir = Path(run_dir)
    histories, finals = collect(run_dir)
    result = dict(kind='posthoc_protocol_separated_peaks_not_checkpoint_selection',
                  histories={protocol: dict(evaluated_epochs=len(rows), peaks=peaks(list(rows.values())))
                             for protocol, rows in histories.items()}, final_reports=finals)
    if (run_dir/'selection.json').is_file():
        result['checkpoint_selection'] = json.loads((run_dir/'selection.json').read_text())
    if write:
        atomic_json(run_dir/'results_summary.json', result)
    return result


def print_peaks(summary):
    for protocol, data in summary['histories'].items():
        print(f'PEAKS protocol={protocol} evaluated_epochs={data["evaluated_epochs"]}', flush=True)
        for key, row in data['peaks'].items():
            if row is not None:
                print(f'Max {key}: epoch={row["epoch"]+1} '
                      f'All={100*row["all_acc"]:.2f}% Old={100*row["old_acc"]:.2f}% '
                      f'New={100*row["new_acc"]:.2f}% status={row["status"]}', flush=True)
