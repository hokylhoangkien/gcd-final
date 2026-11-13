"""Evaluation-label-selected SS maxima; never feed scores back into SGD."""
import json
import math
import os
from pathlib import Path
import shutil

from .metrics import atomic_json, read_rows, write_epoch
from .readiness import sha256

PROTOCOL = 'balanced_ss_kmeans_train_unlabeled'
UNBALANCED_PROTOCOL = 'unbalanced_ss_kmeans_train_unlabeled'
METRICS = dict(all='all_acc', old='old_acc', new='new_acc')


def ss_protocol(config):
    return UNBALANCED_PROTOCOL if config.get('unbalanced', False) else PROTOCOL


def enabled(config):
    return config.get('ss_eval_every_epochs', 0) > 0


def copy_verified(source, destination, digest):
    source, destination = Path(source), Path(destination)
    temporary = destination.with_suffix('.tmp')
    shutil.copyfile(source, temporary)
    if sha256(temporary) != digest:
        raise ValueError('SS best checkpoint hash mismatch; existing best preserved')
    os.replace(temporary, destination)


def update_best(run_dir, config, record):
    """Rank scheduled SS + fixed last only; earliest epoch wins exact ties."""
    protocol = ss_protocol(config)
    if (not enabled(config) or record['protocol'] != protocol
            or record['purpose'] not in ('monitor_only', 'final')):
        return
    for metric in METRICS.values():
        value = record[metric]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('SS best requires finite fraction metrics')
    if record.get('unit') != 'fraction':
        raise ValueError('SS best requires fraction metrics')
    checkpoint = Path(record['checkpoint'])
    digest = record['checkpoint_sha256']
    if sha256(checkpoint) != digest:
        raise ValueError('SS report does not match its evaluated checkpoint')
    run_dir = Path(run_dir)
    index = run_dir/'ss_best.json'
    summary = json.loads(index.read_text()) if index.is_file() else dict(
        protocol=protocol, rule='max_each_ss_metric_earliest_epoch_on_ties',
        uses_evaluation_labels=True, candidates='scheduled_ss_and_fixed_last', best={})
    if summary['protocol'] != protocol:
        raise ValueError('Cannot mix balanced and unbalanced SS best histories')
    for name, metric in METRICS.items():
        old = summary['best'].get(name)
        improved = (old is None or record[metric] > old[metric]
                    or (record[metric] == old[metric] and record['epoch'] < old['epoch']))
        if improved:
            target = run_dir/f'backbone_best_ss_{name}.pt'
            copy_verified(checkpoint, target, digest)
            best = dict(record, checkpoint=str(target), source_checkpoint=str(checkpoint),
                        purpose=f'best_ss_{name}', selected_metric=metric,
                        selected_using_evaluation_labels=True,
                        checkpoint_selection_uses_evaluation_labels=True,
                        used_for_checkpoint_selection=True)
            atomic_json(run_dir/f'evaluation_best_ss_{name}.json', best)
            summary['best'][name] = best
            atomic_json(index, summary)
        best = summary['best'][name]
        if sha256(Path(best['checkpoint'])) != best['checkpoint_sha256']:
            raise ValueError('Stored SS best checkpoint missing or changed')
        print(f'SS_BEST metric={name} epoch={best["epoch"]+1} '
              f'All={100*best["all_acc"]:.2f}% Old={100*best["old_acc"]:.2f}% '
              f'New={100*best["new_acc"]:.2f}% updated={improved} '
              f'checkpoint={best["checkpoint"]}', flush=True)


def restore_best(source, destination, config, epoch):
    """Carry best weights and SS history into a resumed run without resetting maxima."""
    if not enabled(config):
        return
    source, destination = Path(source), Path(destination)
    index = source/'ss_best.json'
    rows = [row for row in read_rows(source/'epoch_metrics.jsonl')
            if row['protocol'] == ss_protocol(config) and row['purpose'] in ('monitor_only', 'final')]
    if not index.is_file():
        if rows:
            raise ValueError('Missing SS best artifacts for existing SS history')
        return
    summary = json.loads(index.read_text())
    if summary['protocol'] != ss_protocol(config):
        raise ValueError('Cannot restore a different SS clustering protocol')
    if set(summary['best']) != set(METRICS):
        raise ValueError('Incomplete SS best index; restore original run artifacts')
    for name, best in summary['best'].items():
        if best['epoch'] > epoch:
            raise ValueError('SS best is ahead of resume checkpoint')
        checkpoint = source/f'backbone_best_ss_{name}.pt'
        if sha256(checkpoint) != best['checkpoint_sha256']:
            raise ValueError('SS best weights do not match saved report')
        target = destination/checkpoint.name
        if source.resolve() != destination.resolve():
            copy_verified(checkpoint, target, best['checkpoint_sha256'])
        best['checkpoint'] = str(target)
        atomic_json(destination/f'evaluation_best_ss_{name}.json', best)
    atomic_json(destination/'ss_best.json', summary)
    for row in rows:
        if row['epoch'] <= epoch:
            write_epoch(destination/'epoch_metrics.jsonl', row)
            if row['purpose'] == 'monitor_only':
                atomic_json(destination/'evaluations'/f'epoch_{row["epoch"]+1:04d}.json', row)
