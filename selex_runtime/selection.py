"""Secondary SelEx best-Old-v1 report; fixed-final SS remains primary.

Scores are read only by evaluation workers and resume integrity checks. They
never change SGD, repairs, stopping criteria or the predeclared primary result.
"""
import json
import math
import os
from pathlib import Path
import shutil

from gcd_runtime.config import code_fingerprint
from gcd_runtime.metrics import atomic_json, read_rows, write_epoch
from gcd_runtime.readiness import sha256

BEST_NAME = 'backbone_best_old_test_v1.pt'
BEST_FORMAT = 'selex_best_old_plain_test_v1_v1'
RULE = 'max_old_v1_disjoint_plain_test_strict_improvement'


def enabled(config):
    return config.get('baseline_family') == 'selex' and config.get('report_best_old_test', False)


def read_best(path, config):
    import torch
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if (bundle.get('format') != BEST_FORMAT or bundle.get('config') != config
            or bundle.get('code_fingerprint') != code_fingerprint()):
        raise ValueError('SelEx best checkpoint requires the exact source/config')
    return bundle


def old_score(record):
    if (record.get('protocol') != 'plain_kmeans_disjoint_test'
            or record.get('selection_matching') != 'hungarian_v1_separate'):
        raise ValueError('SelEx best selection requires plain disjoint-test Old-v1')
    for score in record['metrics_v1'].values():
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid SelEx v1 selection metric')
    score = record['metrics_v1']['old_acc']
    return score


def write_selection(run_dir, bundle):
    record = bundle['selection']
    atomic_json(Path(run_dir)/'selection.json', dict(rule=RULE,
        uses_evaluation_labels=True, matching='hungarian_v1_separate',
        primary='fixed_final_ss_v2', primary_selected_using_evaluation_labels=False,
        best_epoch=record['epoch'], best_epoch_number=record['epoch']+1,
        best_old_acc=old_score(record), best_test_metrics=record,
        checkpoint=str(Path(run_dir)/BEST_NAME), checkpoint_sha256=sha256(Path(run_dir)/BEST_NAME)))


def select_checkpoint(run_dir, config, checkpoint, record):
    """Earliest epoch wins ties; first finite epoch is an explicit zero fallback."""
    import torch
    if not enabled(config):
        raise ValueError('SelEx secondary best reporting is disabled')
    score = old_score(record)
    run_dir, checkpoint = Path(run_dir), Path(checkpoint)
    if sha256(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Selection report does not match the evaluated snapshot')
    path = run_dir/BEST_NAME
    previous = read_best(path, config) if path.is_file() else None
    if previous and previous['selection']['epoch'] > record['epoch']:
        raise ValueError('SelEx best checkpoint is ahead of this resume epoch')
    if previous and previous['selection']['epoch'] == record['epoch']:
        # Idempotent replay after a failed RNG handoff. Serialization can differ
        # across processes, so compare actual tensors as well as the score.
        model = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (score != old_score(previous['selection']) or model.keys() != previous['model'].keys()
                or any(not torch.equal(model[k], previous['model'][k]) for k in model)):
            raise ValueError('Different weights/Old-v1 score at an already selected epoch')
    improved = previous is None or score > old_score(previous['selection'])
    bundle = previous
    if improved:
        bundle = dict(format=BEST_FORMAT, config=config, code_fingerprint=code_fingerprint(),
            model=torch.load(checkpoint, map_location='cpu', weights_only=True), selection=record)
        temporary = path.with_suffix('.tmp')
        torch.save(bundle, temporary)
        os.replace(temporary, path)
    write_selection(run_dir, bundle)
    print(f'SELEX_SELECTION epoch={record["epoch"]+1} best_epoch={bundle["selection"]["epoch"]+1} '
          f'Old_v1={100*old_score(bundle["selection"]):.2f}% updated={improved} primary=last', flush=True)


def load_checkpoint(path, config, runtime):
    bundle = read_best(path, config)
    runtime.evaluation_epoch = bundle['selection']['epoch']
    runtime.evaluation_role = 'best_old_test'
    runtime.selection_rule = RULE
    runtime.selection_matching = 'hungarian_v1_separate'
    return bundle['model']


def restore_artifacts(source, destination, config, epoch, *, pending):
    """Preserve best-so-far; require complete original reports before next SGD."""
    source, destination = Path(source), Path(destination)
    records = {}
    for path in sorted((source/'test_evaluations').glob('epoch_*.json')):
        row = json.loads(path.read_text())
        if row['epoch'] <= epoch:
            old_score(row)
            records[row['epoch']] = row
    required = range(epoch if pending else epoch+1)
    if any(e not in records for e in required):
        raise ValueError('SelEx resume needs the original test_evaluations and best checkpoint')
    path = source/BEST_NAME
    if records and not path.is_file():
        raise ValueError('Missing SelEx best-so-far checkpoint; refusing to reset selection')
    if path.is_file():
        bundle = read_best(path, config)
        best = bundle['selection']
        if best['epoch'] > epoch:
            raise ValueError('SelEx best checkpoint is ahead of resume.pt')
        candidates = list(records.values())+[best]
        expected = max(sorted(candidates, key=lambda row: row['epoch']), key=old_score)
        if expected['epoch'] != best['epoch'] or old_score(expected) != old_score(best):
            raise ValueError('SelEx best checkpoint disagrees with the restored Old-v1 history')
        if source.resolve() != destination.resolve():
            shutil.copy2(path, destination/BEST_NAME)
        write_selection(destination, bundle)
    for e, row in sorted(records.items()):
        atomic_json(destination/'test_evaluations'/f'epoch_{e+1:04d}.json', row)
        write_epoch(destination/'test_metrics.jsonl', row)
    for name in ('metrics.jsonl', 'epoch_metrics.jsonl', 'train_plain_metrics.jsonl', 'best_old_ss_metrics.jsonl'):
        for row in read_rows(source/name):
            if row['epoch'] <= epoch:
                write_epoch(destination/name, row)
