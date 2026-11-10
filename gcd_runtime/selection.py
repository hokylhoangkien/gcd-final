"""Evaluation-only Old-test-v2 selection; no scores control SGD or repairs."""
import json
import os
from pathlib import Path
import shutil

from .config import code_fingerprint
from .metrics import atomic_json, read_rows, write_epoch
from .readiness import sha256

BEST_NAME = 'backbone_best_old_test.pt'
BEST_FORMAT = 'hypcd_best_old_test_v2_v1'
TEST_PROTOCOL = 'plain_kmeans_disjoint_test'


def enabled(config):
    return config.get('checkpoint_selection') == 'best_old_test_v2'


def read_best(path, config):
    import torch
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    validate_best(bundle, config)
    return bundle


def validate_best(bundle, config):
    if (bundle.get('format') != BEST_FORMAT or bundle.get('config') != config
            or bundle.get('code_fingerprint') != code_fingerprint()):
        raise ValueError('Best checkpoint requires the exact source and config')


def load_evaluation_checkpoint(path, config, runtime):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if config.get('baseline_family') == 'selex':
        from selex_runtime.selection import BEST_FORMAT as SELEX_BEST_FORMAT, load_checkpoint
        if checkpoint.get('format') == SELEX_BEST_FORMAT:
            return load_checkpoint(path, config, runtime)
    if checkpoint.get('format') == BEST_FORMAT:
        validate_best(checkpoint, config)
        runtime.evaluation_epoch = checkpoint['selection']['epoch']
        runtime.evaluation_role = 'best_old_test'
        return checkpoint['model']
    return checkpoint


def select_checkpoint(run_dir, config, checkpoint, record):
    """Strict improvement; earliest epoch wins ties. Bundle replacement is atomic.

    The bundle contains weights and their selection record together, so a crash
    cannot pair new weights with an old score. First finite epoch is a fallback
    even at Old=0, where the released code would otherwise produce no best file.
    """
    import torch
    if not enabled(config) or record['protocol'] != TEST_PROTOCOL:
        raise ValueError('Selection requires explicit best_old_test_v2 and disjoint-test metrics')
    run_dir, checkpoint = Path(run_dir), Path(checkpoint)
    best_path = run_dir/BEST_NAME
    previous = read_best(best_path, config) if best_path.is_file() else None
    if previous and previous['selection']['epoch'] > record['epoch']:
        raise ValueError('Best checkpoint is ahead of this epoch; use a matching resume bundle')
    improved = previous is None or record['old_acc'] > previous['selection']['old_acc']
    if improved:
        bundle = dict(format=BEST_FORMAT, model=torch.load(checkpoint, map_location='cpu', weights_only=True),
                      selection=record, config=config, code_fingerprint=code_fingerprint())
        temporary = best_path.with_suffix('.tmp')
        torch.save(bundle, temporary)
        os.replace(temporary, best_path)
    else:
        bundle = previous
    best = bundle['selection']
    if best['epoch'] == record['epoch'] and best['checkpoint_sha256'] != record['checkpoint_sha256']:
        raise ValueError('A different snapshot is already selected at this epoch')
    selection = dict(rule='max_old_v2_disjoint_test_strict_improvement',
                     uses_evaluation_labels=True, matching='hungarian_v2_global',
                     best_epoch=best['epoch'], best_epoch_number=best['epoch']+1,
                     best_old_acc=best['old_acc'], best_test_metrics=best,
                     checkpoint=str(best_path), checkpoint_sha256=sha256(best_path))
    atomic_json(run_dir/'selection.json', selection)
    print(f'SELECTION epoch={record["epoch"]+1} best_epoch={best["epoch"]+1} '
          f'best_Old={100*best["old_acc"]:.2f}% updated={improved}', flush=True)


def restore_artifacts(source, destination, config, epoch):
    """Restore a matching run bundle; never silently reset the best-so-far score."""
    source, destination = Path(source), Path(destination)
    records = {}
    for path in sorted((source/'test_evaluations').glob('epoch_*.json')):
        row = json.loads(path.read_text())
        if row['epoch'] <= epoch:
            records[row['epoch']] = row
    # The just-saved epoch may still be waiting for its selector. Every earlier
    # epoch must have completed evaluation before the next SGD epoch can start.
    if not all(e in records for e in range(epoch)):
        raise ValueError('Resume selection needs the original run directory, including test_evaluations and best checkpoint')
    path = source/BEST_NAME
    if not path.is_file() and records:
        raise ValueError('Missing best checkpoint for the saved evaluation history')
    if path.is_file():
        bundle = read_best(path, config)
        best = bundle['selection']
        if best['epoch'] > epoch:
            raise ValueError('Best checkpoint is ahead of resume.pt; use matching artifacts')
        candidates = list(records.values())+[best]
        expected = max(sorted(candidates, key=lambda r: r['epoch']), key=lambda r: r['old_acc'])
        if expected['epoch'] != best['epoch']:
            raise ValueError('Best checkpoint does not match the restored selection history')
        if source.resolve() != destination.resolve():
            shutil.copy2(path, destination/BEST_NAME)
        selection = dict(rule='max_old_v2_disjoint_test_strict_improvement', uses_evaluation_labels=True,
                         best_epoch=best['epoch'], best_epoch_number=best['epoch']+1,
                         best_old_acc=best['old_acc'], best_test_metrics=best,
                         checkpoint=str(destination/BEST_NAME), checkpoint_sha256=sha256(destination/BEST_NAME))
        atomic_json(destination/'selection.json', selection)
    for e, row in sorted(records.items()):
        atomic_json(destination/'test_evaluations'/f'epoch_{e+1:04d}.json', row)
        write_epoch(destination/'test_metrics.jsonl', row)
    # Keep optional SS history for later peak inspection as well.
    for name in ('metrics.jsonl', 'epoch_metrics.jsonl'):
        for row in read_rows(source/name):
            if row['epoch'] <= epoch:
                write_epoch(destination/name, row)
