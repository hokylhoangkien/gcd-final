"""Offline-only export of train-pool ground truth as sample_id,true_label.

Evaluation labels: never import this from a training, dose-selection or
checkpoint-selection path. It instantiates the pinned SelEx dataset directly
(not sanitize_training) on CPU, on a host that has the data. IDs are the same
uq_idxs that SupervisionState.ids carries; labels use the same native offset
and target_transform as the legal labeled targets, so anchors agree.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def label_rows(merged, name):
    """(sample_id, true_label) for labeled then unlabeled rows; offline audit only."""
    from gcd_runtime.datasets import dataset_spec, raw_targets
    offset = dataset_spec(name).target_offset
    rows, seen = [], set()
    for part in (merged.labelled_dataset, merged.unlabelled_dataset):
        targets = raw_targets(part, name)
        if len(targets) != len(part.uq_idxs):
            raise ValueError('Target/ID lengths differ')
        transform = getattr(part, 'target_transform', None)
        for sample_id, target in zip(part.uq_idxs, targets):
            label = int(target)-offset
            if transform is not None:
                label = int(transform(label))
            sample_id = int(sample_id)
            if sample_id in seen or label < 0:
                raise ValueError('Duplicate sample ID or invalid label')
            seen.add(sample_id)
            rows.append((sample_id, label))
    return rows


def export(dataset, output, config=None):
    import argparse as _argparse
    from gcd_runtime.config import load_config
    from selex_runtime.reference import configure_environment
    cfg = load_config(config or ROOT/f'configs/selex_{dataset}_crish_200.json')
    if cfg['dataset_name'] != dataset:
        raise ValueError('Config dataset differs from --dataset')
    configure_environment(ROOT/'runs/offline-labels-placeholder', cfg)
    from data.get_datasets import get_class_splits, get_datasets
    args = get_class_splits(_argparse.Namespace(dataset_name=dataset, use_ssb_splits=True, prop_train_labels=.5))
    merged, _, _, _ = get_datasets(dataset, None, None, args)
    rows = label_rows(merged, dataset)
    output = Path(output)
    with output.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'true_label'])
        writer.writerows(rows)
    return dict(status='offline_labels_exported', uses_evaluation_labels=True, dataset=dataset,
                rows=len(rows), output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['scars', 'cub', 'aircraft'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Default: configs/selex_{dataset}_crish_200.json')
    arguments = parser.parse_args()
    import json
    print(json.dumps(export(arguments.dataset, arguments.output, arguments.config), indent=2))
