"""Offline protocol checks; does not import training code or expose labels to it."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.datasets import DATASETS, dataset_spec
SPLIT_SHA = '28d471f0fb9e2c773e1255b8b447e542974f6617952203056a0635c638513d39'


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Split must contain only primitive data')


def check_split(path, dataset_name='scars'):
    spec = dataset_spec(dataset_name)
    if dataset_name == 'cifar100':
        # Upstream get_class_splits defines ranges; no SSB pickle exists here.
        protocol = dict(dataset_name='cifar100', known_classes=list(range(80)),
                        novel_classes=list(range(80, 100)), prop_train_labels=0.5)
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        return dict(sha256=digest, kind='upstream_fixed_ranges_not_split_file',
                    known_count=80, novel_count=20, protocol=protocol)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec.split_sha256:
        raise ValueError('Official pinned split checksum mismatch')
    with path.open('rb') as handle:
        split = DataOnlyUnpickler(handle).load()
    known = split['known_classes']
    groups = split['unknown_classes']
    novel = groups['Hard'] + groups['Medium'] + groups['Easy']
    if len(known) != len(set(known)) or len(novel) != len(set(novel)):
        raise ValueError('Duplicate classes')
    if set(known) & set(novel) or set(known + novel) != set(range(spec.num_classes)):
        raise ValueError('Invalid class partition')
    return {'sha256': digest, 'known_count': len(known), 'novel_count': len(novel),
            'novel_groups': {k: len(v) for k, v in groups.items()}}


def check_membership(rows, known_classes, root, num_classes=196):
    """Audit-only manifest: id, path, partition, label (zero based).

    Labels are used for integrity checks only. Never load this manifest for training.
    Accepts the train pool only; labeled/unlabeled must form its disjoint partition.
    """
    if not rows:
        raise ValueError('Empty train manifest')
    ids, paths = set(), set()
    counts = {'labeled': 0, 'unlabeled': 0}
    for row in rows:
        partition = row['partition']
        if partition not in counts:
            raise ValueError('Invalid partition')
        if row['id'] in ids:
            raise ValueError('Duplicate sample ID across train pool')
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Missing image or path outside dataset root')
        if path in paths:
            raise ValueError('Duplicate image path')
        label = row['label']
        if type(label) is not int or not 0 <= label < num_classes:
            raise ValueError('Invalid zero-based dataset label')
        if partition == 'labeled' and label not in known_classes:
            raise ValueError('Novel label in supervised partition')
        ids.add(row['id'])
        paths.add(path)
        counts[partition] += 1
    if not all(counts.values()):
        raise ValueError('Both train partitions must be nonempty')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--dataset', default='scars', choices=tuple(DATASETS))
    args = parser.parse_args()
    filename = dataset_spec(args.dataset).split_file
    path = ROOT / 'upstream/HypCD/data/ssb_splits' / filename if filename else None
    report = {'dataset_name': args.dataset, 'split': check_split(path, args.dataset), 'image_integrity': 'not_checked_no_manifest'}
    if args.manifest:
        if args.dataset == 'cifar100':
            parser.error('CIFAR-100 array membership is audited by remote preflight, not a file-image manifest')
        if not args.data_root:
            parser.error('--data-root is required with --manifest')
        with path.open('rb') as handle:
            known = DataOnlyUnpickler(handle).load()['known_classes']
        report['image_integrity'] = check_membership(
            json.loads(args.manifest.read_text()), set(known), args.data_root,
            num_classes=dataset_spec(args.dataset).num_classes)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
