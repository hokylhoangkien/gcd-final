"""Fail-fast checks for remote baseline inputs; no training."""
import hashlib
import subprocess
from .config import ROOT, UPSTREAM_SHA, paths
from .datasets import dataset_spec, image_paths, raw_targets, whole_training_dataset
from scripts.check_integrity import check_split, check_membership


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_upstream():
    """Verify either a Git submodule or the hash manifest in a portable release."""
    import json
    upstream = ROOT / 'upstream/HypCD'
    if (upstream / '.git').exists():
        sha = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
        # Enumerate files (not collapsed directories) and use NUL separators so
        # spaces/quoting cannot hide an unexpected source modification.
        dirty = subprocess.check_output(['git', '-C', str(upstream), 'status',
            '--porcelain', '-z', '--untracked-files=all'], text=True, timeout=10)
        unexpected = []
        from pathlib import PurePosixPath
        for entry in dirty.split('\0'):
            if not entry:
                continue
            relative = PurePosixPath(entry[3:])
            generated = (entry.startswith('?? ') and '__pycache__' in relative.parts
                         and relative.suffix in ('.pyc', '.pyo')
                         and not (upstream/relative).is_symlink())
            if not generated:
                unexpected.append(entry)
        if sha != UPSTREAM_SHA or unexpected:
            raise RuntimeError('Upstream must match the pinned, clean commit; '
                               f'HEAD={sha}; unexpected changes={unexpected!r}')
        return {'sha': sha, 'kind': 'git_submodule'}
    manifest = ROOT / 'upstream/HypCD.SNAPSHOT.json'
    if not manifest.is_file():
        raise RuntimeError('Missing upstream Git checkout or verified release snapshot')
    snapshot = json.loads(manifest.read_text())
    if snapshot['commit'] != UPSTREAM_SHA:
        raise RuntimeError('Release upstream commit differs from pin')
    for relative, digest in snapshot['files'].items():
        path = (upstream / relative).resolve()
        if not path.is_relative_to(upstream.resolve()) or not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f'Upstream snapshot mismatch: {relative}')
    return {'sha': snapshot['commit'], 'kind': 'verified_release_snapshot'}


def inspect_inputs(cfg=None):
    upstream = ROOT / 'upstream/HypCD'
    is_selex = (cfg or {}).get('baseline_family') == 'selex'
    if is_selex:
        from selex_runtime.reference import verify_reference, SOURCE_ROOT
        provenance, upstream = verify_reference(), SOURCE_ROOT
    else:
        provenance = verify_upstream()
    sha = provenance['sha']
    name = (cfg or {}).get('dataset_name', 'scars')
    spec = dataset_spec(name)
    split = check_split(upstream / 'data/ssb_splits'/spec.split_file if spec.split_file else None, name)
    data, weights, _ = paths()
    variant = (cfg or {}).get('dino', 'v1')
    weight = weights / ('dinov2_vitb14_reg4_pretrain.pth' if variant == 'v2' else 'dino_vitb16_pretrain.pth')
    root = data/spec.folder
    annotations = [root/relative for relative in spec.metadata]
    required = annotations+[weight]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError('Missing required files: ' + ', '.join(missing))
    result = dict(dataset_name=name, upstream_sha=sha, upstream_kind=provenance['kind'], split=split,
                  weight_sha256=sha256(weight),
                  annotation_sha256={str(p.relative_to(root)): sha256(p) for p in annotations})
    if cfg and 'method' in cfg and not is_selex:
        from audit.protocol import compare_worker
        result['reference_protocol'] = compare_worker(cfg, smoke=bool(cfg.get('max_steps_per_epoch')))
    return result


def audit_dataset(cfg=None):
    # Instantiate the actual pinned loader. Ground truth is restricted to this audit.
    import argparse
    from .worker import configure_environment
    configure_environment(ROOT / 'runs/probe-placeholder', cfg)
    from data.get_datasets import get_class_splits, get_datasets
    name = (cfg or {}).get('dataset_name', 'scars')
    spec = dataset_spec(name)
    args = get_class_splits(argparse.Namespace(dataset_name=name, use_ssb_splits=True,
                                              prop_train_labels=0.5))
    merged, test, _, _ = get_datasets(name, None, None, args)
    if name == 'cifar100':
        return audit_cifar_membership(merged, test, args, paths()[0]/spec.folder)
    rows = []
    for partition, dataset in [('labeled', merged.labelled_dataset),
                               ('unlabeled', merged.unlabelled_dataset)]:
        for sample_id, path, label in zip(dataset.uq_idxs, image_paths(dataset, name), raw_targets(dataset, name)):
            rows.append(dict(id=int(sample_id), path=path, label=int(label)-spec.target_offset, partition=partition))
    data, _, _ = paths()
    root = data/spec.folder
    counts = check_membership(rows, set(args.train_classes), root, num_classes=spec.num_classes)
    # Whole train IDs must be covered once. Derive expected count from annotations,
    # independently of the two returned subsets.
    whole = whole_training_dataset(name, root)
    if {r['id'] for r in rows} != set(map(int, whole.uq_idxs)):
        raise RuntimeError('Train pool coverage mismatch')
    targets = raw_targets(whole, name)
    if {int(label)-spec.target_offset for label in targets} != set(args.train_classes) | set(args.unlabeled_classes):
        raise RuntimeError('Train class coverage differs from the official split')
    known_count = sum(int(label)-spec.target_offset in args.train_classes for label in targets)
    if counts['labeled'] != int(known_count * 0.5):
        raise RuntimeError('Label fraction differs from baseline')
    from pathlib import Path
    test_paths, train_paths = image_paths(test, name), image_paths(whole, name)
    if not all(Path(p).is_file() for p in test_paths):
        raise RuntimeError('Missing disjoint test images')
    if set(train_paths) & set(test_paths):
        raise RuntimeError('Train/test path overlap')
    counts['disjoint_test'] = len(test)
    counts['known_classes'] = len(args.train_classes)
    counts['novel_classes'] = len(args.unlabeled_classes)
    return counts


def audit_cifar_membership(merged, test, args, root):
    """Audit-only targets; prove ID/pixel/label alignment against the full pool.

    Train and test use separate official files/namespaces, although both native
    loaders number IDs from zero. No per-image filesystem paths are invented.
    File integrity is checked by the offline upstream torchvision loader.
    """
    import numpy as np
    whole = whole_training_dataset('cifar100', root)
    known, novel = set(args.train_classes), set(args.unlabeled_classes)
    if known != set(range(80)) or novel != set(range(80, 100)):
        raise RuntimeError('CIFAR-100 requires the pinned 80/20 class partition')
    if len(whole) != 50000 or len(test) != 10000:
        raise RuntimeError('CIFAR-100 requires 50000 train / 10000 disjoint test rows')
    truth, test_truth = np.asarray(whole.targets), np.asarray(test.targets)
    for dataset, labels, expected in ((whole, truth, 500), (test, test_truth, 100)):
        pixels = np.asarray(dataset.data)
        if pixels.shape != (len(labels), 32, 32, 3) or pixels.dtype != np.uint8:
            raise RuntimeError('Invalid CIFAR-100 RGB pixel layout')
        if not np.array_equal(np.asarray(dataset.uq_idxs), np.arange(len(labels))):
            raise RuntimeError('Whole CIFAR-100 pool has unexpected sample IDs')
        if labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels >= 100):
            raise RuntimeError('Invalid CIFAR-100 zero-based targets')
        if not np.array_equal(np.bincount(labels, minlength=100), np.full(100, expected)):
            raise RuntimeError('CIFAR-100 class coverage/count mismatch')
    seen, counts = set(), {}
    for name, dataset in (('labeled', merged.labelled_dataset), ('unlabeled', merged.unlabelled_dataset)):
        ids = np.asarray(dataset.uq_idxs)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(whole)):
            raise RuntimeError('Invalid CIFAR-100 training IDs')
        unique = set(map(int, ids))
        if len(unique) != len(ids) or seen & unique:
            raise RuntimeError('Duplicate CIFAR-100 ID across training membership')
        if not np.array_equal(np.asarray(dataset.targets), truth[ids]):
            raise RuntimeError('CIFAR-100 target/ID alignment mismatch')
        if not np.array_equal(np.asarray(dataset.data), whole.data[ids]):
            raise RuntimeError('CIFAR-100 pixels/ID alignment mismatch')
        if name == 'labeled' and not set(map(int, dataset.targets)).issubset(known):
            raise RuntimeError('Novel label in supervised CIFAR-100 partition')
        seen.update(unique)
        counts[name] = len(ids)
    if seen != set(range(len(whole))) or counts != dict(labeled=20000, unlabeled=30000):
        raise RuntimeError('CIFAR-100 train coverage or 50% known-label fraction mismatch')
    if getattr(whole, 'train', None) is not True or getattr(test, 'train', None) is not False:
        raise RuntimeError('CIFAR-100 train/test namespace mismatch')
    return dict(counts, disjoint_test=len(test), known_classes=80, novel_classes=20,
                sample_identity='official_file_namespace_and_zero_based_row_id')
