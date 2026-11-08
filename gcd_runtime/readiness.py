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
        dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True, timeout=10)
        if sha != UPSTREAM_SHA or dirty:
            raise RuntimeError('Upstream must match the pinned, clean commit')
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
    provenance = verify_upstream()
    sha = provenance['sha']
    name = (cfg or {}).get('dataset_name', 'scars')
    spec = dataset_spec(name)
    split = check_split(upstream / 'data/ssb_splits'/spec.split_file, name)
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
    if cfg and 'method' in cfg:
        from audit.protocol import compare_worker
        result['reference_protocol'] = compare_worker(cfg, smoke=bool(cfg.get('max_steps_per_epoch')))
    return result


def audit_dataset(cfg=None):
    # Instantiate the actual pinned loader. Ground truth is restricted to this audit.
    import argparse
    from .worker import configure_environment
    configure_environment(ROOT / 'runs/probe-placeholder')
    from data.get_datasets import get_class_splits, get_datasets
    name = (cfg or {}).get('dataset_name', 'scars')
    spec = dataset_spec(name)
    args = get_class_splits(argparse.Namespace(dataset_name=name, use_ssb_splits=True,
                                              prop_train_labels=0.5))
    merged, test, _, _ = get_datasets(name, None, None, args)
    rows = []
    for partition, dataset in [('labeled', merged.labelled_dataset),
                               ('unlabeled', merged.unlabelled_dataset)]:
        for sample_id, path, label in zip(dataset.uq_idxs, image_paths(dataset, name), raw_targets(dataset, name)):
            rows.append(dict(id=int(sample_id), path=path, label=int(label) - 1, partition=partition))
    data, _, _ = paths()
    root = data/spec.folder
    counts = check_membership(rows, set(args.train_classes), root, num_classes=spec.num_classes)
    # Whole train IDs must be covered once. Derive expected count from annotations,
    # independently of the two returned subsets.
    whole = whole_training_dataset(name, root)
    if {r['id'] for r in rows} != set(map(int, whole.uq_idxs)):
        raise RuntimeError('Train pool coverage mismatch')
    targets = raw_targets(whole, name)
    if {int(label)-1 for label in targets} != set(args.train_classes) | set(args.unlabeled_classes):
        raise RuntimeError('Train class coverage differs from the official split')
    known_count = sum(label - 1 in args.train_classes for label in targets)
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
