"""Fail-fast checks for remote baseline inputs; no training."""
import hashlib
import subprocess
from .config import ROOT, UPSTREAM_SHA, paths
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


def inspect_inputs():
    upstream = ROOT / 'upstream/HypCD'
    provenance = verify_upstream()
    sha = provenance['sha']
    split = check_split(upstream / 'data/ssb_splits/scars_osr_splits.pkl')
    data, weights, _ = paths()
    weight = weights / 'dino_vitb16_pretrain.pth'
    required = [data / 'cars/devkit/cars_train_annos.mat',
                data / 'cars/devkit/cars_test_annos_withlabels.mat', weight]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError('Missing required files: ' + ', '.join(missing))
    return dict(upstream_sha=sha, upstream_kind=provenance['kind'], split=split, weight_sha256=sha256(weight),
                annotation_sha256={p.name: sha256(p) for p in required[:2]})


def audit_dataset():
    # Instantiate the actual pinned loader. Ground truth is restricted to this audit.
    import argparse
    from .worker import configure_environment
    configure_environment(ROOT / 'runs/probe-placeholder')
    from data.get_datasets import get_class_splits, get_datasets
    args = get_class_splits(argparse.Namespace(dataset_name='scars', use_ssb_splits=True,
                                              prop_train_labels=0.5))
    merged, test, _, _ = get_datasets('scars', None, None, args)
    rows = []
    for partition, dataset in [('labeled', merged.labelled_dataset),
                               ('unlabeled', merged.unlabelled_dataset)]:
        for sample_id, path, label in zip(dataset.uq_idxs, dataset.data, dataset.target):
            rows.append(dict(id=int(sample_id), path=path, label=int(label) - 1, partition=partition))
    data, _, _ = paths()
    counts = check_membership(rows, set(args.train_classes), data / 'cars')
    # Whole train IDs must be covered once. Derive expected count from annotations,
    # independently of the two returned subsets.
    from data.stanford_cars import CarsDataset
    whole = CarsDataset(train=True)
    if {r['id'] for r in rows} != set(map(int, whole.uq_idxs)):
        raise RuntimeError('Train pool coverage mismatch')
    known_count = sum(label - 1 in args.train_classes for label in whole.target)
    if counts['labeled'] != int(known_count * 0.5):
        raise RuntimeError('Label fraction differs from baseline')
    from pathlib import Path
    if not all(Path(p).is_file() for p in test.data):
        raise RuntimeError('Missing disjoint test images')
    if set(map(str, whole.data)) & set(map(str, test.data)):
        raise RuntimeError('Train/test path overlap')
    counts['disjoint_test'] = len(test)
    return counts
