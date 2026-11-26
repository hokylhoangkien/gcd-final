"""RP2K preparation and runtime adapter; custom GCD split, never an SSB split.

Preparation may inspect all labels for indexing/audit. Training loads only the
public index, whose unlabeled targets are -1. Evaluation truth lives separately.
No upstream source is modified and no network or GPU work happens on import.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = 'rp2k-gcd-v1'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def dataset_root(values):
    from .config import paths
    subdir = values.get('rp2k_subdir', 'rp2k') if isinstance(values, dict) else getattr(values, 'rp2k_subdir', 'rp2k')
    if subdir not in ('rp2k', 'rp2k/smoke'):
        raise ValueError('Invalid RP2K data subdirectory')
    return paths()[0]/subdir


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def resolve_image(root, relative):
    root = Path(root).resolve()
    value = Path(relative)
    path = (root/value).resolve()
    if value.is_absolute() or not path.is_relative_to(root):
        raise ValueError('RP2K image path escapes dataset root')
    return path


def _rank(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode('utf-8')).hexdigest()


def prepare(root, source, *, seed=0):
    """Index native train/test folders in place, hashing and decoding every image.

    Exact byte duplicates are removed within/between pools (train wins), and a
    duplicate with conflicting SKU labels is rejected. Classes need >=4 distinct
    train and >=1 distinct test image after filtering. No images are reassigned.
    """
    from PIL import Image
    root, source = Path(root).resolve(), Path(source).resolve()
    if not source.is_relative_to(root):
        raise ValueError('Place the raw dataset inside the RP2K root for portable indexes')
    if (root/'manifest.json').exists():
        raise FileExistsError('RP2K manifest already exists; use a fresh root to change the split')
    for pool in ('train', 'test'):
        if not (source/pool).is_dir():
            raise ValueError(f'Missing RP2K native pool: {source/pool}')
    rows, seen, duplicates = [], {}, Counter()
    # Fixed native pool order ensures duplicate removal never moves test into train.
    for pool in ('train', 'test'):
        folders = sorted(p for p in (source/pool).iterdir() if p.is_dir())
        for folder in folders:
            for image in sorted(folder.rglob('*')):
                if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                    continue
                relative = image.relative_to(root).as_posix()
                image = resolve_image(root, relative)
                checksum = digest(image)
                if checksum in seen:
                    prior = seen[checksum]
                    if prior['sku'] != folder.name:
                        raise ValueError(f'Identical image has conflicting SKU labels: {relative}')
                    duplicates[f"{prior['pool']}->{pool}"] += 1
                    continue
                with Image.open(image) as decoded:
                    decoded.verify()
                stat = image.stat()
                row = dict(path=relative, sku=folder.name, pool=pool, sha256=checksum,
                           size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                seen[checksum] = row
                rows.append(row)
        print(f'RP2K scan {pool}: {len(rows)} unique images so far', flush=True)
    counts = {pool: Counter(row['sku'] for row in rows if row['pool'] == pool)
              for pool in ('train', 'test')}
    all_classes = sorted(set(counts['train']) | set(counts['test']))
    eligible = [sku for sku in all_classes if counts['train'][sku] >= 4 and counts['test'][sku] >= 1]
    if len(eligible) < 4:
        raise ValueError('RP2K requires >=4 eligible SKU classes after duplicate filtering')
    classes = sorted(eligible, key=lambda sku: _rank(seed, sku))
    mapping = {sku: index for index, sku in enumerate(classes)}
    known = len(classes)//2
    rows = [row for row in rows if row['sku'] in mapping]
    selected, grouped = set(), defaultdict(list)
    for row in rows:
        if row['pool'] == 'train':
            grouped[row['sku']].append(row)
    for sku in classes[:known]:
        group = grouped[sku]
        group.sort(key=lambda row: _rank(seed, row['path']))
        selected.update(row['path'] for row in group[:len(group)//2])
    public, train_truth, test = [], [], []
    for sample_id, row in enumerate(rows):
        target = mapping[row['sku']]
        base = {key: row[key] for key in ('path', 'sha256', 'size', 'mtime_ns')}
        base.update(sample_id=sample_id, target=target)
        if row['pool'] == 'train':
            train_truth.append(dict(base))
            public.append(dict(base, target=target if row['path'] in selected else -1))
        else:
            test.append(base)
    root.mkdir(parents=True, exist_ok=True)
    indexes = {'train.jsonl': public, 'train_evaluation.jsonl': train_truth, 'test.jsonl': test}
    for name, values in indexes.items():
        with (root/name).open('x', encoding='utf-8') as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
    manifest = dict(schema=SCHEMA, protocol='custom_rp2k_known_half_labeled_half_v1',
        split_seed=seed, label_fraction=0.5, classes=classes, class_count=len(classes),
        known_count=known, novel_count=len(classes)-known, known_k_assumption=True,
        counts=dict(train=len(public), labeled=len(selected), unlabeled=len(public)-len(selected), test=len(test)),
        duplicate_policy='exact_bytes_train_first_no_pool_reassignment',
        removed_duplicates=dict(duplicates),
        excluded_classes=[dict(sku=sku, train=counts['train'][sku], test=counts['test'][sku])
                          for sku in all_classes if sku not in mapping],
        indexes={name: digest(root/name) for name in indexes})
    (root/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest


def prepare_smoke(root, *, class_count=16, train_per_class=48):
    """Copy a bounded image subset for GPU readiness, never a benchmark result."""
    import shutil
    root = Path(root).resolve()
    manifest = load_manifest(root)
    grouped = defaultdict(list)
    for row in read_rows(root, 'train_evaluation.jsonl'):
        grouped[row['target']].append(row)
    eligible = [c for c in range(manifest['class_count']) if len(grouped[c]) >= train_per_class]
    if len(eligible) < class_count:
        raise ValueError('Not enough RP2K classes for the declared bounded smoke subset')
    smoke = root/'smoke'
    if smoke.exists():
        raise FileExistsError('RP2K smoke root exists; preserve it or use a fresh dataset root')
    selected = set(eligible[:class_count])
    rows = []
    for c in sorted(selected):
        rows.extend(grouped[c][:train_per_class])
    test = defaultdict(list)
    for row in read_rows(root, 'test.jsonl'):
        if row['target'] in selected:
            test[row['target']].append(row)
    for pool, pool_rows in (('train', rows), ('test', [r for c in sorted(selected) for r in test[c][:2]])):
        for row in pool_rows:
            sku = manifest['classes'][row['target']]
            target = smoke/'raw/all'/pool/sku/f"{row['sample_id']}{Path(row['path']).suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolve_image(root, row['path']), target)
    return prepare(smoke, smoke/'raw/all', seed=manifest['split_seed'])


def load_manifest(root, expected=None, *, verify_indexes=True):
    root = Path(root)
    path = root/'manifest.json'
    if expected is not None and digest(path) != expected:
        raise ValueError('RP2K manifest fingerprint mismatch; do not change data on resume')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    k, known = manifest.get('class_count', 0), manifest.get('known_count', 0)
    if (manifest.get('schema') != SCHEMA or type(k) is not int or k < 4
            or type(known) is not int or known != k//2
            or manifest.get('novel_count') != k-known
            or len(manifest.get('classes', [])) != k or len(set(manifest['classes'])) != k
            or manifest.get('label_fraction') != 0.5
            or set(manifest.get('indexes', {})) != {'train.jsonl', 'train_evaluation.jsonl', 'test.jsonl'}):
        raise ValueError('Invalid RP2K manifest protocol/class partition')
    if verify_indexes:
        names = manifest['indexes'] if verify_indexes is True else verify_indexes
        for name in names:
            checksum = manifest['indexes'][name]
            if digest(root/name) != checksum:
                raise ValueError(f'RP2K index fingerprint mismatch: {name}')
    return manifest


def read_rows(root, name):
    rows = [json.loads(line) for line in (Path(root)/name).read_text(encoding='utf-8').splitlines()]
    for row in rows:
        resolve_image(root, row['path'])
        if type(row['sample_id']) is not int or row['sample_id'] < 0 or type(row['target']) is not int:
            raise ValueError('Invalid RP2K sample ID/target')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate RP2K sample ID')
    return rows


def audit(root, expected=None, *, rehash=False):
    """Audit-only access to withheld truth; never return annotation rows to policy."""
    root = Path(root)
    manifest = load_manifest(root, expected)
    public, truth, test = [read_rows(root, name) for name in
                           ('train.jsonl', 'train_evaluation.jsonl', 'test.jsonl')]
    if len(public) != len(truth):
        raise ValueError('RP2K train truth coverage mismatch')
    known, k = manifest['known_count'], manifest['class_count']
    per_class, supervised = Counter(), Counter()
    hashes, paths, ids = set(), set(), set()
    for row in truth+test:
        path = resolve_image(root, row['path'])
        if row['sha256'] in hashes or row['path'] in paths or row['sample_id'] in ids:
            raise ValueError('RP2K duplicate image/ID across native pools')
        if not 0 <= row['target'] < k:
            raise ValueError('RP2K truth label out of range')
        stat = path.stat()
        # Hashing on the GPU host is opt-in; normal preflight checks size. mtime
        # is provenance only because copying/extraction across hosts can change it.
        if stat.st_size != row['size'] or (rehash and digest(path) != row['sha256']):
            raise ValueError(f'RP2K image changed: {path}')
        hashes.add(row['sha256']); paths.add(row['path']); ids.add(row['sample_id'])
    for row, hidden in zip(public, truth):
        if any(row[key] != hidden[key] for key in ('sample_id', 'path', 'sha256', 'size')):
            raise ValueError('RP2K public/truth row alignment mismatch')
        target = hidden['target']
        per_class[target] += 1
        if row['target'] != -1:
            if not 0 <= row['target'] < known or row['target'] != target:
                raise ValueError('Novel/incorrect label in RP2K supervised pool')
            supervised[target] += 1
    if set(per_class) != set(range(k)) or set(row['target'] for row in test) != set(range(k)):
        raise ValueError('RP2K native pool class coverage mismatch')
    if any(supervised[c] != per_class[c]//2 or supervised[c] < 2 for c in range(known)):
        raise ValueError('RP2K known anchors do not match the declared per-class half split')
    counts = dict(train=len(public), labeled=sum(supervised.values()),
                  unlabeled=len(public)-sum(supervised.values()), test=len(test))
    if counts != manifest['counts']:
        raise ValueError('RP2K manifest counts differ from indexes')
    return dict(labeled=counts['labeled'], unlabeled=counts['unlabeled'], disjoint_test=len(test),
                known_classes=known, novel_classes=k-known, manifest_sha256=digest(root/'manifest.json'),
                protocol=manifest['protocol'], known_k_assumption=True, exact_byte_dedup=True,
                image_hashes_rechecked=rehash)


class ProductRows:
    """Image dataset compatible with the existing sanitization boundary."""
    def __init__(self, root, rows, transform):
        import numpy as np
        self.root, self.transform, self.target_transform = str(root), transform, None
        self.samples = [(str(resolve_image(root, row['path'])), row['target']) for row in rows]
        self.uq_idxs = np.array([row['sample_id'] for row in rows], dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image
        path, target = self.samples[index]
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target, int(self.uq_idxs[index])


def class_splits(args):
    manifest = load_manifest(dataset_root(args), getattr(args, 'rp2k_manifest_sha256', None),
                             verify_indexes=('train.jsonl',))
    args.train_classes = list(range(manifest['known_count']))
    args.unlabeled_classes = list(range(manifest['known_count'], manifest['class_count']))
    args.image_size, args.use_ssb_splits, args.prop_train_labels = 224, False, 0.5
    return args


def datasets(train_transform, test_transform, args):
    root = dataset_root(args)
    evaluation = getattr(args, 'rp2k_evaluation', False)
    load_manifest(root, args.rp2k_manifest_sha256,
                  verify_indexes=True if evaluation else ('train.jsonl',))
    public = read_rows(root, 'train.jsonl')
    labeled_ids = {row['sample_id'] for row in public if row['target'] >= 0}
    # Withheld labels are never opened by the training dataset constructor.
    rows = read_rows(root, 'train_evaluation.jsonl') if evaluation else public
    labeled = ProductRows(root, [r for r in rows if r['sample_id'] in labeled_ids], train_transform)
    unlabeled = ProductRows(root, [r for r in rows if r['sample_id'] not in labeled_ids], train_transform)
    # Match the pinned wrapper's four-field interface, then sanitize for training.
    from data.data_utils import MergedDataset
    merged = MergedDataset(labeled, unlabeled)
    test = ProductRows(root, read_rows(root, 'test.jsonl'), test_transform) if evaluation else None
    unlabeled_eval = ProductRows(root, [r for r in rows if r['sample_id'] not in labeled_ids], test_transform) if evaluation else None
    return merged, test, unlabeled_eval, {}
