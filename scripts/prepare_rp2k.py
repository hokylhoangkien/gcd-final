"""Download/extract/index RP2K or create one local-CRiSH run plan. No training."""
import argparse
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.config import paths, load_config
from gcd_runtime.rp2k import prepare, prepare_smoke, audit, digest, load_manifest

ARCHIVE_URL = 'https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip'


def download(root):
    archive = root/'downloads/rp2k_dataset.zip'
    receipt = archive.with_suffix('.receipt.json')
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        if digest(archive) != recorded['sha256']:
            raise ValueError('Downloaded archive differs from its receipt')
        return recorded
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--retry-all-errors',
        '--retry-delay', '5', '--retry-max-time', '7200', '--max-time', '7200',
        '--speed-limit', '1024', '--speed-time', '90', '--continue-at', '-',
        '--output', str(archive), ARCHIVE_URL], check=True, timeout=7500)
    with zipfile.ZipFile(archive) as zipped:
        if not zipped.namelist():
            raise ValueError('Empty RP2K archive')
    result = dict(url=ARCHIVE_URL, sha256=digest(archive), bytes=archive.stat().st_size,
                  published_checksum_verified=False)
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    return result


def extract(root):
    archive, target = root/'downloads/rp2k_dataset.zip', root/'raw'
    if target.exists():
        raise FileExistsError('RP2K raw directory exists; it will not be overwritten')
    with zipfile.ZipFile(archive) as zipped:
        total = 0
        names = set()
        for entry in zipped.infolist():
            destination = (target/entry.filename).resolve()
            if (not destination.is_relative_to(target.resolve()) or '\\' in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError('Unsafe RP2K zip member path/symlink')
            if entry.filename in names:
                raise ValueError('Duplicate archive member path')
            names.add(entry.filename)
            total += entry.file_size
        if total > 100 * 1024**3:
            raise ValueError('Archive expands beyond the 100 GiB preparation limit')
        target.mkdir(parents=True, exist_ok=False)
        zipped.extractall(target)
    if not (target/'all/train').is_dir() or not (target/'all/test').is_dir():
        raise ValueError('Expected raw/all/train and raw/all/test; inspect this archive release')
    return dict(raw=str(target), uncompressed_bytes=total)


def plan(root, output):
    """Freeze the exact manifest checksums into a fresh portable run config."""
    from selex_runtime.config import validate_config
    manifest = load_manifest(root)
    smoke = load_manifest(root/'smoke')
    template = json.loads((ROOT/'configs/selex_rp2k_crish_200.json').read_text())
    template.update(rp2k_manifest_sha256=digest(root/'manifest.json'),
                    rp2k_smoke_manifest_sha256=digest(root/'smoke/manifest.json'))
    cfg = validate_config(template)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(cfg, indent=2)+'\n')
    load_config(output)
    return dict(status='prepared_not_trained', config=str(output),
                known_classes=manifest['known_count'], novel_classes=manifest['novel_count'],
                counts=manifest['counts'], smoke_counts=smoke['counts'],
                cuda_verified=False, training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['download', 'extract', 'index', 'smoke-index', 'audit', 'plan'])
    parser.add_argument('--root', type=Path, default=paths()[0]/'rp2k')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--rehash', action='store_true', help='Audit every image byte again')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/plans/rp2k/crish.json')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.mode == 'download':
        result = download(root)
    elif args.mode == 'extract':
        result = extract(root)
    elif args.mode == 'index':
        result = prepare(root, root/'raw/all', seed=args.split_seed)
    elif args.mode == 'smoke-index':
        result = prepare_smoke(root)
    elif args.mode == 'audit':
        result = audit(root, rehash=args.rehash)
    else:
        if root != (paths()[0]/'rp2k').resolve():
            parser.error('Set GCD_DATA_ROOT so --root equals $GCD_DATA_ROOT/rp2k before generating a plan')
        result = plan(root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
