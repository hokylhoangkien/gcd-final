"""Package only the current CRiSH/B0 source and its single presentation."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcd_runtime.config import ROOT, UPSTREAM_SHA
from gcd_runtime.readiness import verify_upstream, sha256

# Exact extensions prevent caches, temporary reports or artifacts leaking in.
SOURCE_DIRS = {
    'gcd_runtime': {'.py'}, 'crish': {'.py'}, 'audit': {'.py'},
    'scripts': {'.py', '.sh'}, 'tests': {'.py'}, 'configs': {'.json'},
    'environments': {'.txt'}, 'LICENSES': {'.txt'},
    'presentation': {'.py', '.md', '.pdf'},
}
ROOT_FILES = ('AGENTS.md', 'README.md', 'NOTICE', '.gitignore', '.gitattributes', '.gitmodules')
RECEIPTS = ('results/local_validation.json', 'results/cleanup_parity.json',
            'results/workspace_audit.json')


def release_files():
    files = [ROOT/name for name in ROOT_FILES]
    for directory, extensions in SOURCE_DIRS.items():
        files.extend(p for p in (ROOT/directory).rglob('*')
                     if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts)
    files.extend(ROOT/name for name in RECEIPTS if (ROOT/name).is_file())
    return sorted(set(files))


def build(output):
    provenance = verify_upstream()
    files = release_files()
    upstream = ROOT/'upstream/HypCD'
    if provenance['kind'] == 'git_submodule':
        tracked = subprocess.check_output(['git', '-C', str(upstream), 'ls-files', '-z'], timeout=10).decode().split('\0')
    else:
        tracked = list(json.loads((ROOT/'upstream/HypCD.SNAPSHOT.json').read_text())['files'])
    upstream_files = [upstream/p for p in tracked if p]
    files = sorted(set(files+upstream_files))
    release_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    prefix = 'crish-gcd-v2.1-clean-'+release_id
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output/(prefix+'.tar.gz')
    snapshot = dict(commit=UPSTREAM_SHA, files={str(p.relative_to(upstream)): sha256(p) for p in upstream_files})
    manifest = dict(release_id=release_id, upstream_sha=UPSTREAM_SHA,
                    methods=['crish', 'baseline'], cuda_verified=False, benchmark_status='not_run',
                    entrypoint='bash scripts/train_crish.sh',
                    files={str(p.relative_to(ROOT)): sha256(p) for p in files})
    with tarfile.open(destination, 'x:gz') as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError(f'Release must not include symlinks: {path}')
            archive.add(path, arcname=prefix+'/'+str(path.relative_to(ROOT)), recursive=False)
        for name, data in [('upstream/HypCD.SNAPSHOT.json', snapshot), ('release_manifest.json', manifest)]:
            payload = json.dumps(data, indent=2).encode()
            info = tarfile.TarInfo(prefix+'/'+name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    digest = sha256(destination)
    destination.with_suffix(destination.suffix+'.sha256').write_text(digest+'  '+destination.name+'\n')
    print(json.dumps(dict(archive=str(destination), sha256=digest, release_id=release_id), indent=2))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist')
    build(parser.parse_args().output)
