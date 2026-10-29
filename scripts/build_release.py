"""Build an allowlisted portable source release, with no Git history/data/weights."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gcd_runtime.config import ROOT, UPSTREAM_SHA
from gcd_runtime.readiness import verify_upstream, sha256


def build(output):
    provenance=verify_upstream()
    files=[]
    for directory in ('gcd_runtime','scripts','tests','configs','reports','environments','LICENSES'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and '.pyc'!=p.suffix)
    files.extend(ROOT/name for name in ('AGENTS.md','README.md','RUNPOD.md','THIRD_PARTY_NOTICES.md','.gitignore','.gitmodules'))
    upstream=ROOT/'upstream/HypCD'
    if provenance['kind']=='git_submodule':
        tracked=subprocess.check_output(['git','-C',str(upstream),'ls-files','-z'],timeout=10).decode().split('\0')
    else:
        tracked=list(json.loads((ROOT/'upstream/HypCD.SNAPSHOT.json').read_text())['files'])
    upstream_files=[upstream/p for p in tracked if p]
    files.extend(upstream_files)
    release_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    prefix='gcd-final-runpod-'+release_id
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    destination=output/(prefix+'.tar.gz')
    snapshot=dict(commit=UPSTREAM_SHA,files={str(p.relative_to(upstream)):sha256(p) for p in upstream_files})
    manifest=dict(release_id=release_id,upstream_sha=UPSTREAM_SHA,
                  cuda_verified=False,benchmark_status='not_run',
                  files={str(p.relative_to(ROOT)):sha256(p) for p in files})
    with tarfile.open(destination,'x:gz') as archive:
        for path in sorted(set(files)):
            archive.add(path,arcname=prefix+'/'+str(path.relative_to(ROOT)),recursive=False)
        for name,data in [('upstream/HypCD.SNAPSHOT.json',snapshot),('release_manifest.json',manifest)]:
            payload=json.dumps(data,indent=2).encode()
            info=tarfile.TarInfo(prefix+'/'+name)
            info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload))
    digest=sha256(destination)
    destination.with_suffix(destination.suffix+'.sha256').write_text(digest+'  '+destination.name+'\n')
    print(json.dumps(dict(archive=str(destination),sha256=digest,release_id=release_id),indent=2))
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist')
    args=parser.parse_args()
    build(args.output)
