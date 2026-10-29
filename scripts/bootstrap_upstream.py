"""Populate a missing pinned upstream checkout; never reset existing work."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcd_runtime.config import ROOT, UPSTREAM_SHA

path = ROOT / 'upstream/HypCD'
if (ROOT/'upstream/HypCD.SNAPSHOT.json').is_file() and not (path/'.git').exists():
    from gcd_runtime.readiness import verify_upstream
    print(verify_upstream())
    raise SystemExit(0)
if not (path / '.git').exists():
    if path.exists() and any(path.iterdir()):
        raise SystemExit('Nonempty upstream without Git metadata; preserved, inspect manually')
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=ROOT,
                   timeout=180, check=True)
sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                              timeout=10, text=True).strip()
if sha != UPSTREAM_SHA:
    raise SystemExit('Existing upstream differs from pin; preserved, inspect manually')
print(sha)
