"""Explicit installed-dependency repair; preserves project checkpoint fingerprints.
Use after the failed worker exits. Retains previous center for empty clusters;
finite inputs/centers are mandatory. Evaluation implementation differs from PyPI.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.find_spec('kmeans_pytorch')
if spec is None or not spec.origin:
    raise SystemExit('kmeans_pytorch missing')
p = Path(spec.origin)
s = p.read_text()
marker = '# GCD_EMPTY_CLUSTER_REPAIR_V1'
if marker in s:
    print('Repair already installed:', p)
    raise SystemExit(0)
old = '        initial_state[index] = selected.mean(dim=0)'
if s.count(old) != 1:
    raise SystemExit('Unrecognized dependency source; no change made')
new = '''        # GCD_EMPTY_CLUSTER_REPAIR_V1
        if not torch.isfinite(X).all():
            raise FloatingPointError('Non-finite kmeans input; stop rather than hide model failure')
        if selected.shape[0] > 0:
            initial_state[index] = selected.mean(dim=0)
        # Empty cluster: retain its previous finite center (no RNG change).
        if not torch.isfinite(initial_state).all():
            raise FloatingPointError('Non-finite kmeans centers')'''
# Check input once per call, not once per center (avoid GPU synchronization).
new = new.replace("        if not torch.isfinite(X).all():\n            raise FloatingPointError('Non-finite kmeans input; stop rather than hide model failure')\n", '')
new = new.replace("        if not torch.isfinite(initial_state).all():\n            raise FloatingPointError('Non-finite kmeans centers')", '')
out = s.replace(old, new)
needle = '    initial_state = initialize(X, num_clusters)'
if out.count(needle) != 1:
    raise SystemExit('Unrecognized initialization; no change made')
out = out.replace(needle, "    if not torch.isfinite(X).all():\n        raise FloatingPointError('Non-finite kmeans input')\n" + needle)
compile(out, str(p), 'exec')
backup = p.with_suffix('.py.before_gcd_empty_repair')
if backup.exists():
    raise SystemExit('Backup already exists; inspect before changing')
backup.write_text(s)
p.write_text(out)
record = {'dependency_file': str(p), 'before_sha256':hashlib.sha256(s.encode()).hexdigest(),
          'after_sha256':hashlib.sha256(out.encode()).hexdigest(),
          'policy':'retain previous center for empty clusters; reject nonfinite input',
          'comparison_note':'Apply same repair to matched baseline; not untouched PyPI evaluation'}
p.with_name('gcd_empty_cluster_repair.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
