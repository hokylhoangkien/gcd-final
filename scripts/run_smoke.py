"""Bounded local checks with discoverable log/config/source metadata. No weights."""
import ast
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from gcd_runtime.config import code_fingerprint, UPSTREAM_SHA

run_id = 'local-smoke-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
run_dir = root / 'logs' / run_id
run_dir.mkdir(parents=True, exist_ok=False)
commands = [[sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
            [sys.executable, 'scripts/check_integrity.py']]
revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, timeout=10)
metadata = dict(run_id=run_id, kind='local_smoke_not_benchmark',
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                git_sha=revision.stdout.strip() if revision.returncode == 0 else 'uncommitted',
                upstream_sha=UPSTREAM_SHA, source_hashes=code_fingerprint(),
                python=sys.version, platform=platform.platform(), commands=commands,
                config=dict(timeout_per_command_seconds=60, pretrained_weights=False,
                            fixture_seeds={'geometry': 7, 'resume': 123}),
                log_path=str(run_dir / 'smoke.log'), status='running')
try:
    with (run_dir / 'smoke.log').open('w') as log:
        for command in commands:
            result = subprocess.run(command, cwd=root, timeout=60, capture_output=True, text=True)
            output = result.stdout + result.stderr
            print(output, end='', flush=True)
            log.write(output)
            log.flush()
            result.check_returncode()
        for directory in ('gcd_runtime', 'scripts', 'tests'):
            for path in (root / directory).glob('*.py'):
                ast.parse(path.read_text(), filename=str(path))
        print('Static syntax checks: passed')
        log.write('Static syntax checks: passed\n')
    metadata['status'] = 'passed'
except BaseException as exc:
    metadata.update(status='failed', error=repr(exc))
    raise
finally:
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Smoke metadata: {run_dir / "metadata.json"}')
