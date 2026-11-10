"""Bounded CPU pytest + source/split checks, with versioned run provenance."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.config import TRAIN_SHA as SHA256
from gcd_runtime.readiness import verify_upstream
SOURCE = ROOT / 'upstream/HypCD/train/train_HypSelEx.py'


def git(*args, cwd=ROOT):
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    result.check_returncode()
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/local_smoke.json')
    parser.add_argument('--report', type=Path, default=ROOT/'results/local_validation.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['device'] != 'cpu' or cfg['full_training_enabled'] or cfg['pretrained_weights'] or cfg['dataset'] is not None:
        raise ValueError('This runner only supports weight-free synthetic CPU checks')
    timeout = cfg['timeout_per_command_seconds']
    if type(timeout) is not int or not 1 <= timeout <= 60:
        raise ValueError('Bounded command timeout must be 1..60 seconds')
    now = datetime.now(timezone.utc)
    run_id = 'crish-v21-' + now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    run_dir = ROOT / 'logs' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_names = ('AGENTS.md', 'README.md', 'NOTICE', 'reports/04_method_and_research_plan.md',
                   'paper/manuscript.pdf', 'paper/manuscript.build.json')
    tracked = [ROOT / name for name in input_names if (ROOT / name).is_file()]
    for directory, pattern in [('crish', '*.py'), ('audit', '*.py'), ('gcd_runtime', '*.py'),
                               ('selex_runtime', '*.py'), ('tests', '*.py'), ('configs', '*.json'),
                               ('scripts', '*.py'), ('scripts', '*.sh'),
                               ('reports', '*.md'), ('paper', '*.tex'), ('paper', '*.bib')]:
        tracked.extend((ROOT / directory).rglob(pattern))
    tracked.extend([Path(__file__), SOURCE, args.config.resolve()])
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(tracked))}
    provenance = verify_upstream()
    has_git = (ROOT / '.git').exists()
    metadata = dict(run_id=run_id, kind=cfg['kind'], timestamp_utc=now.isoformat(),
                    config=cfg, git_sha=git('rev-parse', 'HEAD') if has_git else 'verified_source_archive',
                    branch=git('branch', '--show-current') if has_git else None,
                    git_status_before=git('status', '--short') if has_git else None, source_hashes=hashes,
                    upstream_sha=provenance['sha'], upstream_kind=provenance['kind'],
                    absent_source_inputs=[name for name in input_names if not (ROOT / name).is_file()],
                    python=sys.version, platform=platform.platform(), log_path=str(run_dir / 'smoke.log'),
                    benchmark_metrics=None, status='running')
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    commands = [[sys.executable, '-m', 'pytest', cfg['tests'], '-q', '--tb=short', '-o', 'cache_dir=.cache/pytest',
                 '--junitxml=' + str(run_dir / 'pytest.xml')],
                [sys.executable, 'scripts/check_integrity.py'],
                [sys.executable, 'scripts/check_integrity.py', '--dataset', 'cub'],
                [sys.executable, 'scripts/check_integrity.py', '--dataset', 'cifar100'],
                [sys.executable, 'scripts/check_integrity.py', '--dataset', 'aircraft'],
                [sys.executable, 'scripts/check_protocol.py'],
                [sys.executable, 'scripts/check_manuscript.py']]
    metadata['commands'] = commands
    try:
        if metadata['upstream_sha'] != cfg['upstream_sha']:
            raise RuntimeError('Upstream pin/cleanliness gate failed')
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('Training-source hash mismatch')
        with (run_dir / 'smoke.log').open('w') as log:
            for command in commands:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
                output = result.stdout + result.stderr
                print(output, end='', flush=True)
                log.write(output)
                log.flush()
                result.check_returncode()
            for directory in ('gcd_runtime', 'crish', 'selex_runtime', 'audit', 'scripts', 'tests'):
                for path in (ROOT / directory).rglob('*.py'):
                    ast.parse(path.read_text(), filename=str(path))
            log.write('Static syntax, upstream SHA/source hash: passed\n')
        suites = ET.parse(run_dir / 'pytest.xml').getroot().findall('testsuite')
        metadata['pytest'] = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
        metadata['status'] = 'passed'
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    finally:
        metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
        (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('Run metadata:', run_dir / 'metadata.json')
    summary = {k: metadata[k] for k in ('run_id', 'kind', 'status', 'git_sha', 'upstream_sha', 'pytest', 'timestamp_utc', 'completed_utc', 'benchmark_metrics')}
    summary.update(metadata_path=str((run_dir / 'metadata.json').relative_to(ROOT)),
                   log_path=str((run_dir / 'smoke.log').relative_to(ROOT)),
                   real_data_gate='not_run_no_dataset_weights_or_local_cuda',
                   config=cfg, python=sys.version, platform=platform.platform(), cuda_verified=False,
                   commands=[[str(part).replace(str(ROOT), '.') for part in command] for command in commands],
                   source_hashes=hashes)
    report_path = args.report if args.report.is_absolute() else ROOT/args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
