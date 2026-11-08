"""Bounded remote launcher. --dry-run works without PyTorch/data/CUDA."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import uuid
from .config import ROOT, load_config, paths, code_fingerprint


def final_evaluations(snapshot, run_dir, cfg, env, log_handle):
    """Evaluate both locked candidates; each worker has its own strict deadline."""
    from .selection import BEST_NAME, enabled
    from .metrics import print_peaks, summarize_run
    candidates = [('last', run_dir/'backbone_final.pt', 'evaluation_last.json')]
    if enabled(cfg):
        candidates.append(('best_old_test', run_dir/BEST_NAME, 'evaluation_best_old_test.json'))
    commands = []
    for role, checkpoint, report in candidates:
        if not checkpoint.is_file():
            raise RuntimeError(f'Missing {role} checkpoint; no substitute evaluation started')
        command = [sys.executable, '-m', 'gcd_runtime.worker', str(snapshot),
                   str(run_dir), 'evaluate', str(checkpoint)]
        commands.append(dict(role=role, command=command))
        print(f'Final SS evaluation role={role}; timeout={cfg["final_eval_timeout_seconds"]}s', flush=True)
        subprocess.run(command, cwd=ROOT, env=env, stdout=log_handle,
                       stderr=subprocess.STDOUT, timeout=cfg['final_eval_timeout_seconds'], check=True)
        if not (run_dir/report).is_file():
            raise RuntimeError(f'{role} evaluation produced no {report}')
    if not (run_dir/'evaluation.json').is_file():
        raise RuntimeError('Primary SS evaluation report missing')
    summary = summarize_run(run_dir, write=True)
    print_peaks(summary)
    return commands


def run_revision():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else 'uncommitted'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'evaluate', 'preflight'])
    parser.add_argument('--config', default=str(ROOT / 'configs/crish.json'))
    parser.add_argument('--checkpoint', type=Path, help='Trusted resume.pt for train; backbone state_dict for evaluate')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--eval-every', type=int, help='Override ss_eval_every_epochs; 0 disables intermediate evaluation')
    args = parser.parse_args()
    cfg = load_config(args.config, eval_every_epochs=args.eval_every)
    if args.mode == 'evaluate' and not args.checkpoint:
        parser.error('evaluate requires --checkpoint (backbone state_dict, not resume.pt)')
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error('checkpoint does not exist')
    print(json.dumps(dict(mode=args.mode, config=cfg, paths=list(map(str, paths())),
                         status='dry_run' if args.dry_run else 'readiness_pending'), indent=2), flush=True)
    if args.dry_run:
        return
    if platform.system() != 'Linux':
        raise RuntimeError('Full baseline runs are restricted to Linux + NVIDIA; use local smoke tests')
    _, _, runs = paths()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    run_dir = runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir / 'config.json'
    snapshot.write_text(json.dumps(cfg, indent=2))
    # Gate is a child with a strict deadline, including data and CUDA checks.
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(cfg['seed']),
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env['XFORMERS_DISABLED'] = '1'
    metadata = dict(run_id=run_id, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    git_sha=run_revision(), config=cfg, command=sys.argv, seed=cfg['seed'],
                    python=sys.version, platform=platform.platform(), status='preflight',
                    code_fingerprint=code_fingerprint(),
                    git_status=subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or 'clean_or_release_snapshot',
                    checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None)
    log = run_dir / 'run.log'
    print(f'Preflight run={run_id}; log={log}; timeout=180s; full-batch VRAM chưa đo', flush=True)
    try:
        frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, timeout=30)
        (run_dir / 'environment.txt').write_text(frozen)
        with log.open('w') as handle:
            subprocess.run([sys.executable, '-m', 'gcd_runtime.remote_gate', str(snapshot), str(run_dir)],
                           cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=180, check=True)
            if args.mode != 'preflight':
                command = [sys.executable, '-m', 'gcd_runtime.worker', str(snapshot), str(run_dir),
                           args.mode, str(args.checkpoint.resolve()) if args.checkpoint else '']
                metadata['worker_command'] = command
                print(f'Worker command: {json.dumps(command)}', flush=True)
                worker_timeout = (cfg['final_eval_timeout_seconds'] if args.mode == 'evaluate'
                                  else cfg['max_runtime_seconds'])
                print(f'Run {run_id}; log={log}; max_runtime={worker_timeout}s', flush=True)
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                      start_new_session=True) as process:
                    metadata['pid'] = process.pid
                    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
                    print(f'PID={process.pid}; theo dõi: tail -f {log}', flush=True)
                    try:
                        code = process.wait(timeout=worker_timeout)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        # Stop only this launcher's own process group, including
                        # its synchronous monitor and DataLoader children.
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                    if code:
                        raise RuntimeError(f'Worker exited {code}; inspect {log}')
                if args.mode == 'evaluate':
                    # An explicit single-checkpoint evaluation always provides
                    # evaluation.json, including when the caller chose last.
                    from .metrics import atomic_json, print_peaks, summarize_run
                    if not (run_dir/'evaluation.json').is_file() and (run_dir/'evaluation_last.json').is_file():
                        atomic_json(run_dir/'evaluation.json', json.loads((run_dir/'evaluation_last.json').read_text()))
                    if not (run_dir/'evaluation.json').is_file():
                        raise RuntimeError('Explicit evaluation produced no metrics file')
                    print_peaks(summarize_run(run_dir, write=True))
                if args.mode == 'train' and cfg['final_ss_evaluation']:
                    metadata['final_evaluation_commands'] = final_evaluations(snapshot, run_dir, cfg, env, handle)
        metadata['status'] = 'preflight_passed' if args.mode == 'preflight' else 'completed'
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        metadata['log_path'] = str(log)
        (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print(f'Status={metadata["status"]}; metadata={run_dir / "metadata.json"}', flush=True)


if __name__ == '__main__':
    main()
