"""Explicitly authorized stop at a fresh, consistent checkpoint; Linux only."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time


def worker_identity(pid, source):
    proc = Path('/proc')/str(pid)
    command = (proc/'cmdline').read_bytes().split(b'\0')
    if b'gcd_runtime.worker' not in command or str(source).encode() not in command:
        raise ValueError('PID is not the specified run worker; refusing any signal')
    stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    if os.getpgid(pid) != pid:
        raise ValueError('Worker must own its process group')
    return stat[19]  # field22: start time, after pid and comm


def latest_epoch(source):
    path = source/'metrics.jsonl'
    if not path.is_file():
        return -1
    rows = path.read_text().splitlines()
    for line in reversed(rows):
        try:
            return int(json.loads(line)['epoch'])
        except (json.JSONDecodeError, KeyError):
            continue
    return -1


def validate_checkpoint(source, minimum_epoch):
    # Import only after freeze. CUDA_VISIBLE_DEVICES='' for this controller
    # avoids allocating another CUDA context alongside the training worker.
    import torch
    root = source.parent.parent
    sys.path.insert(0, str(root))
    from gcd_runtime.config import code_fingerprint
    from selex_runtime.selection import BEST_NAME, enabled, read_best
    state = torch.load(source/'resume.pt', map_location='cpu', weights_only=False)
    if state['epoch'] < minimum_epoch or state['code_fingerprint'] != code_fingerprint():
        raise ValueError('Checkpoint epoch/source mismatch; original job will continue')
    if state.get('plain_rng_pending', False):
        raise ValueError('Plain evaluation is pending; original job will continue')
    if enabled(state['config']):
        best = read_best(source/BEST_NAME, state['config'])
        if best['selection']['epoch'] > state['epoch']:
            raise ValueError('Best artifacts ahead of checkpoint; original job will continue')
        for epoch in range(state['epoch']+1):
            report = source/'test_evaluations'/f'epoch_{epoch+1:04d}.json'
            if not report.is_file():
                raise ValueError('Missing completed plain evaluation; original job will continue')
    return state['epoch']


def stop_after_checkpoint(source, pid, *, max_seconds=600):
    source = Path(source).resolve()
    identity = worker_identity(pid, source)
    previous = latest_epoch(source)
    deadline = time.monotonic()+max_seconds
    print(f'WAIT pid={pid} completed_epochs={previous+1}; waiting for next saved epoch; timeout={max_seconds}s', flush=True)
    while time.monotonic() < deadline:
        if worker_identity(pid, source) != identity:
            raise ValueError('Worker identity changed; refusing any signal')
        observed = latest_epoch(source)
        if observed > previous:
            frozen, terminating = False, False
            try:
                os.killpg(pid, signal.SIGSTOP)
                frozen = True
                epoch = validate_checkpoint(source, observed)
                os.killpg(pid, signal.SIGTERM)
                terminating = True
                print(f'STOP_REQUESTED pid={pid} completed_epochs={epoch+1} resume={source / "resume.pt"}', flush=True)
                return epoch
            finally:
                if frozen:
                    # Resume on validation failure; deliver pending TERM only
                    # after checkpoint validation succeeded. Never force-kill.
                    try:
                        os.killpg(pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                    if not terminating:
                        print('Checkpoint validation failed; continued original worker', flush=True)
        time.sleep(.2)
    raise TimeoutError('No fresh checkpoint within limit; no stop signal sent')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--confirm-stop', action='store_true', required=True)
    args = parser.parse_args()
    if args.pid <= 1 or args.timeout <= 0:
        parser.error('Positive worker PID and timeout required')
    stop_after_checkpoint(args.run, args.pid, max_seconds=args.timeout)
