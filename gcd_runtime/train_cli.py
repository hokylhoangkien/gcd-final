"""User entrypoint: CRiSH by default; bounded real-data smoke before full train."""
import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys

from .config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='train', choices=['train', 'smoke', 'check', 'evaluate'])
    parser.add_argument('--baseline', action='store_true', help='Matched B0, with CRiSH repairs disabled')
    parser.add_argument('--resume', type=Path, help='Trusted resume.pt from the same source/config')
    parser.add_argument('--checkpoint', type=Path, help='Locked backbone checkpoint for evaluate')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.resume and args.mode != 'train':
        parser.error('--resume is only valid for train')
    if args.mode == 'evaluate' and not args.checkpoint:
        parser.error('evaluate requires --checkpoint')
    name = 'baseline' if args.baseline else 'crish'
    full, trial = (ROOT/'configs'/f'{name}.json', ROOT/'configs'/f'{name}_smoke.json')
    stages = []
    if args.mode == 'train':
        if not args.resume:
            stages.append(('train', trial, None))
        stages.append(('train', full, args.resume))
    elif args.mode == 'smoke':
        stages.append(('train', trial, None))
    elif args.mode == 'check':
        stages.append(('preflight', full, None))
    else:
        stages.append(('evaluate', full, args.checkpoint))
    print(json.dumps(dict(selected_method=name, training_protocol='crish_v2_1',
                          status='dry_run' if args.dry_run else 'readiness_pending',
                          stages=[dict(mode=mode, config=str(path.relative_to(ROOT))) for mode, path, _ in stages],
                          note='CRiSH research implementation; no accuracy/novelty claim'), indent=2), flush=True)
    for mode, path, checkpoint in stages:
        cfg = load_config(path)
        if not args.dry_run:
            if platform.system() != 'Linux':
                raise RuntimeError('Run training on Linux NVIDIA; use --dry-run or CPU smoke tests on this Mac')
            processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                       capture_output=True, text=True, timeout=10, check=True)
            if any(line.strip().isdigit() for line in processes.stdout.splitlines()):
                raise RuntimeError('GPU process already present; inspect it before starting. No process was stopped.')
        command = [sys.executable, '-m', 'gcd_runtime.launch', mode, '--config', str(path)]
        if checkpoint:
            command.extend(['--checkpoint', str(checkpoint)])
        if args.dry_run:
            command.append('--dry-run')
        # Launcher also enforces worker deadlines. This outer bound includes
        # its preflight/evaluation children; no detached jobs are started.
        timeout = 240+cfg['max_runtime_seconds']+cfg['final_eval_timeout_seconds']
        subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


if __name__ == '__main__':
    main()
