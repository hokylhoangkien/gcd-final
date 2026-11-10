"""SelEx+CRiSH entrypoint, separate from the legacy Hyp-SelEx command."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from gcd_runtime.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='train', choices=['train', 'check', 'smoke', 'evaluate'])
    parser.add_argument('--baseline', action='store_true', help='Updated SelEx without CRiSH')
    parser.add_argument('--config', type=Path, help='Exact SelEx config snapshot; also supports other seeds')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, help='Matched run seed; default comes from config')
    parser.add_argument('--eval-every', type=int, help='Extra SS monitoring interval; default 0')
    args = parser.parse_args()
    if args.resume and args.mode != 'train':
        parser.error('--resume is for train only')
    if (args.seed is not None and args.seed < 0) or (args.eval_every is not None and args.eval_every < 0):
        parser.error('Seed and SS interval must be nonnegative')
    if args.eval_every is not None and args.mode != 'train':
        parser.error('--eval-every is for train only')
    if args.mode == 'evaluate' and not args.checkpoint:
        parser.error('evaluate requires --checkpoint')
    name = 'baseline' if args.baseline else 'crish'
    full = args.config or ROOT/f'configs/selex_{name}.json'
    cfg = load_config(full, eval_every_epochs=args.eval_every)
    if args.seed is not None:
        cfg['seed'] = args.seed
    if cfg.get('baseline_family') != 'selex' or cfg['method'] != name:
        parser.error('Config must match SelEx and the selected baseline/CRiSH arm')
    smoke = ROOT/f'configs/selex_{name}_smoke.json'
    if args.config or args.seed is not None or args.eval_every is not None:
        directory = ROOT/'runs/plans'/('selex-config-'+uuid.uuid4().hex[:12])
        directory.mkdir(parents=True, exist_ok=False)
        full = directory/'full.json'
        full.write_text(json.dumps(cfg, indent=2)+'\n')
        trial = json.loads(json.dumps(cfg))
        trial.update(epochs=1, max_steps_per_epoch=2, max_runtime_seconds=1800, ss_eval_every_epochs=0)
        trial['crish']['warmup_epochs'] = 0
        smoke = directory/'smoke.json'
        smoke.write_text(json.dumps(trial, indent=2)+'\n')
    stages = [('preflight', full, None)] if args.mode == 'check' else []
    if args.mode == 'train':
        if not args.resume:
            stages.append(('train', smoke, None))
        stages.append(('train', full, args.resume))
    elif args.mode == 'smoke':
        stages.append(('train', smoke, None))
    elif args.mode == 'evaluate':
        stages.append(('evaluate', full, args.checkpoint))
    print(json.dumps(dict(baseline_family='selex', method=name, primary='fixed-final SS-v2',
        epochs=cfg['epochs'], seed=cfg['seed'], head_dimension=cfg['mlp_out_dim'],
        status='dry_run' if args.dry_run else 'readiness_pending'), indent=2), flush=True)
    for mode, config, checkpoint in stages:
        stage = load_config(config)
        if not args.dry_run:
            if platform.system() != 'Linux':
                raise RuntimeError('Only bounded CPU synthetic tests on this Mac; train on Linux NVIDIA')
            busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=10, check=True)
            if any(line.strip().isdigit() for line in busy.stdout.splitlines()):
                raise RuntimeError('GPU process present; inspect before starting. No job was stopped.')
        command = [sys.executable, '-m', 'gcd_runtime.launch', mode, '--config', str(config)]
        if checkpoint:
            command += ['--checkpoint', str(checkpoint)]
        if args.dry_run:
            command += ['--dry-run']
        subprocess.run(command, cwd=ROOT, check=True,
                       timeout=240+stage['max_runtime_seconds']+stage['final_eval_timeout_seconds'])


if __name__ == '__main__':
    main()
