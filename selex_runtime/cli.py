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
    parser.add_argument('--dataset', choices=['scars', 'cub', 'aircraft', 'cifar100', 'herbarium_19'],
                        help='Dataset for the matched 200-epoch DINOv2 profiles; default scars')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, help='Matched run seed; default comes from config')
    parser.add_argument('--eval-every', type=int, help='Extra SS monitoring interval; default 0')
    parser.add_argument('--pair-reliability', choices=['off', 'rn', 'cross_view', 'both'],
                        help='Online base pseudo-SSE ablation; default from the selected config')
    parser.add_argument('--no-repair', action='store_true',
                        help='Disable local and population repair for a weighting-only control')
    parser.add_argument('--variant', choices=['local', 'online', 'policy', 'hybrid'],
                        help='Explicit experiment variant; default comes from config')
    parser.add_argument('--probe-known-per-class', type=int,
                        help='1: historical sampling; >=2: exact structured legal class quotas')
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
    dataset = args.dataset or 'scars'
    full = args.config or ROOT/f'configs/selex_{dataset}_{name}_200.json'
    cfg = load_config(full, eval_every_epochs=args.eval_every)
    if args.dataset is not None and cfg['dataset_name'] != args.dataset:
        parser.error('--dataset must match the supplied --config')
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.variant is not None:
        from crish.population_settings import PopulationSettings
        from crish.config import PairReliabilitySettings
        from dataclasses import asdict
        cfg['population_policy'] = PopulationSettings(**dict(cfg.get('population_policy', {}),
            enabled=args.variant in ('policy', 'hybrid'))).to_dict()
        cfg['pair_reliability'] = asdict(PairReliabilitySettings(**dict(cfg.get('pair_reliability', {}),
            enabled=args.variant in ('online', 'hybrid'))))
        cfg['crish']['enabled'] = args.variant == 'local'
    if args.probe_known_per_class is not None:
        cfg['crish']['probe_known_per_class'] = args.probe_known_per_class
        from dataclasses import asdict
        from crish.config import CrishSettings
        cfg['crish'] = asdict(CrishSettings(**cfg['crish']))
    if args.pair_reliability is not None:
        from dataclasses import asdict
        from crish.config import PairReliabilitySettings
        values = dict(cfg.get('pair_reliability', {}))
        values.update(enabled=args.pair_reliability != 'off',
                      reciprocal=args.pair_reliability in ('rn', 'both'),
                      cross_view=args.pair_reliability in ('cross_view', 'both'))
        cfg['pair_reliability'] = asdict(PairReliabilitySettings(**values))
    if args.no_repair:
        if args.baseline:
            parser.error('--no-repair belongs to the CRiSH weighting-only control')
        cfg['crish']['enabled'] = False
        if 'population_policy' in cfg:
            cfg['population_policy']['enabled'] = False
    if cfg.get('baseline_family') != 'selex' or cfg['method'] != name:
        parser.error('Config must match SelEx and the selected baseline/CRiSH arm')
    # Derive smoke from the selected dataset/arm, never a Cars-only preset.
    directory = ROOT/'runs/plans'/('selex-config-'+uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    full = directory/'full.json'
    full.write_text(json.dumps(cfg, indent=2)+'\n')
    trial = json.loads(json.dumps(cfg))
    trial.update(epochs=1, max_steps_per_epoch=2, max_runtime_seconds=1800, ss_eval_every_epochs=0,
                 require_intervention_smoke=True)
    trial['crish']['warmup_epochs'] = 0
    if trial.get('pair_reliability', {}).get('enabled'):
        trial['pair_reliability'].update(warmup_epochs=0, ramp_epochs=1)
    if trial.get('population_policy', {}).get('enabled'):
        trial['population_policy']['warmup_epochs'] = 0
        trial['population_policy']['schedule_epochs'] = 1
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
        dataset_name=cfg['dataset_name'], dino=cfg['dino'],
        epochs=cfg['epochs'], seed=cfg['seed'], head_dimension=cfg['mlp_out_dim'],
        unbalanced=cfg.get('unbalanced', False),
        pair_reliability=cfg.get('pair_reliability', dict(enabled=False)),
        discrete_repair=name == 'crish' and cfg['crish']['enabled'],
        population_policy=cfg.get('population_policy', dict(enabled=False)),
        secondary='SS-v2 of best plain-test Old-v1' if cfg['report_best_old_test'] else None,
        protocol='matched_extended_200_epochs' if cfg['epochs'] == 200 else 'explicit_config_schedule',
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
        final_count = 2 if mode == 'train' and stage['report_best_old_test'] else 1
        subprocess.run(command, cwd=ROOT, check=True,
                       timeout=240+stage['max_runtime_seconds']+final_count*stage['final_eval_timeout_seconds'])


if __name__ == '__main__':
    main()
