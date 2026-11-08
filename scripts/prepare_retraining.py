"""Freeze matched configs and commands for manual, sequential GPU reruns.

This prepares a plan only. It never downloads inputs or starts a worker/sweep.
Generate the plan again on the target Linux checkout before executing commands.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.config import load_config, code_fingerprint
from gcd_runtime.datasets import dataset_spec
from gcd_runtime.readiness import verify_upstream


def prepare(dataset, profile, seeds, selection=None, output=None):
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    prefix = '' if dataset == 'scars' else dataset+'_'
    suffix = '' if profile == 'reference' else '_upgraded'
    templates = {(method, smoke): load_config(ROOT/'configs'/
                 f'{prefix}{method}{suffix}{"_smoke" if smoke else ""}.json')
                 for method in ('baseline', 'crish') for smoke in (False, True)}
    selection = selection or templates['baseline', False]['checkpoint_selection']
    plan_id = 'matched-'+dataset+'-'+profile+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    directory = Path(output) if output else ROOT/'runs/plans'/plan_id
    directory.mkdir(parents=True, exist_ok=False)
    directory = directory.resolve()
    provenance = verify_upstream()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True, timeout=10)
    spec = dataset_spec(dataset)
    manifest = dict(plan_id=plan_id, status='prepared_not_executed', dataset=dataset,
        profile=profile, seeds=seeds, checkpoint_selection=selection,
        upstream=provenance, git_sha=revision.stdout.strip() if revision.returncode == 0 else 'verified_release',
        code_fingerprint=code_fingerprint(), split_sha256=spec.split_sha256,
        pretraining='dinov2_vitb14_reg4_pretrain.pth', final_clustering_n_init=1,
        final_clustering_max_iterations=10, primary_metric='All', executions=[])
    commands = ['# Prepared only; run one command at a time on Linux NVIDIA.',
                '# Confirm data/weights and inspect nvidia-smi before starting each job.',
                '# Do not run these commands concurrently or change configs during a run.']
    for seed in seeds:
        pair = {}
        for method in ('baseline', 'crish'):
            paths = {}
            for smoke in (False, True):
                cfg = json.loads(json.dumps(templates[method, smoke]))
                cfg.update(seed=seed, checkpoint_selection=selection,
                           test_every_epochs=1 if selection == 'best_old_test_v2' else 0)
                name = f'{method}-seed-{seed}'+('-smoke' if smoke else '')+'.json'
                path = directory/name
                path.write_text(json.dumps(cfg, indent=2)+'\n')
                cfg = load_config(path)
                paths['smoke' if smoke else 'full'] = path
                if not smoke:
                    pair[method] = cfg
            job = dict(seed=seed, method=method, stages=[])
            commands.append(f'\n# {method}, seed={seed}: preflight -> bounded real-data smoke -> full run')
            for mode, config in (('preflight', paths['full']), ('train', paths['smoke']), ('train', paths['full'])):
                target = str(config.relative_to(ROOT)) if config.is_relative_to(ROOT) else str(config)
                command = ['.venv/bin/python', '-m', 'gcd_runtime.launch', mode, '--config', target]
                job['stages'].append(dict(mode=mode, config=target, command=command))
                commands.append(shlex.join(command))
            manifest['executions'].append(job)
        if {key for key in pair['baseline'].keys() | pair['crish'].keys()
                if pair['baseline'].get(key) != pair['crish'].get(key)} != {'method'}:
            raise ValueError('Prepared pair is not matched; no training was started')
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Deliberately not an executable shell script: these are reviewable commands,
    # not an automatic multi-seed GPU sweep.
    (directory/'commands.txt').write_text('\n'.join(commands)+'\n')
    print(json.dumps(dict(status=manifest['status'], directory=str(directory.relative_to(ROOT))
        if directory.is_relative_to(ROOT) else str(directory), matched_pairs=len(seeds),
        train_runs=2*len(seeds), selection=selection, next='Inspect commands.txt on Linux; execute one stage at a time',
        resources='batch 128, 200 epochs; host VRAM/data/weights/CUDA not verified by this preparer'), indent=2))
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['scars', 'cub'], default='scars')
    parser.add_argument('--profile', choices=['reference', 'upgraded'], default='reference')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--selection', choices=['best_old_test_v2', 'final_epoch'],
                        help='Default inherits the shipped protocol; do not change after seeing scores')
    parser.add_argument('--output', type=Path, help='New directory only; defaults to ignored runs/plans/')
    args = parser.parse_args()
    prepare(args.dataset, args.profile, args.seeds, args.selection, args.output)
