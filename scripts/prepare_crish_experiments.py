"""Prepare A1/A2/policy controls; never install, download or launch training."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.config import load_config, code_fingerprint
from crish.population_settings import PopulationSettings
from selex_runtime.config import validate_config


def variants(template, suite='all'):
    """Axes change one declared field relative to the same local reference."""
    reference = copy.deepcopy(template)
    reference['method'] = 'crish'
    reference['pair_reliability']['enabled'] = False
    reference['population_policy']['enabled'] = False
    reference['crish'].update(enabled=True, boundaries=4, max_accepted=1,
                              evidence_samples=128, refresh_every=5, probe_known_per_class=2)
    result = {}
    if suite in ('all', 'dose'):
        result['local_reference'] = reference
        for key, values in (('max_accepted', (2, 4)), ('refresh_every', (1, 2)),
                            ('evidence_samples', (64, 256)), ('boundaries', (2, 8))):
            for value in values:
                cfg = copy.deepcopy(reference); cfg['crish'][key] = value
                result[f'local_{key}_{value}'] = cfg
    if suite in ('all', 'controls'):
        legacy = copy.deepcopy(reference); legacy['crish']['probe_known_per_class'] = 1
        result['known_16way_1shot'] = legacy
        result['known_8way_2shot'] = copy.deepcopy(reference)
        for mode in ('counterfactual', 'fixed', 'scheduled', 'random'):
            cfg = copy.deepcopy(template)
            cfg['method'] = 'crish'; cfg['crish']['enabled'] = False
            cfg['pair_reliability']['enabled'] = False
            cfg['population_policy'] = PopulationSettings(enabled=True, mode=mode).to_dict()
            result[f'policy_{mode}'] = cfg
        for name, policy in (('online_only', False), ('policy_online_hybrid', True)):
            cfg = copy.deepcopy(template); cfg['method'] = 'crish'
            cfg['crish']['enabled'] = False
            cfg['population_policy']['enabled'] = policy
            cfg['pair_reliability']['enabled'] = True
            result[name] = cfg
    # Baseline matches the local reference except the method switch. All its
    # intervention settings are ignored, rather than using different SGD/data.
    baseline = copy.deepcopy(reference); baseline['method'] = 'baseline'
    result['baseline'] = baseline
    return {name: validate_config(cfg) for name, cfg in result.items()}


def prepare(dataset, seeds, output=None, suite='all'):
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    run_id = 'crish-experiments-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output = Path(output) if output else ROOT/'runs/plans'/run_id
    output.mkdir(parents=True, exist_ok=False)
    base = load_config(ROOT/f'configs/selex_{dataset}_crish_200.json')
    version = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                             text=True, timeout=10).stdout.strip() or 'source_archive'
    entries, commands = [], []
    for seed in seeds:
        for name, cfg in variants(base, suite).items():
            cfg = dict(cfg, seed=seed, experiment_name=name)
            path = output/f'{name}_seed_{seed}.json'
            payload = json.dumps(cfg, indent=2)+'\n'; path.write_text(payload)
            command = ['bash', 'scripts/train_selex.sh', '--config', str(path)]
            if cfg['method'] == 'baseline': command.append('--baseline')
            entries.append(dict(name=name, seed=seed, config=path.name,
                config_sha256=hashlib.sha256(payload.encode()).hexdigest(), command=command))
            commands.append('# '+name+f', seed {seed}; execute ONE prepared job at a time\n'+shlex.join(command))
    manifest = dict(run_id=run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status='prepared_not_executed', suite=suite, dataset=dataset, seeds=seeds,
        git_sha=version, code_fingerprint=code_fingerprint(), entries=entries,
        primary='fixed_final_ss_v2', benchmark_metrics=None)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'commands.txt').write_text('\n\n'.join(commands)+'\n')
    return dict(status=manifest['status'], run_id=run_id, directory=str(output), jobs=len(entries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['scars', 'cub', 'aircraft', 'cifar100'], default='scars')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--suite', choices=['all', 'dose', 'controls'], default='all')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.seeds, args.output, args.suite), indent=2))
