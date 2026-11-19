"""Prepare CRiSH-G (global reliability policy) profiles; never launch jobs or download inputs."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.config import code_fingerprint, load_config
from selex_runtime.config import validate_config

PROFILES = ('g_fixed', 'g_cf', 'g_cf_move', 'g_random', 'g_cf_nopersist')
OPTIONAL = ('g_cf_local',)


def common(template):
    if template.get('baseline_family') != 'selex':
        raise ValueError('CRiSH-G profiles require the pinned SelEx path')
    cfg = copy.deepcopy(template)
    cfg['method'] = 'crish'
    cfg['crish']['enabled'] = False
    if 'pair_reliability' in cfg:
        cfg['pair_reliability']['enabled'] = False
    cfg.pop('crish_ablation', None)
    policy = dict(cfg.get('population_policy', {}))
    policy.update(enabled=True, ranking='knn_agreement', selection='threshold', levels='active',
                  persist=True, warmup_epochs=10, refresh_every=5, scoring_batches=4,
                  scoring_sampling='train_sampler', probe_float64=True, save_arrays=True,
                  thresholds=[0., .3, .5, .7], fixed_threshold=.5, sample_choice='evidence',
                  actions='abstain', gate='counterfactual', mode='counterfactual', compose_local=False)
    cfg['population_policy'] = policy
    return cfg


def global_variants(template, profiles=PROFILES):
    base = common(template)
    out = {}
    def make(**changes):
        cfg = copy.deepcopy(base)
        cfg['population_policy'].update(changes)
        return cfg
    builders = dict(
        g_fixed=lambda: make(mode='fixed', gate='none', fixed_threshold=.5, sample_choice='evidence'),
        g_cf=lambda: make(),
        g_cf_move=lambda: make(actions='abstain_move'),
        g_random=lambda: make(mode='fixed', gate='none', fixed_threshold=.5, sample_choice='random'),
        g_cf_nopersist=lambda: make(persist=False),
        g_cf_local=lambda: local(make(compose_local=True)))
    for name in profiles:
        if name not in builders:
            raise ValueError(f'Unknown CRiSH-G profile: {name}')
        out[name] = validate_config(builders[name]())
    return out


def local(cfg):
    """Same local adaptive budgets as scripts/prepare_local_adaptive.py."""
    cfg['crish'].update(enabled=True, max_accepted=8, confirmation_budget=8, boundaries=16,
        train_batch=128, evidence_samples=128, evidence_trials=2, k=5,
        probe_known=16, probe_unlabeled=32, probe_sampling='class_balanced',
        probe_known_per_class=2, max_refresh_seconds=120)
    return cfg


def prepare(base, output, seeds=None, profiles=PROFILES):
    template = load_config(base)
    seeds = [template['seed']] if not seeds else list(seeds)
    if len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for seed in seeds:
        for name, cfg in global_variants(dict(template, seed=seed), profiles).items():
            path = output/f'{name}_seed{seed}.json'
            payload = json.dumps(cfg, indent=2)+'\n'
            path.write_text(payload)
            variant = 'global_local' if cfg['population_policy'].get('compose_local') else 'global'
            command = ['bash', 'scripts/train_selex.sh', '--config', str(path), '--variant', variant,
                       '--eval-every', '1']
            entries.append(dict(name=name, seed=seed, config=path.name, variant=variant,
                sha256=hashlib.sha256(payload.encode()).hexdigest(), command=command))
    version = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
        text=True, timeout=10).stdout.strip() or 'source_archive'
    manifest = dict(status='prepared_not_executed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=version, code_fingerprint=code_fingerprint(), base=str(base), seeds=seeds,
        profiles=list(profiles), entries=entries, dataset=template['dataset_name'],
        epochs=template['epochs'], cuda_verified=False, benchmark_metrics=None)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'commands.txt').write_text('\n\n'.join(
        f'# {entry["name"]} seed{entry["seed"]}; run one job at a time\n'+shlex.join(entry['command'])
        for entry in entries)+'\n')
    return dict(status=manifest['status'], directory=str(output), jobs=len(entries), profiles=list(profiles))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'configs/selex_scars_crish_200.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--profiles', nargs='+', choices=PROFILES+OPTIONAL, default=list(PROFILES))
    parser.add_argument('--with-local', action='store_true', help='Also write optional g_cf_local')
    args = parser.parse_args()
    profiles = tuple(args.profiles)+(OPTIONAL if args.with_local and 'g_cf_local' not in args.profiles else ())
    print(json.dumps(prepare(args.base, args.output, args.seeds, profiles), indent=2))
