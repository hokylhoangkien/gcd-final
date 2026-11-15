"""Prepare matched SelEx local profiles; never launch jobs or download inputs."""
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


def local_variants(template):
    """Cap controls reserve identical probe rows, evidence and scoring budgets."""
    if template.get('baseline_family') != 'selex':
        raise ValueError('Local adaptive profiles require the pinned SelEx path')
    legacy = copy.deepcopy(template)
    legacy['method'] = 'crish'
    legacy['crish']['enabled'] = True
    for key in ('population_policy', 'pair_reliability'):
        if key in legacy:
            legacy[key]['enabled'] = False
    adaptive = copy.deepcopy(legacy)
    adaptive['crish'].update(max_accepted=8, confirmation_budget=8, boundaries=16,
        train_batch=128, evidence_samples=128, evidence_trials=2, k=5,
        probe_known=16, probe_unlabeled=32, probe_sampling='class_balanced',
        probe_known_per_class=2, max_refresh_seconds=120)
    context = copy.deepcopy(adaptive)
    context['crish']['max_accepted'] = 1
    baseline = copy.deepcopy(adaptive)
    baseline['method'] = 'baseline'
    return {name: validate_config(cfg) for name, cfg in
            [('baseline', baseline), ('legacy', legacy), ('cap1_context', context), ('adaptive', adaptive)]}


def prepare(base, output):
    configs = local_variants(load_config(base))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, cfg in configs.items():
        path = output/f'{name}.json'
        payload = json.dumps(cfg, indent=2)+'\n'
        path.write_text(payload)
        command = ['bash', 'scripts/train_selex.sh', '--config', str(path)]
        command += ['--baseline'] if cfg['method'] == 'baseline' else ['--variant', 'local']
        entries.append(dict(name=name, config=path.name,
            sha256=hashlib.sha256(payload.encode()).hexdigest(), command=command))
    version = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
        text=True, timeout=10).stdout.strip() or 'source_archive'
    manifest = dict(status='prepared_not_executed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=version, code_fingerprint=code_fingerprint(), entries=entries,
        dataset=configs['adaptive']['dataset_name'], epochs=configs['adaptive']['epochs'],
        cuda_verified=False, benchmark_metrics=None)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'commands.txt').write_text('\n\n'.join(
        '# '+entry['name']+'; run one job at a time\n'+shlex.join(entry['command'])
        for entry in entries)+'\n')
    return dict(status=manifest['status'], directory=str(output), profiles=list(configs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'configs/selex_scars_crish_200.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.output), indent=2))
