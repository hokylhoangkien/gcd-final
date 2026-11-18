"""Prepare the six Table-II configurations without launching any jobs."""
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
from crish.ablation import VARIANTS
from gcd_runtime.config import code_fingerprint, load_config
from selex_runtime.config import validate_config


def prepare(base, output, seeds=(1, 2, 3), variants=VARIANTS):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Require unique nonnegative seeds')
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('Require unique supported variants')
    if str(base).endswith(('.zip', '.tar.gz')):
        from scripts.paper_registry import Source
        source = Source(base)
        try:
            template = validate_config(source.obj('config.json'))
        finally:
            source.close()
    else:
        template = load_config(base)
    if template['epochs'] != 200:
        raise ValueError('Paper plan requires 200 epochs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for seed in seeds:
        for variant in variants:
            cfg = copy.deepcopy(template)
            cfg.update(seed=seed, crish_ablation=variant)
            cfg = validate_config(cfg)
            name = f'{variant}_seed{seed}.json'
            payload = json.dumps(cfg, indent=2)+'\n'
            (output/name).write_text(payload)
            cmd = ['bash', 'scripts/train_selex.sh', '--config', str(output/name), '--variant', 'local']
            entries.append(dict(variant=variant, seed=seed, config=name,
                config_sha256=hashlib.sha256(payload.encode()).hexdigest(), command=cmd))
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                         text=True, timeout=10).stdout.strip()
    manifest = dict(status='prepared_not_executed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=sha, code_fingerprint=code_fingerprint(), base=str(base),
        base_sha256=hashlib.sha256(Path(base).read_bytes()).hexdigest(), entries=entries,
        cuda_verified=False, benchmark_metrics=None, primary='fixed_final_ss_v2')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'commands.txt').write_text('\n\n'.join(shlex.join(e['command']) for e in entries)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'configs/selex_scars_local_adaptive_200.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    args = parser.parse_args()
    manifest = prepare(args.base, args.output, args.seeds, args.variants)
    print(json.dumps(dict(status=manifest['status'], jobs=len(manifest['entries']))))
