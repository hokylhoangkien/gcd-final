"""Prepare an explicit SS-cadence-only resume bundle after the source job stops."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from gcd_runtime.config import code_fingerprint, load_config
from gcd_runtime.readiness import sha256
from gcd_runtime.ss_best import restore_best
from selex_runtime.selection import BEST_NAME, read_best, restore_artifacts, write_selection


def assert_stopped(source):
    """Refuse migration while any worker/evaluator references this run on Linux."""
    proc = Path('/proc')
    if not proc.is_dir():
        raise RuntimeError('Run migration on the Linux training host after stopping the source job')
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if str(source) in command and any(name in command for name in ('gcd_runtime.worker', 'gcd_runtime.remote_gate')):
            raise RuntimeError(f'Source run still has worker/evaluator PID {entry.name}; no checkpoint changed')


def prepare(source, output, every=1, *, stopped_check=assert_stopped):
    source, output = Path(source).resolve(), Path(output).resolve()
    if type(every) is not int or every < 1:
        raise ValueError('SS interval must be a positive integer')
    if source == output or output.exists():
        raise ValueError('Output must be a new directory; source is never overwritten')
    stopped_check(source)
    checkpoint = source/'resume.pt'
    original_hash = sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    original = load_config(source/'config.json')
    if (original.get('baseline_family') != 'selex' or state['config'] != original
            or state['code_fingerprint'] != code_fingerprint()):
        raise ValueError('Original source/config do not match checkpoint; refusing migration')
    if original['ss_eval_every_epochs'] == every:
        raise ValueError('Checkpoint already uses the requested SS cadence')
    if state['epoch']+1 >= original['epochs']:
        raise ValueError('Training is complete; evaluate the saved checkpoint instead')
    pending = state.get('plain_rng_pending', False)
    target = copy.deepcopy(original)
    target['ss_eval_every_epochs'] = every
    output.mkdir(parents=True, exist_ok=False)
    # Validate/copy the complete original evaluation history before adapting its
    # config-bound best bundle. Never weaken the runtime's strict resume guards.
    restore_best(source, output, original, state['epoch'])
    restore_artifacts(source, output, original, state['epoch'], pending=pending)
    if (output/BEST_NAME).is_file():
        best = read_best(output/BEST_NAME, original)
        best['config'] = target
        torch.save(best, output/BEST_NAME)
        write_selection(output, best)
    stopped_check(source)
    if sha256(checkpoint) != original_hash:
        raise RuntimeError('Source checkpoint changed during migration; do not use output')
    manifest = dict(kind='ss_cadence_only_migration', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_run=str(source), source_checkpoint_sha256=original_hash,
        saved_epoch=state['epoch'], completed_epochs=state['epoch']+1,
        old_interval=original['ss_eval_every_epochs'], new_interval=every,
        changed_config_fields=['ss_eval_every_epochs'], original_config=original,
        state_preserved=['model', 'head', 'optimizer', 'scheduler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'],
        historical_missing_ss='not reconstructed; resumed runtime evaluates latest saved epoch if due')
    state['config'] = target
    state['ss_cadence_migration'] = manifest
    torch.save(state, output/'resume.pt')
    (output/'config.json').write_text(json.dumps(target, indent=2)+'\n')
    (output/'migration.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--every', type=int, default=1)
    options = parser.parse_args()
    print(json.dumps(prepare(options.source_run, options.output, options.every), indent=2))
