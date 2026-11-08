"""Synchronous, bounded SS evaluators; no oracle metrics return to training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import ROOT


def monitor_due(config, epoch):
    every = config.get('ss_eval_every_epochs', 0)
    if not every:
        return False
    if epoch + 1 == config['epochs']:
        # The launcher evaluates the fixed final checkpoint after training exits.
        return not config['final_ss_evaluation']
    return (epoch + 1) % every == 0


def monitor_epoch(runtime, epoch, model, *, selection_only=False):
    from .selection import enabled
    modes = ['select'] if enabled(runtime.config) else []
    if not selection_only and monitor_due(runtime.config, epoch):
        modes.append('monitor')
    if not modes:
        return
    import torch
    run_dir = Path(runtime.run_dir).resolve()
    checkpoint = run_dir / 'backbone_monitor.pt'
    temporary = run_dir / 'backbone_monitor.tmp'
    # Keep one snapshot, not hundreds of full backbones. CPU serialization must
    # neither change model modes/parameters nor consume the training RNG.
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
    os.replace(temporary, checkpoint)
    config_path = run_dir / 'monitor_config.json'
    config_path.write_text(json.dumps(runtime.config, indent=2))
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    timeout = runtime.config['ss_eval_timeout_seconds']
    for mode in modes:
        command = [sys.executable, '-m', 'gcd_runtime.worker', str(config_path),
                   str(run_dir), mode, str(checkpoint), str(epoch)]
        print(f'EVAL start kind={mode} epoch={epoch+1}/{runtime.config["epochs"]} '
              f'timeout={timeout}s checkpoint={checkpoint} command={json.dumps(command)}', flush=True)
        start = time.monotonic()
        try:
            # Inherit run.log. Training never reads evaluator scores.
            subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)
            folder = 'test_evaluations' if mode == 'select' else 'evaluations'
            report = run_dir/folder/f'epoch_{epoch+1:04d}.json'
            if not report.is_file():
                raise RuntimeError(f'Evaluator exited without an epoch report: {report}')
        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
            print(f'EVAL failed kind={mode} epoch={epoch+1}; resume.pt and snapshot preserved: {exc}', flush=True)
            raise
        finally:
            print(f'EVAL worker_seconds={time.monotonic()-start:.3f} kind={mode} epoch={epoch+1}', flush=True)
