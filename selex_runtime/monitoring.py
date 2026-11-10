"""Bounded plain-test monitoring without test-dependent checkpoint selection."""
import json
import os
import subprocess
import sys
import torch
from gcd_runtime.config import ROOT


def monitor_plain(runtime, epoch, model):
    snapshot = runtime.run_dir/'backbone_monitor.pt'
    temporary = snapshot.with_suffix('.tmp')
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
    os.replace(temporary, snapshot)
    cfg_path = runtime.run_dir/'monitor_config.json'
    cfg_path.write_text(json.dumps(runtime.config, indent=2))
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.empty_cache()
    from .rng import upstream_policy, create_request, commit_plain_result
    request = None
    if upstream_policy(runtime.config):
        from gcd_runtime.readiness import sha256
        runtime.plain_rng_pending = True
        request = create_request(runtime, epoch, sha256(snapshot))
    subprocess.run([sys.executable, '-m', 'gcd_runtime.worker', str(cfg_path),
        str(runtime.run_dir), 'select', str(snapshot), str(epoch)], cwd=ROOT,
        timeout=runtime.config['ss_eval_timeout_seconds'], check=True)
    if not (runtime.run_dir/'test_evaluations'/f'epoch_{epoch+1:04d}.json').is_file():
        raise RuntimeError('SelEx plain monitor produced no report; resume checkpoint preserved')
    if request is not None:
        commit_plain_result(runtime, request)
        runtime.plain_rng_pending = False
