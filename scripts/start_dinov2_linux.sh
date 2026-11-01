#!/usr/bin/env bash
# Run only after the previous v1 job has stopped; keep its artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
: "${GCD_DATA_ROOT:?Set GCD_DATA_ROOT}"
: "${GCD_WEIGHTS_ROOT:?Set GCD_WEIGHTS_ROOT}"
: "${GCD_RUNS_ROOT:?Set GCD_RUNS_ROOT}"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo 'GPU process present. Inspect and stop the previous job before migration; nothing killed.'; exit 2
fi
# A separate environment preserves the v1 environment for later reproduction.
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
.venv/bin/python -m pip install --only-binary=:all: --no-deps xformers==0.0.28.post1
.venv/bin/python -m pip check
export XFORMERS_DISABLED=1
# Use upstream PyTorch attention fallback consistently; xformers is still needed
# for its unconditional SwiGLU import. Actual GPU path must pass the gate.
weight="$GCD_WEIGHTS_ROOT/dinov2_vitb14_reg4_pretrain.pth"
if [[ ! -f "$weight" ]]; then
 curl --fail --location --connect-timeout 20 --max-time 600 --output "$weight.part" \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
 mv "$weight.part" "$weight"
fi
sha256sum "$weight"
.venv/bin/python -m gcd_runtime.launch preflight --config configs/cars_proposed_dino_v2.json
# Bounded active-reliability trial before authorizing a full-batch long run.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
cfg=json.loads(Path('configs/cars_proposed_dino_v2.json').read_text())
cfg.update(epochs=1,max_runtime_seconds=1800)
cfg['reliability'].update(warmup_epochs=0,ramp_epochs=0)
p=Path('configs/cars_proposed_dino_v2_trial.json')
if p.exists():
    assert json.loads(p.read_text())==cfg, 'Existing trial differs'
else:
    p.write_text(json.dumps(cfg,indent=2)+'\n')
PY
.venv/bin/python -m gcd_runtime.launch train --config configs/cars_proposed_dino_v2_trial.json
# The trial is operational only; full training starts fresh from DINOv2.
echo 'Active v2 trial passed. Starting proposed v2: 200 epochs, batch128, max86400s; run/log/PID printed below.'
exec .venv/bin/python -m gcd_runtime.launch train --config configs/cars_proposed_dino_v2.json
