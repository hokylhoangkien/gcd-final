#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit('Use a Linux NVIDIA image with Python 3.10, 3.11 or 3.12. Set PYTHON_BIN if needed.')
PY
command -v nvidia-smi >/dev/null || { echo 'An NVIDIA GPU host is required'; exit 1; }
nvidia-smi
"$PYTHON_BIN" -m venv .venv-rpc
.venv-rpc/bin/python -m pip install --upgrade pip
.venv-rpc/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv-rpc/bin/python -m pip install -r requirements-rpc.txt
.venv-rpc/bin/python - <<'PY'
import torch, torchvision
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable after installation. Check the NVIDIA driver/CUDA image before training.')
x = torch.randn(16, 16, device='cuda', requires_grad=True)
x.square().mean().backward()
torch.cuda.synchronize()
print('CUDA forward/backward passed:', torch.cuda.get_device_name(), torch.__version__, torchvision.__version__)
PY
.venv-rpc/bin/python -m unittest discover -s tests_rpc -v
.venv-rpc/bin/python validate_global_crish.py
.venv-rpc/bin/python -m planogram_rpc.assets
