#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-local}"
case "$mode" in local|cuda) ;; *) echo 'usage: bash scripts/bootstrap.sh local|cuda'; exit 2 ;; esac
if [[ "$mode" == cuda && "$(uname -s)" != Linux ]]; then
  echo 'CUDA environment is restricted to Linux NVIDIA hosts'; exit 2
fi
python_bin="${GCD_PYTHON:-python3}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] in ((3,9),(3,10),(3,11)), "Profile requires Python 3.9, 3.10 or 3.11"'
# Avoid replacing an existing environment; bootstrap only creates it when absent.
if [[ ! -d .venv ]]; then "$python_bin" -m venv .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] in ((3,9),(3,10),(3,11)), "Unsupported existing venv; select a separate checkout/environment"'
export PIP_DEFAULT_TIMEOUT=30
export PIP_RETRIES=1
.venv/bin/python -m pip install pip==24.3.1
if [[ "$mode" == cuda ]]; then
  .venv/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
else
  .venv/bin/python -m pip install torch==2.4.1 torchvision==0.19.1
fi
.venv/bin/python -m pip install -r environments/requirements-common.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/run_smoke.py
# No data/weights are downloaded and no training is started.
