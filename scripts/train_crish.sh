#!/usr/bin/env bash
# Default: CRiSH preflight -> bounded real-data smoke -> fresh full training.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv. On Linux NVIDIA: GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda'
  exit 2
fi
exec .venv/bin/python -m gcd_runtime.train_cli "$@"
