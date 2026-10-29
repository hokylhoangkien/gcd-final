#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m gcd_runtime.launch train --config configs/cars_proposed_dino_v1.json "$@"
