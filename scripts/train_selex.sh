#!/usr/bin/env bash
# Updated SelEx + CRiSH; --baseline selects its matched SelEx control.
set -euo pipefail
cd "$(dirname "$0")/.."
selex_python="${SELEX_PYTHON:-.venv/bin/python}"
if [[ ! -x "$selex_python" ]]; then
  echo 'Missing Python environment; set SELEX_PYTHON to the prepared interpreter.'
  exit 2
fi
exec "$selex_python" -m selex_runtime.cli "$@"
