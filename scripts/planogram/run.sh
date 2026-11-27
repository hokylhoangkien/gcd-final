#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${RPC_PYTHON:-.venv-rpc/bin/python}"
RPC_DATA="${RPC_DATA:-datasets/rpc40}"
RPC_RUN="${RPC_RUN:-runs/rpc40-e3-s1}"
RPC_ASSETS="${RPC_ASSETS:-assets_local}"
action="${1:-help}"
if (($#)); then shift; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
case "$action" in
  prepare)
    "$PYTHON_BIN" -m planogram_rpc.prepare --root "$RPC_DATA" "$@"
    ;;
  preflight)
    "$PYTHON_BIN" -m planogram_rpc.prepare --root "$RPC_DATA" --verify-only
    timeout --signal=TERM --kill-after=1m 20m "$PYTHON_BIN" -m planogram_rpc.train --data "$RPC_DATA" --assets "$RPC_ASSETS" --preflight "$@"
    ;;
  check-repair)
    # Exercise real E3 immediately in an independent bounded run.
    timeout --signal=TERM --kill-after=5m 1h "$PYTHON_BIN" -m planogram_rpc.train --data "$RPC_DATA" --assets "$RPC_ASSETS" \
      --output "${RPC_RUN}-repair-check" --crish_warmup 0 --stop-after-epochs 1 --max-hours 0.75 "$@"
    ;;
  train|pilot|resume)
    mkdir -p logs_rpc
    args=(--data "$RPC_DATA" --assets "$RPC_ASSETS" --output "$RPC_RUN")
    if [[ "$action" == pilot ]]; then args+=(--stop-after-epochs 1); fi
    if [[ "$action" == resume ]]; then args+=(--resume); fi
    log="logs_rpc/$(basename "$RPC_RUN")-${action}-$(date -u +%Y%m%dT%H%M%SZ).log"
    echo "Run: $RPC_RUN | log: $log | 200 epochs, batch 128 by default | bounded to 24h per invocation"
    timeout --signal=TERM --kill-after=10m 24h "$PYTHON_BIN" -m planogram_rpc.train "${args[@]}" --max-hours 23.5 "$@" 2>&1 | tee "$log"
    ;;
  evaluate)
    timeout --signal=TERM --kill-after=2m 4h "$PYTHON_BIN" -m planogram_rpc.evaluate --data "$RPC_DATA" --assets "$RPC_ASSETS" --run "$RPC_RUN" "$@"
    ;;
  export)
    "$PYTHON_BIN" -m planogram_rpc.export --assets "$RPC_ASSETS" --run "$RPC_RUN" "$@"
    ;;
  *)
    echo 'Usage: bash scripts/planogram/run.sh {prepare|preflight|check-repair|pilot|train|resume|evaluate|export} [extra CLI args]'
    [[ "$action" == help ]]
    ;;
esac
