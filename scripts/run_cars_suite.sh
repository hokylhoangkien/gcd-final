#!/usr/bin/env bash
# Run from any directory on the Linux GPU host.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=false
if [[ ${1:-} == --dry-run ]]; then dry=true; shift; fi
if (( $# )); then echo "Usage: bash scripts/run_cars_suite.sh [--dry-run]" >&2; exit 2; fi
seed=${SIMGCD_SEED:-1}
root=${SIMGCD_RUN_ROOT:-dev_outputs/cars-suite}
work="$root/simgcd/log"
reference="$work/cars-E3-fixed-s$seed/checkpoints/crish-global.jsonl"
if ! "$dry"; then
    : "${SIMGCD_CARS_ROOT:?Set SIMGCD_CARS_ROOT to the official-layout Stanford Cars directory}"
    test -d "$SIMGCD_CARS_ROOT/cars_train"
    test -f "$SIMGCD_CARS_ROOT/devkit/cars_train_annos.mat"
    test -f "$SIMGCD_CARS_ROOT/devkit/cars_test_annos_withlabels.mat"
    if [[ -n ${SIMGCD_DINO_REPO:-} ]]; then test -f "${SIMGCD_DINO_WEIGHTS:?Set pretrained ViT-B/16 weights}"; fi
    mkdir -p "$root"
    exec 9>"$root/.cars-suite-s$seed.lock"
    flock -n 9 || { echo "Another suite owns this output root and seed." >&2; exit 1; }
fi
run() {
    local mode=$1 name=$2 cap=$3
    local opts=(--crish_global_suite --resume_run --crish_mode "$mode" --crish_cap "$cap"
                --exp_name "cars-$name-s$seed" --exp_root "$root" --run_seed "$seed")
    if [[ $mode == E1 || $mode == E2 ]]; then opts+=(--crish_dose "$reference"); fi
    if "$dry"; then printf '%q ' bash scripts/run_cars_crish.sh "${opts[@]}"; printf '\n'
    else bash scripts/run_cars_crish.sh "${opts[@]}"; fi
}
run E4 E4-adaptive 8
run E3 E3-fixed 1
run E1 E1-random 1
run E2 E2-evidence 1
if ! "$dry"; then python scripts/summarize_cars_suite.py "$work" "$seed"; fi
