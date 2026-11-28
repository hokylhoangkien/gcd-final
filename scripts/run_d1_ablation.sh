#!/usr/bin/env bash
# One stage per SSH-safe tmux session. Reuse the same run root for exact resume.
set -euo pipefail

case "${1:-}" in
    50|100|150|200) milestone=$1; shift ;;
    *) echo "Usage: bash scripts/run_d1_ablation.sh {50|100|150|200} [--micro]" >&2; exit 2 ;;
esac
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --micro ]]; }; then
    echo "Only --micro is allowed after the milestone" >&2
    exit 2
fi
cd "${D1_ABLATION_REPO:-$(dirname "$0")/..}"
: "${SIMGCD_CARS_ROOT:?Set the official Cars root}" "${SIMGCD_DINO_REPO:?Set the local DINOv1 source}"
: "${SIMGCD_DINO_WEIGHTS:?Set the official DINOv1 weights}" "${D1_CLIP_CACHE:?Set the verified CLIP cache}"
: "${D1_ABLATION_ROOT:?Set a persistent, dedicated production run directory}"

test -f "$SIMGCD_CARS_ROOT/devkit/cars_train_annos.mat"
test -f "$SIMGCD_CARS_ROOT/devkit/cars_test_annos_withlabels.mat"
test -f "$SIMGCD_DINO_REPO/hubconf.py"
test -f "$SIMGCD_DINO_WEIGHTS"
test -f "$D1_CLIP_CACHE"
root=$(realpath -m "$D1_ABLATION_ROOT")
session=${D1_ABLATION_SESSION:-crish-d1-ablation}
python=$(command -v "${D1_ABLATION_PYTHON:-python}")
mkdir -p "$root/logs"
log=$(mktemp "$root/logs/epoch-$milestone.XXXXXX.log")

# Do not depend on environment variables inherited from an old tmux server.
cmd=(env "SIMGCD_CARS_ROOT=$(realpath "$SIMGCD_CARS_ROOT")"
    "SIMGCD_DINO_REPO=$(realpath "$SIMGCD_DINO_REPO")"
    "SIMGCD_DINO_WEIGHTS=$(realpath "$SIMGCD_DINO_WEIGHTS")"
    "PATH=$PATH" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
    "OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}" "MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
    timeout --signal=TERM --kill-after=90s "${D1_ABLATION_TIMEOUT:-72h}"
    "$python" -u -m dual.ablation_ops --root "$root"
    --clip "$(realpath "$D1_CLIP_CACHE")" --until "$milestone" "$@")
printf -v invocation '%q ' "${cmd[@]}"
printf -v redirect '%q' "$log"
tmux new-session -d -s "$session" -c "$PWD" bash -c "exec $invocation >>$redirect 2>&1"
printf 'tmux session %s\nPersistent log %s\nStage status %s/status.json\n' "$session" "$log" "$root"
