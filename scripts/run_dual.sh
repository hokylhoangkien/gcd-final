#!/usr/bin/env bash
# Start/resume D1 in an SSH-safe tmux session. The controller owns milestone decisions.
set -euo pipefail
cd "${D1_REPO:-$(dirname "$0")/..}"
: "${SIMGCD_CARS_ROOT:?Set official Cars layout}" "${SIMGCD_DINO_REPO:?Set local DINOv1 source}" "${SIMGCD_DINO_WEIGHTS:?Set official DINOv1 ViT-B/16 weights}" "${D1_CLIP_CACHE:?Set verified CLIP .npz path}"
test -f "$SIMGCD_CARS_ROOT/devkit/cars_train_annos.mat"
test -f "$SIMGCD_CARS_ROOT/devkit/cars_test_annos_withlabels.mat"
test -f "$SIMGCD_DINO_REPO/hubconf.py"
test -f "$SIMGCD_DINO_WEIGHTS"
test -f "$D1_CLIP_CACHE"
root=$(realpath -m "${D1_RUN_ROOT:-$HOME/runs/crish-dual}")
session=${D1_SESSION:-crish-d1}
mkdir -p "$root/D1"
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Session $session already exists" >&2
    exit 1
fi
printf -v command '%q ' env "SIMGCD_CARS_ROOT=$(realpath "$SIMGCD_CARS_ROOT")" \
    "SIMGCD_DINO_REPO=$(realpath "$SIMGCD_DINO_REPO")" "SIMGCD_DINO_WEIGHTS=$(realpath "$SIMGCD_DINO_WEIGHTS")" \
    "PATH=$PATH" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    timeout --signal=TERM --kill-after=90s "${D1_TIMEOUT:-72h}" "$(command -v python)" -u -m dual.run \
    --root "$root" --cache "$(realpath "$D1_CLIP_CACHE")" "$@"
printf -v logfile '%q' "$root/D1/controller.log"
tmux new-session -d -s "$session" -c "$PWD" "$command >>$logfile 2>&1"
echo "tmux attach -t $session"
echo "tail -f '$root/D1/training.log'"
