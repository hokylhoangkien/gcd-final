#!/usr/bin/env bash
# Start or resume one isolated Linux session; the controller reads committed epochs.
set -euo pipefail
cd "${REL_REPO:-$(dirname "$0")/..}"
variant=${1:?Usage: bash scripts/run_relational.sh R0|R1|R2 [controller options]}
shift
case "$variant" in R0|R1|R2) ;; *) exit 2 ;; esac
: "${SIMGCD_CARS_ROOT:?Set official Cars layout}" "${SIMGCD_DINO_REPO:?Set local facebookresearch/dino checkout}" "${SIMGCD_DINO_WEIGHTS:?Set DINOv1 ViT-B/16 weights}"
test -f "$SIMGCD_CARS_ROOT/devkit/cars_train_annos.mat"
test -f "$SIMGCD_CARS_ROOT/devkit/cars_test_annos_withlabels.mat"
test -f "$SIMGCD_DINO_WEIGHTS"
test -f "$SIMGCD_DINO_REPO/hubconf.py"
root=$(realpath -m "${REL_RUN_ROOT:-$HOME/runs/crish-relational}")
mkdir -p "$root/$variant"
session="${REL_SESSION:-crish-rel-$variant}"
if tmux has-session -t "=$session" 2>/dev/null; then echo "Session $session already exists"; exit 1; fi
# tmux may have an older environment than the current SSH shell.
printf -v command '%q ' env "SIMGCD_CARS_ROOT=$(realpath "$SIMGCD_CARS_ROOT")" \
    "SIMGCD_DINO_REPO=$(realpath "$SIMGCD_DINO_REPO")" "SIMGCD_DINO_WEIGHTS=$(realpath "$SIMGCD_DINO_WEIGHTS")" \
    "PATH=$PATH" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}" CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout --signal=TERM --kill-after=90s "${REL_TIMEOUT:-48h}" \
    "$(command -v python)" -u -m relational.run "$variant" --root "$root" "$@"
printf -v logfile '%q' "$root/$variant/controller.log"
tmux new-session -d -s "$session" -c "$PWD" "$command >>$logfile 2>&1"
echo "tmux attach -t $session"
echo "tail -f '$root/$variant/training.log'"
