#!/usr/bin/env bash
# Fail if the requested run has no resumable epoch checkpoint.
set -euo pipefail
cd "${REL_REPO:-$(dirname "$0")/..}"
variant=${1:?Usage: bash scripts/resume_relational.sh R0|R1|R2 [controller options]}
case "$variant" in R0|R1|R2) ;; *) exit 2 ;; esac
root=${REL_RUN_ROOT:-$HOME/runs/crish-relational}
if ! compgen -G "$root/$variant/epoch-[0-9][0-9][0-9].pt" >/dev/null; then
    echo "No committed checkpoint found in $root/$variant" >&2
    exit 1
fi
exec bash scripts/run_relational.sh "$@"
