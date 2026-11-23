# CRiSH-G v2 runbook

Run from the `feat/crish-g-v2` checkout on Linux/NVIDIA. Keep other running jobs
in their original checkouts. The Mac implementation worktree is separate from
the owner's manuscript checkout. Preregistration: report 25, commit `f70279b`.
Freeze the final source commit, all generated configs and their manifest before
G1. Do not change source or config while a job is running or before resume.

## Environment and Phase 0

```bash
git submodule update --init --recursive
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
export GCD_DATA_ROOT=/path/to/datasets
export GCD_WEIGHTS_ROOT=/path/to/weights
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/export_offline_labels.py --dataset scars --output /tmp/scars_labels.csv
```

Extract any supplied archive into a new analysis directory; never replace a run.
Paths below are actual run directories containing NPZ/checkpoints, not summary
imports. Scripts fail rather than reconstruct missing partitions or weights.

```bash
.venv/bin/python scripts/prepare_posthoc_snapshots.py "$LOCAL_RUN" \
  --labels /tmp/scars_labels.csv --output /tmp/local-snapshots-NEW.jsonl
.venv/bin/python scripts/paper_posthoc.py "$LOCAL_RUN" \
  --snapshots /tmp/local-snapshots-NEW.jsonl --output /tmp/local-audit-NEW.json
.venv/bin/python scripts/audit_probe_scale.py "$LOCAL_RUN/crish.jsonl" "$GLOBAL_RUN/crish_p.jsonl" \
  --output /tmp/probe-scale-NEW.json
.venv/bin/python scripts/audit_global_policy.py "$GLOBAL_RUN" \
  --labels /tmp/scars_labels.csv --replay-ema --output /tmp/global-audit-NEW.json
.venv/bin/python scripts/evaluate_kmeans_seeds.py --config "$LOCAL_RUN/config.json" \
  --checkpoint "$LOCAL_RUN/backbone_final.pt" --output /tmp/local-five-seeds-NEW.json
```

For the one-checkpoint two-view AUROC comparison, export a new diagnostic
partition and both evidence channels without any SGD step:

```bash
.venv/bin/python scripts/export_policy_evidence.py --config "$GLOBAL_RUN/config.json" \
  --checkpoint "$GLOBAL_RUN/backbone_final.pt" --epoch 199 --output /tmp/two-view-evidence-NEW
.venv/bin/python scripts/audit_global_policy.py /tmp/two-view-evidence-NEW \
  --labels /tmp/scars_labels.csv --output /tmp/two-view-auroc-NEW.json
```

The partition is re-extracted at that locked checkpoint; it is not substituted
for historical run partitions. `single_view_reliability` and `reliability` in
that export respectively identify the one-view and intersected two-view scores.
Offline EMA replay is hypothetical score smoothing, not an observed EMA training
run. Gaps reset history and are reported; do not use incomplete history to assert
that the exact Phase-0 EMA gate was verified.

The repeated evaluator is offline, CUDA-required, and has a 7,200-second deadline
per seed. It never substitutes another checkpoint. Old/new CSV labels are
permitted only in offline audit scripts and the explicitly marked oracle run.
Turn on reliability EMA, two-view evidence or asymmetric Move only if report 25's
Phase 0/posthoc conditions pass. Defaults leave all three off. Freeze any such
choice consistently before G1; do not inspect G1 to tune those switches.

## G1 profiles and readiness

Create a new output directory (the preparer refuses to overwrite it):

```bash
.venv/bin/python scripts/prepare_global_policy.py --base configs/selex_scars_crish_200.json \
  --output /tmp/cars-g2-G1-NEW --seeds 1 --oracle-labels /tmp/scars_labels.csv \
  --profiles matched_selex g2_fixed g2_random g2_cf g2_oracle_dose
GCD_CONFIG=/tmp/cars-g2-G1-NEW/g2_cf_seed1.json
bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant global --eval-every 1 --dry-run
bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant global
bash scripts/train_selex.sh smoke --config "$GCD_CONFIG" --variant global
```

The launcher accepts an explicit bounded two-epoch smoke. Prepare the three
profiles with warmup 0 and refresh_every 1 in a separate smoke config directory;
these overrides are for smoke only. Then run each profile serially:

```bash
bash scripts/train_selex.sh smoke --config "$SMOKE_CONFIG" --variant global --smoke-epochs 2
```

Each epoch has at most two minibatches. Keep max_runtime_seconds bounded.
Do not copy smoke metrics into benchmark tables. A real CUDA smoke must report
finite backward, non-fallback scoring, all diagnostic fields, memory/timing
and zero deadline triggers. Default full-run launcher behavior is unchanged.

After readiness, launch one G1 job at a time:

```bash
bash scripts/train_selex.sh --config /tmp/cars-g2-G1-NEW/matched_selex_seed1.json --baseline --eval-every 1
bash scripts/train_selex.sh --config /tmp/cars-g2-G1-NEW/g2_fixed_seed1.json --variant global --eval-every 1
# Remaining jobs use g2_random_seed1.json, g2_cf_seed1.json, g2_oracle_dose_seed1.json.
```

All remain 200-epoch cosine configs, stopped after 100 completed epochs. Do not
change epochs to 100: that would change optimization. The checkpoint controller
requires the exact worker PID printed by the launcher, not its parent shell PID.
Its explicit stop flag authorizes that command; do not signal other processes.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/stop_after_checkpoint.py \
  --run "$RUN_DIR" --pid "$WORKER_PID" --after-epoch 100 --timeout 21600 --confirm-stop
```

The controller stops at the first fresh validated checkpoint at/after the target;
inspect completed_epoch and retain all artifacts. It cannot restore an overwritten
epoch-100 checkpoint if invoked too late. Use a long-lived controller tab before
that boundary. Resume in the same frozen source checkout:

```bash
bash scripts/train_selex.sh --config "$RUN_DIR/config.json" --resume "$RUN_DIR/resume.pt"
# Add --baseline for matched_selex.
```

Read `crish_p.jsonl`, `crish_policy_epochs.jsonl`, `crish_policy_exposure.jsonl`,
`use_saturation.jsonl`, `metrics.jsonl`, and `crish_g_epoch_*.npz`. q=0 is a valid
rejected dose, not a missing result. A fallback/deadline is a separate condition.

```bash
.venv/bin/python scripts/compare_runs.py --treatment "$CF_RUN" --control "$MATCHED_RUN" \
  --window 61 100 --output /tmp/g1-cf-vs-matched-NEW.json
.venv/bin/python scripts/compare_runs.py --treatment "$FIXED_RUN" --control "$MATCHED_RUN" \
  --window 61 100 --output /tmp/g1-fixed-vs-matched-NEW.json
.venv/bin/python scripts/audit_global_policy.py "$FIXED_RUN" --compare-run "$RANDOM_RUN" \
  --labels /tmp/scars_labels.csv --bootstrap 2000 --output /tmp/enrichment-ratio-NEW.json
```

Apply report 25's rules in order, including the oracle stop rule, Old/New checks,
gray-zone second seed, selection-enrichment gate and realized-dose checks.
Missing epochs/arrays/labels block the corresponding conclusion; they are not zero.

## Phase 3 and final analysis

Only after G1 passes all gates, prepare seed 1–3 configs per dataset for the
chosen method and matched_selex, then Cars controls/ablations from report 25.
Use `--phase phase3` to set SS cadence 5 and the final 10 epochs. Existing G1
runs must resume with their original exact config/cadence; changing it violates
resume checks. If strict same-cadence final comparisons are required, continue
all matched G1 arms at cadence 1, and report that preregistration deviation.

```bash
.venv/bin/python scripts/prepare_global_policy.py --base configs/selex_scars_crish_200.json \
  --output /tmp/cars-g2-P3-NEW --seeds 1 2 3 --phase phase3 \
  --profiles matched_selex g2_cf g2_fixed g2_random g2_cf_nopersist g2_cf_noconfirm g2_cf_move
# Replace scars by cub or aircraft for matched_selex + the method selected at G1.
```

On every final checkpoint, save the repeated evaluation next to its run so
`compare_runs.py` reads the five-seed medians as the separate final primary:

```bash
.venv/bin/python scripts/evaluate_kmeans_seeds.py --config "$RUN_DIR/config.json" \
  --checkpoint "$RUN_DIR/backbone_final.pt" --output "$RUN_DIR/evaluation_kmeans_seeds.json"
.venv/bin/python scripts/compare_runs.py \
  --treatment "$CARS_T1" "$CARS_T2" "$CARS_T3" "$CUB_T1" "$CUB_T2" "$CUB_T3" "$AIR_T1" "$AIR_T2" "$AIR_T3" \
  --control "$CARS_C1" "$CARS_C2" "$CARS_C3" "$CUB_C1" "$CUB_C2" "$CUB_C3" "$AIR_C1" "$AIR_C2" "$AIR_C3" \
  --window 191 200 --output /tmp/final-paired-NEW.json
```

Pairing keys are (dataset, seed). Missing five-seed reports leave the final
primary missing, without substituting a best epoch or a single-seed score.
The JSON includes paired t/Wilcoxon/sign tests and per-dataset t intervals.
Pooled tests treat the nine dataset/seed differences as independent, an assumption
to disclose. Keep Old/New and every per-seed value. No G1/Phase 3 result exists
from this implementation session.
