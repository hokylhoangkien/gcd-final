# Per-epoch SS and independent best checkpoints

The SS/best feature applies to **both local CRiSH and CRiSH-P**, and baseline.
The policy command below records the earlier run request; it is not the
Keep/Abstain/Move method. Use [report 14](14_selex_local_training.md) for the
current local-method training commands.

2026-11-12. User requested a fresh Cars CRiSH-P run, SS every epoch and
separate best All / Old / New weights, epochs and paired metric triplets.
Existing remote process is not controlled from this local workspace.

## New run

After stopping the old launcher in its original tmux pane and confirming its
workers have exited, update source and start a fresh run (do not resume an
old-source checkpoint):

```bash
bash scripts/train_selex.sh --dataset scars --variant policy --eval-every 1
```

This retains the existing preflight/smoke gates and full 200-epoch schedule.
Smoke keeps SS monitoring disabled. Epochs 1–199 receive synchronous bounded
SS monitoring; epoch 200 is included through the existing fixed-last final
evaluator. The separate plain-test-selected best evaluator remains available
but is excluded from the SS-best candidate pool. SS scores never enter SGD,
repair or early stopping. Default configs without SS monitoring stay unchanged.

## Outputs in the new run directory

- `evaluations/epoch_NNNN.json`: scheduled SS reports (fraction units).
- `epoch_metrics.jsonl`: SS history; final-last report included at run end.
- `ss_best.json`: three independent maxima, full All/Old/New, zero-based
  `epoch` and one-based `epoch_number`, checkpoint paths and SHA256.
- `backbone_best_ss_{all,old,new}.pt`: raw backbone state dicts matching the
  exact evaluated weights, usable for inference/evaluation, not optimizer resume.
- `evaluation_best_ss_{all,old,new}.json`: corresponding selected reports.
- `results_summary.json`: existing histories plus `ss_best_checkpoint_selection`.

Each SS report prints `EVAL ... protocol=balanced_ss_kmeans_train_unlabeled`.
`SS_BEST metric=all|old|new` prints the current best epoch and full metric triplet
after each scheduled/fixed-last evaluation. Ties keep the earliest epoch.
Only three SS-best backbone files are retained (roughly 1 GB total for this
backbone, excluding temporary copies and existing resume/monitor/best files).
Finite fraction scores and checkpoint hashes are checked. Writes replace
individual files atomically; a partially completed multi-file update is an
explicit recovery error rather than a silently reset best. Resume copies
best weights and SS reports and refuses missing/inconsistent artifacts.

## Interpretation and resource limits

These are evaluation-label-selected maxima among evaluated checkpoints, not
unbiased held-out or fixed-final estimates. Retain the last report separately;
match the SS evaluation schedule and checkpoint selection budget in baseline.
Do not combine three independent maxima into one All/Old/New row.
SS uses existing SelEx BSSK, ten initializations, global Hungarian-v2 and
unlabeled training-pool evaluation. No upstream geometry/loss changes.
Additional runtime and full CUDA behavior must be measured remotely; no new
GPU run or benchmark was executed locally. The old remote run remains intact
until the user stops it. Local paper files are preserved.

## Validation

Targeted CPU regressions cover independent maxima/paired scores/weights,
strict ties and final inclusion, exclusion of plain-test-selected best,
disabled-path behavior, invalid scores/hash mismatch, resume preservation,
missing-artifact errors, scheduling, RNG parity and existing selection behavior.
Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest tests/test_ss_best.py tests/test_checkpoint_selection.py
tests/test_periodic_evaluation.py tests/test_selex_selection.py
tests/test_selex_rng.py -q --tb=short`. The initial targeted set passed 45 tests;
an added missing-SS resume replay regression is included in the final check.
Final check added `tests/test_selex_runtime.py` and
`tests/test_selex_ss_protocol.py`: **58 passed** in 15.26 seconds. Subprocess
tests required shared-memory access outside the local sandbox for OpenMP;
the default sandbox produced SHM errors, not CUDA verification.
