# Local scoring support audit and correction

2026-11-12. Remote local smoke `20261112T121600Z-e525d57d` failed its
counterfactual-exercise gate before any real SGD batch. Config was verified
`method=crish`, local true, population policy false. User-supplied diagnostics:
four actions for samples 3429/6657, two Abstains with zero scoring pseudo-SSE
change and two Moves exceeding protected-coefficient cap. `gains=[]`, status
completed, accepted zero. This is not gain-based rejection, an OOM report or
evidence that local CRiSH was evaluated for full epochs.

## Cause and scope

Old scoring-batch construction reserved action IDs, then their measured
neighbors, then evidence-cache fill. Boundary neighbors can belong to other
pseudo-clusters. An action can have global effect but no same-cluster positive
peer in this small batch: Abstain then has zero target drift. Move on tiny
within-cluster batch support can also induce large changes in protected
same-ID coefficient denominators. The supplied log establishes these guard
failures; it does not establish that the Move failures were solely caused by
sampling, because their original drift values were not exported.

## Implementation change

`crish/scoring_batch.py` reserves action IDs, then prioritizes active source
pseudo-cluster peers (one for Abstain, two for Move) and two target-cluster
peers for Move, then measured neighbors and cache fill within the original
`train_batch` budget. It prefers cached peers, otherwise nearest frozen-feature
peers from the training snapshot. All discovery and confirmation probe IDs
are excluded. Additional peers receive actual cached augmented features;
original cached views are reused. No hidden targets, evaluation metrics or
invented evidence edges are used. Peer inclusion does not legalize a Move;
the measured-neighbor/support and coefficient-drift guards still apply.

This changes local counterfactual batch sampling, not only logging. Record
the new source fingerprint in any new local experiment. The baseline path,
population policy, objective, acceptance thresholds and loss definitions are
unchanged. No valid peer means zero-change remains a rejection; acceptance
is never forced. Larger real refreshes still need measured runtime/VRAM.

`crish.jsonl` now includes `scoring_batch` (selection rule, actual source/target
peer counts per action and added support IDs), and full `target_drift` for
pre-scoring coverage/cap rejections under `scoring_rejections`. Existing action
decisions/gains remain available. The smoke gate still refuses no-scoring.

## Verification and remote next step

CPU tests demonstrate an old-style two-row batch gives zero Abstain drift,
while a legal source-peer batch yields nonzero drift and zero protected shift.
A real Move fixture with source/destination peers passes the existing .25 cap.
Other cases check no-probe-peer fallback, deterministic budgeted IDs, probe
overlap rejection, complete engine candidate identity and baseline/training
regressions. Targeted command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_crish_scoring_batch.py tests/test_crish_candidate_identity.py tests/test_full_crish_method.py tests/test_crish_training.py tests/test_selex_runtime.py tests/test_ss_best.py tests/test_periodic_evaluation.py -q --tb=short
```

**74 passed**, 6.86 seconds. One initial subprocess test hit local sandbox
OpenMP SHM denial; the targeted run passed with shared-memory access. No
new CUDA smoke, benchmark, accuracy improvement or acceptance is claimed.

Do not update runtime under an active job. After confirming no GPU job is
running, pull the corrected source and run only:

```bash
bash scripts/train_selex.sh smoke --dataset scars --variant local
```

Inspect actual remote `scoring_batch`, `gains`, decisions and any failure.
Old-source checkpoint resume requires original source/config; this change
does not migrate the stopped policy run or local failed smoke.
