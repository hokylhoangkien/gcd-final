# CRiSH-G v2 implementation — 2026-11-23

## Scope and provenance

Isolated worktree: `gcd-final-crish-g-v2`, branch `feat/crish-g-v2`, requested base
`119d073e57b7d95d443e4dd1be32ae2594d821ba`. The original dirty checkout remains
on its existing branch, including the owner's manuscript edits. No push,
benchmark, data download, checkpoint replacement or process termination.

This implements Phase A of the current owner-supplied `finetuneok.md`. It does
not implement the previously reverted relational-discovery proposal. The
verbatim thresholds and decision rules are in report 25, preregistered at
commit `f70279b` before any G1 run (none has been launched).

## Implementation map

| WP | Changed/new implementation | Behavior |
|---|---|---|
| 1 | `crish/probe.py`, `crish/config.py`, SelEx config validation | Optional per-row squared norm; default arithmetic unchanged |
| 2 | `crish/lookahead.py`, `selex_runtime/objective.py` | SGD assumptions checked; fp64 dot operands; direct removed-pair loss |
| 3 | `crish/normalized_selector.py`, `crish/population_settings.py` | Pooled within-dose sigma, historical floor, view-only one-SE ranking, known/variance constraints |
| 4 | `crish/rank_policy.py` | Hashed ties, nested rank doses, quotas, exact retained mass, permutation-invariant EMA |
| 5 | `crish/rank_moves.py` | Base-level ranked Move and parent propagation, support/target/source/protected guards |
| 6 | `crish/evidence_v2.py` | Optional augmented pass with private loader RNG; intersection edge weights |
| 7 | `scripts/prepare_global_policy.py`, `crish/oracle.py`, launcher | Eight explicit profiles; oracle label access only through a lazy diagnostic import |
| 8 | `crish/population_v2.py`, small dispatch in `crish/population.py` | Fixed-work refresh, disjoint folds/chunks, emergency Keep, persisted dose/scales/EMA |
| 9 | `crish/diagnostics.py`, `crish/training.py` | No-grad USE/exposure; per-refresh and per-epoch records, score arrays |
| 10 | `scripts/evaluate_kmeans_seeds.py`, monitoring/config | Separate bounded seed workers, final median summary, opt-in last-N schedule |
| 11 | `scripts/audit_probe_scale.py`, `scripts/export_policy_evidence.py`, `scripts/global_policy_statistics.py`, `scripts/paired_statistics.py`, audit/compare scripts | Component shares, AUROC, paired image bootstrap, temporal error, paired inference |

`crish/engine.py`, legacy `global_moves.py`, `rebuild.py`, SelEx `selex_loss`,
`evaluate_ss`, upstream sources and existing configs are preserved. All v2
behavior is opt-in. Existing historical settings omit default new keys from
serialized configuration. No evaluator scores return to the training policy.

## Key mapping

New keys retain their names from section 6. `population_policy.selection`
remains `threshold` solely to route to the existing global controller;
`dose_mode=rank_quantile` selects the new implementation. `ranking=knn_agreement`
is retained, with the measured signal logged as `single_view_reciprocal_agreement`
or `two_view_reciprocal_agreement`. `actions`, `persist`, `known_guard`,
`scoring_batches`, `scoring_sampling`, cluster guards and `save_arrays` are
reused. `experiment_profile` is metadata only. `diagnostics` and
`reliability_ema` are dictionaries under `population_policy`.

Checkpoint key `population_policy_state` is reused: v1 stores `tau`; v2 stores
`q`, `selected_epoch`, `sigma_history`, and optional CPU EMA tensors. Missing
v2 policy state starts at Keep, without relaxing source/config resume checks.

## Explicit implementation/protocol distinctions

- With-replacement weighted chunk sampling uses the training sampler's exact
  relative weights. Both probe folds are excluded; confirmation excludes every
  discovery ID. Disjointness conditioning changes its eligible population,
  necessarily differing from an unrestricted training batch.
- One initial projector copy isolates weight-norm derived tensors; a second,
  shared Keep-step copy produces all component gradients. There is no cloned
  SGD step per dose or per component. This extra origin copy favors strict live
  state isolation over the plan's literal single-copy wording.
- Two-view mode obtains the augmented pass from the sanitized training dataset;
  graph A uses the first normalized extraction pass. That pass is produced by
  the existing training augmentation, not a newly introduced clean transform.
  Upstream extraction is unchanged. Log labels identify the measured signals.
- Random/oracle target counts use the specified boundary-based n_l. Constraints
  and the evidence ceiling can cause unequal *realized* counts across controls;
  every level logs requested count and shortfall. Do not claim matched realized
  dose when those logs differ.
- Exposure is written to `crish_policy_exposure.jsonl` and the epoch training
  summary after SGD batches finish. The earlier policy epoch record cannot
  include exposure from batches that have not yet been sampled.
- The five-seed script seeds immediately before the existing fitter. Historical
  upstream seed 0 occurs before dataset creation and can advance before fitting;
  the new seed-0 diagnostic is not falsely claimed bit-identical to that stream.
  Standard single-seed run evaluation remains available unchanged.
- The oracle is one privileged fixed-dose heuristic, not a mathematical upper
  bound over all possible unlabeled policies. Report 25 preserves the owner's
  preregistered stop rule verbatim; interpreting it as a theorem is unwarranted.
- First-order gain is a projector proxy. CPU numerical tests establish its
  implementation, not its calibration to full-backbone learning or accuracy.

- Historical configuration validation imported a Torch-dependent ablation module.
  An existing mock-Linux CLI test exposed a Mac shared-library import failure.
  The same variant constants now live in tensor-free settings; local policy
  logic and its default decisions are unchanged.
- `GCD_PLAN_ROOT` optionally redirects disposable CLI plans outside the checkout;
  `--smoke-epochs 2`, preparer `--phase`, and stop controller `--after-epoch 100`
  provide concrete commands requested by the plan. Defaults remain historical.
- Diagnostic exports add an offline no-SGD two-view checkpoint pass and optional
  EMA score replay from archived arrays. Replay resets across missing epochs.
- G1-to-Phase-3 cadence differs in the supplied plan, whereas existing resume
  requires exact configuration. Existing G1 jobs must retain their frozen
  cadence on resume; the runbook records this instead of bypassing resume checks.
- Sigma history stores one raw discovery scale per component per refresh.
  Confirmation uses only earlier refresh history and logs its independent scale;
  it does not receive extra weight in history merely because a dose was selected.

## Validation and remaining blockers

Final CPU regression: **438 passed, 0 failed, 0 skipped, 0 timeouts**, across
46 independent pytest groups (each file, plus `tests/crish_v21`, timeout 120s).
Runtime source commit: `07a7c97`. Four v2 test modules cover 33 cases. A previous
run exposed the CLI import issue documented above; the final full rerun passes.
Eight generated profiles passed actual launcher dry-run. No GPU execution.
The release archive was built and tested outside Git by the existing release test.

Exact regression command, one bounded subprocess per group:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GCD_PLAN_ROOT=/tmp/crish-g-v2-plans
export SELEX_PYTHON="$PWD/.venv/bin/python"
.venv/bin/python - <<'PYTEST'
from pathlib import Path
import subprocess, sys
for group in sorted(Path('tests').glob('test_*.py')) + [Path('tests/crish_v21')]:
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(group)],
                   check=True, timeout=120)
PYTEST
```

The three two-epoch CPU smoke cases (`cf`, `fixed`, `random`) each execute
actual model/head forward, SelEx backward and SGD on synthetic 4D inputs,
one train minibatch per epoch. They require exercised policy paths and check
finite losses, diagnostic files and no fallback/deadline. These are not image
benchmarks and establish no full-data GPU latency/accuracy result.

The 18 mandatory contracts are covered by the v2 modules plus preserved source
parity, label-sanitization, optimizer, evaluation and checkpoint tests. The
legacy JSON fixture was captured at `119d073` before edits and compared with
final default output, excluding only timing. Resume tests cover fixed and CF
policies with EMA, including model/head weights, train RNG, dose, score history
and applied masks. Poisoned unknown labels leave CF/fixed/random decisions
unchanged. Two-view stochastic augmentation is reproducible with zero workers
and preserves Python/NumPy/Torch training RNG. Failed confirmation does not
search a runner-up. All changed files pass `git diff --check`.

Final group results (nested checks inside release are not counted again):

| Group | Result |
|---|---|

Local hardware: Apple M5 / 16 GiB / PyTorch 2.4.1 CPU; no CUDA. Real dataset,
pretrained weights and final checkpoints are not available in this worktree.
GPU smoke, full-pool timing, peak CUDA memory, final five-seed scores, Phase 0
semantic evidence, G1 and Phase 3 are NOT RUN. No accuracy improvement claim.

| `tests/test_aircraft_runtime.py` | 5 passed in 2.53s |
| `tests/test_change_ss_cadence.py` | 3 passed in 1.01s |
| `tests/test_checkpoint_selection.py` | 15 passed |
| `tests/test_cli_shared_gpu.py` | 6 passed in 0.03s |
| `tests/test_cluster_alignment.py` | 1 passed in 0.41s |
| `tests/test_compare_runs.py` | 7 passed in 1.37s |
| `tests/test_crish_candidate_identity.py` | 1 passed in 0.72s |
| `tests/test_crish_g_v2.py` | 4 passed |
| `tests/test_crish_g_v2_runtime.py` | 15 passed |
| `tests/test_crish_g_v2_selector.py` | 9 passed |
| `tests/test_crish_g_v2_tools.py` | 5 passed |
| `tests/test_crish_scoring_batch.py` | 4 passed in 0.42s |
| `tests/test_crish_training.py` | 13 passed |
| `tests/test_cub_runtime.py` | 3 passed in 1.23s |
| `tests/test_dinov2_config.py` | 1 passed in 0.02s |
| `tests/test_evaluation_policy.py` | 1 passed in 0.41s |
| `tests/test_evaluation_protocol_audit.py` | 3 passed |
| `tests/test_full_crish_method.py` | 27 passed |
| `tests/test_global_policy.py` | 61 passed |
| `tests/test_herbarium_runtime.py` | 6 passed in 6.14s |
| `tests/test_integrity.py` | 3 passed in 0.01s |
| `tests/test_intervention_audit.py` | 8 passed in 0.04s |
| `tests/test_kmeans.py` | 7 passed |
| `tests/test_layout.py` | 3 passed in 0.02s |
| `tests/test_local_adaptive.py` | 25 passed |
| `tests/test_loss_geometry.py` | 4 passed in 0.43s |
| `tests/test_move_batch_experiment.py` | 8 passed in 1.07s |
| `tests/test_observed_figures.py` | 5 passed in 0.01s |
| `tests/test_online_pair_reliability.py` | 20 passed |
| `tests/test_paper_ablation.py` | 16 passed |
| `tests/test_paper_registry.py` | 11 passed in 0.47s |
| `tests/test_periodic_evaluation.py` | 11 passed |
| `tests/test_protocol_alignment.py` | 3 passed in 0.12s |
| `tests/test_release.py` | 1 passed in 2.78s |
| `tests/test_runtime.py` | 6 passed |
| `tests/test_selex_empty_clusters.py` | 12 passed |
| `tests/test_selex_multidataset.py` | 10 passed in 5.33s |
| `tests/test_selex_rng.py` | 5 passed in 3.09s |
| `tests/test_selex_runtime.py` | 11 passed |
| `tests/test_selex_selection.py` | 8 passed |
| `tests/test_selex_ss_protocol.py` | 1 passed in 13.44s |
| `tests/test_ss_best.py` | 8 passed in 0.85s |
| `tests/test_stop_after_checkpoint.py` | 3 passed in 0.02s |
| `tests/test_training_upgrades.py` | 14 passed |
| `tests/test_upstream_pin.py` | 4 passed in 0.35s |
| `tests/crish_v21` | 41 passed in 0.84s |

## Preservation verification

SHA256 comparison against the initial worktree inventory: zero changes to all
previously present files in `upstream/`, `paper/`, `configs/`, `results/`, `logs/`.
`git diff 119d073 -- upstream paper configs results logs` is empty. The local
controller, legacy Move and rebuild files also match the requested base.
Original dirty checkout files were not staged or committed. Paper receipt was
not regenerated: the preserved manuscript corresponds to its historical source.
No push and no existing process stop/restart occurred.
