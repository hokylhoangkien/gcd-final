# Workspace audit — 2026-11-05 (Asia/Ho_Chi_Minh)

## Scope and preserved state

Investigate the newly supplied `crish_share/` training failure and make the
smallest tested runtime correction. The existing project is CRiSH-GCD v2.1,
with a shared baseline/CRiSH training loop, isolated from the pinned HypCD
submodule. This session does not introduce a research method or claim novelty.

- Working directory: repository root `gcd-final`.
- Starting HEAD: `918011cfa0d65fbf2bcab17fe050ca7636a38527`.
- Starting git status: only untracked `crish_share/`; no tracked edits.
- Upstream HypCD: `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`.
- Inspected `AGENTS.md`, README, runtime/training files, existing CPU test
  receipt, scripts/config inventory and supplied failure artifacts.
- Existing logs, source archives and release packages are preserved.

## Process and hardware inventory

The sandbox initially denied `ps`; a read-only process query succeeded with
escalated access. No Python training or pytest job was visible on this Mac.
VS Code Python environment/Pylance services were present. No process was
stopped, restarted or replaced. The remote A40 process state cannot be inferred
from a local process list.

Local host: macOS arm64, MacBook Air Apple M5, 16 GB RAM. `.venv` provides
Python 3.9.6, torch 2.4.1 and pytest 8.3.5. CUDA is unavailable; `nvidia-smi`
is absent. No local dataset/pretrained/checkpoint files were found in the
active project inventory. Only bounded CPU synthetic checks are appropriate.

## Supplied remote evidence

`crish_share/` contains `run.log`, `metadata.json`, `metrics.jsonl`,
`partition_audit.jsonl`, and `checkpoint.sha256` (all supplied files dated
2026-11-05 11:52 local time). It does not contain the checkpoint itself.

The remote log reports PyTorch 2.4.1+cu121 on NVIDIA A40, verified CUDA probes,
2,000 labeled / 6,144 unlabeled training rows and 8,041 disjoint test rows.
These are supplied remote observations, not local verification.
Run `20261105T042348Z-66079b2b` completed zero-based epochs 0–2 and wrote
`resume.pt` after each. The next hierarchy extraction raised
`ValueError: Invalid prototypes for radii` from `crish/rebuild.py`, through
`EpochRuntime.capture_hierarchy`. No final benchmark accuracy is available.
The last completed local test receipt records 72 passed synthetic CPU tests
on 2026-11-04; it is not evidence that this real-data failure was covered.

## Missing evidence and next safe action

The failing hierarchy tensors and remote checkpoint are absent, so the exact
invalid tensor cannot be inspected directly. The radius guard rejects both
nonfinite prototypes and fewer than two prototypes; identify which producer
paths can generate these conditions before changing it. Inspect upstream
hierarchy behavior and reproduce reachable edge cases with bounded CPU tests.
Preserve finite/label-safety checks and the unchanged upstream source. Record
the protocol, patch, regression evidence and remote rerun instructions.
Do not start local benchmark training or modify the user's supplied logs.

## Completed safe work

The pinned producer's empty-cluster `mean()` bug was reproduced on CPU. A
runtime-only adapter now fills empty clusters from eligible non-anchor donors,
guards collapsed k-means++ initialization, preserves source behavior on tested
nondegenerate fixtures and serves both B0 and CRiSH. No upstream files changed.
The precise missing remote tensor remains unverified.

Bounded full CPU validation: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/run_smoke.py` (60-second timeout per child command).
Run `crish-v21-20261105T051250Z-6e24b2b2`: 79 passed, zero failed/skipped;
split SHA/protocol, syntax and source-archive execution checks passed.
Log: `logs/crish-v21-20261105T051250Z-6e24b2b2/smoke.log`.
Receipt: `results/local_validation.json`. No benchmark or CUDA run was started.
Remote next action is patched-source `bash scripts/train_crish.sh check`, then
`bash scripts/train_crish.sh smoke`; old-source checkpoint resume stays blocked
by the existing fingerprint check. See `reports/05_training_failure_20261105.md`.

## Authorized periodic evaluation update — 2026-11-05

Starting HEAD: `1ce56f5d813f3aea40e2b67cd4c299baad7a1d68`; only the
supplied `crish_share/` directory is untracked. Upstream remains clean. A fresh
read-only process inventory found no local Python training/pytest job. Codex and
VS Code Python services are active; no process was changed. Remote GPU process
status is unknown. The host remains macOS arm64, 16 GiB RAM, 10 CPUs, without
local NVIDIA CUDA. Supplied logs end after epochs 0–2 and a hierarchy failure;
they do not establish current remote progress or a completed benchmark.

The user now explicitly requests an appropriate code change after discussing
monitoring and checkpoint selection. The safe scope is periodic All/Old/New
monitoring in a bounded separate evaluator (default every five epochs, CLI
override to one or zero), using the same balanced SS k-means/Hungarian-v2 as
final evaluation. Training stays sanitized, final-epoch checkpoint selection
stays fixed, and B0/CRiSH share the schedule. Preserve upstream and supplied
artifacts; validate only bounded synthetic CPU tests locally. Do not start or
replace a remote run. Full-data/CUDA verification remains blocked by unavailable
local dataset, weights and NVIDIA hardware.

### Completed periodic-evaluation change

- Shared B0/CRiSH configs enable balanced SS monitoring every five epochs;
  `--eval-every 1` evaluates each epoch, `0` retains final-only evaluation.
  Both real-data smoke profiles keep monitoring off. No loss, optimizer,
  dataset sanitization or upstream implementation was changed.
- `gcd_runtime/monitoring.py` serializes one replaceable CPU backbone snapshot,
  waits synchronously for a separate evaluator with a 1,800-second timeout,
  and never reads its metrics. The existing resume checkpoint is written first.
  Evaluator failure/missing report is visible and stops the run with checkpoint
  preserved. Launcher training deadlines now include the owned child process
  group, so an in-flight monitor is covered by the same bounded lifecycle.
- `epoch_metrics.jsonl` and per-epoch JSON reports retain fraction metrics,
  sample counts, elapsed evaluation time, run seed, actual evaluation RNG seed,
  timestamp, checkpoint SHA256, protocol and final-vs-monitor purpose. Live log
  output and `scripts/show_metrics.py` show percentages. Missing epochs remain
  explicitly unevaluated. Only the latest monitor backbone is retained.
- The final epoch is evaluated once by the launcher, after training exits;
  `evaluation.json` is never overwritten by an intermediate report. Final-epoch
  selection remains fixed. Upstream's test-Old-based `_best.pt` behavior is
  documented as a protocol difference, not adopted or asserted to explain all
  published table entries. Periodic label exposure is documented in the runbook
  and presentation; method/novelty claims are unchanged.

Validation command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python
scripts/run_smoke.py`. Run `crish-v21-20261105T151922Z-fc973aef` completed
**90 passed, zero failed/skipped** in the CPU synthetic suite, with the original
split checksum, upstream/source checks, syntax and archive execution checks
passing. New tests cover schedule/CLI, real child-process RNG isolation, matched
training updates, resume parity, snapshot persistence after timeout/failure,
missing reports, real balanced SS metric computation, report separation and
saved-table formatting. The initial sandbox-only subprocess test hit macOS
OpenMP shared-memory denial; rerunning with granted execution access passed.
The full runner retains its 60-second deadline per child command.

Receipt: `results/local_validation.json`; log:
`logs/crish-v21-20261105T151922Z-fc973aef/smoke.log`. Updated presentation PDF
rendered successfully (21 slides, layout bounds checked by renderer).
`scripts/show_metrics.py crish_share` correctly shows only the three observed
losses and no accuracy; original supplied artifacts were not modified.

Remote next action remains `bash scripts/train_crish.sh check`, then
`bash scripts/train_crish.sh smoke` using the new source. To explicitly test
the full evaluator on that smoke backbone with the correct one-epoch metadata:
`.venv/bin/python -m gcd_runtime.launch evaluate --config configs/crish_smoke.json
--checkpoint runs/SMOKE_RUN_ID/backbone_final.pt` (bounded at 1,800 seconds;
smoke accuracy is not a benchmark result). Do not run alongside another GPU job.
If those pass, start a new run with `bash scripts/train_crish.sh` (every five
epochs) or `bash scripts/train_crish.sh --eval-every 1`. No remote process was
started, stopped or replaced; no CUDA/real-data/accuracy claim was verified.
Existing source/config fingerprint restrictions on old checkpoints remain.

## Authorized Hyp-SelEx checkpoint-selection update — 2026-11-06

The user now explicitly requests implementing the released Hyp-SelEx code's
selection procedure, after discussing its use of evaluation labels. This
supersedes the earlier final-epoch-only selection constraint for this requested
reproduction mode. Keep a final-only mode available and never pass annotation
values or scores back into SGD/CRiSH action screening. This is an explicit
evaluation-label-based checkpoint-selection protocol, not a label-free or purely
held-out research claim. Novelty/algorithm claims remain unvalidated.

Fresh inventory: the prior periodic-monitor changes remain uncommitted and are
preserved; `crish_share/` remains untouched. HEAD is still the previously
recorded commit; upstream is pinned and unchanged. No local Python training or
pytest process was found; Codex/VS Code are active. Hardware remains macOS
arm64, 16 GiB RAM, 10 CPUs, no local NVIDIA CUDA. No remote process was inspected,
stopped or restarted. A newly pasted remote evaluation reports All .82177734375,
Old .9225387306346826, New .7731112720251025; config/source/checkpoint artifacts
for that run have not been supplied, so this is user-provided observed evidence,
not a locally verified run or a matched B0 comparison.

Source findings: Hyp-SelEx runs plain Euclidean k-means on normalized backbone
features over the disjoint test set after every epoch, saves best when Old-v2
strictly increases, and retains last weights. Its helper returns v2. SelEx's
default helper returns the first metric (v1 with default ['v1','v2']); its final
report explicitly evaluates best and last by SS k-means. The SelEx author
confirms best-checkpoint selection in issue #8. HypCD's evaluation script loads
`model_test.pt`, so its published table-to-checkpoint mapping remains unverified.

Next safe change: one shared B0/CRiSH runtime, isolated per-epoch disjoint-test
evaluators, strict Old-v2 selection, portable best/last checkpoints and resume
state, final balanced SS evaluations for both, histories and max-All/Old/New
summaries split by protocol. Preserve all original losses and the upstream tree.
Only bounded synthetic CPU validation will run here. Real-data/CUDA cost and
end-to-end behavior require bounded verification on the user's Linux machine.

### Completed selection implementation and validation

- Full and bounded real-data smoke profiles for both B0/CRiSH now select on
  strict Old-v2 improvement after each completed epoch, retaining last weights.
  Both smoke profiles exercise selector and final best/last evaluators; smoke
  has one epoch/two SGD steps, and its accuracy is explicitly non-benchmark.
  Additional SS monitoring is off by default, configurable with --eval-every.
- New `disjoint_evaluation.py` calls the official ordinary k-means backend on
  normalized backbone features; global Hungarian-v2 is evaluated afterwards.
  `selection.py` stores CPU backbone plus selection metadata in one atomic
  best bundle. Neither labels nor scores enter the training objective or repair
  scorer. `final_epoch` with test interval 0 is still a supported configuration.
- `test_metrics.jsonl` stores every test epoch; optional SS histories are in
  `epoch_metrics.jsonl`. `evaluation_best_old_test.json` and
  `evaluation_last.json` preserve both final SS reports. `evaluation.json`
  aliases best in selection mode, last in final-only mode, or the explicitly
  requested model for standalone evaluation. `results_summary.json` and
  `scripts/show_metrics.py` retain separate per-protocol max-All/Old/New triplets,
  winning epochs and number of actually evaluated epochs.
- Resume requires the original run directory's best bundle and test reports.
  It restores best-so-far/history and recovers a pending selector before more
  SGD. This includes recovering a failed last-epoch selector with no extra SGD.
  Incomplete or future selection bundles are rejected. Source/config checks on
  all resume/best checkpoints remain strict; no old artifact was migrated or
  deleted. Only the best and last model weights are kept, not all peak models.
- CUDA bootstrap now installs the official Git `kmeans_pytorch` at pinned SHA
  `f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805`, plus Numba 0.60.0. Existing PyPI
  0.3 lacks the released call's tqdm_flag and differs in GPU/empty-cluster
  behavior. CUDA preflight rejects that legacy backend and runs a real device
  tensor clustering probe. Each test report records backend version/Git SHA /
  function hash. The Git SHA was verified through GitHub's commit API; the
  dependency was not installed or executed on local CUDA. Local CPU fixtures use
  the pre-existing PyPI package on small nondegenerate data.
- RNG remains isolated per seed/epoch/protocol. This deliberate difference from
  the source's shared stream is documented. Matching the selector does not
  erase pre-existing optimizer/hyperparameter/numerical differences or prove
  table reproduction. SelEx-v1 selection and Hyp-SelEx-v2 selection are
  distinguished. No new novelty/effectiveness claim is made.

Validation: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python
scripts/run_smoke.py` completed **104 passed, 0 failed/errors/skipped**.
Run ID: `crish-v21-20261106T033914Z-6911fee6`.
Log: `logs/crish-v21-20261106T033914Z-6911fee6/smoke.log`.
Receipt: `results/local_validation.json`. Child commands retain 60-second
deadlines. The suite covers actual plain-k-means CPU execution, v1/v2 selection
distinction, strict Old-v2 vs max-All selection, ties, three paired peaks,
training/RNG/hidden-label invariance, pending/final-epoch recovery, missing
artifacts, best/last orchestration, and explicit source/config validation.
Original loss AST/gradient, SS evaluation, split/source hash and portable archive
checks also pass. New protocol defaults required updating the old final-only
configuration test expectations; no algorithm regression was observed.
Shell syntax and git diff whitespace checks passed, and upstream remains clean.
The 21-slide presentation PDF regenerated with layout checks passing.

Next safe remote commands, only after moving the new source to an idle Linux
CUDA environment: install `environments/requirements-selector-cuda.txt`, then
`bash scripts/train_crish.sh check`, then `bash scripts/train_crish.sh smoke`.
Both must pass before starting the finite full run. Do not update an environment
in use by a running job. This session started no training or remote evaluator,
did not stop a process, and did not verify CUDA/real-image throughput or memory.
The previously pasted All 82.18% run remains separate user-provided evidence.

## Authorized Cars/DINOv2 configuration alignment — 2026-11-06

The user requests a definite standard configuration and source alignment rather
than further parameter choices. Fresh read-only inventory found the preceding
uncommitted runtime/evaluation work unchanged, no local Python training/pytest
job, and a clean pinned HypCD tree. Hardware remains macOS arm64, 16 GiB RAM,
10 CPUs; no NVIDIA CUDA is available. No process or supplied artifact was changed.

The authoritative target is the pinned `scripts/train_HypSelEx.cmd` Cars-v2
invocation: `scars v2 0.5 0.5 0.1 2.0` (USE smoothing, maximum distance weight,
curvature, radius respectively). The command fixes batch 128, trainable blocks
from 10, 200 epochs, SGD lr .1, supervised weight .35, weight decay 5e-5,
ImageNet transforms, SSB split, v2 matching and projector dimension 8192.
Local full/smoke B0 and CRiSH configs currently differ at hyper_max_weight=1.0;
other inspected command parameters match. CLI defaults in the Python file and
paper prose are not substituted for this dataset/backbone-specific command.

Next safe change: set .5 consistently in the four profiles; add an independent
source-derived protocol check against the pinned command and parser defaults,
including the actual local worker flags, so future drift is visible. Keep
numerical fixes, isolated evaluation RNG, label boundaries and the new shared
best-Old-v2/last evaluation procedure; document those remaining differences.
No retuning, expensive sweep, checkpoint migration or new benchmark is started.

### Completed alignment

All four shipped profiles now use hyper_max_weight=.5. Worker flags explicitly
set the reference backbone and labeled fraction. `audit/protocol.py` derives the
Cars-v2 command's shell substitutions and source parser defaults using AST and
shlex only, then compares them with the actual local worker flags/defaults.
`scripts/check_protocol.py` checks all four profiles; only the one-epoch smoke
limit is exempt. A regression to weight 1.0, curvature .05, block 9, smoothing
1.0 or DINOv1 is detected. CUDA preflight records the same comparison in
readiness.json without falsely claiming complete paper reproduction.

The independent comparison passed: no differences for full B0/CRiSH or their
baseline hyperparameters in smoke. The upstream script checksum is
`75e31263b78e6d6b981cd82701e96f870047a8c1aed974361df568ff04b752e3`.
The original upstream source/tree remain unchanged. Existing correction/RNG
policies and the selected best/last protocol remain documented differences;
there is no promise of equal or better paper accuracy.

Bounded validation command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/run_smoke.py`. Run
`crish-v21-20261106T035459Z-73c88e69`: **107 passed, 0 failures/errors/skipped**;
split/source checks, new protocol comparison, syntax and source-archive execution
passed. Log: `logs/crish-v21-20261106T035459Z-73c88e69/smoke.log`;
receipt: `results/local_validation.json`. Shell syntax/whitespace checks passed;
updated 21-slide PDF rendered within layout bounds. Only synthetic CPU checks
ran locally, with 60-second deadlines per child; CUDA/data readiness is still
unverified here. Previous logs/receipts in logs and source archives were preserved.

Remote next steps: use this new source/config in an idle environment, install
the pinned selector dependencies if needed, then `bash scripts/train_crish.sh
check` and `bash scripts/train_crish.sh smoke`. Start fresh matched B0/CRiSH
runs only after these pass. Old checkpoints/configs retain their original 1.0
setting and must use their original source to resume; no resume guard is bypassed.

## Authorized shared Cars/CUB runtime — 2026-11-06

The user explicitly requests implementing CUB support after discussing one
shared method with dataset-specific inputs/configuration. Existing uncommitted
work is preserved. Fresh read-only process inspection found no Python training
or pytest job on this macOS arm64 host (16 GiB RAM, 10 CPUs, no NVIDIA CUDA).
Codex is active; no process was stopped or replaced. Upstream remains clean and
pinned. Dataset images/weights and current remote process state are unavailable.

Inspected the actual CUB loader: it stores rows in a DataFrame, shifts native
1-based class targets, keeps source uq_idxs through subsampling, and requires
images.txt, image_class_labels.txt, train_test_split.txt and
attributes/image_attribute_labels.txt under CUB_200_2011. Its default download
path can fetch/extract an archive; disable that runtime behavior without editing
upstream, so missing CUB inputs fail immediately instead of downloading.
The official CUB SSB split SHA256 is
48bdcaa05603d932037738d3df776ab500945e9b7aeff85f4668305bf71bd1e7:
100 known / 100 novel (Hard 34, Medium 34, Easy 32), verified from the local
pinned file. Use its class identities, not an assumed contiguous half split.
The shared get_datasets wrapper sets labeled fraction .5 and original train/test
partition; do not use the CUB loader's standalone defaults (.8/range(160)).

Implement dataset selection plus registry/adapters, CUB configs matching the
released command `cub v2 1.0 1.0 0.1 1.5`, generic split/membership/data preflight,
offline-only CUB loading, and dataset-specific source-protocol comparison. Keep
Cars the default, preserve CRiSH/loss/selection algorithm, and verify Cars parity
and CUB label/path/ID isolation with bounded synthetic CPU fixtures. No training,
large data download or generalization claim is authorized by this implementation.

### Shared adapter validation completed

Added four CUB configs and dataset selection through CLI/worker, native path/target adapters, label-isolated training materialization, offline CUB loader guard, metadata integrity validation, dataset-aware preflight/CUDA image probe and source-protocol checker. Existing Cars defaults and upstream checkout preserved. No active training was stopped or started; no data/weights downloaded. CUB is an additional supported input, not evidence of method generalization.

Executed `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py`: **110 tests passed**, both pinned splits passed, all eight profile CLI comparisons passed. Run metadata: `logs/crish-v21-20261106T040857Z-1b262235/metadata.json`. Synthetic native CUB fixture verifies full membership/label mapping/decoding, training hidden-target isolation, and missing-input rejection without network download. CUDA remains unverified locally. Next remote command after preparing CUB data and weights: `bash scripts/train_crish.sh check --dataset cub`.

## Upgrade review inventory — 2026-11-06 (Asia/Ho_Chi_Minh)

The current request asks which upgrades are worthwhile. Review the existing
implementation before recommending additional method complexity; this request
does not require changing the training objective or starting an experiment.

- HEAD remains `1ce56f5d813f3aea40e2b67cd4c299baad7a1d68`. Existing uncommitted
  runtime, selection, protocol, Cars/CUB, documentation and test work is present
  and preserved. The upstream HypCD checkout is clean at the pinned commit.
- A fresh read-only process query, retried with approved access after the
  sandbox denied `ps`, found Codex and VS Code Python services but no local
  Python training/pytest job. No process was changed. Remote status is unknown.
- The current hardware query confirms Apple M5, arm64 macOS, 16 GiB RAM and
  10 CPUs. `nvidia-smi` is absent. Only bounded local CPU checks are appropriate.
- The latest existing validation receipt records 110 passed synthetic CPU tests
  from `crish-v21-20261106T040857Z-1b262235`; this is historical verification,
  not a new run or CUDA/benchmark evidence.
- The supplied remote `crish_share/metrics.jsonl` still records only epochs 0–2,
  zero repairs and no accuracy values. Its approximately 23 GB allocated CUDA
  memory is from that historical source/config, not a current resource estimate.
- Existing runtime features include matched B0/CRiSH training, isolated
  evaluators, best/last selection, source/config resume checks and Cars/CUB
  adapters. Dataset/weight availability on the remote machine and matched
  current-source B0/CRiSH effectiveness remain unverified here.

Next safe action: inspect the actual repair budget, decision logs and locked
evaluation protocol, then prioritize a measured B0/CRiSH comparison and missing
controls. Reuse the existing methodology/novelty discussion in
`presentation/CRISH_GCD.md`; do not claim that extra reliability or hierarchy
modules are new. No training source or supplied artifact is modified for this
review. The next remote readiness command remains
`bash scripts/train_crish.sh check`, followed by the bounded `smoke` only on an
idle Linux NVIDIA environment with data and weights prepared.

### Upgrade review findings and bounded verification

The full CRiSH profile warms up for 10 epochs, refreshes every five epochs,
samples at most 128 evidence rows and accepts at most one action per refresh.
`crish/training.py` extracts a new HSSK stack each epoch, so repaired targets
are not explicitly carried into the next stack. These are conservative design
choices, not evidence of a defect or of insufficient effect: one action can
change multiple pair terms and influence subsequent model parameters. Measure
that effect before increasing budgets. The separate real-data smoke profile
already overrides warmup to zero and refresh interval to one; it does exercise
the repair attempt path rather than waiting for the full-run warmup.

Priorities for subsequent work:

1. Aggregate existing `crish.jsonl` candidates, accepted/rejected actions,
   fallback reasons, changed mass and elapsed time; extend diagnostics only
   where needed to measure effects on the actual SGD batches. Report observed
   target/loss coverage as well as accepted counts. Existing projector-only
   one-step probe gains do not establish downstream discovery improvements.
2. Establish current-source matched B0/CRiSH runs on Cars after readiness and
   bounded smoke. Keep the same split, initialization, seed, optimizer and
   declared checkpoint-selection rule. Report best/last All/Old/New, time and
   peak memory; use paired seeds when resources permit.
3. Implement declared controls before adding complexity: abstention-only and
   simple supported-neighbor/random action selection with matched intervention
   mass and budget. Both are future work; shipped runtime profiles currently
   select only B0 or CRiSH. Optional stability remains off pending evidence.
4. Use CUB for subsequent validation after Cars works. Its adapter passing CPU
   fixtures does not demonstrate accuracy or generalization.

Re-read the official HypCD method and Hyp-SelEx appendix via
https://arxiv.org/html/2504.06120v1 (the CVF page returned HTTP 403). HypGCD's
full PDF, section on local hierarchy, already uses hyperbolic K-reciprocal
neighbors and triplets: https://duannyu.github.io/assets/papers/hypgcd.pdf .
The RCA proceedings page confirms the confidence/cross-view overlap lead, but
its full PDF could not be fetched by the browser; LEAH's full PDF was also
access-blocked. These access limits do not close the novelty gate. Existing
full-method review requirements in the presentation remain open; no novel
contribution is claimed from this upgrade review.

Executed `scripts/check_protocol.py` with `.venv/bin/python` under a 30-second
subprocess timeout: exit 0, all eight shipped Cars/CUB profiles match the
released dataset-specific CLI hyperparameters (only finite smoke epoch limits
are exempt). Log: `logs/upgrade-review-protocol-20261106T080923Z.log`.
This checks source/config alignment, not data, CUDA, accuracy or exact paper
reproduction. No full suite was rerun, no training source changed and no job
started. This session only appended this audit; prior edits and artifacts remain
preserved.

### Accuracy-oriented code review — 2026-11-06

The user clarifies that the question concerns concrete training changes likely
to improve results. A fresh process query again found no local training job;
source and upstream state remain preserved. Reviewed `training_data.py`, the
shared loss, confusion-factor rebuild, evidence sampling and legal probes.

The highest-priority correctness experiment is the already documented legacy
USE normalization: `factor / factor.sum(dim=1)` divides columns by the vector
of row sums. A bounded 20-second CPU arithmetic fixture confirmed differing
soft-target masses (example sums 1.667/.950/.767 versus 1/1/1 with row
normalization); both variants produced finite losses/gradients. This establishes
the arithmetic difference, not an accuracy benefit. Log:
`logs/use-normalization-review-20261106T081345Z.log`. A future corrected mode
must normalize the actual non-self target support, handle zero mass explicitly,
and be applied in matched B0/CRiSH controls while retaining the source-faithful
default. No training loss was changed in this review.

Next method experiments should improve evidence support: current kNN is computed
only among 128 randomly selected evidence images rather than the full extracted
feature pool. A blockwise full-pool shortlist with bounded cross-view checks is
a specific candidate, not an implemented or validated improvement. Separately,
the 4-known/8-unlabeled probes per fold could be enlarged and legally
class-balanced on the known side to test decision robustness. Neither change
may use hidden known/novel identities, oracle scores or carry arbitrary cluster
IDs across rebuilt stacks. Keep the original split/backbone/optimizer/selection
settings fixed when comparing each change; no GPU run or sweep was started.

### Authorized coordinated training upgrade — 2026-11-06

The user now requests implementing the discussed improvements together. Fresh
read-only inventory confirms the same HEAD and existing uncommitted work, a
clean pinned upstream, no local Python training process, and Apple M5/16 GiB
without NVIDIA CUDA. Dataset/weights/checkpoints are absent at project defaults;
remote process/readiness remain unverified. No process or artifact is replaced.

Implementation scope: selectable non-self row-normalized USE targets with a
positive-view fallback; a full frozen-feature neighbor shortlist followed by
bounded local two-view verification; and larger legally class-balanced known
probes. Preserve existing default profiles and exact legacy arithmetic. Add
explicit upgraded and matched-control profiles rather than silently redefining
the published baseline. Keep existing optimizer, curvature, repair acceptance
guards, supervision boundaries, evaluator and checkpoint selection. Test each
component and integrated CPU forward/backward/resume under deadlines, then
prepare remote check/smoke commands. These are research experiments, not a
validated accuracy gain or a closed novelty gate. No expensive run is started.

### Coordinated upgrade completed and locally validated

- Added opt-in `--profile upgraded` with eight separate Cars/CUB B0/CRiSH
  full/smoke configs. Existing profile files/default selection are preserved.
- Shared USE now supports `legacy` (unchanged source arithmetic) and `row`
  (non-self unit-mass smoothing, positive-view zero-support fallback). The
  original supervised loss classes and terms remain intact; drift/scoring and
  SGD share the chosen mode. Upgraded B0 uses the same corrected objective.
- Added bounded full frozen-feature query kNN and cache construction in
  `crish/reliability.py`/`crish/sampling.py`, plus legal class-balanced probes.
  Reference blocks are 1,024 rows; cache remains at most 128. Augmented-view
  evidence is explicitly local to that cache. Full probes use 16 labeled / 32
  unlabeled images per fold. No hidden known/novel identity enters sampling.
- Shared engine logs modes, reference/cache counts and legal-probe coverage.
  Real DINOHead synthetic preflight exercises the selected modes at reduced
  fixture sizes. Optimizer, geometry, warmup, repair/confirmation guards and
  checkpoint-selection protocol remain the existing values.
- Updated source-protocol reporting to distinguish matched upstream CLI from
  changed USE arithmetic, and check all 16 profiles. Added method/equation and
  open-novelty details in `reports/02_novelty_and_method.md`, README commands,
  and a successfully rendered 23-slide presentation. No accuracy table filled.

Focused validation initially passed 28 tests, then all 13 new upgrade tests
passed including real DINOHead loss/scoring, finite backbone/head gradients,
untouched supervised terms, dense-neighbor comparison, probe disjointness,
hidden-label poisoning, disabled-repair parity and exact epoch resume.

The first full synthetic suite had 122 passed and one sandbox-only OpenMP
shared-memory failure in the separate-evaluator test. Its log/metadata are
preserved in `logs/crish-v21-20261106T083012Z-36978537/`. Re-running the same
bounded suite with approved execution access passed **123 tests, 0 failures,
0 errors, 0 skipped**. Run: `crish-v21-20261106T083145Z-a919091c`.
Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python
scripts/run_smoke.py` (60-second timeout per child). Receipt:
`results/local_validation.json`; log:
`logs/crish-v21-20261106T083145Z-a919091c/smoke.log`.
Both official splits, 16 protocol comparisons, syntax/source checks and the
archive execution test passed. `git diff --check` passed; pinned upstream is
still clean. No benchmark, data download or NVIDIA execution occurred.

Next exact commands on an idle Linux NVIDIA machine with data/weights prepared:
`bash scripts/train_crish.sh check --profile upgraded`, then
`bash scripts/train_crish.sh smoke --profile upgraded`. Only after readiness and
bounded smoke pass, `bash scripts/train_crish.sh --profile upgraded` creates a
fresh finite run; `--baseline` selects its matched corrected-loss control.
Add `--dataset cub` only for prepared CUB inputs. Resume from old source stays
restricted to its original source/config; no checkpoint migration is attempted.

## Remote selector-install failure — 2026-11-06

User-supplied log: run `20261106T094101Z-2eba9a32` fails in the final
disjoint-selector preflight probe with `CUDA selector needs pinned Git
kmeans_pytorch`. Earlier supplied RTX 3090 tensor forward/backward succeeded;
the gate reaches this selector only after input/data/backbone/method checks.
This is not a completed readiness gate or a training/accuracy result. Dataset
and weights are preserved; no training process is stopped or restarted.

Local HEAD/previous edits are preserved, upstream is clean. A local process
query was sandbox-denied; local/remote ongoing process state is not newly
verified beyond the failed foreground command supplied by the user.

The common requirements first install PyPI kmeans-pytorch 0.3. pip documents
that an already satisfied version may prevent a VCS requirement replacing it.
Thus listing the Git dependency alone does not ensure the origin/signature
required by the selector. Bootstrap must resolve pinned selector dependencies,
then force-reinstall those two packages with --no-deps, preserving Torch and
other resolved dependencies. Current remote environment can use that same
bounded reinstall, pip check, then a fresh check/smoke with existing inputs.
No source change to training/selection or checkpoint fingerprint is needed.

Implemented the scoped forced reinstall in `scripts/bootstrap.sh` after normal
dependency resolution, and corrected README recovery commands. `bash -n
scripts/bootstrap.sh` and whitespace checks passed. No local environment was
reinstalled and no training test was rerun for this installer-only change.
The remote corrective install and complete preflight/smoke are pending the
user's execution; the original source archive remains identifiable and unchanged.

## Observed RTX 3090 backward OOM — 2026-11-06

The next supplied log reaches the real-data smoke training worker after
preflight, then fails at loss.backward(). Physical capacity is 23.56 GiB;
process use 23.19 GiB, with 19.93 GiB allocated by PyTorch, 2.90 GiB reserved
but unallocated, and only 345.19 MiB free while requesting another 1 GiB.
The failed foreground command returned to its shell. This is direct evidence
that the current batch-128 smoke does not fit this observed allocator state;
the earlier A40 allocation receipt was not proof of 24-GB readiness.

First bounded recovery is one fresh smoke with
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, preserving batch, objective
and both comparison arms. PyTorch 2.4 documents this allocator option as an
experimental mitigation for unusable memory slices. The reserved amount alone
does not prove fragmentation is the sole cause or guarantee recovery. No
training source/config is changed, no checkpoint is resumed, and no full run
or repeated trial loop is started. If this still OOMs, actual activation-memory
reduction or a separately documented matched lower-batch protocol is needed.

### Supplied smoke progress after allocator change

Run `20261106T095355Z-63c8f5fb` now reaches the final best-checkpoint SS
evaluator. The pasted report states epoch 1/1, All 53.37%, Old 61.92%, New
49.24%, purpose best_old_test, on train-unlabeled using balanced SS/Hungarian-v2.
The shipped smoke has batch 128, one epoch and at most two SGD steps. Thus this
is observed user-supplied smoke evidence, not a benchmark or proof of full-run
memory stability. It demonstrates progress past the earlier backward OOM in
this attempt. Exact metric fractions, peak memory and final launcher status
have not been supplied; check metadata.status == completed before a full run.
No remote process was changed by the assistant and no further smoke was started.

## Manuscript alignment review — 2026-11-08 (Asia/Ho_Chi_Minh)

Scope: compare the current working tree with the user-supplied English IEEE
CRiSH-GCD manuscript. This is an implementation review, not authorization to
change the algorithm or launch benchmark training. Preserve all earlier audit
entries and the substantial pre-existing tracked/untracked changes.

- HEAD: `1ce56f5d813f3aea40e2b67cd4c299baad7a1d68`; the working tree has
  existing changes across CRiSH, runtime, configs, tests, reports and presentation.
- HypCD remains clean at `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`.
- A read-only process query succeeded outside the process-list sandbox after
  the sandbox denied `ps`. Only Codex and VS Code Python tooling matched;
  no local Python training/pytest process was visible. Remote status is unknown.
- Hardware: Apple M5 MacBook Air, 16 GB RAM, macOS arm64; `nvidia-smi`
  absent. No training, dataset download or large weight download is started.
- Active source includes `crish/engine.py`, reliability, partition stability,
  constrained actions, supervision rebuild, legal probe and projector lookahead.
  Cars/CUB configs and synthetic tests exist. No image dataset, pretrained
  `.pt`/`.pth` weights or actual checkpoint appeared in the active-file inventory
  (historical/cache/venv areas were excluded). Supplied remote logs remain
  historical evidence, not proof of current remote readiness.
- The IDE-selected archive is the 2026-11-06 03:56:40 UTC release; newer
  archives exist. Current working-tree findings must not automatically be
  attributed to that older archive.
- Initial material discrepancy: active CRiSH configs select
  `best_old_test_v2`, whereas the supplied manuscript requires checkpoint
  selection independent of test/hidden novel labels. Repair-probe label safety
  and checkpoint-selection label safety must be assessed separately.

Next safe action: inspect the pinned loss/refresh code and CRiSH wiring, write
a source-referenced alignment report, and run bounded CPU regression checks.
Do not modify existing training behavior to make it match prose during review.

Review result: `reports/06_manuscript_source_alignment.md` records implemented
components, disabled M2, label-dependent checkpoint selection, projector-only
discovery/confirmation scoring, bounded local evidence, incomplete ablations
and dataset coverage, and the older archive/worktree differences. Selected
archive SHA256 matches its sidecar. The bounded CPU regression run
`manuscript-alignment-20261108T092537Z` passed 82 tests in 3.60 seconds
(13 dependency deprecation warnings), with a 60-second subprocess timeout.
Log and metadata are in `logs/manuscript-alignment-20261108T092537Z/`.
Python 3.9.6 / torch 2.4.1 reported CUDA=False and MPS=False in this process.
No algorithm/config change, real-data training, download or remote operation
was performed. Remaining work is a concrete manuscript/protocol alignment
decision and matched real-data experiments; synthetic success is not evidence
of benchmark gains or of completed novelty gates.

## Source housekeeping authorized by owner — 2026-11-08

The owner explicitly requested removal of surplus presentations, saved copies
and generated clutter while retaining `.venv` and the ability to run locally.
This supersedes the default prohibition on deleting anything; it does not
authorize removing unique experiment evidence, changing the method or claiming
paper readiness. A fresh read-only `ps` showed only Codex/IDE Python services,
no local training job. Current tracked/untracked source changes are preserved.

Inventory: `.venv` about 673 MB; historical experimental backup about 3 GB;
generated releases/archives about 21 MB; presentation previews about 4.3 MB.
The large historical backup contains actual checkpoints and an observed
evaluation from the retired reliability method, so it is retained separately
from CRiSH results. Research-paper downloads and the supplied current paper
draft are useful inputs, not disposable caches. Planned cleanup removes old
generated releases, redundant synthetic run logs, slide PDF/render/previews
and obsolete workspace receipts; retains source/config/tests/environments,
upstream, all real-data checkpoints/logs, current validation evidence and the
method/protocol/novelty reports. Packaging/docs references will be adjusted
and bounded CPU smoke plus source-hash preservation checked afterwards.

Cleanup run `source-cleanup-20261108T093324Z` removed 203 inventoried obsolete
or generated files (25,648,653 bytes), with per-file SHA256 in
`logs/source-cleanup-20261108T093324Z/cleanup_manifest.json`. The removed files
are old distribution/archive copies, superseded synthetic smoke logs, slide
PDF/renderer/previews, old workspace-only receipts and OS metadata. The
original touched documents/scripts/tests were saved under that run's `before/`
for review of the pre-existing edits. No weight/checkpoint file was deleted.

The presentation's unique method content now lives in
`reports/04_method_and_research_plan.md`; the supplied manuscript PDF moved to
`paper/manuscript.pdf`, with its existing user-supplied LaTeX copied to
`paper/manuscript.tex`. These are research drafts, not completed experimental
papers. README, source packaging, smoke provenance and layout/archive tests
now reference those canonical locations. `.gitignore` excludes LaTeX build
intermediates. Distribution archives are built on demand instead of kept as
multiple stale versions. The older wording in `crish/rebuild.py`'s docstring
refers to the former method-document path; runtime bytes were deliberately
preserved because checkpoints compare their source fingerprint exactly.

The retired-method archive `gcd-final-runpod-20261031T034654Z-87f188c3.tar.gz`
matches all 12 recorded runtime hashes of the historical completed run.
It and its checksum were retained in `logs/final-backup-20261101/source/`.
Other real-data evidence in `logs/final-backup-20261101`, `logs/remote-audit`
and `crish_share` remains untouched, with sizes/mtimes checked for 80 files.
64 protected current method/runtime/audit/config/environment/AGENTS/NOTICE
files match their pre-cleanup SHA256; `.venv/pyvenv.cfg` also matches.
The environment was not recreated or reinstalled. Current validation and
manuscript-review logs, the failure-regression receipt, research PDFs and
unique debugging evidence remain available. Full bounded CPU validation is
the next command; no training/data download is part of this cleanup.

Validation completed: run `crish-v21-20261108T093606Z-c1ae6403` passed all
123 CPU tests (zero failures/errors/skips), source/split integrity checks for
Cars and CUB, all profile/CLI protocol checks and syntax/upstream verification.
The archive integration test built/extracted the new layout, checked its
manifest, and ran synthetic smoke and CRiSH/B0 dry-runs without Git. No
release artifact was left in `dist/`; the integration archive was temporary.
The first sandboxed attempt had 122 passes and one OpenMP shared-memory
SIGABRT in a real subprocess test, not an algorithm assertion failure. The
same bounded CPU runner passed outside that sandbox constraint; both failure
and success logs are kept under the cleanup run directory. Current receipt
is `results/local_validation.json`. Final checks reconfirmed all 64 protected
source hashes, all 80 historical evidence sizes/mtimes and the preserved
environment; `git diff --check` passed. No algorithm, hyperparameter, weight,
checkpoint or upstream byte was changed by this housekeeping task.

## Code/manuscript synchronization — 2026-11-08

Owner requires algorithm/code and paper to agree. Current reference/upgraded
runtime and configs are the factual implementation contract; do not invent
missing experiments or turn optional M2 on to fit the old prose. Fresh `ps`
again showed no local training job. Existing changes, `.venv`, checkpoints and
the pinned upstream are preserved. Backups of the current manuscript/PDF and
documents plus protected method/config hashes are in the `logs/paper-sync-*`
run for this session. The local environment remains Mac M5/16 GB, CPU-only
development; no full benchmark is started.

Plan: replace design-only claims with the actual loss, evidence cache, optional
stability, projector SGD discovery/confirmation policy and label-dependent
Old-test-v2 selector; separate implemented Cars/CUB paths from future controls.
Generate the settings table from actual configs and check a source digest so
later implementation drift is detected. Rebuild and inspect the PDF if a
compiler is available; no LaTeX compiler was found on PATH at the audit.

Completed synchronization: `paper/manuscript.tex` now describes actual USE/SSE
coefficients and ramp, legacy versus row-normalized USE, cache-local Euclidean
M1, disabled-by-default subset M2, legal probe MSE/variance components,
projector-only exact SGD ranking and winner-only independent confirmation.
Pseudocode includes action budgets, guard checks, expiry at next epoch HSSK,
and whole-refresh Keep fallback. The test-dependent default selector, separate
best/last anchored SS evaluation, exploratory Cars exposure and planned-only
controls/datasets are explicitly disclosed. No benchmark values were added.

`scripts/check_manuscript.py` generates the profile table and binds its review
to implementation/config hashes, checks cross-reference/citation keys and
validates the LaTeX/PDF receipt. Its digest does not replace semantic review.
`scripts/build_paper.py` compiles in a temporary directory with finite timeouts,
checks that inputs did not change during compilation, publishes the PDF and
writes `paper/manuscript.build.json`. Both the new checker and receipt are
covered by archive/smoke execution. README and method/alignment reports were
updated to describe current behavior rather than the superseded blueprint.

A standalone official Tectonic 0.15.0 aarch64 binary was downloaded into
`/private/tmp` (archive SHA256
`24bd46566fa30d41101848405e9cbc4645edb92d8f857c9d21262174fb70cd33`).
It used its normal TeX resource cache; `.venv` was neither recreated nor
reinstalled. The final build produced a seven-page PDF; sampled pages were
visually inspected and the log has zero overfull boxes or undefined-reference
warnings (underfull/typesetting notices are not benchmark failures). Successful
build log: `logs/paper-build-20261108T095322Z-d6c2a7a6/build.log`.

Final validation run `crish-v21-20261108T095356Z-0aed688d` passed all 123 CPU
tests, Cars/CUB integrity, profile/CLI protocol, source/syntax checks and the
new manuscript/PDF check. Negative fixtures correctly rejected a stale table
and a mismatched PDF hash. CPU runner timeout was 60 seconds; subprocess
tests ran outside the known OpenMP shared-memory sandbox constraint. The
current receipt is `results/local_validation.json`. All 61 protected
algorithm/runtime/config/environment/audit source hashes remained unchanged;
upstream stayed clean, `.venv` is preserved, and no dataset/weights/training
or remote process was altered. Session evidence is under
`logs/paper-sync-20261108T093957Z/`. Remaining empirical/novelty gates are still
open despite the completed documentation alignment.

## Evaluation audit and retraining preparation — 2026-11-08

Owner requests a direct paper/source comparison with HypCD/Hyp-SelEx and SelEx
before retraining for submission. Current working-tree changes and completed
paper synchronization are preserved. Fresh read-only process inspection found
no local training process; remote state is not inferred from this Mac. The
local M5/16 GB environment has no NVIDIA CUDA or configured image/weight inputs;
the last recorded validation is 123 passing CPU synthetic tests. No full
training or dependency reinstall is started.

Audit scope: official evaluation pool, anchored versus plain k-means, metric
v1/v2, checkpoint selection, feature/backbone settings, published-paper versus
released-code discrepancies, class split/label fraction, and evidence needed
for a fair matched rerun. Read the pinned HypCD implementation and the official
papers; inspect SelEx at a recorded immutable commit through read-only GitHub
tools. Any concrete evaluation defect will receive a bounded fixture before a
minimal fix; protocol choices will remain explicitly separated from bugs.

Audit outcome is in `reports/07_evaluation_and_retraining_audit.md`. SelEx
commit `c3d3b42a0aa7d7c9ea748a8305a6890a34100f73` warns of ordering sensitivity;
its author confirms the coarse pseudo-label mapping bug in issue #11. HypCD
main is still the pinned SHA and contains the same un-restored coarse output
pattern. The current runtime already restores each level. A new integration
fixture exercises interleaved predicted membership and permuted sample IDs;
it passes, without claiming whole-kmeans permutation invariance. Final metric
is correctly global-v2 on train-unlabeled with legal SS anchors. Ground truth
for post-prediction Hungarian scoring is distinguished from selection/tuning.

Released-code comparison also finds SelEx Old-v1 versus Hyp-SelEx Old-v2
selection, and standalone final SS default restarts 10 versus Hyp-SelEx 1.
Those defaults do not prove which settings produced every published table.
The report separates plain disjoint-test selector scores, final anchored SS
scores and retrospective peaks. No protocol/default, loss or optimizer was
silently changed to chase published results.

A concrete metadata defect was reproduced by a failing fixture: fixed-last
reports inherited `selected_using_evaluation_labels=True` from the run's best
selector. Only `gcd_runtime/checkpoint.py` changed among existing runtime/
method files, to separate actual best selection, run selector policy, matching
label use and a report's participation in selection. Accuracy and updates are
unchanged. Old artifacts were not rewritten; old-source checkpoints still
require their original source to resume. Paper LaTeX/PDF and digest were
rebuilt consistently, adding the ordering/restart/selector caveats.

`scripts/prepare_retraining.py` now freezes matched B0/CRiSH full/smoke configs
and reviewable commands, without executing a sweep. Prepared Cars reference
plan: `runs/plans/matched-scars-reference-20261108T100459Z-b2ca3351/`, seeds
1/2/3, inherited Old-test-v2 selection, 3 matched pairs. Pair/smoke configs and
six seed-1 stages passed dry-run verification. An alternative CUB/upgraded/
fixed-last plan also passed preparation checks in a temporary directory.
Targeted regression: 18 passed; full run
`crish-v21-20261108T101031Z-85b7ce2e`: 126 passed, zero failures/errors/skips,
plus integrity, protocol, archive execution, syntax and manuscript/PDF checks.
Tests are bounded CPU only; actual CUDA/data/VRAM gates remain unverified.

Local input inspection explicitly failed on missing Cars annotations and
DINOv2-reg4 weights, and CUDA availability is False. Historical checkpoints,
`.venv`, dataset/weight inputs and running processes were not modified. A
fresh clean source archive is being prepared for transfer; regenerate the
plan on the matching Linux checkout, inspect GPU usage, then run B0 preflight
and real-data bounded smoke before any full run. Session evidence is under
`logs/evaluation-audit-20261108T100159Z/`.

## GitHub source checkpoint — 2026-11-08

Owner requests moving the prepared source to GitHub before training. Current
branch is `main`, origin is `hokylhoangkien/gcd-final` (public), and remote
main matches local HEAD `1ce56f5d813f3aea40e2b67cd4c299baad7a1d68` at the
read-only check. No local training process was visible. The upstream submodule
is clean and remains pinned. Existing source/config changes are exactly the
inputs covered by the 126-test validation receipt; manuscript/PDF checks pass.

The publication scope is source, configs, tests, environments, reproducibility
reports and the aligned manuscript. Local environments, image datasets,
weights/checkpoints, raw logs, generated release archives and run plans are
excluded. Add `crish_share/` to ignore rules because it contains supplied raw
debug logs/metadata. Candidate text files were checked for credential/private
key patterns and personal absolute paths; no matches were found. GitHub's
connected API has read-only access to this repository; local Git credentials
will be used for an ordinary push. Stage a bounded path list and inspect it
before committing. Do not force-push, rewrite history, alter repository
visibility, or modify any training process.

## Evidence/probe experiment preparation — 2026-11-08

The owner asks which change is worth testing before paying for another run.
Initial working tree is clean at `1cfe181`. A fresh read-only process inspection
shows IDE/Codex services and no Python training job. Local architecture is
arm64; the previously audited Mac has no NVIDIA CUDA and benchmark inputs are
still unverified locally. A restricted `sysctl hw.memsize` read was denied;
the earlier hardware audit remains the source for the 16 GB memory inventory.

The reference CRiSH profile samples a random 128-row evidence cache and uses
4 labeled / 8 unlabeled probe rows per fold. Existing optional code supports
global frozen-feature neighbor shortlisting and 16 / 32 probes with legal
class-balanced labeled sampling. Prepare an isolated seed-1 comparison using
these options while preserving legacy USE normalization, loss coefficients,
checkpoint selection and repair limits. This tests evidence/probe changes
separately from the upgraded profile's row normalization. It is a hypothesis,
not an estimate of accuracy gain or a new contribution. Main configs, upstream,
manuscript and historical artifacts remain unchanged. Only bounded synthetic
CPU validation is authorized here; no GPU job or sweep is started.

Prepared four ignored configs under
`runs/plans/evidence-probe-only-20261108T110242Z-fe28609c/`, with checksummed
manifest and reviewable sequential commands. Verified that only evidence
sampling, labeled-probe sampling and the two probe sizes differ within each
reference/candidate pair. Four launcher dry runs and the bounded scaled
synthetic CPU forward/backward + repair scorer passed. The fixture scored one
proposal and accepted zero; no benchmark gain is inferred. Recorded the
decision and limits in `reports/02_novelty_and_method.md`. No source algorithm,
shipped config, manuscript, checkpoint or environment was changed. Real-data
smoke must still establish runtime/probe readiness before any full GPU run.

## Final training handoff preparation — 2026-11-08

The owner requests the final version after the evidence/probe decision. Initial
changes are limited to the two preceding audit/method reports; HEAD remains
`1cfe181`. Fresh process inspection again shows no local training process.
The baseline source and prior CPU validation are available, but real-data
CUDA readiness and benchmark gains remain unverified on this Mac.

Promote the tested combination into explicit portable `final` profiles for
Cars/CUB and both methods; preserve reference and row-normalized upgraded
profiles. The default user entrypoint will select final, with an explicit
reference option for the earlier settings. Final preserves legacy USE, the
official dataset loss/optimizer parameters, acceptance guards and budgets;
it enables global frozen-feature shortlisting and larger legal probes.
Update the manuscript/config table, commands and release provenance together.
Run bounded CPU checks, rebuild the PDF with the existing temporary compiler,
then package and verify a fresh source archive. No full training, downloads of
large inputs, checkpoint mutation or remote publication is part of this step.

Final handoff checks passed in CPU run
`crish-v21-20261108T110851Z-869972ef`: 127 tests, zero failures/errors/skips;
Cars/CUB source and split checks; all 24 full/smoke config CLI comparisons;
static syntax; archive extraction/execution without Git; manuscript/PDF hashes.
The existing temporary Tectonic compiler rebuilt the PDF (7 pages, no overfull
boxes or undefined references; compiler font/underfull warnings remain).
Matched seed-1 plan `runs/plans/matched-scars-final-20261108T110832Z-9818ccfa/`
has six successful launcher dry runs and remains `prepared_not_executed`.
All 109 recorded executable/config/test/paper input hashes still match the
validation receipt. Only these completion notes are added after the run.

Shipped changes: eight final configs; default CLI and plan profile; protocol
and source-archive metadata; matching README, manuscript/PDF and research
notes; default-profile and final/legacy synthetic regressions. Reference and
upgraded configs, upstream, environments, logs and historical weights remain
preserved. CUDA, actual images/weights, full-size probe runtime and benchmark
accuracy remain unverified here. Next remote command after extracting the
handoff and preparing data/weights/environment:
`bash scripts/train_crish.sh check`, followed by `bash scripts/train_crish.sh smoke`.
Full training is finite but is not started by this handoff.

## Final profile GitHub publication — 2026-11-08

The owner explicitly requests pushing the final source. Current branch is
`main`, remote `hokylhoangkien/gcd-final` is public, and a fresh remote read
matches local parent `1cfe18133dadbc08dee36c5a9b5bd7ac729b9ff0`. No training
process is visible locally. The 25 reviewed candidate paths contain source,
eight final configs, tests, research notes, manuscript/PDF and the CPU receipt;
all 109 recorded executable/config/test/paper input hashes still match the
127-test passing receipt. Manuscript/PDF consistency and diff whitespace checks
pass; candidate text contains no detected credential/private-key patterns.
No data, weights, checkpoints, environments, raw logs, generated plans or
release archives are included. The GitHub connector has read permission only;
use configured local Git credentials for a normal commit/push to `main`.
Do not force-push or alter history, visibility, upstream, or running processes.

## Per-epoch metric output inspection — 2026-11-10

Owner asks whether All/Old/New are exported for each epoch. Starting HEAD:
`722c9bb4c6fb1c7e21fc3eb5e7ca55055b3fefc4`; working tree and pinned HypCD
submodule were clean. Read-only process inspection required sandbox escalation
and succeeded: Codex/VS Code services are present, with no visible local Python
training process. Remote training state was not inspected. Host is macOS arm64;
`nvidia-smi` is absent, and unprivileged hardware `sysctl` was denied. Prior
audits record an M5/16 GB development machine. No training was started or stopped.

Current baseline/CRiSH profiles set `test_every_epochs=1`: completed disjoint-test
evaluations write `test_metrics.jsonl`, per-epoch `test_evaluations/` reports,
and percentage All/Old/New lines in `run.log`. `ss_eval_every_epochs=0` disables
intermediate anchored SS monitoring by default; `--eval-every 1` enables it for
each epoch, writing `epoch_metrics.jsonl`. These are different protocols and
must remain separate. Final best/last SS evaluation is already supported.

Verification: `.venv/bin/python scripts/show_metrics.py crish_share` exited 0
and showed three observed epoch losses with `not_evaluated` accuracy cells.
Those supplied historical files contain no per-epoch All/Old/New; missing
accuracy cannot be reconstructed from loss alone. Active `runs/` contains plans,
not completed training runs. Historical backup checkpoints/reports are preserved.
No runtime/config/test/manuscript changes or benchmark evaluations were needed.
This audit note is the only edit. Full baseline readiness on Linux NVIDIA still
requires the existing data/weight/CUDA preflight and bounded real-data smoke.

Next read-only command on the machine holding a real run:
`.venv/bin/python scripts/show_metrics.py runs/RUN_ID`.

## Epoch 188 versus best epoch 90 inspection — 2026-11-10

Scope: explain a reported accuracy regression; preserve all active training,
source, configs and pre-existing audit edits. Starting HEAD remains
`722c9bb4c6fb1c7e21fc3eb5e7ca55055b3fefc4`. Only this audit file was already
modified at turn start. Read-only escalated process inspection found IDE/Codex
services and no visible local Python training process. Remote job health and
source/config identity remain unverified. No process was stopped or started.
Local checks: macOS arm64, 10 CPUs, torch 2.4.1, CUDA=False, MPS=False,
`nvidia-smi` absent. Earlier audit records M5/16 GB; this session's unprivileged
memory query was denied. Active `runs/`, supplied `crish_share/` and
`logs/remote-audit/` have no checkpoint files. Historical backup artifacts are
preserved and are not evidence for the current run. Local full-data/CUDA
baseline readiness is still blocked; no benchmark is run here.

User-reported observations, not locally verified: epoch 188
All/Old/New = 68.08/75.79/60.64 percent, selected epoch 90 Old = 84.3 percent.
The Old difference is -8.51 percentage points. No epoch-90 All/New, current
run history or repair diagnostics are available to establish the All/New
trend or a causal explanation. Searches found no current artifact matching
this complete triplet. Do not attribute the regression to overfitting,
CRiSH repairs, data corruption or k-means randomness without further evidence.

Verified current implementation: `gcd_runtime/selection.py` selects on strict
Old improvement and preserves the earlier best bundle. `crish/training.py`
continues SGD on the current model; selection does not restore the best model
or stop training. Thus best and current scores can diverge. Per-epoch plain
disjoint-test k-means differs from final anchored SS train-unlabeled evaluation;
scores from these protocols must not be directly compared. Existing protocol
and novelty reports remain applicable; no research-method change is justified
by the supplied two observations. The safest next evidence is the saved
`selection.json`, `test_metrics.jsonl` and config snapshot for this same run;
inspect through `scripts/show_metrics.py` without rerunning training. No loss,
selector, data adapter, schedule or upstream code is changed.

### Clarification: regression against earlier training runs

The owner clarifies that earlier runs were close to paper-reported scores;
the comparison of interest is across versions, not solely epoch 90 versus 188.
Historical artifacts substantiate high earlier SS scores: run
`20261031T055221Z-03368e3d` has SS window means near 82 percent All, and
its final backbone's separate completed evaluation
`20261031T172513Z-881c7f46/evaluation.json` records balanced SS train-unlabeled
All/Old/New = 81.2337/86.6567/78.6145 percent. Evaluation metadata explicitly
names the earlier run's `backbone_final.pt`. The earlier training log's last
plain disjoint-test result is 56.24/65.20/47.59 percent. This is direct evidence
that different evaluation pools/clustering can produce substantially different
numbers for the historical final model; it does not quantify the effect in
the current remote run. The current reported epoch-188 protocol is still
unverified; current shipped defaults print plain disjoint-test each epoch.

Training also differs materially: the archived config uses `method=proposed`
with reciprocal/cross-view pair reliability, `hyper_max_weight=1.0` and
`final_epoch` selection. Current final config uses CRiSH partition repairs,
`hyper_max_weight=0.5`, and Old-test selection. Current runtime also has source
fixes documented in reports 01/07. Therefore neither "only evaluator changed"
nor "new method is worse" is established. First compare saved final SS best/last
reports under matching evaluation; if the regression persists, isolate method,
loss coefficients and source fixes with matched B0 controls. Do not silently
restore upstream ordering bugs or tune to historical test labels. This turn
only reads archives and extends the audit; no code/config/job is changed.

## SelEx + CRiSH implementation — 2026-11-10

Owner clarifies the new objective: apply CRiSH to updated SelEx rather than
Hyp-SelEx. Preserve the legacy HypCD pin, profiles, manuscript, results and
existing audit edits. Starting HEAD is `722c9bb`; initial edits are the two
audit reports from this conversation. Fresh read-only escalated process query
shows IDE/Codex services, no local Python training job. No process is changed.
Local CUDA remains unavailable; only bounded synthetic CPU tests are used.

An isolated official checkout in `/private/tmp` verified current SelEx HEAD
`d8107b5458f3337b2d5342f79345a976037a1265` (2026-11-09). A filtered, unmodified
source/split snapshot is added under `upstream/SelEx` with a hash manifest;
no images, weights or environment installation is included. This resolves
the browser's mixed cached config/training reads noted in report 07. The
actual pinned Cars config specifies 50 epochs, block 9, smoothing .75 and
temperature .5. Source head output is 65,536, and final SS defaults to ten
initializations. These must be tested as a new protocol, not mixed with old
Hyp-SelEx settings. Data, pretrained files and NVIDIA compute are not available
locally; full baseline reproduction and benchmark claims remain blocked.

Implemented the separate `selex_runtime` path and `scripts/train_selex.sh`,
four matched full/smoke configs, immutable SelEx snapshot, documentation and
ten new CPU regressions. Shared CRiSH gained explicit objective injection;
historical Hyp-SelEx retains its defaults. SelEx loss/gradient and coarse
grouping match executed pinned-source statements on synthetic fixtures;
repair/target-drift see only currently active hierarchy levels. Updated SelEx
fixed-final SS-v2 is predeclared as primary; per-epoch plain-test is monitoring
only, not the historical Hyp-SelEx Old-v2 selector.

Full CPU validation `crish-v21-20261110T073222Z-6b4a6807`: 137 passed,
zero failures/errors/skips, including legacy regression and portable archive
execution. The historical manuscript now explicitly states its older scope;
its existing compiler rebuilt the PDF and receipt after provenance updates.
No real images/weights/CUDA were used by the test suite. Read the resource
limits, environment instructions and unverified GPU gates in
`reports/08_selex_crish_migration.md`. No GPU benchmark, long training or sweep
was launched. Next remote action is `bash scripts/train_selex.sh check --baseline`.

## Authorized GitHub handoff — 2026-11-10

Owner explicitly requests pushing the prepared code so a colleague can train.
Branch is `main`, origin `hokylhoangkien/gcd-final` is public, and the GitHub
read-only connector reports remote main at local parent `722c9bb` before push.
All current runtime hashes match the passing 137-test CPU receipt. SelEx
snapshot integrity and historical manuscript/PDF provenance pass; diff
whitespace checks are clean. No algorithm/config changes are made in this
publication step. CUDA, real data and actual-head repair resource use still
require remote preflight and bounded smoke.

Publish the bounded reviewed source/config/test/report paths, pinned SelEx
source/split metadata with MIT notices, and local validation receipt. Exclude
data images, weights, checkpoints, environments, raw logs, local plans and
handoff archives. GitHub connector permissions are read-only; use configured
local Git credentials for a normal commit/push, without force or history rewrite.
The separate SelEx entrypoint and fixed-final SS policy are documented for the
colleague. Do not contact the colleague or start remote training automatically.

Staged review contains 73 files. Runtime bytes match the 137-test receipt and
all staged SelEx files match the pinned snapshot hashes. The vendor snapshot
retains upstream trailing whitespace; own-source diff checks pass with that
unmodified third-party directory excluded. No credential-pattern matches,
images, weights, checkpoint files, raw logs or release archives were staged.
