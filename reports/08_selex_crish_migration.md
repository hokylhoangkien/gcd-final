# SelEx + CRiSH — implementation and verification scope

Date: 2026-11-10. This is a new experimental path, not a benchmark result or a
novelty claim. Legacy Hyp-SelEx/CRiSH commands and profiles remain available.

Current extension: the main launcher now selects matched **200-epoch** profiles
for Cars, CUB and FGVC-Aircraft, with CIFAR-100 retained as an optional dataset.
See [the current runbook](09_selex_200epoch_multidataset.md).
The original 20/50-epoch upstream settings and initial adapter history below
remain provenance, not the default budget of these new experiments.

## Pinned foundation

Official SelEx commit `d8107b5458f3337b2d5342f79345a976037a1265`, dated
2026-11-09. The unmodified 38-file source/split snapshot is hash-verified in
`upstream/SelEx.SNAPSHOT.json`; original MIT/Meta notices are retained. This
checkout resolves stale browser config reads; do not mix sources from main
at different dates. No dataset images, weights or private paths are bundled.

Cars/DINOv2 defaults from this source: 50 epochs, batch 128, SGD lr .1,
momentum .9, weight decay .0001, trainable blocks from 9, smoothing .75,
temperature .5, output dimension 65536. The head is Euclidean; normalization
occurs before USE/SSE and again inside the source InfoNCE helper. The pinned
loss definitions are loaded from their AST without executing source main.
Source-clamped 1D USE broadcasting is preserved in both arms; this is not the
optional row-normalized Hyp-SelEx experiment.

Coarse levels independently group all level-0 prototypes. Level i becomes
active at epoch 10*i (zero-based). Only active levels enter the state supplied
to loss, repair, target drift and protected-coefficient checks. Known labels
remain compressed/legal; all unlabeled training targets are sanitized to -1.

## CRiSH intervention

The existing evidence cache, reciprocal cross-view scores, legal ID-disjoint
probes, Keep/Abstain/Move candidates, exact cloned projector SGD and independent
confirmation are reused. Both keep/action gradients now evaluate the SelEx
objective. Main training still updates the selected backbone blocks and head;
the virtual step updates only a copied projector. No ToPoincare or hyperbolic
distance is used on this path.

Move edits one active level independently, rebuilds the affected prototype
means/radii and rederives USE/SSE. It intentionally intervenes on the original
coarse assignments; it does not propagate a sample across a guaranteed tree.
Abstain retains genuine supervised/same-image positives and original positive
counts. Corrections expire at the next epoch's fresh clustering. Empty or
invalid cluster registries fail explicitly rather than fabricate samples.

`configs/selex_baseline.json` and `selex_crish.json` differ only in `method`.
The original Hyp-SelEx profiles are not overwritten or relabeled. Old resume
checkpoints require their original source; do not resume them into SelEx.

## Evaluation policy, declared before new benchmark runs

The updated SelEx README reports last-checkpoint SS-v2. This path therefore
uses fixed-final SS-v2 as its primary endpoint for both arms, not the old
Hyp-SelEx Old-v2 best selector. The six 200-epoch configs additionally implement
the source's best-Old-v1 selector as a secondary report: plain disjoint test
records both v1/v2, Old-v1 alone selects best, and both best/last get final
SS-v2 evaluation. Selection never stops SGD, feeds repairs or overrides the
fixed-last primary. Historical 50-epoch configs retain secondary selection off.

Final SS fits normalized backbone features on the train pool with legal
labeled anchors and scores only unlabeled rows. Source fitter: ten restarts,
ten iterations per phase, clustering seed 0, balanced capacity. Global
Hungarian-v2 precedes Old/New restriction. Grouped outputs are mapped back to
sample order explicitly. Evaluation runs in bounded workers, with the RNG
protocol below. Physical process separation does not imply RNG isolation for
the mandatory plain-test monitor.

### RNG alignment correction (2026-11-10)

Shipped SelEx configs now declare `evaluation_rng=upstream` for both arms.
Training uses the source main's `np.random.seed(seed)` and
`torch.manual_seed(seed)` calls, without adding Python/cudnn seeding. Each
epoch's mandatory plain-test consumes the training RNG stream as upstream
does: batch size 128, Git-pinned plain k-means, `iter_limit=1000`. A worker
receives the training RNG state after SGD, discards setup-only random draws,
and returns RNG state only. No labels, predictions or metric values are
returned to training. Snapshot hash, config, epoch and a request nonce bind
each exchange; stale results and extra fields are rejected. Metric reports
remain separate and never select the fixed-final checkpoint.

The resume checkpoint is saved after the RNG exchange. A bounded evaluator
failure instead saves completed SGD with `plain_rng_pending=true`; resume
replays only that evaluation before proceeding, including at the last epoch.
Optional extra SS monitoring remains independent of the training RNG stream.

Standalone SS executes the pinned source's `seed_torch(0)` **before dataset
construction**, preserving the NumPy draws made by source subsampling and
train/validation bookkeeping. It does not reset the stream immediately before
fitting. Features are normalized as in source extraction and again as in the
source feature-cache evaluator. Reports record the RNG policy, seed stage
and state digest before fitting. The former `isolated` policy remains
explicitly supported for old config inspection; changing source still
invalidates old resume fingerprints. Use the original release for old runs.

The extended adapter also supports upstream's extra train-unlabeled plain
evaluation every `train_report_interval` (default 200) epochs, before disjoint
test in the same RNG exchange. CPU fixtures verify source seeding, a fresh-process RNG exchange,
next-step RNG parity and interrupted/resumed SGD parity. This is not a CUDA
benchmark or proof of bitwise agreement across hardware/library versions.

Outputs are under `GCD_RUNS_ROOT/RUN_ID` or project `runs/RUN_ID`. Primary
files: `evaluation.json` and `evaluation_last.json`; plain history:
`test_metrics.jsonl`; optional/final SS history: `epoch_metrics.jsonl`;
training loss/time: `metrics.jsonl`; summary: `results_summary.json`.
Keep `config.json`, `metadata.json`, `environment.txt`, `readiness.json`,
`run.log`, `backbone_final.pt` and `resume.pt` for provenance/resume. The
200-epoch profiles additionally retain `backbone_best_old_test_v1.pt`,
`selection.json`, `evaluation_best_old_test.json` and `best_old_ss_metrics.jsonl`.
Their best bundle is for SS evaluation, not optimizer resume. Resume requires
the original run's best checkpoint and test-evaluation history as well as
`resume.pt`; see report 09.

## Remote commands

Use a separate environment to preserve an existing Hyp-SelEx installation.
The upstream environment specifies Python 3.12 and pinned requirements; its
README explains CUDA-driver-compatible Torch installation. No environment or
large weight download was performed locally. An example when those CUDA wheels
are compatible with the target GPU/driver:

```bash
python3.12 -m venv .venv-selex
.venv-selex/bin/python -m pip install -r upstream/SelEx/requirements.txt
export SELEX_PYTHON=.venv-selex/bin/python
export GCD_DATA_ROOT=/path/to/datasets
export GCD_WEIGHTS_ROOT=/path/to/weights
export GCD_RUNS_ROOT=/path/to/runs
```

Run one stage/job at a time, after confirming the GPU is available:

```bash
bash scripts/train_selex.sh check --baseline
bash scripts/train_selex.sh --baseline
bash scripts/train_selex.sh
```

The train command runs bounded real-data smoke before a fresh 50-epoch run;
each launcher stage includes preflight. Nothing above is an automatic sweep.
For review without GPU, append `--dry-run`. For another seed, copy the full
config and change only seed, then use `--config PATH` for the corresponding arm.
For resume use the original run config and original source:

```bash
bash scripts/train_selex.sh --baseline --config runs/RUN_ID/config.json --resume runs/RUN_ID/resume.pt
```

Preflight is 180 seconds; smoke training/monitoring is at most 1800 seconds.
Historical 50-epoch profiles cap full training/monitoring at 86400 seconds and
each SS/plain evaluator at 1800 seconds. New 200-epoch profiles instead use
604800 seconds and 7200 seconds respectively (report 09).
These are stop limits, not expected completion times. Head output
is eight times wider than the old profile; cloned-head memory and actual runtime
must be measured. Do not shrink only one arm's batch/probes after seeing scores.

## Evidence boundary

CPU tests cover executed source-loss/gradient parity across the level-activation
boundary, finite protected Abstain, hidden-target invariance of real repair
scoring, disabled-CRiSH parity, resume, source hashes and CLI selection. Fresh
process imports and an actual source DINOHead CPU backward are checked. These
do not verify CUDA, full-size head repair costs, real-data clustering behavior,
benchmark accuracy, multi-seed gains or cross-dataset generalization. The initial
adapter supported Cars only; the later three-dataset extension is documented
in report 09. Use the checksummed run receipt for actual test
counts/status. The historical manuscript still describes Hyp-SelEx CRiSH;
it must not be submitted as a description of this new SelEx experiment.

## Completed local verification

Full bounded CPU receipt `crish-v21-20261110T073222Z-6b4a6807`:
**137 passed, zero failures/errors/skips**. Command: `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py`; every child
command has a 60-second deadline. Includes old Hyp-SelEx regression tests,
ten SelEx-specific tests, Cars/CUB legacy split/source checks, all legacy CLI
profiles, static syntax, source-archive extraction/execution and manuscript/PDF
provenance. The initial restricted attempt had one macOS OpenMP SHM subprocess
failure and a stale historical manuscript digest. Granted CPU execution and a
reviewed scope note/provenance/PDF rebuild resolved those checks; no loss or
protocol was tuned to make a benchmark score pass.

Receipt: `results/local_validation.json`. Full runner log:
`logs/crish-v21-20261110T073222Z-6b4a6807/smoke.log`. Earlier targeted/source-head
checks and failure evidence: `logs/selex-port-20261110T072611Z/`. No benchmark
or NVIDIA job was started; CUDA remains explicitly unverified.

`--seed N` creates an isolated config snapshot for a matched rerun;
`--eval-every N` enables additional SS monitoring. Custom-config/seed smoke
inherits that config and changes only bounded smoke settings. Each run saves
the effective config. Example dry-run only:
`bash scripts/train_selex.sh --baseline --seed 2 --eval-every 5 --dry-run`.

Next command on the prepared Linux NVIDIA machine:
`bash scripts/train_selex.sh check --baseline`. Do not start full training
until this and the bounded real-data smoke succeed.
