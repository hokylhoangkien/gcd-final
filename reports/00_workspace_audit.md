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
