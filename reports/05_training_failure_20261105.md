# CRiSH hierarchy failure investigation — 2026-11-05

## Observed failure

Supplied run `20261105T042348Z-66079b2b` used source `918011c`, NVIDIA A40
and torch 2.4.1+cu121. Epochs 0, 1 and 2 completed (63 steps each), with losses
4.1413756893, 3.9643610546 and 3.8742356868. Their observed peak allocations
were approximately 23.0 GB decimal. There are no benchmark accuracy results.

During the next HSSK extraction, `EpochRuntime.capture_hierarchy` called
`prototype_radii`, which raised `ValueError: Invalid prototypes for radii`.
The radius guard can reject malformed/fewer-than-two/nonfinite centers. The
fixed Cars hierarchy always requests at least two centers. The supplied files
do not include the failed tensors or `resume.pt`, so the exact failing level
cannot be established from this log alone.

## Reproduced producer defect

The pinned semi-supervised k-means computes `selected.mean(0)` without
checking for empty assignments in both its update phases. An empty cluster
therefore produces a NaN center; later assignments may propagate that NaN.
Its k-means++ also divides by zero when all remaining distances are zero.
This reachable defect is a strong explanation for the log, not proof of the
absent failing tensor's contents. The CPU regression runs the original source
with duplicate starting novel centers and reproduces nonfinite prototypes
followed by exactly `Invalid prototypes for radii`.

## Correction

`gcd_runtime/kmeans.py` preserves the source's two bounded phases and balance
heuristic. Before averaging, each empty cluster receives a non-anchor row
with maximal squared residual, from a donor containing more than one row.
Ties use row order; true anchors remain fixed and the number of clusters stays
unchanged. Subsequent means are over nonempty assignments. A zero-distance
initialization uses a bounded deterministic fallback. Invalid input features
still fail explicitly; NaNs are never silently replaced by zero.

The adapter is imported once by the shared trainer, including final clustering
evaluation, so B0 and CRiSH use the same numerical policy. Source losses,
learning-rate schedule, hyperbolic mapping, configs and upstream files remain
unchanged. Log lines report empty-cluster repairs and degenerate initialization
counts; error messages now identify hierarchy level and bad cluster IDs.
Preflight runs an actual small collapsed-feature HSSK fit on the selected
device; the older synthetic training fixture had mocked HSSK entirely.

## Verification and rerun constraints

Seven new targeted CPU regression tests passed, including direct reproduction
of the original failure, exact labels/centers/inertia parity on nondegenerate
balanced and unbalanced source fixtures, collapsed-feature termination,
anchor/singleton protection, invalid inputs, error context and four actual
hierarchy refresh/loss-backward cycles. Full-suite receipt is recorded in
`results/local_validation.json`: run `crish-v21-20261105T051250Z-6e24b2b2`,
79 passed, zero failures/errors/skips, full suite plus split/SHA/syntax checks.
The suite also builds/extracts a source archive and executes its bounded smoke.
Log: `logs/crish-v21-20261105T051250Z-6e24b2b2/smoke.log`. The layout test was
updated to permit the audit reports explicitly required by `AGENTS.md`; the
existing release whitelist still excludes reports and supplied private logs.

These synthetic tests do not establish CUDA correctness or completion of 200
real-data epochs. Updated preflight and real-data smoke must pass on the A40.
Use the new source package, preserve old run directories and use the commands
in README. A new run starts at epoch zero: the existing source-fingerprint
check intentionally refuses to resume an old-source checkpoint under this
modified source. No checkpoint migration or fingerprint bypass is introduced.
The old checkpoint hash remains evidence; a checksum file is not a checkpoint.
