# A2, A1 and CRiSH-P: implementation contract

Scope clarification, 2026-11-12: this is the **optional CRiSH-P / controls
handoff**, not a claim that population abstention replaces local CRiSH's
Keep/Abstain/Move algorithm. The requested SelEx + local CRiSH training guide
is [report 14](14_selex_local_training.md). Raw defaults below remain policy;
use explicit `--variant local` for the local algorithm.

Date: 2026-11-11. This is an experimental implementation and handoff, not
evidence of increased accuracy, cleared novelty or GPU verification. The owner
explicitly freezes all paper/LaTeX/PDF files. Preserve existing local work.

## A2: structured legal probes

Add `probe_known_per_class`. Value 1 retains historical round-robin sampling;
values above 1 require exact class quotas, at least two classes per fold and
sufficient distinct images per class. Discovery and confirmation use shared
sampled class identities but disjoint image IDs; no repeated image fills a
quota. Rotate class subsets by the refresh seed. Default structured comparison:
16 images/fold, 8 classes x 2 images. Log actual per-class image counts and
cross-image same-class support. This improves the probe's available signal,
not a guarantee about final clustering Old accuracy.

Local repairs with acceptance budget above one reserve a fresh disjoint
confirmation fold for each possible accepted edit, rather than repeatedly
selecting against the same confirmation data. All folds are excluded from
the scoring/evidence image cache. Multiple queries across epochs remain a
limitation; these are training-only probes, not an independent final test set.

## A1: dose plans and measured intervention

Prepare configs/commands for one-axis changes in refresh frequency, evidence
cache, boundary count and accepted-action cap. Disable online weighting and
population policy for local-dose experiments. Log actual accepted edits,
changed IDs, changed positive relations, target mass and fallback reasons;
nominal budgets are not realized dose. `max_accepted` cannot exceed distinct
shortlisted samples in effect. Keep gain/known-guard scores with measured dose
for offline comparison to observed All/Old/New, time and memory. Plans do not
execute sweeps. Matched seeds are generated only; no GPU run starts locally.

## B1: population abstention policy (CRiSH-P)

Counterfactual candidates choose a predeclared fraction q in {0,.05,.1,.2}
of low-reliability unlabeled samples. q=0 is Keep. Unlike the local 128-row
augmentation cache, population ranking must cover the entire frozen training
snapshot. Use a blockwise geometric assignment margin against occupied base
prototypes: r=.5+.5*(d_other-d_own)/(d_other+d_own+eps), with squared Euclidean
distances and score bounds [0,1]. This is not classifier confidence or global
cross-view kNN. Empty prototype slots are excluded as alternative evidence.
No varying geometric evidence returns Keep instead of arbitrary ID-based
rejection. Ranking uses training images/pseudo-assignments only, never unknown
ground truth or evaluation metrics. Its cost is O(N*K*D), bounded in blocks.

Choose lowest-score samples globally, subject to per-pseudo-cluster retention
and minimum retained base pseudo-SSE coefficient mass. Preserve known targets,
same-ID/labeled-labeled positives, original positive denominators, USE,
coarse levels and all predictions at evaluation. q concerns training supervision
only. Actual sample coverage and removed pair mass can differ from nominal q.

Score each nonzero admissible policy on a bounded training-gradient cache,
uniformly drawn without probe IDs, using the real SelEx objective and a copied
projector SGD with its live optimizer state. Ranking is a transductive function
of the baseline's full frozen snapshot; probe augmentations and sample IDs
remain excluded from the gradient cache. This does not establish statistical
independence of an external validation set. Confirm only the top eligible
policy on the second probe; failure returns Keep without runner-up search.
Policy corrections expire at next extraction. Fixed/scheduled/random-ranking
controls use the same guards, not an oracle choice of test-optimal q.

Local and population repair are mutually exclusive in a config. A hybrid may
combine population policy with online pair weights; default experiment should
keep those weights off to identify the new policy's effect. Baseline bypasses
every intervention. Keep original local and online variants selectable.

## Novelty and evidence gate

This is an explicit implementation hypothesis, not a new-name novelty claim.
Counterfactual one-step selection overlaps with
[Learning to Reweight Examples](https://proceedings.mlr.press/v80/ren18a.html);
sample-selection GCD overlaps with
[Let's Start Over](https://ijcai.org/proceedings/2024/532), while coverage-aware
rejection has precedents such as
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).
The potentially distinguishing test is a globally applied, coverage-constrained
base pseudo-supervision policy, selected with training-only structured probes
and winner-only confirmation in non-parametric GCD. Method-level RCA/LEAH
comparisons remain open as recorded in reports 02/11. Required controls include
guarded fixed/scheduled/random policy candidates, local CRiSH, online-only and
baseline with the same source, split, backbone and fixed-final SS-v2 endpoint.
No superiority, causal calibration, novelty or broad generalization is claimed.

## Portable handoff and launch

The main eight 200-epoch configs now select the policy experiment, structured
8-way/2-shot probes, no local repair and no online weighting. Baseline configs
are identical except `method`, which bypasses every intervention. `--variant
local|online|policy|hybrid` selects the other mechanisms explicitly. Historical
20/50-epoch configs stay unchanged. `--no-repair` disables both controllers;
it does not enable online weighting by itself.

Local code checks (no data, weights, CUDA or training):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py --code-only
bash scripts/train_selex.sh --dataset scars --variant policy --dry-run
.venv/bin/python scripts/prepare_crish_experiments.py --dataset scars --suite all --seeds 1
```

The planner writes a unique manifest, per-config hashes, seed, code fingerprint,
Git revision and individual commands under ignored `runs/plans/`. Recreate it
on the training machine after Git transfer; local generated paths are not
portable artifacts. `--suite dose` prepares one-axis A1 contrasts; `controls`
prepares A2 shot counts and guarded fixed/scheduled/random/online/hybrid controls.
Use seeds 1/2/3 only when compute permits. No script automatically executes
the prepared jobs or selects parameters from evaluation scores.

On Linux NVIDIA, use an existing prepared interpreter via `SELEX_PYTHON`, or
the compatibility environment described in the repository (`GCD_PYTHON=python3
bash scripts/bootstrap.sh cuda`, Python 3.9–3.11). Bootstrap now runs code-only
checks to respect the frozen paper. It does not download data/weights. Its
Torch 2.4.1 compatibility profile is distinct from upstream's newer Python 3.12
requirements; record the actual environment and compare both arms within it.
No CUDA compatibility is asserted from the local CPU suite.

With official dataset layout/pretrained DINOv2 weights already present:

```bash
bash scripts/train_selex.sh check --dataset scars --baseline
bash scripts/train_selex.sh smoke --dataset scars --baseline
bash scripts/train_selex.sh smoke --dataset scars --variant policy
bash scripts/train_selex.sh --dataset scars --baseline
bash scripts/train_selex.sh --dataset scars --variant policy
```

These are individual foreground stages; run one GPU job at a time. The full
train command repeats preflight and bounded smoke before starting a fresh full
run. Derived smoke configs use controller warmup 0 and policy schedule length
1 to exercise the branch within two batches; full configs retain warmup 10,
refresh 5 and scheduled-control ramp 30. Smoke accepts a legitimate Keep result
after actual scoring, but refuses fallback/no-scoring before full training.
Do not bypass that gate to claim the method ran. Full worker/evaluator deadlines,
checkpoint/resume and best/last separation remain unchanged (report 09).

New outputs: `crish_p.jsonl`, `intervention_diagnostics.jsonl`, optional repaired
`supervision_epoch_*.npz`, and existing `metrics.jsonl`/`crish.jsonl`/pair histograms.
Global target mass is computed analytically before online weighting. In hybrid,
an additional actual weighted scoring-batch mass guard is enforced; no claim
about full-population online-weighted mass is made. Positive-relation topology
counts and changed coefficient mass are separate quantities. Online diagnostics
exclude positives already removed by abstention, avoiding double-counted dose.

After completed experiments:

```bash
.venv/bin/python scripts/summarize_crish_experiments.py runs --output results/crish_observed.csv
```

Only observed fixed-final SS-v2 results enter the CSV. The companion summary
retains exact per-seed values and computes mean/sample-std only for distinct
seeds with matching config/source, pretrained/annotation/split provenance and
environment. Single-seed std is null; duplicate seeds or missing provenance
are not pooled. No empty result is replaced with an estimate.

Git preparation (review and execute deliberately; this task does not push):

```bash
git diff --check
git status --short
git add README.md configs crish gcd_runtime selex_runtime scripts tests reports results/local_validation.json
git diff --cached --stat
git commit -m "Add structured probes and population CRiSH experiments"
git push origin HEAD
```

The explicit paths leave frozen paper edits from earlier sessions unstaged.
Datasets, weights, logs, checkpoints, plans and archives remain ignored. Keep
old checkpoints with their original source/config; the new fingerprint will
reject old-source optimizer resume. Baseline/accuracy reproduction, multi-seed
benefit and the novelty gate remain empirical/research work after this handoff.

## Completed local verification

Final command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python
scripts/run_smoke.py --code-only` — **231 passed, zero failures/errors/skips**.
Each child command has a 60-second timeout. Receipt:
`results/local_validation.json`; run ID
`crish-v21-20261111T035336Z-c7ba71b6`; log:
`logs/crish-v21-20261111T035336Z-c7ba71b6/smoke.log`.
Split/protocol/static checks, source hash verification, source DINOHead
counterfactual scoring and archive execution outside Git all passed.

Tests cover exact structured quotas and fresh confirmation folds for multiple
local acceptances; global occupied-center margins, empty slots and block
deadlines; quota/mass guards, protected labels and coefficient semantics;
real scoring hidden-target invariance and no live optimizer/RNG mutation;
winner-only confirmation, Keep fallback and no runner-up search; actual SGD
consumption/expiry of a population mask and exact synthetic resume; one-axis
plans, CLI variants, smoke gating, and provenance-aware per-seed aggregation.
An earlier full attempt had 229 passes and one test-fixture failure: the fixture
assumed every q was scored, while the new retained-mass guard correctly rejected
one candidate. The fixture now identifies the actual confirmation fold and
counts only scored candidates; the guard was preserved. Original failed logs
remain available. This is not evidence of a benchmark accuracy change.

Actual A1/A2/control plan created, not executed:
`runs/plans/crish-experiments-20261111T035332Z-c6db35d7/` (18 named profiles,
seed 1). `local_reference` and `known_8way_2shot` describe the same reference
settings under different analysis roles; reuse that result rather than treating
the aliases as independent experiments. Portable generator source is tracked;
ignored generated local plans should be recreated on the target checkout.

SHA256 comparisons confirm all three paper files unchanged from task start.
The manuscript gate is `not_run_code_only`, deliberately not reported as passed.
CUDA/data/weights and full-size real-data policy costs remain unverified. No
training sweep, large download, process termination, commit or push occurred.
Only documentation evidence was appended after the final test snapshot; tested
runtime/config/test source remained unchanged.
