# SelEx / SelEx + CRiSH: three datasets, 200 epochs

Date: 2026-11-10. User-authorized extended-budget experiments; no benchmark
results are asserted. Both methods use the corrected, pinned **Euclidean
SelEx** source `d8107b5458f3337b2d5342f79345a976037a1265`. Pinned upstream files
are unchanged. Historical Hyp-SelEx profiles/checkpoints are separate.

Current main matrix: **Stanford Cars, CUB and FGVC-Aircraft**. The user replaced
CIFAR-100 with Aircraft; CIFAR configs/CLI support remain optional prior work.

## Matched experiment matrix

| Dataset flag | Known / novel classes | Labeled fraction of known-class images | Epochs | Temperature | Smoothing | Trainable blocks from | Workers |
|---|---|---|---|---|---|---|---|
| `scars` | 98 / 98, pinned SSB | 0.5 | 200 | 0.5 | 0.75 | 9 | 4 |
| `cub` | 100 / 100, pinned SSB | 0.5 | 200 | 0.25 | 0.75 | 10 | 4 |
| `aircraft` | 50 / 50, pinned SSB | 0.5 | 200 | 0.5 | 0.75 | 9 | 4 |

All six main configs use DINOv2 ViT-B/14 **with four registers**, input 224,
backbone feature dimension 768, projector dimension 65536, batch 128,
SGD lr 0.1, momentum 0.9, weight decay 0.0001, supervised weight 0.35 and
cosine annealing over all 200 epochs to 0.001 times the initial lr. Within
each dataset, baseline/CRiSH configs differ only in `method`. CRiSH is the
sole intervention; its settings are shared across datasets, warmup 10 epochs.
The baseline skips CRiSH even though both config snapshots retain its settings.

These 200-epoch runs **do not reproduce the corrected 2026 README training budget**:
the pinned source's defaults are Cars/CUB/Aircraft 50 epochs (optional CIFAR-100
20 epochs). Compare
our matched baseline and CRiSH at 200 epochs; label any separately quoted
upstream scores with their own budget/backbone/protocol. More epochs do not
guarantee higher All or New accuracy. No tuning uses unlabeled evaluation labels.

Clarification after checking original sources: released **2024 code defaults to
200 epochs**. The 20/50 schedules above refer only to the corrected 2026 source,
not evidence about the original paper's training budget. There are other method,
hyperparameter, backbone-row and checkpoint differences; see
[the paper-comparability audit](10_paper_comparability_audit.md).

## Inputs and compute

Use the separate Linux NVIDIA/Python environment described in report 08.
Set `SELEX_PYTHON` to that environment's Python. Paths are portable through
`GCD_DATA_ROOT`, `GCD_WEIGHTS_ROOT`, `GCD_RUNS_ROOT`; defaults are the project
`datasets/`, `weights/`, `runs/` directories. Required layout:

```text
datasets/
  cars/
    cars_train/                         # original train images
    cars_test/                          # original disjoint test images
    devkit/cars_train_annos.mat
    devkit/cars_test_annos_withlabels.mat
  cub/CUB_200_2011/
    images/
    images.txt
    image_class_labels.txt
    train_test_split.txt
    attributes/image_attribute_labels.txt
  aircraft/fgvc-aircraft-2013b/
    data/images/                       # original JPEG images
    data/images_variant_trainval.txt
    data/images_variant_test.txt
weights/dinov2_vitb14_reg4_pretrain.pth
```

Aircraft uses the pinned **100-variant** task with `trainval` as the training
pool and the separate `test` pool. Metadata must contain the same 100 native
variant names/indices in both pools, unique/disjoint image IDs, and existing
JPEG paths. Preflight rejects a different task/class set rather than remapping
it. The pinned SSB split selects 50 known / 50 novel classes; 50% of known-class
training images are labeled. Native targets are **zero-based**, unlike Cars/CUB.
No family/manufacturer labels, class prompts, bbox or extra Aircraft-specific
crop is added. Use the native SelEx image/transform protocol.
Acquire the release separately from the [FGVC-Aircraft dataset page](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
and prepare the extracted layout above; runtime download/extraction is disabled.

**Optional CIFAR-100, outside the main matrix:** retained `--dataset cifar100`
and its old 200-epoch configs use `datasets/cifar100/cifar-100-python/{train,test,meta}`.
CIFAR is the extracted official **Python** format, not image folders. Training
has 50,000 images (20,000 labeled / 30,000 unlabeled); the disjoint test set
has 10,000. Both official file checksums and ID/pixel/label alignment are
checked. Native train/test IDs each start at zero in distinct official-file
namespaces. CIFAR's fixed class ranges and their deterministic descriptor hash
are reported as such, not as a nonexistent SSB pickle. Cars/CUB/Aircraft use the pinned
split files and their SHA256s. No runtime automatically downloads datasets.

On training entry, file-backed Cars/CUB/Aircraft rows retain only paths, legal labels
and IDs. CIFAR rows own copied RGB arrays, legal labeled targets and `-1` for
all unlabeled rows; they retain no source annotation-bearing dataset or hidden
target transform. Evaluation-only datasets are released from the training scope.
Integrity audits and evaluators are the only readers of hidden ground truth.

Each run needs one available NVIDIA CUDA GPU, pretrained weights and its
dataset. Full-batch VRAM and actual wall time are not measured locally: the
65536-output projector, gradients, optimizer, CRiSH copies, activations and
backbone all contribute to memory. CIFAR images are resized to 224; its
larger training pool need not be cheaper than Cars/CUB. Start no competing GPU
job. The launcher refuses visible occupied GPUs and requires real-image,
actual-head CUDA forward/backward preflight, then bounded smoke before full SGD.

## Commands: six independent jobs, default seed 1

Inspect a plan on any machine, without training or loading weights:

```bash
bash scripts/train_selex.sh --dataset aircraft --baseline --dry-run
```

For each dataset first run preflight for both arms (one command at a time):

```bash
bash scripts/train_selex.sh check --dataset scars --baseline
bash scripts/train_selex.sh check --dataset scars
bash scripts/train_selex.sh check --dataset cub --baseline
bash scripts/train_selex.sh check --dataset cub
bash scripts/train_selex.sh check --dataset aircraft --baseline
bash scripts/train_selex.sh check --dataset aircraft
```

Then select the appropriate individual job command. This is a command list,
not an automatic sweep; do not paste all jobs into concurrent sessions:

```bash
bash scripts/train_selex.sh --dataset scars --baseline
bash scripts/train_selex.sh --dataset scars
bash scripts/train_selex.sh --dataset cub --baseline
bash scripts/train_selex.sh --dataset cub
bash scripts/train_selex.sh --dataset aircraft --baseline
bash scripts/train_selex.sh --dataset aircraft
```

Append `--seed 2` or `--seed 3` to both arms for matched additional seeds when
compute permits. Six full single-seed runs become 18 for three seeds. No
multi-seed sweep is started automatically. No new DINOv1 runs are implicit in
these commands. `--config PATH` still supports exact custom/historical SelEx
configs; if `--dataset` is also given it must match, as must `--baseline`.
Every smoke inherits the chosen full config/dataset/seed and changes only
epochs to 1, training steps to 2, CRiSH warmup to 0, extra SS off and its
training deadline. It still extracts/clusters the complete dataset: synthetic
CPU checks are the only intended local Mac tests.

## Evaluation, epoch 200 and resume

Every epoch runs plain k-means on the disjoint test set. At epoch 200 (zero-based
199), source `train_report_interval=200` additionally runs plain k-means on
train-unlabeled **before** disjoint test. Both execute in one bounded evaluator
with the exact ordered RNG exchange. Neither metric values, targets nor
predictions return to training; only validated RNG state returns. Train-plain
reports are distinct from final **semi-supervised** SS-v2 reports.

Primary endpoint is normalized-backbone, balanced semi-supervised k-means,
10 restarts / 10 iterations per phase, seeded at source stage 0, scoring only
unlabeled-train rows with global Hungarian-v2 before Old/New restriction.
Always report All, Old, New. All uses sample weighting, not a simple Old/New
average. The primary checkpoint is fixed at 200, with no test-based primary
selection or early stopping. Optional `--eval-every N` adds isolated SS monitoring equally
for both arms and never selects a checkpoint.

All six main 200-epoch configs set `report_best_old_test=true` (retained CIFAR
profiles do as well). Like the pinned
GitHub default `eval_funcs=['v1','v2']`, each plain disjoint-test evaluation also
computes **Old-v1 (separate Hungarian assignments)**. Strict improvements in
that Old-v1 score save `backbone_best_old_test_v1.pt`; ties keep the earlier
epoch. It is not selected by Old-v2, All, SS metrics or train-unlabeled scores.
The only edge-case extension is keeping the first finite epoch even at zero
Old-v1 so that a secondary report exists; upstream's initial threshold of zero
would otherwise save no best. This fallback is documented rather than called
exact behavior for an all-zero run.

At completion, two independently bounded workers evaluate **SS-v2** for last
and best. Both use identical SS settings and seed stage. The best result is
secondary, recorded with `selection_matching=hungarian_v1_separate` and
`matching=hungarian_v2_global`; its JSON explicitly declares selection used
evaluation labels. Primary `evaluation.json` remains SS-last even if best has
higher All. No evaluator metric changes SGD, CRiSH, warmup or stopping.
Historical 50-epoch configs default this additional reporting to false.

Each job prints its PID, run ID and log location. Inspect
`GCD_RUNS_ROOT/RUN_ID/{config.json,metadata.json,environment.txt,readiness.json,run.log}`.
Training loss/time/peak memory: `metrics.jsonl`; CRiSH decisions: `crish.jsonl`;
plain test: `test_metrics.jsonl`; epoch-200 train-plain:
`train_plain_metrics.jsonl`; primary SS: `evaluation.json`, `evaluation_last.json`;
secondary SS: `evaluation_best_old_test.json`, `best_old_ss_metrics.jsonl`;
best checkpoint/provenance: `backbone_best_old_test_v1.pt`, `selection.json`;
posthoc summary: `results_summary.json`; resume state: `resume.pt`; final
backbone: `backbone_final.pt`. All are observed outputs only when jobs run.

Preflight has a 180-second deadline. New full configs cap training and monitors
at **604800 seconds (7 days) per invocation**, and each plain/SS evaluator at
**7200 seconds per evaluator**; final best and last each have that bound, and
the parent CLI allows both. These are stop limits, not completion-time estimates. Smoke
SGD/monitors are limited to 1800 seconds. If an owned worker reaches its bound,
the launcher stops its own process group and preserves the last completed
epoch checkpoint; an epoch in progress is not saved as complete. Increase a
limit before starting a new run if the machine requires it; changing config
afterward invalidates exact-config resume.

After a failure, inspect the log/cause and resume with the exact original
source/config, not a newly edited 50/200-epoch config. Example for baseline:

```bash
bash scripts/train_selex.sh --baseline --config runs/RUN_ID/config.json --resume runs/RUN_ID/resume.pt
```

Omit `--baseline` for CRiSH; use the actual external run path when configured.
Keep the original run folder, including `test_evaluations/` and the best
bundle: resume refuses to silently reset the best-so-far score. Old source
checkpoints cannot be migrated by adding this config flag; use their original
source/config for any already-running experiment.
Resume launches a new provenance directory, restores optimizer/scheduler/RNG
and starts at the next epoch. If a plain evaluator failed, the saved
`plain_rng_pending` phase replays the entire ordered evaluation before SGD
continues, including at epoch 200. Fully completed resume exports the final
backbone and permits retrying final SS without extra SGD. Prior run logs are
preserved in their original directory. Checkpoints from older source releases
still require those original releases and must not be mixed with these runs.

## Local evidence and remaining gate

CPU fixtures test matched profiles, six CLI plans, pinned CUB and CIFAR split
construction, hidden-target access guards, owned pixels, native-label offsets,
official-shape CIFAR membership/coverage failures, source loss/head backward,
epoch-200 pool ordering, RNG handoff and recovery. These are synthetic fixtures,
not real-data metrics or CUDA certification. The complete validation receipt
is `results/local_validation.json` and contains exact test count/log/source
hashes after validation. Local Mac has no prepared dataset/weights or NVIDIA
hardware; full-size memory, duration and benchmark All/Old/New remain unmeasured.

The historical manuscript describes Hyp-SelEx and must be rewritten for the
new SelEx method and populated with observed experiments before submission.

Final local validation run `crish-v21-20261110T130309Z-747ad77d`: **157 passed,
zero failures/errors/skips**, including ten new multidataset/epoch-200 tests
and archive execution of all six plans. All three split checks, source/protocol,
syntax and historical manuscript/PDF provenance checks passed. Command:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py`;
60-second deadline per child. Log:
`logs/crish-v21-20261110T130309Z-747ad77d/smoke.log`.

Best/last extension validation: `crish-v21-20261110T132854Z-e0e51225`,
**166 passed, zero failures/errors/skips**, via the same bounded CPU runner.
Includes Old-v1-vs-v2 selection, strict ties, zero fallback, exact-source resume,
pending evaluator replay, unchanged SGD/RNG, separate primary/secondary reports
and both final-worker deadlines. Log:
`logs/crish-v21-20261110T132854Z-e0e51225/smoke.log`; current receipt:
`results/local_validation.json`. These are CPU fixtures, not benchmark results.

Aircraft replacement validation: `crish-v21-20261110T134317Z-52b19aa4`,
**171 passed, zero failures/errors/skips**, including five Aircraft-specific
tests and main-matrix archive execution. The actual pinned loader is checked
with synthetic JPEGs: zero-based legal labels, unlabeled target guards,
trainval/test coverage, class mappings, duplicate/missing/unsafe paths, native
224 transforms and finite CPU backward. Optional CIFAR regression coverage is
retained. All four split descriptors and source/protocol/syntax/archive checks
passed; log: `logs/crish-v21-20261110T134317Z-52b19aa4/smoke.log`.
No real dataset/weights/CUDA benchmark was launched.

## Online pair reliability — 2026-11-11

This section records the earlier online-weighting stage. The current default
200-epoch experiment is CRiSH-P with structured probes and online weighting
off; see [the full A2/A1/CRiSH-P handoff](12_full_method_handoff.md). Explicit
variant flags below preserve the earlier comparisons without relying on the
changed default. No historical validation receipt is a current benchmark.

Owner explicitly requests an algorithm implementation change while freezing
all LaTeX/.tex/PDF. The four dataset pairs of 200-epoch configs now include
the same online-pair settings; the baseline ignores them completely. The
default CRiSH arm combines online weighting with the existing confirmed
discrete repair. Historical 20/50-epoch configs omit the field and retain
the prior objective. No observed benchmark gain or novelty is claimed.

After 10 warmup epochs, ramp the online effect over 10 epochs, with minimum
weight .25. At each minibatch, build detached normalized-backbone kNN graphs
on distinct sample IDs for both augmentations; repeated sampler IDs are
aggregated. For measured edges use mean reciprocal status (RN), persistent
directed edges (CVS), or their product. Unmeasured edges retain weight one.
Apply weights only to base-level pseudo-positive numerators involving an
unlabeled endpoint. Original denominators, same-ID/labeled-labeled weights,
genuine supervised loss, USE and every coarse objective are preserved.

The new mechanism is evaluated on every batch after warmup; it is not limited
to the discrete repair's 38 possible refreshes. The complete intervention is
still a hypothesis. Loss reduction after downweighting is partly a coefficient
mass change and is not accuracy evidence. Local neighborhoods can reinforce
semantic errors or reflect density differences, especially for rare classes.
The fixed floor, warmup and component controls allow that trade-off to be tested.

Commands below select one experiment at a time on the prepared NVIDIA machine;
inspect availability/data/weights first. No sweep is started here.

```bash
# B0: unchanged corrected-SelEx objective
bash scripts/train_selex.sh check --dataset scars --baseline
bash scripts/train_selex.sh smoke --dataset scars --baseline

# Earlier online + local CRiSH combination, selected explicitly
bash scripts/train_selex.sh smoke --dataset scars --variant local --pair-reliability both
bash scripts/train_selex.sh --dataset scars --variant local --pair-reliability both

# Original CRiSH objective, online weighting off
bash scripts/train_selex.sh --dataset scars --variant local --pair-reliability off

# Online-only controls, no discrete repair
bash scripts/train_selex.sh --dataset scars --variant online --pair-reliability rn --no-repair
bash scripts/train_selex.sh --dataset scars --variant online --pair-reliability cross_view --no-repair
bash scripts/train_selex.sh --dataset scars --variant online --pair-reliability both --no-repair
```

Use `--dry-run` to inspect any command locally and `--seed` for predeclared
matched seeds when compute permits. Smoke forces online warmup 0/ramp 1
to exercise the branch within its bounded two batches; full profiles retain
10/10. This smoke override is not a benchmark setting. `--no-repair` disables
only discrete repair; `--pair-reliability off` independently disables weighting.

Outputs: `pair_reliability.jsonl` every 20 steps and epoch summaries in
`metrics.jsonl` include eligible/changed directed pair counts, weight mass,
histogram, and mean weights by legally labeled/unlabeled anchor. They do not
use evaluation known/novel identity. Retain `crish.jsonl`, hierarchy support,
memory/time and final All/Old/New as before. Zero changed pairs can be a valid
no-evidence/clean-neighborhood outcome; investigate distributions rather than
assuming an accuracy gain.

Executed CPU code validation:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py --code-only`:
**204 passed; 0 failures/errors/skips**, run
`crish-v21-20261111T031200Z-2f99aa23`. The 60-second child deadlines, source/split,
static and archive checks all passed. New tests cover score/weight bounds,
components, duplicate IDs, unknown/collapsed evidence, independent coefficient
and gradient arithmetic, baseline/off/warmup parity, protected/coarse loss
preservation, weighted repair drift, hidden-target invariance and resumed SGD.
Receipt: `results/local_validation.json`; log:
`logs/crish-v21-20261111T031200Z-2f99aa23/smoke.log`.

All three paper files retain the exact SHA256 recorded at task start in
`logs/online-pair-change-20261111/paper_before.json`. The manuscript gate is
explicitly `not_run_code_only`, not passed. Its historical source digest is
now stale by design; no paper regeneration or full GPU training occurred.
Old-source checkpoints must retain their original source/config for resume.
