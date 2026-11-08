# Protocol relevant to the training failure — 2026-11-05

Current evaluation/retraining review (2026-11-08): see
`reports/07_evaluation_and_retraining_audit.md` for author-confirmed upstream
ordering defects, SelEx-vs-Hyp-SelEx selector/restart differences and the small
last/best provenance correction. Historical entries below retain their dates.

## Sources and scope

- Pinned official code: [Visual-AI/HypCD](https://github.com/Visual-AI/HypCD),
  commit `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73` (MIT).
- [Hyperbolic Category Discovery, arXiv full method and supplement](https://arxiv.org/html/2504.06120v1),
  especially sections 3.4–3.5 and 6.3: Hyp-SelEx combines hyperbolic/angle
  learning with hierarchical pseudo-supervision and nonparametric clustering.
  The CVF HTML/PDF requests returned 403; the arXiv method was accessible.
- Local references: `upstream/HypCD/train/train_HypSelEx.py`,
  `upstream/HypCD/scripts/train_HypSelEx.cmd`,
  `upstream/HypCD/methods/clustering/faster_mix_k_means_pytorch.py`.

This is a runtime/protocol audit, not a new method or a completed baseline reproduction.
Existing novelty risks and unclosed method gates remain in
`reports/04_method_and_research_plan.md`, section 13 (RCA, LEAH, HypGCD and further leads).
Those methods were not re-reviewed in this debugging session; no novelty claim
depends on the numerical correction.

## Exact local configuration and boundaries

`configs/crish.json` and `configs/baseline.json` differ only in `method`.
The active pipeline is `scripts/train_crish.sh` → `gcd_runtime/train_cli.py`
→ `launch.py` → `worker.py` → `trainer.py` → `crish/training.py`.
Upstream Slurm scripts are inspected, not executed on this Mac.

- Stanford Cars original train pool; official SSB split
  `upstream/HypCD/data/ssb_splits/scars_osr_splits.pkl`.
- SHA256: `28d471f0fb9e2c773e1255b8b447e542974f6617952203056a0635c638513d39`.
- 98 known / 98 novel classes (7 Hard, 15 Medium, 76 Easy); loader selects
  50% of known-class training rows, not 50% of every class independently.
- Supplied run: 2,000 labeled + 6,144 unlabeled = 8,144 train rows;
  8,041 original-test rows are disjoint. Local images are unavailable.
- DINOv2 ViT-B/14 reg4, backbone dimension 768, input 224, projector 8192,
  two ImageNet augmentations; last blocks from index 10 trainable.
- 196 base clusters, hierarchy widths `[196,98,48,24,12,6,2]` as observed
  in the supplied successful epoch audits. Normalized backbone features feed
  Euclidean HSSK; projected features use existing `ToPoincare` operations.
- Seed 1, 200 epochs, batch 128, SGD lr .1 / momentum .9 / weight decay 5e-5,
  cosine schedule with eta_min .0001, supervised weight .35, USE smoothing .5,
  curvature .1, clipping radius 2, maximum distance weight **.5** in the aligned
  profiles. The earlier supplied failed-run config used 1.0 and is historical.
- CRiSH warmup 10 epochs / refresh every 5. The failed extraction precedes
  CRiSH action scoring; all three completed epochs report zero repairs.
- Training strips unlabeled targets to -1. HSSK receives only legal anchor
  labels; higher-level anchors derive from predictions. No hidden targets
  enter initialization, empty-cluster relocation or acceptance scoring.

## Evaluation and comparability caveats

### Claim boundary, rechecked 2026-11-06 during the remote run

`SOURCE: OK` verifies packaged source integrity, not scientific equivalence to
the paper. The active remote run `20261106T101304Z-19f82348` is CRiSH upgraded:
row-normalized USE plus the selected evidence/probe extensions. It is not an
unchanged Hyp-SelEx reproduction. No completed matched B0 result for that run
has been supplied, so an algorithmic gain over Hyp-SelEx is unestablished.

Direct pinned-source evidence:

- `train/train_HypSelEx.py:391-392`: ordinary k-means on the disjoint test pool
  after each epoch; `:416-425`: save best on strict Old-test improvement.
- `:585-619`: normalize backbone features, ordinary Euclidean k-means with
  the known total cluster count, then matching/All/Old/New. The local selector
  preserves this rule while isolating RNG and worker lifecycle.
- `:530-582`: separate balanced semi-supervised k-means with legal labeled
  anchors and scoring restricted to train-unlabeled rows. The local final
  evaluator uses this protocol family for best and last. Its scores are not
  interchangeable with ordinary k-means disjoint-test histories.
- `scripts/eval_HypSelEx.cmd:60` names `model_test.pt`. This does not establish
  which best/last checkpoint generated each published table entry.

Selecting on test Old is evaluation-label-based model selection even though
neither labels nor scores enter SGD/repair. Reproducing released code does not
remove that selection bias. Report the exposure and criterion explicitly.
The retained final-epoch checkpoint permits a fixed-epoch comparison; it does
not erase earlier researcher exposure/tuning to evaluation curves.

Paper/code differences also remain material. The accessible arXiv v1 supplement
6.2 specifies Cars last-three-block tuning, smoothing 1 and fine-grained max
distance weight 1; the pinned Cars-v2 released script uses block index 10,
smoothing .5 and max weight .5. Paper main-text fine-grained curvature is .05,
whereas this released Cars command uses .1. Our reference is explicitly the
released command. CVF main/supplement PDFs returned HTTP 403 in this recheck;
do not silently identify arXiv-v1 parameter discrepancies with verified final
CVPR text. The table/checkpoint/hyperparameter mapping still needs resolution
before claiming exact reproduction of published numbers.

Required comparisons: legacy-USE local B0, corrected-USE B0 and CRiSH with the
same corrected USE, then matched seeds/inputs/selection and final SS best/last.
If a validation/fixed-epoch primary protocol is chosen for a manuscript, declare
it separately and apply it symmetrically; do not rewrite this active run's
selection rule after inspecting its metrics. All remains the primary outcome,
with Old/New reported. No active process/config/source was altered for this audit.

The earlier final-only rule is retained as an optional `final_epoch` mode. On
2026-11-06 the user explicitly authorized implementing the released Hyp-SelEx
checkpoint-selection rule for comparison. All shipped CRiSH/B0 profiles now use
`best_old_test_v2`, `test_every_epochs=1`, and no additional SS monitoring by
default. Smoke is one epoch/two steps and exercises the selector plus both final
evaluators, so its accuracy is not a benchmark result.

After every epoch, a separate bounded worker extracts normalized backbone
features from the full disjoint test pool, runs ordinary Euclidean k-means with
196 clusters, computes global Hungarian-v2, and selects the checkpoint on strict
Old-v2 improvement (earliest epoch wins ties). All/Old/New are logged in
`test_metrics.jsonl` and `test_evaluations/epoch_NNNN.json`. Test targets enter
matching and selection, never feature extraction, clustering initialization,
SGD, or CRiSH acceptance. Global matching uses both known/novel test labels;
this is explicitly evaluation-label-based checkpoint selection. It is not an
oracle-free validation rule. First finite Old=0 still gets a usable best
checkpoint; the upstream script would otherwise save no best in that edge case.

After training exits, final evaluators fit balanced semi-supervised k-means
with legal labeled anchors over the train pool for both best and last backbones,
scoring only unlabeled rows. Hungarian-v2 uses one global matching; Old/New
select the known/novel subsets afterwards. `All = (n_old*Old+n_new*New)/(n_old+
n_new)`, not the unweighted average. `evaluation_best_old_test.json` and
`evaluation_last.json` preserve both reports; `evaluation.json` aliases best
under the new mode and last under final-only mode. Standalone evaluation aliases
the explicitly requested checkpoint. Accuracies are fractions; display uses %.

`--eval-every 1` adds per-epoch balanced SS monitoring; `0` (default) adds none
between the best/last final evaluations. These SS records live separately in
`epoch_metrics.jsonl`. `results_summary.json` and `scripts/show_metrics.py` show
max-All/max-Old/max-New with each winning epoch's full triplet, separated by
protocol and restricted to actually evaluated checkpoints. Peaks are diagnostic
and do not override the Old-test selection criterion. Training metrics still
have null All/Old/New because scores are not returned to the SGD loop.

Best weights and selection metadata are stored atomically in one weights-only
loadable bundle. Resume copies the original run's best and evaluation history,
checks completeness/config/source, and recovers a pending selector for the saved
epoch before further SGD. A lone resume.pt is insufficient for restoring the
best history; missing/future/inconsistent selection artifacts are rejected.
Resume after a failed last-epoch selector performs no extra SGD, recovers that
selection and proceeds to both final evaluators. No early stopping is introduced.

CUDA uses the official `subhadarship/kmeans_pytorch` Git implementation pinned
at `f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805`. This supports the upstream
`tqdm_flag` and device/empty-cluster behavior that PyPI 0.3 lacks. Bootstrap's
CUDA selector requirements add that Git pin and Numba; preflight checks the VCS
commit and executes the actual backend on CUDA tensors. Version, Git commit and
function hash are recorded per test report. Legacy PyPI backend is permitted
only for CPU fixtures. Dependency's convergence loop is bounded by the separate
worker deadline; no new unbounded training process is introduced. The original
paper does not pin that dependency's SHA, so it is our reproducibility pin.

Evaluation RNG is isolated and keyed by seed/epoch/protocol, matching the local
runtime's previous design. This prevents selection/monitor schedules changing
SGD, but differs from the released script's shared RNG stream. Numerical HSSK
fixes and original local hyperparameters remain shared across B0/CRiSH. Therefore
matching this selection rule alone does not prove exact table reproduction.

In pinned upstream `train/train_HypSelEx.py:416`, `_best.pt` is saved when
`old_acc_test` on the disjoint test set improves; the final print reports
train-unlabeled metrics associated with that epoch. The upstream evaluation
script instead names `model_test.pt`. Code inspection does not prove which
checkpoint produced each paper table entry. The paper's implementation details
do not resolve this in the inspected text. Our fixed-final-epoch rule is an
explicit protocol difference, not proof of identical published model selection.
The previous local no-test-label-selection constraint was explicitly superseded
by the user's reproduction request; final-only remains configurable. Additional
evaluation changes cost and label exposure, not the loss or optimizer.

SelEx's author confirms best-checkpoint reporting in
[issue #8](https://github.com/SarahRastegar/SelEx/issues/8#issuecomment-2629402874).
Its default helper returns the first metric in `['v1','v2']`, so its Old selector
is v1, unlike HypCD's explicit v2 return. SelEx's best/last final SS reports use
the reported v2 metric. Do not label this runtime's Hyp-SelEx Old-v2 mode an
exact reproduction of SelEx's selector.

The full and smoke configs were aligned on 2026-11-06 to the upstream DINOv2
Cars invocation: `scars v2 0.5 0.5 0.1 2.0`. All four now use
`hyper_max_weight=.5`; the previous 1.0 profiles/runs are not silently reclassified.
The paper's general fine-grained curvature description differs from the Cars
command's .1; the command is the explicit reference used here. The independent
`audit/protocol.py` parser derives shell values plus Python defaults and compares
them with the actual worker flags and local parser defaults, without executing
Slurm. `scripts/check_protocol.py` checks the four shipped profiles during CPU
smoke; CUDA readiness records the same comparison. It catches weight, curvature,
backbone, trainable-block and other CLI drift. Only the one-epoch limit is exempt
for bounded smoke. It does not label custom settings as source-matched.
The source also contains a USE broadcasting defect; this patch does not change
that arithmetic. Runtime B0 is a matched control, not a claim of an unchanged
official reproduction. Both arms now share empty-cluster handling, so compare
new B0 and CRiSH runs with the same patched source and inputs.

## Next commands

Local bounded CPU validation: `.venv/bin/python scripts/run_smoke.py`.
On Linux NVIDIA, update `environments/requirements-selector-cuda.txt` dependencies
only when no run is using that environment. With data/weights:
`bash scripts/train_crish.sh check`, then
`bash scripts/train_crish.sh smoke`. Only after those pass, start a new finite
training run with `bash scripts/train_crish.sh`. This session starts none.

## Shared Cars/CUB adapter (2026-11-06)

The method and optimization pipeline are shared. `gcd_runtime/datasets.py` selects native loaders without modifying upstream; `training_data.py` materializes image-only unlabeled rows for both layouts. `train_cli.py --dataset cub` selects the four `cub_*` profiles; omission retains Cars. Preflight audits native train/test membership, image paths, official split checksum, known/novel coverage, and 50% labeled-known membership. CUB metadata joins require matching unique image IDs, valid class IDs and binary train/test flags; duplicate/missing/escaping paths fail. Automatic CUB download/extraction is disabled in the runtime.

## Optional coordinated upgrade (2026-11-06)

The reference profiles above retain their legacy USE arithmetic. New
`*_upgraded*.json` profiles are selected explicitly with `--profile upgraded`.
They normalize non-self USE smoothing targets per row, fall back to the other
view when support is empty, and use full frozen-feature neighbor shortlisting
plus bounded local cross-view verification and larger class-balanced labeled
probes for CRiSH. These are documented runtime/objective changes, not upstream
CLI flags or an exact reproduction. B0 upgraded shares the corrected objective
and skips repair entirely. Both arms retain the same dataset-specific source
hyperparameters, optimization and label-based Old-test-v2 checkpoint selector.

`scripts/check_protocol.py` now covers all 16 shipped profiles. A true
`cli_hyperparameters_match` does not imply that the objective is unchanged:
`reference_use_arithmetic_preserved` is false for upgraded profiles and the
runtime extensions/caveats are recorded alongside it. See README and
`reports/02_novelty_and_method.md` for equations, limits and the comparison plan.
No benchmark outcome is inferred from synthetic checks.

Pinned CUB SSB: `cub_osr_splits.pkl`, SHA256 `48bdcaa05603d932037738d3df776ab500945e9b7aeff85f4668305bf71bd1e7`, 100 known / 100 novel (Hard 34, Medium 34, Easy 32). Known classes are the actual source split IDs, not the first 100 classes. Native targets are one-based; the existing native known-first target transform is retained for legal training labels and evaluation only. Unlabeled training rows contain target -1.

Pinned `scripts/train_HypSelEx.cmd` CUB DINOv2 example: `cub v2 1.0 1.0 0.1 1.5`. Thus smoothing=1, hyper_max_weight=1, c=.1, cr=1.5; other launch parameters are derived independently from the same script/parser. No Slurm command is executed. B0/CRiSH use the same dataset-specific settings. Source CLI alignment is not proof of reproducing the paper's checkpoint/table results.

Required native layout under `$GCD_DATA_ROOT/cub/CUB_200_2011`: images/, images.txt, image_class_labels.txt, train_test_split.txt, attributes/image_attribute_labels.txt. The attributes file is required by the pinned loader even though the method does not use semantic attributes for supervision. Fixture validation uses 600 synthetic images/200 classes (400 train, 200 test, 100 labeled, 300 unlabeled); these are implementation checks, not a CUB benchmark result.
