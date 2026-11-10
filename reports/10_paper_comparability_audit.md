# SelEx paper-comparability audit — 2026-11-10

**Kết luận:** pipeline hiện tại có evaluator phù hợp với SelEx đã sửa, nhưng
chưa đủ điều kiện gọi là tái lập trực tiếp số trong paper ECCV 2024. Không cần
đổi về code có lỗi hierarchy; cần khai báo đúng phiên bản và các khác biệt.
Không có benchmark mới hoặc NVIDIA verification trong audit này.

Current experiment choice: Cars/CUB/**FGVC-Aircraft** replace the earlier
Cars/CUB/CIFAR matrix. CIFAR rows below remain an optional historical comparison,
not part of the main three-dataset schedule. Both main methods still use the
corrected source, DINOv2, 200 epochs and SS-best/SS-last reporting.

## References and version boundary

- [Original ECCV paper, Table 1, Table 3 and section 5.1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08926.pdf).
  Tables were also inspected visually, not inferred from a flattened search snippet.
- [Original arXiv version, appendix D.1](https://arxiv.org/html/2408.14371v1).
- [Released 2024 training code](https://github.com/SarahRastegar/SelEx/blob/62e88d1e76803cfe822ee9e617529368197486cd/methods/contrastive_training/contrastive_training.py),
  retrieved through the GitHub connector; the file's last commit before the end
  of 2024 was dated 2024-08-29. Static inspection only; no old training executed.
- [Original checkpoint filenames](https://github.com/SarahRastegar/SelEx/blob/62e88d1e76803cfe822ee9e617529368197486cd/config.py).
- [Pinned corrected source and updated results](https://github.com/SarahRastegar/SelEx/blob/d8107b5458f3337b2d5342f79345a976037a1265/README.md).
- [Confirmed hierarchy ordering bug, issue 11](https://github.com/SarahRastegar/SelEx/issues/11).

Current upstream snapshot `d8107b5458f3337b2d5342f79345a976037a1265` passes
hash verification. Upstream files and existing training settings were preserved.
The corrected source repairs grouped-vs-loader row ordering, changes coarse
levels to joint grouping of level-0 prototypes, introduces level activation at
10-epoch intervals, and revises hyperparameters. It is not the original method
implementation underlying the 2024 table, even when training lasts 200 epochs.

## Which paper numbers are relevant?

These values are **paper-reported**, not observed in this repository:

| Original paper row | Backbone | All (%) | Old/Known (%) | New/Novel (%) | Relation to current runs |
|---|---|---:|---:|---:|---|
| Table 1, CUB | DINOv2 | 87.4 | 85.1 | 88.5 | Same broad backbone family; corrected method/settings differ |
| Table 1, Stanford Cars | DINOv2 | 92.6 | 93.5 | 91.8 | Same broad backbone family; corrected method/settings differ |
| Table 1, FGVC-Aircraft | DINOv2 | 79.8 | 82.3 | 78.6 | Same broad backbone family; corrected method/settings differ |
| Table 3, CIFAR-100 | DINOv1 | 82.3 | 85.3 | 76.3 | Current default DINOv2 is a backbone mismatch |

The inspected original Table 3 has no SelEx CIFAR-100 DINOv2 row. Do not label
82.3 as a DINOv2 baseline or attribute a DINOv2-vs-DINOv1 difference to CRiSH.
The updated 2026 README's DINOv2 means (CUB 83.5, Cars 79.1, Aircraft 77.2)
belong to corrected-code 50-epoch runs over seeds 1–3 (optional CIFAR-100 is
90.7 at 20 epochs). They must be labeled
as updated repository results rather than copied into an original-paper row.

## Stage-by-stage assessment

| Stage | Current implementation | Assessment |
|---|---|---|
| Class split | Pinned SSB CUB 100/100, Cars 98/98, Aircraft 50/50; optional CIFAR ranges 80/20 | Matches declared GCD partition definitions at source/fixture level; real inputs still need preflight |
| Label availability | 50% of known-class train images labeled; all unlabeled train targets sanitized to -1 | Appropriate; unknown labels do not enter training/repair scores |
| Backbone | DINOv2 ViT-B/14 with four registers; 768-D features; 224 input | Original released config references the same reg4 filename, but actual checkpoint SHA/state dict remains unverified locally |
| Optimization | Batch 128, lambda .35, SGD .1/.9, cosine to .0001 over 200 epochs | Several defaults align with released code; not proof of all paper-run settings |
| Fine-tuning | CUB blocks 10–11; Cars/Aircraft 9–11; optional CIFAR 10–11 | Cars/Aircraft use three blocks; paper prose describes two. Current choice follows corrected dataset settings |
| Smoothing alpha | CUB/Cars/Aircraft .75; optional CIFAR .5 | Differs from paper prose: .5 fine-grained / .1 generic; older README additionally recommended 1 for CUB/Cars, a paper/code discrepancy |
| InfoNCE temperature | CUB .25, Cars/Aircraft .5; optional CIFAR .15 | Corrected source settings; 2024 parser default was 1.0 |
| Hierarchy and loss | Corrected hierarchy; active levels ramp every 10 epochs; source USE/SSE definitions and gradients checked | Appropriate for corrected SelEx comparison, not exact original hierarchy reproduction |
| Feature used to score | Eval-mode normalized backbone output; projector excluded | Matches source feature-extraction/evaluator structure |
| Final clustering | Balanced semi-supervised k-means; known labels only as anchors; official total K; 10 restarts / 10 iterations per phase | Matches pinned source fitter arguments; actual CUDA kernel/data not exercised locally |
| Scored samples | Only unlabeled part of the train pool after SS fitting | Correct `Train ACC Unlabelled_v2` endpoint; disjoint test monitoring is a different protocol |
| ACC matching | Global Hungarian-v2, then Old/New restriction | Correct; All is sample-weighted, not an arithmetic mean of Old/New |
| Checkpoint | Primary fixed epoch 200; secondary best plain-test Old-v1 also evaluated with SS-v2 | Matches updated README's last endpoint and implements both source outputs. Exact original-paper best/last provenance remains unresolved |
| Seeds | Default 1; CLI supports 2/3 | Single-seed plans exist; no observed multi-seed result or mean/std yet |
| Real-data reproduction | Local data/weights/CUDA absent; no new benchmark outputs | Not verified; synthetic test passes cannot certify paper-score reproduction |

Epoch clarification: the 2024 released parser explicitly defaults to **200**
epochs (line 662), while corrected 2026 `dataset_configs` uses 50/50/20.
The inspected paper text does not explicitly state epoch count. Thus the
earlier conversation must not be read as evidence that the original paper
used 20/50 epochs, or that 200 alone makes a run incomparable. The important
remaining differences are method version, settings, backbone rows and endpoint
provenance. The original shell example also passed weight decay 5e-5 while the
parser defaults to 1e-4; current 1e-4 is not proof of the exact paper-run value.

## Evaluation path and leakage checks

1. `selex_runtime/trainer.py:28` constructs the pinned dataset/split and the
   selected backbone. Training discards annotation-bearing evaluation datasets
   after `sanitize_training`; only legal labels remain accessible to SGD/CRiSH.
2. `selex_runtime/hierarchy.py:37` clusters normalized training features with
   legal anchors, restores row order and builds the corrected active hierarchy.
3. `selex_runtime/objective.py:15` executes source USE/SSE arithmetic and the
   CRiSH intervention; baseline bypasses CRiSH. Matched configs differ only in
   `method`, with no extra labels or dataset-specific class prompts.
4. At final evaluation `selex_runtime/trainer.py:189` encodes the train pool,
   normalizes features as in the source cache/evaluator, and calls the pinned
   fitter with `truth[mask]` only. Unlabeled truth is used after predictions
   solely for ACC. Predicted labels are restored to sample order before masking.
5. `upstream/SelEx/project_utils/cluster_and_log_utils.py:29` computes global
   Hungarian matching. Old/New fractions are derived from that one assignment.
6. `evaluation.json` / `evaluation_last.json` are primary final SS metrics;
   `test_metrics.jsonl` is plain disjoint-test monitoring and now supplies
   Old-v1 for secondary best selection; `evaluation_best_old_test.json` is its
   separate final SS-v2 result. Epoch-200
   `train_plain_metrics.jsonl` is also plain clustering, not final SS. Do not
   substitute their posthoc maximum All for a declared fixed-final result.

Original released training defaults to eval functions `[v1, v2]` and selects
its best checkpoint using the first returned Old score; it reports both best
and last afterward. The paper text alone does not establish which candidate
produced each table number. Current fixed-last/global-v2 is an explicit,
defensible declaration, but not evidence that this ambiguity has disappeared.

## Executed verification and limits

Fresh targeted regression command:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest
tests/test_selex_runtime.py tests/test_selex_rng.py
tests/test_selex_multidataset.py tests/test_evaluation_protocol_audit.py -q`:
**29 passed**. No GPU/training process was started.

Added `tests/test_selex_ss_protocol.py`: it executes the pinned standalone
SS evaluator function against the adapter, on the same CPU feature fixture.
Only the CUDA fitter kernel is substituted. It checks twice-normalized features,
10/10 fitter arguments, anchor targets, RNG consumption, unlabeled-truth
invariance of fitter inputs, row restoration and sample-weighted global-v2 ACC.
The initial mock duplicated an `is_code` keyword; that fixture error was fixed.
Its bounded rerun passed. This checks the evaluation interface/arithmetic,
not numerical parity of real CUDA clustering or the full 200-epoch trajectory.

Full-suite count/log/source receipt after this addition is recorded in
`results/local_validation.json`. Differences between local test libraries and
the pinned CUDA environment can affect clustering, including sklearn's default
`n_init`. No bitwise parity across these environments is asserted.

Completed full validation: `crish-v21-20261110T131403Z-5d2b952b`, **158 passed,
zero failures/errors/skips**, via `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/run_smoke.py`. Every child command has a 60-second
deadline. Log: `logs/crish-v21-20261110T131403Z-5d2b952b/smoke.log`; receipt:
`results/local_validation.json`. Reference excerpts/versions:
`logs/paper-comparability-20261110T131231Z/references.json`.

## Decision before spending compute

Keep the user's matched 200-epoch corrected-SelEx / corrected-SelEx+CRiSH matrix
on Cars/CUB/Aircraft. Retain CIFAR configs as optional prior work.
Describe it with that exact version and report all observed outcomes. Treat
original paper numbers as historical, version-annotated references, not an
identical-conditions test of CRiSH. Do not restore the confirmed ordering bug
to chase old scores. A strict CIFAR-100 comparison with Table 3 additionally
requires matched DINOv1 runs; that is separate compute, not started here.

Before any claim: complete remote data/weight/CUDA preflight and bounded smoke,
run the corrected baseline, preserve all final metrics/configs/checksums,
investigate discrepancies, then compare matched CRiSH and baseline seeds.
Exact original-paper checkpoint selection would require further logs/artifacts
or author clarification. The next safe remote command remains
`bash scripts/train_selex.sh check --dataset scars --baseline`.
