# Source and algorithm re-audit — 2026-11-11

**Kết luận:** sau bản sửa dưới đây, đường SelEx + CRiSH qua 184 kiểm thử CPU
tổng hợp và các kiểm tra source/protocol/package. Có thể chuyển sang preflight
và bounded smoke trên Linux NVIDIA đã chuẩn bị data/weights. Chưa đủ bằng
chứng kết luận thuật toán tăng accuracy, tính mới đã đạt hoặc benchmark sẵn sàng.

## Confirmed defect and minimal fix

`crish/engine.py` previously keyed cached candidate gradients by
`(version, sample_id, level, kind)`. Two Move destinations for the same sample
overwrote one another. If an earlier destination won discovery, confirmation
could evaluate the last destination's gradients and attach its result to the
wrong diagnostic entry, while committing the earlier destination's state.

The cache now uses the complete immutable `Action`, including source, target
and version. Gain records also include source/target. This preserves loss,
candidate ranking, optimizer, geometry, split, acceptance thresholds and the
baseline bypass. Shipped profiles have `max_targets=1`, so the collision
requires increasing that supported setting; the fix makes that setting safe.

`tests/test_crish_candidate_identity.py` executes the real engine/repair policy
with two supported Move destinations, real distinct autograd gradients, and
controlled discovery gains. Before the fix, confirmation gradients disagreed
with the selected destination in all 12 fixture weight entries. After the fix,
confirmation uses exactly the selected gradients/trace and commits that Move.
The comparison scores are controlled fixtures, not measured learning gains.

## Reviewed implementation contracts

| Component | Source and evidence | Assessment |
|---|---|---|
| Baseline | Pinned SelEx `d8107b5458f3337b2d5342f79345a976037a1265`; `reference.py` verifies snapshot hashes | Upstream preserved. HypCD remains separate and pinned. |
| Matched arms | Loaded Cars/CUB/Aircraft 200-epoch config pairs | Each pair differs only in `method`. This establishes config comparability, not a paper reproduction. |
| Label availability | `gcd_runtime/training_data.py`, hierarchy and LegalProbe; hidden-target poison/accessor regressions | Unlabeled training rows store -1. SGD, evidence and repair receive legal known targets only. |
| Sample alignment | Stable sample IDs, grouped-to-input restoration, batch queries may repeat under replacement sampling | Tested on interleaved/sparse IDs; no new issue found in reviewed paths. |
| Hierarchy | All coarse partitions group level-0 prototypes directly; active levels enter every 10 epochs | Follows corrected upstream; not a guaranteed nested tree. Move edits one level independently. |
| Empty slots | Pre-existing `allow_empty_clusters` patch and empty-cluster regression suite | Preserve source assignments, IDs and prototype budget; finite fallback only for nonfinite unsupported base slots. Nonfinite occupied centers/invalid anchors still fail. Move requires source/destination support. This is numerical compatibility, not recovered semantic classes. |
| Objective | `selex_runtime/objective.py`, executed source USE/SSE and gradient fixtures | Known-label term preserved; no-action/disabled CRiSH parity tested. Source's 1D USE broadcasting and clamp intentionally remain. Finite arithmetic does not prove normalized probabilistic targets. |
| Reliability | `crish/reliability.py`, `sampling.py` | Bounded reciprocal-in-both-views joint score. Global frozen-feature shortlist; augmented-view graph is local to cache. No classifier confidence is invented. |
| Actions | `candidates.py`, `rebuild.py`, `policy.py` | Labeled assignments protected; Abstain retains same-ID/labeled-labeled positives and original positive denominator. Move rebuilds means/radii and rederives targets. |
| Lookahead | `lookahead.py`, `probe.py`, engine and new regression | One copied projector SGD step, real momentum/LR/weight decay, ID-disjoint discovery/confirmation. Confirmation checks only the winner; failed confirmation stops refresh. Backbone is not virtually updated. |
| Training/resume | Epoch checkpoints, source/config fingerprints and RNG exchange tests | Synthetic interrupted/resumed updates match. New source fingerprint requires original source for old checkpoints. |
| Evaluation | SS and plain-test fixtures, global Hungarian-v2, best/last selection tests | Primary fixed-final SS-v2 on train-unlabeled; All is sample-weighted Old/New. Secondary best uses plain disjoint-test Old-v1 and therefore evaluation labels. Do not treat that secondary selection as independent evaluation. |

Cars protocol remains the checksum-verified SSB split (98 known / 98 novel),
50% of known training images labeled, official total K=196. Real image counts,
paths and checkpoint checksums still require remote data preflight. The current
corrected-SelEx 200-epoch protocol differs from the original paper implementation;
see report 10 before comparing published numbers.

## Design and evidence still open

- With warmup 10, refresh interval 5, 200 epochs and max accepted 1, at most
  38 actions can be accepted (zero-based epochs 10 through 195). Corrections
  expire at the next epoch's fresh extraction. Prototype/radius changes can
  affect many USE coefficients, so this is not a bound on affected pairs.
  Measure accepted counts, changed mass and actual gains before deciding whether
  this conservative budget is useful; do not silently relax it after scores.
- The full 65,536-dimensional head, cloned optimizer states, actual data and
  CUDA clustering have not been exercised here. A time-budget fallback can
  leave CRiSH behaving as Keep. Inspect `crish.jsonl` reasons/scoring counts
  in real-data smoke; zero accepted actions alone need not be an error.
- Projector-only one-step probe improvement is a proxy. It gives no guarantee
  of improved backbone representations, novel-class recovery or All accuracy.
  Class-balanced known folds do not cover every known class at these sizes.
- Stability trials default to zero. RN-only/cross-view-only weighting controls
  and the broader proposed ablations are not completed experiments. The active
  score uses their joint persistence; do not describe this as measured M1+M2.
- The manuscript remains a historical Hyp-SelEx draft, with an explicit scope
  notice. Its confirmation statement/digest/PDF were updated because the shared
  engine changed; this does not turn it into a SelEx paper or validate results.

## Primary-source and novelty checks

Revisited [HypCD appendix 6.3](https://arxiv.org/html/2504.06120v1),
[SelEx method/appendix C](https://arxiv.org/html/2408.14371v1) and the pinned
training source. Paper-era recursive hierarchy and corrected source's joint
coarse grouping must remain distinguished. Existing report 10 retains the
paper/code discrepancies and benchmark version boundary.

[HypGCD, equations 15–16](https://duannyu.github.io/assets/papers/hypgcd.pdf)
already uses K-reciprocal neighborhoods to construct local hierarchical
triplets. Reciprocal neighbors alone cannot establish novelty here. CRiSH's
Euclidean evidence and discrete probe-confirmed intervention differ in mechanism;
that observation is not proof of a novel or effective contribution.

[RCA](https://proceedings.mlr.press/v306/yu26l.html) remains an overlapping
confidence/cross-view lead. Its linked PDF could not be parsed in this review.
[LEAH](https://ojs.aaai.org/index.php/AAAI/article/view/39236) and its PDF
endpoint returned access denial; indexed text does not satisfy a full-method
comparison. Additional searches for GCD/counterfactual repair did not clear
the novelty gate. Full RCA/LEAH comparisons and controls against simpler repair
or meta-reweighting remain required; no novelty claim is made.

## Executed checks and next action

Final command:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py`

- **184 passed; 0 failed/errors/skipped**. Each child command has a 60-second
  timeout. Source/split/protocol/static checks, archive execution outside Git
  and manuscript/PDF receipt all passed.
- Receipt: `results/local_validation.json`; run ID
  `crish-v21-20261111T025722Z-81f40790`; log:
  `logs/crish-v21-20261111T025722Z-81f40790/smoke.log`.
- Initial targeted run: 44 passed, one worker blocked by sandbox OpenMP SHM.
  Approved CPU execution resolved it. First full rerun: 183 passed, one archive
  check failed on the intentionally stale manuscript fingerprint. After semantic
  review, regenerated the settings digest and rebuilt the PDF using the existing
  Tectonic binary; the final full suite above passed.
- No training, dataset/weight download, remote action, process termination,
  upstream change or empirical accuracy output occurred. Initial worktree edits
  were preserved. CUDA remains unverified locally.

On the prepared Linux NVIDIA machine, run sequentially:

```bash
bash scripts/train_selex.sh check --dataset scars --baseline
bash scripts/train_selex.sh smoke --dataset scars --baseline
bash scripts/train_selex.sh smoke --dataset scars
```

Inspect loss, hierarchy support, repair/fallback reasons, peak VRAM and final
evaluation artifacts before a full matched baseline/CRiSH run. Retain the
original source/config for any existing checkpoints.
