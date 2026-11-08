# Code–paper alignment — 2026-11-08

The current `paper/manuscript.tex` describes the existing reference/upgraded
runtime. It no longer presents the original blueprint as implemented behavior.
The original manuscript and earlier comparison report are preserved under the
session's `logs/paper-sync-20261108T093957Z/before/` for provenance.
This revision changes documentation and verification tools, not the method,
training settings, runtime source fingerprint or benchmark results.

## Reviewed implementation contract

| Paper topic | Source | What the manuscript now says |
|---|---|---|
| Representation and partitions | `gcd_runtime/trainer.py`, `crish/partition_audit.py` | Frozen normalized backbone HSSK; raw backbone inputs for training head; level-wise partitions, not a guaranteed tree |
| Objective and level indexing | `crish/loss.py`, `gcd_runtime/trainer.py` | USE/SSE branch weights, prefix widths and pseudo weights `2^(-(ell+1))`; actual zero-based hyperbolic ramp |
| USE targets | `crish/rebuild.py:confusion_factor` | Legacy column-broadcast denominator retained in reference; non-self row normalization and positive-view fallback only in upgraded |
| M1 | `crish/reliability.py`, `crish/sampling.py`, `crish/engine.py` | Joint reciprocal cross-view score; Euclidean backbone evidence in a bounded cache; global frozen-feature shortlist is an upgraded option, not global augmented-view kNN |
| M2 | `crish/stability.py`, `crish/engine.py:_optional_stability` | Optional noisy subset KMeans/IoU/coverage; all shipped profiles set zero trials; ARI/NMI not claimed |
| Actions and rebuild | `crish/candidates.py`, `crish/rebuild.py` | Legal unlabeled Keep/Abstain/Move, minimum support, coherent centers/radii/masks; unchanged other levels |
| Abstain | `crish/rebuild.py:pseudo_sse_loss` | Remove only eligible pseudo-positive numerators; preserve original positive denominator, same-image/genuine supervised positives and USE |
| Probe | `crish/probe.py`, `crish/engine.py` | Two reserved ID-disjoint folds; class-averaged known loss, frozen teacher cross-view embedding MSE and population-variance floor; no EMA/classifier confidence |
| Lookahead | `crish/lookahead.py`, `crish/engine.py` | One cloned projector SGD step with actual momentum/weight decay; first-order alignment is diagnostic only |
| Acceptance | `crish/policy.py`, `crish/engine.py` | Rank exact discovery gain, confirm only its winner; failed confirmation stops the refresh without searching runners-up; no averaged multi-fold vote |
| Integration | `crish/training.py`, `crish/config.py` | Fresh HSSK each epoch; repair after warmup on its interval; accepted edits expire at the next epoch's extraction |
| Selection/evaluation | `gcd_runtime/selection.py`, `gcd_runtime/disjoint_evaluation.py`, `gcd_runtime/trainer.py` | Default best-Old test-v2 selection uses evaluation labels, including novel labels for global matching; final balanced SS on train-unlabeled reports best and last separately |
| Dataset/experiment scope | `gcd_runtime/datasets.py`, `configs/` | Cars/CUB implemented; Cars exploratory, not pristine held-out; other datasets and B1–B5/B7/B0+ controls remain planned |

## Automatic consistency checks

```bash
.venv/bin/python scripts/check_manuscript.py
```

The checker loads real configs with dataclass defaults, checks matched B0/CRiSH
settings, compares the generated table, verifies a digest of method/runtime/
audit/upstream/training-config sources, and checks bibliography/reference keys.
It also verifies the LaTeX/PDF hashes in `paper/manuscript.build.json`.
Changing source requires renewed semantic review of equations and pseudocode;
only after that review regenerate the marked table/digest:

```bash
.venv/bin/python scripts/check_manuscript.py --write
.venv/bin/python scripts/build_paper.py --compiler PATH_TO_TEX_COMPILER
.venv/bin/python scripts/check_manuscript.py
```

The build helper uses an existing Tectonic/pdfLaTeX/XeLaTeX compiler, compiles in
a temporary directory with finite timeouts, publishes only a successful PDF,
and writes its provenance. It installs nothing and leaves intermediate files
out of `paper/`. Before publishing it verifies that source/implementation did
not change while compiling. The normal CPU smoke runner also runs the checker.
A hash check detects drift; it does not prove that prose or equations are
semantically correct. The source-by-source review above remains necessary.

## Remaining scientific limits

- M2 remains disabled; this is now described consistently, not silently enabled.
- Test-dependent checkpoint selection is explicitly disclosed. Describing it
  correctly does not make it an independent validation protocol.
- B0 remains a matched runtime control with shared numerical fixes, not a
  verified reproduction of every published table entry.
- No completed CRiSH/B0 matched multi-seed benchmark or generalization evidence
  has been added. Synthetic tests and historical retired-method runs cannot
  fill the pending accuracy table.
- Full novelty/related-work comparison and author/affiliation metadata are
  still required before a submission. The bibliography is not evidence that
  these novelty gates have been met.

The subsequent evaluation audit in `07_evaluation_and_retraining_audit.md`
documents the upstream-confirmed ordering issue and released-code selector/
restart differences. The manuscript includes those caveats. A reporting-only
fix now distinguishes fixed-last checkpoint provenance from a run's separate
test-dependent selector; this does not change accuracy or repair behavior.
