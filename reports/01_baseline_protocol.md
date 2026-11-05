# Protocol relevant to the training failure — 2026-11-05

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

This is a bug-fix audit, not a new method or a completed baseline reproduction.
Existing novelty risks and unclosed method gates remain in
`presentation/CRISH_GCD.md`, section 13 (RCA, LEAH, HypGCD and further leads).
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
  curvature .1, clipping radius 2, maximum distance weight 1.
- CRiSH warmup 10 epochs / refresh every 5. The failed extraction precedes
  CRiSH action scoring; all three completed epochs report zero repairs.
- Training strips unlabeled targets to -1. HSSK receives only legal anchor
  labels; higher-level anchors derive from predictions. No hidden targets
  enter initialization, empty-cluster relocation or acceptance scoring.

## Evaluation and comparability caveats

Final fixed-epoch backbone evaluation runs in a separate worker. It fits
balanced semi-supervised k-means with labeled anchors over the train pool and
scores only the unlabeled rows. Hungarian-v2 uses one global matching; Old/New
select the known/novel subsets after matching. `All = (n_old*Old + n_new*New)
/ (n_old+n_new)`, not the unweighted average. Saved accuracies are fractions.
Training All/Old/New are null; no test label selects a checkpoint.

The upstream DINOv2 Cars example passes `hyper_max_weight=.5`, whereas the
existing local configs use 1.0. The paper's general fine-grained curvature
description also differs from the Cars command's .1. These existing differences
are preserved and must be disclosed before comparing with published scores.
The source also contains a USE broadcasting defect; this patch does not change
that arithmetic. Runtime B0 is a matched control, not a claim of an unchanged
official reproduction. Both arms now share empty-cluster handling, so compare
new B0 and CRiSH runs with the same patched source and inputs.

## Next commands

Local bounded CPU validation: `.venv/bin/python scripts/run_smoke.py`.
On Linux NVIDIA with data/weights: `bash scripts/train_crish.sh check`, then
`bash scripts/train_crish.sh smoke`. Only after those pass, start a new finite
training run with `bash scripts/train_crish.sh`. This session starts none.
