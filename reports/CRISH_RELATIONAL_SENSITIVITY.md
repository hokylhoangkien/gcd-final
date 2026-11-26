# Cars relational gradient sensitivity before training

**Training lock: λ = 1.0 for R1 and R2, R2 edge delta = 0.25.** At this setting the median edit-only gradient reached 0.121% of the original block-11 gradient with the fresh head and 0.129% with the learned head. The median pairwise displacement increased about tenfold over the paired λ = 0.1 reference. The largest full weighted R2 gradient across all eight batch/head combinations was 0.908% of SimGCD. The response is measurable and distributed across the backbone. Semantic benefit and long-run stability remain open.

## Decision rule

Use only label-blind, paired block-11 responses. Fix the following descriptive engineering tolerances before reading the receipt. On **each** head snapshot seek a median edit gradient at least 0.10% of the SimGCD gradient and median pairwise cosine RMS at least three times the original (0.1, 0.25) grid cell on the same batches. Across **every** measured batch and head require the weighted R2 gradient below 5% of the SimGCD norm and less than 1% relative loss of held-out normalized feature variance against same-coefficient KEEP. Prefer delta 0.25, which retains positive attraction on WEAKEN, then the smallest coefficient meeting these bounds. Report signed effects and spillover even if unfavorable. These thresholds express a budget for a one-step perturbation; they have no established relationship to accuracy.

## Paired protocol

* Start from the seed-1 DINOv1/SimGCD initialization. Build one full 6,144-image reciprocal 5-NN graph with the production image-only refresh, center view and two independent augmented views. Save digests of the graph keys, edges, votes and each batch's two input crops. Sample four batch-128 draws with replacement from the original weighted sampler. Duplicate occurrences and labeled rows remain in the denominator.
* Compare the fresh head with a copy trained for eight mixed batch-128 SimGCD steps at the initial warmup temperature, SGD LR .1 and momentum .9. The backbone stays frozen during those eight steps. Both snapshots therefore share identical keys, graph, block parameters and measurement inputs. Both use the first active epoch's temperature for measurement. A learned head changes the ordinary SimGCD gradient and the shared factual update. It cannot change the relational gradient at fixed features. This isolates one source of sensitivity. It does not reconstruct the production epoch-11 backbone or its optimizer history.
* Each sampled batch has a 48-image center-view panel drawn by a fixed random ordering of unlabeled IDs, excluding every head-warmup ID and the current mixed batch. Panels can overlap across batches. The graph itself includes the panel images. They measure response outside the gradient batch and are not a held-out dataset for generalization. Unlabeled rows supply images, stable IDs and a sentinel. Only the original supervised terms see legal labeled targets. The diagnostic never iterates the annotated evaluation loader or scores accuracy.

For snapshot parameters $\theta$, let $c_{ivj}=1-\hat h_{iv}^{\top}\hat k_j$ with detached peer keys. A reciprocal edge has initial weight $w=.5$ and sign $s_{ij}=\mathrm{votes}_{ij}-1$. Padded slots have zero weight and zero sign. With $D=2Bk$, the two exact relational gradient bases at block 11 are

$$
g_K=\nabla_\theta\frac{\sum w c_{ivj}}{D},\qquad
g_E=\nabla_\theta\frac{\sum s_{ij} c_{ivj}}{D}.
$$

The weighted R1, R2 and edit-only gradients are $\lambda g_K$, $\lambda(g_K+\delta g_E)$ and $\lambda\delta g_E$. Compute these two bases once per batch using the production graph loss and the full mixed-batch normalization, then sum microbatch vector–Jacobian products through block 11. Compute the SimGCD gradient $g_S$ once per head. This uses four full block-gradient calculations per batch for both snapshots and all ten grid cells. The linear identity applies to gradients before the step. The feature response requires a new forward for each distinct updated block.

For each arm apply one fresh-momentum SGD block step of size .1 with ordinary parameter-group weight decay, from the same saved FP32 parameters. Compare R2 against R1 at the **same** coefficient and rerun the final block on the complete batch and panel. The head update cannot alter immediate backbone features on this step. Identical KEEP replays are checked bitwise and the block is restored after each batch. Pairwise cosine and variance differences use FP64 reductions on features produced by FP32 forwards.

RMS pairwise displacement uses the full matrix of both views, including repeated IDs. Directed STRENGTHEN, WEAKEN and unchanged-edge statistics compare online batch features with the same frozen full-graph keys. Panel spillover is measured both against full-graph reciprocal keys and through changes to online–online cosines for fixed top-5 peers within the panel and the remaining off-diagonal pairs. Variance is the mean per-coordinate population variance of unit CLS features; the resultant is the norm of their mean. Both measure global contraction on the sampled inputs. Effective edit mass counts each edited anchor–peer occurrence once, giving $\lambda\delta(n_+ + n_-)/(Bk)$, whereas signed cosine statistics count both views.

## Full-graph sensitivity results

The full graph contained 4,570 reciprocal edges and 2,044 isolated nodes. Counts for zero, one and two crop votes were 1,502, 1,572 and 1,496. The four batches had 60, 67, 54 and 78 unlabeled draws, with STRENGTHEN/WEAKEN occurrences of 29/31, 31/24, 18/24 and 35/36. These four ordinary draws characterize within-run variation only. The head moved 3.177 in parameter L2 over the eight warmup steps.

Each pair of values is the median across four batches with the **fresh / learned head**. Gradient columns give percentages of the matching SimGCD gradient. Pairwise RMS values are in millionths of cosine. Peak R2 is the maximum ratio over both heads and all batches. Mass is the median per mixed batch. All gradients, directional responses and geometry statistics for every cell are available in the [raw receipt](crish_relational_sensitivity.json).

| λ | Delta | R1 % | R2 % | Edit % | Pairwise RMS × 10⁶ | Peak R2 % | Effective mass | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.1 | 0.25 | 0.069 / 0.073 | 0.067 / 0.073 | 0.012 / 0.013 | 3.45 / 3.48 | 0.091 | 0.00225 | no |
| 0.1 | 0.50 | 0.069 / 0.073 | 0.068 / 0.073 | 0.024 / 0.026 | 6.91 / 6.95 | 0.094 | 0.00449 | no |
| 0.5 | 0.25 | 0.344 / 0.367 | 0.333 / 0.364 | 0.060 / 0.064 | 17.28 / 17.37 | 0.454 | 0.01123 | no |
| 0.5 | 0.50 | 0.344 / 0.367 | 0.341 / 0.364 | 0.121 / 0.129 | 34.56 / 34.73 | 0.469 | 0.02246 | yes |
| **1** | **0.25** | **0.689 / 0.734** | **0.666 / 0.728** | **0.121 / 0.129** | **34.55 / 34.72** | **0.908** | **0.02246** | **yes** |
| 1 | 0.50 | 0.689 / 0.734 | 0.683 / 0.727 | 0.242 / 0.258 | 69.11 / 69.45 | 0.939 | 0.04492 | yes |
| 2 | 0.25 | 1.378 / 1.467 | 1.333 / 1.456 | 0.242 / 0.258 | 69.11 / 69.41 | 1.815 | 0.04492 | yes |
| 2 | 0.50 | 1.378 / 1.467 | 1.366 / 1.455 | 0.483 / 0.515 | 138.20 / 138.82 | 1.878 | 0.08984 | yes |
| 5 | 0.25 | 3.444 / 3.668 | 3.332 / 3.640 | 0.604 / 0.644 | 172.72 / 173.23 | 4.538 | 0.11230 | yes |
| 5 | 0.50 | 3.444 / 3.668 | 3.414 / 3.637 | 1.208 / 1.289 | 345.36 / 346.46 | 4.694 | 0.22461 | yes |

The original λ = 0.1, delta = 0.25 smoke measured 0.0464% edit, 0.2094% total and 19.22 millionths pairwise RMS on a reduced 128-image graph and one draw. The paired full-graph reference above is weaker. Both the neighborhoods and sampled anchors differ, so the difference between receipts cannot be attributed to graph size alone. Use the first table row as the comparator for the lock.

## Mechanism and limits at the selected dose

| Median across batches | Fresh | Learned head |
|---|---:|---:|
| SimGCD block-11 gradient norm | 1.474 | 1.349 |
| Weighted R1 / R2 / edit-only norms | .01009 / .01002 / .001753 | .01009 / .01002 / .001753 |
| Cosine with SimGCD for R1 / R2 / edit | −.468 / −.457 / +.140 | −.154 / −.153 / +.011 |
| Cosine of edit with KEEP | −.313 | −.313 |
| Paired unit-CLS RMS L2, batch / panel | .0001210 / .0001165 | .0001202 / .0001146 |
| Mean cosine change on STRENGTHEN / WEAKEN vs KEEP, × 10⁶ | +1.19 / −26.35 | +2.68 / −21.77 |
| Fraction of STRENGTHEN views with increased cosine | 61.6% | 68.7% |
| Mean cosine change on unchanged batch edges, × 10⁶ | −11.56 | −8.14 |
| Panel reciprocal-anchor/frozen-peer mean change, × 10⁶ | −10.48 | −7.55 |
| Panel within-top-5 / other pairs, RMS change × 10⁶ | 27.29 / 35.21 | 26.55 / 35.28 |
| Panel variance after R0 / KEEP / R2 | .00060490 / .00060437 / .00060439 | .00057225 / .00057162 / .00057167 |
| Panel resultant length after KEEP / R2 | .732013 / .732001 | .748988 / .748965 |

* **Direction.** The entire graph term partly opposes SimGCD in both snapshots. The edit-only direction is closer to orthogonal to SimGCD after head warmup (batch cosine range −.048 to +.079). Gradient magnitude scales exactly with λ at fixed delta, so increasing λ cannot correct an unfavorable direction. WEAKEN had negative mean response in all eight batch/head pairs. STRENGTHEN had positive mean response in seven. Shared parameters allow an edit to affect edges of every sign.
* **Spillover and collapse.** Panel displacement is close to in-batch displacement. Non-neighbor pairs inside the panel move at least as much in median RMS as its nearest pairs. This supplies evidence of a distributed geometric response and limits any claim of localized correction. The minimum per-batch relative panel variance change versus KEEP was −0.00078%, and the median was +0.0070% / +0.0074%. The median panel variance before any step was .00055636. A one-step global variance check cannot exclude later contraction, partial class collapse or semantic damage.
* **Choice.** At delta .25, λ .5 missed the median edit threshold on both heads. The alternative (.5, .5) gave the same edit mass and edit gradient as (1, .25), and a smaller total gradient. Its WEAKEN edges have weight zero and STRENGTHEN edges weight one. We keep interior weights .25 and .75 and accept the larger ordinary KEEP term, then use the smallest passing λ with that property. This preference was specified before examining results. Median absolute edit mass is 14.375 and effective mass ranges from .01641 to .02773 on the four batches.
* **Scope.** Fresh momentum, fixed pretrained backbone, four replacement draws, and one graph limit extrapolation to a trained model, stale keys, later teacher temperatures and accumulated optimizer momentum. The eight head updates cannot stand in for ten production warmup epochs. Paired RMS varies across fresh-head batches from 9.12 to 39.77 millionths. These measurements establish a detectable perturbation and estimate its size at the sampled states. They do not test whether the perturbed neighbors are semantically correct.

## Lock, validation and reproduction

R1 and R2 share λ = 1.0. R2 retains the signed delta .25, exact k = 5, fixed divisor 1,280, epoch-level refresh and original sampler. The 200-epoch cosine schedule, LR .1, ordinary losses, evaluator and standalone guards at **100 and 150** remain as in the Linux runbook. Start future matched runs on this checkout and a new run root. The GPU run was a local bounded gradient diagnostic, independent of the pending server benchmark. No full training or ACC-based selection was performed.

Set the paths and Python environment in the [Linux runbook](CRISH_RELATIONAL_LINUX.md), then run from the checkout root:

~~~bash
python -m relational.sensitivity dev_outputs/sensitivity.json
python -m relational.sensitivity_summary dev_outputs/sensitivity.json
python validate_relational.py
python validate_standalone.py
python validate_relational.py --cuda-parity
~~~

Validation after applying the lock passed 10 relational tests and seven standalone controller/replay tests. CUDA parity reproduced the upstream R0 one-step weights and momentum bit-for-bit. R2 CUDA replay reproduced sampled IDs, exposures, parameters, momentum, scheduler and RNG.

The receipt records parent commit 22468ac and the pre-lock λ .1 in its protocol. Its individual cells explicitly specify their own λ and delta. Re-running on the lock commit changes protocol provenance and the production default, while the diagnostic continues to evaluate the same ten cells. The direct-autograd fixture covers both edit signs, a duplicate anchor, a labeled row and all ten combinations. Repeated identical KEEP updates yielded exactly equal features for all eight measured batch/head comparisons.
