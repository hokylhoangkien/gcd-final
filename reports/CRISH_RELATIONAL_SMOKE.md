# CRiSH-R validation and pre-training decision

**GO for guarded exploratory R0/R1/R2.** The evidence policy changed actual pairwise backbone geometry under a matched step. The edit-only gradient was small and partly opposed the ordinary SimGCD gradient. The smoke supports testing efficacy; it provides no trained Cars ACC, no evidence about generalization and no basis to enable R4. Complete R0 before interpreting either relational variant. [Locked design, guard thresholds and launch instructions](CRISH_RELATIONAL_DESIGN.md)

## Real Cars, one update

Implementation commit 69fda264a644b156bf277edaae13da9b0b77544c, based on ed4b388. NVIDIA GeForce RTX 3050 Ti Laptop GPU, torch 2.6.0+cu124, FP32. DINOv1 pretrained weights, seed-1 fresh SimGCD head. The label-blind graph used 128 randomly selected official unlabeled training IDs (fixed seed 701), plus the official labeled pool for mixed training draws. A single ordinary replacement-sampled batch had 128 occurrences, including 57 unlabeled draws. Graph encoding and frozen-prefix caching used microbatches of four. The smoke used the first active epoch's teacher temperature and a local diagnostic SGD step of .1 with fresh momentum, at pretrained initialization. This was not a replay of an epoch-11 model.

We held the batch, both crops, peer keys, initial parameters and ordinary SimGCD gradient fixed. Full-batch losses saw all 256 views. Microbatch vector–Jacobian products were summed at block 11. We evaluated the exact block parameter updates for baseline plus R1 and baseline plus R2, re-encoded the same inputs and restored original block weights. At this one-step horizon simultaneous head updates cannot change the just-updated backbone features. A separate integration fixture used the production optimizer path.

| Graph and exposure measurement | Observed |
|---|---:|
| Baseline reciprocal edges / isolated nodes | 152 / 18 |
| Votes 0 / 1 / 2 | 27 / 45 / 80 |
| Two-crop reciprocal-edge Jaccard | .46262 |
| Proposed STRENGTHEN / WEAKEN | 80 / 27 |
| Sampled graph-edge occurrences including KEEP | 128 |
| Applied STRENGTHEN / WEAKEN occurrences | 84 / 13 |
| Distinct edited edges / sampled edited anchors | 60 / 33 |
| Absolute intervention mass / summed effective coefficient | 24.25 / .00378906 |
| Mean feature-coordinate variance / mean resultant length | .00050530 / .78226 |
| Diagnostic runtime / peak allocated GPU memory | 15.33 s / 809.46 MiB |

The runtime starts after constructing the pretrained model/datasets. It covers the graph refresh, image forwards, gradient calculations and paired block steps. The full Stanford Cars graph has a different neighborhood distribution and substantially higher search/encoding cost. The full production training batch is not microbatched; this memory number is specific to the diagnostic.

| Quantity on the same batch | Loss before weighting | Weighted block-11 gradient norm | Ratio to SimGCD | Cosine with SimGCD |
|---|---:|---:|---:|---:|
| SimGCD | 9.788596 | 1.096127 | 100% | 1 |
| R1 graph term | .0212356 | .00184267 | .1681% | −.4232 |
| R2 graph term | .0272744 | .00229543 | .2094% | −.4130 |
| R2 minus R1 graph term | .0060388 | .00050840 | .0464% | −.3308 |

After one matched update, the RMS L2 difference in unit CLS vectors was **0.00004548**, and the RMS difference of entries in their pairwise cosine matrix was **0.00001922**. Mean R2-minus-R1 cosine with cached peers was **+0.00001024 on strengthened occurrences and +0.00000782 on weakened occurrences**. The latter also increased at this batch level. A WEAKEN reduces the individual attraction gradient relative to KEEP; it does not guarantee lower observed cosine once all edited anchors share the same backbone update. The direction-isolation unit test and the multi-edge smoke answer different questions. No classifier output was used to calculate these geometric effects.

## Sealed geometry measurement

An independent evaluator measured the pretrained reference on the fixed 980-image panel from the parent audit, with five official unlabeled training images per class. The ground-truth join occurs inside the offline process after configurations were fixed. None of these values enters the graph, policy, lambda choice or budget evaluator.

| Subgroup | n | Cosine kNN-5 purity | Leave-one-out own-centroid cosine | Nearest different centroid distance | Own vs nearest-other margin | Silhouette |
|---|---:|---:|---:|---:|---:|---:|
| Old | 490 | .06653 | .75368 | .08677 | −.06220 | −.19627 |
| Novel | 490 | .04735 | .76035 | .08089 | −.06503 | −.20518 |
| All | 980 | .05694 | .75701 | .08383 | −.06362 | −.20073 |

Five images per class implies at most four same-class neighbors and a maximum purity@5 of .8. These are **initialization measurements**. There are no R0/R1/R2 endpoint geometry or historical gradients yet. The [machine-readable receipt](crish_relational_smoke.json) includes the source hash, exact panel IDs, per-edge exposure counts and both diagnostics.

## Executed checks

| Check | Result |
|---|---|
| Nine CPU contract/integration tests | Passed. KEEP loss/gradient parity, positive and negative differential attraction, zero-weight behavior, fixed denominator, detached peers, duplicate draws, invalid weights, poisoned hidden-label accessor, sampler parity and actual R1/R2 optimizer-path differences |
| External guard/controller fixtures (part of the nine) | Passed both guard boundaries, one-based windows, matched/missing/mismatched references, R2 comparisons, clean BLOCKED and NO-GO exits before another training stage, explicit R1 reference-only continuation |
| State and export fixtures (part of the nine) | Passed optimizer, scheduler, sampling, RNG restoration, journal rollback, latest-plus-milestone retention, archive member verification and checksum corruption rejection |
| Official R0 CUDA parity | Identical model tensors and SGD momentum buffers after the same update as upstream f843f27195701b7c8c5713cfd17499ad67c86247 on a synthetic batch. First-step teacher temperature matched. Long-run equality was not tested. |
| CUDA R2 resume | A synthetic resumed epoch reproduced sampled IDs, edge exposures, all model parameters, momentum buffers, cosine scheduler state and subsequent CUDA random draws bit-for-bit on this machine |
| Local Cars smoke and sealed evaluator | Both completed under individual 300 s watchdogs. The first development run exposed Windows integer ID width; the final graph uses the repository's stable-ID conversion before checking uniqueness. |
| Static checks | Python compilation, Ruff and Bash syntax for launch/resume passed |

The controller test substitutes tiny committed fixtures for training workers and a geometry receipt for the offline GPU process. This original smoke used a reduced graph. Subsequent full-graph timing and generic Linux launch instructions are in the [standalone runbook](CRISH_RELATIONAL_LINUX.md). The original smoke did not exercise remote tmux, network download, real 100/150-epoch guards or a 200-epoch resume. The archive was created and verified locally with fixture checkpoints. A sealed-evaluator error on a future server run is recorded in the bundle so that the checkpoint can still be retrieved.

## Reproduction

Use the environment/path exports in the design and run from the checkout root. CPU tests create only temporary fixture checkpoints. The two GPU measurements below do not run the Cars trainer.

~~~bash
python validate_relational.py
python validate_relational.py --cuda-parity
mkdir -p dev_outputs/rel-smoke
timeout 300s python -m relational.smoke dev_outputs/rel-smoke/smoke.json
timeout 300s python -m relational.offline "$SIMGCD_DINO_WEIGHTS" dev_outputs/rel-smoke/sealed.json --reference
~~~

This receipt documents the historical coefficient .1 on its original commit. The later full-graph, multi-batch diagnostic determines the pre-training coefficient for both R1 and R2. To reproduce the historical smoke, use the implementation commit above. [Current dose and supporting measurements](CRISH_RELATIONAL_SENSITIVITY.md)
