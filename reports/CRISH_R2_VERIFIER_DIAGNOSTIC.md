# Bounded verifier diagnostic on R2 checkpoints

**Result.** V2 improved within-batch rank correlation over V0 and V1, but its guarded choices had worse semantic gain than V0 and its own dose-matched random comparator. These checkpoints do not support replacing V0 with V2.

| Semantic probes | Verifier | Pooled Spearman | Mean within-batch Spearman | Sign matches / determinate | Indeterminate proxy / semantic | Selected / expected random gain ×10⁶ |
|---|---|---:|---:|---:|---:|---:|
| All | V0 | +.014 | +.182 | 8 / 17 | 7 / 0 | +5.304 / +.669 |
| All | V1 | −.709 | −.200 | 6 / 23 | 1 / 0 | −4.504 / +.897 |
| All | V2 | +.090 | +.467 | 13 / 24 | 0 / 0 | −2.323 / −1.136 |
| Old | V0 | −.024 | +.045 | 10 / 17 | 7 / 0 | +9.220 / −.216 |
| Old | V1 | −.548 | −.200 | 6 / 23 | 1 / 0 | −7.977 / −.403 |
| Old | V2 | +.177 | +.233 | 13 / 24 | 0 / 0 | −5.893 / −2.829 |
| New | V0 | +.141 | +.322 | 9 / 16 | 7 / 1 | +2.605 / +.993 |
| New | V1 | −.667 | −.300 | 6 / 22 | 1 / 1 | −2.935 / +1.421 |
| New | V2 | +.037 | +.467 | 13 / 23 | 0 / 1 | −.690 / −.356 |

All 24 candidate positions were filled (4 MOVE, 20 ABSTAIN). Four of six batches had no eligible MOVE. The other two had four each. V0/V1 selected in four batches each, and V2 in three. Each relative-margin probe had 126–128 covered rows (763/768 total) and 189–190 supported slots. The semantic probe covered all 1,536 queries across the six batches (512 Old, 1,024 New, counting repeated panel membership across batches). The detached RTX 3050 Ti job took 163.1 s wall time, including 155.1 s in proxy prediction and virtual steps and 1.36 s in sealed scoring. It exited successfully.

The negative selected gain despite a positive mean rank coefficient shows that ranking and deciding when to intervene are separate problems. Several proxy-consistent candidates can rank above worse alternatives while still moving semantic neighbors in the wrong direction. The common semantic outcome also measures whether V0 predicts a block-11 update's effect. This run cannot establish that an actual V0-only step improves backbone geometry.

## Design

The audited source is `b8554f9` on `feat/crish-relational`. This diagnostic reads the completed R2 audit's extracted checkpoints at epochs 100 and 150, ordered Cars feature cache and three batch/probe manifests per checkpoint. It reconstructs only the inputs needed by the verifier. Batch ID order is asserted against the previous audit. R2 trained with an edge-weight loss. The edits in this diagnostic are E3 pseudo-target edits applied to the resulting R2 checkpoints.

Keep the E3 classification objective in every virtual step: labeled cross-entropy, distillation and mean entropy maximization with their original weights and temperature schedule. The audited checkpoints are past the end of teacher temperature warmup, so the teacher temperature is .04. KEEP uses unmodified targets. MOVE uses the original .2 mass transfer and both-view source-mass check. ABSTAIN zeros its two target rows while preserving the distillation denominator. Use the repository's E3 batch proposal routine, exclude all repeated IDs from intervention, and take up to two candidates of each action type in its existing order. If a type supplies fewer than two, fill remaining places using the shared proposal order. This is a bounded batch-local proposal comparison. The global E3 shortlist, greedy group selection and independent confirmation are outside this diagnostic.

| Variant | Virtual parameters | Proxy score, relative to KEEP |
|---|---|---|
| V0 | E3 weight-normalized classifier | Reduction in original probe cross-view consistency loss plus known CE |
| V1 | Same classifier and ViT block 11 | Reduction in the same original probe objective, with updated probe features |
| V2 | Identical update to V1 | Increase in frozen target-minus-competitor centroid cosine margin |

The frozen opposite-view classifier distributions used by V0/V1 come from the pre-step state. Known CE is recorded and guarded separately in all variants. V2 gives known CE no weight in its proxy. Every branch reloads the saved model parameters and relevant SGD momentum, decay and learning rate. The classifier tensor from the joint step is checked against the original functional E3 step for every action. The head projector is held fixed. Frozen-prefix token caches are reused for batch VJPs and probe re-encoding. V1/V2 share both the same gradient calculations and updated probe features.

Use 32 disjoint labeled images with legal labels plus 128 unlabeled proxy images, both with two deterministic training crops. The remaining 256 audit probe images per batch are separate from the proxy and training anchors. In a fixed reference bank, exclude every training and probe ID. Retain reference rows whose cached two augmented views predict the same slot with probability at least 0.5 in both views. Form each centroid by averaging normalized center features assigned to that slot, then normalize. A slot needs at least three reference rows. A V2 probe needs an agreeing two-view predicted slot with a supported centroid and at least one supported alternative. The competing slot maximizes pre-step cosine with the mean pre-step probe feature across the two views, after excluding its target. Slot assignments, competitor indices and centroids all remain fixed. The V2 gain is the mean change in that same cosine difference against KEEP for covered probe views.

## Sealed comparison and limitations

Before importing the label-join function, write all proxies and known CE values to a receipt and hash the cached post-step semantic features. The separate sealed stage verifies those hashes. Its outcome is the mean paired change versus joint-update KEEP in the nearest same-class minus nearest different-class cosine, measured on the 256 separate probe IDs with fixed reference features. V0 serves as a predictor of this **same** semantic outcome, so all rank comparisons have one outcome definition. V0's own feature gain is identically zero because its backbone is frozen. Semantic queries use one deterministic augmented view and a frozen center-view bank.

Rank statistics use every evaluated action with a defined proxy. Average within-batch Spearman excludes batches with fewer than two comparable actions or constant scores. Pooled Spearman is descriptive and can mix endpoint scales. For sign agreement, proxy or semantic absolute gain at or below 10⁻⁷ is indeterminate. Selection chooses the highest positive proxy gain above that tolerance with non-increasing known CE, allowing no-op. This tolerance is diagnostic and does not replace the production E3 gate. The random comparator is the exact expected gain of a uniform eligible action in the batches where each verifier chooses to act, otherwise zero. It matches intervention count and does not apply the CE guard. Old/New refer to semantic probe classes, not predicted slots.

Two checkpoints from one trajectory and six shared batches produce dependent observations. The study tests immediate motion relative to a fixed bank and does not evaluate persistence, long-run accuracy or an independently trained E3 setting.

## Reproduction

Use the same dataset/DINO environment and cache from the [mechanistic audit](CRISH_RELATIONAL_MECHANISTIC_AUDIT.md). From the repository root:

~~~bash
WORK=dev_outputs/r2-audit
nohup python -u -m scripts.audit_r2_watchdog "$WORK/verifier.status" 3600 \
  scripts.audit_r2_verifier "$WORK" reports/r2_audit_virtual.json \
  "$WORK/verifier" reports/r2_verifier.json >"$WORK/verifier.log" 2>&1 &
echo $!
~~~

The watchdog records the CPU supervisor and GPU subprocess PID. Disposable tensors and frozen predictions stay under the work directory. [Machine-readable results](r2_verifier.json)
