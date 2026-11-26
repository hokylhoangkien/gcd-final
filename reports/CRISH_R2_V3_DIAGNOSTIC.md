# V3 bounded verifier decision diagnostic

Twelve fresh batch-128 weighted draws across R2 checkpoints 100/150. KEEP, MOVE (.2) and ABSTAIN use E3 targets. Joint virtual SGD uses full SimGCD, including InfoNCE and SupCon through block 11, with the saved optimizer state and without the R2 relational term. The classifier result is checked against the functional V0 update for each arm. V0 serves as a classifier-only predictor of the same sealed joint-update effect assessed by the other policies.

Train, reference, discovery, confirmation and sealed probe IDs are disjoint within each batch. Each proxy fold has 32 known and 96 unlabeled images. Centroids, target slots and competitors are fixed before intervention. The discovery top candidate alone is eligible for acceptance, with no fallback. V0 ranks the original objective. V2 ranks relative margin. V3-A adds confirmation to strict V2, V3-B relaxes only the CE guard, and V3 combines both. Positive margin must exceed 1e-7 in both FP64 and FP32 scoring, with their gains agreeing within max(1e-7, 10% of gain). The backbone forward uses FP32 in both. Strict CE requires non-increase; relaxed CE permits +0.1% of fold KEEP CE. With confirmation the gate applies separately on both folds.

Decisions and guard pools are frozen before the sealed label join. Both random controls match each policy's exact intervention batches and count. The guard-matched random pool further restricts eligible candidates by that policy's CE rule. Entries below average over all 12 batches, with zero for no action. Harmful means sealed All gain below −1e-7. The small, checkpoint-conditioned sample cannot establish trajectory-level effects.

| Policy | Actions | Harmful / accepted | Gain All / Old / New (×10⁶) | Count-matched random All / Old / New | Guard-matched random All / Old / New |
|---|---:|---:|---|---|---|
| V0 | 8 | 4/8 | +6.506 / +4.973 / +8.386 | +1.293 / -2.340 / +3.506 | +2.196 / -0.522 / +3.822 |
| V2 | 6 | 3/6 | +1.885 / +2.499 / +1.805 | +1.536 / +0.772 / +2.016 | +0.575 / +0.048 / +1.092 |
| V3-A | 3 | 2/3 | -0.671 / -3.188 / +0.907 | +0.553 / -0.737 / +1.334 | -0.671 / -3.188 / +0.907 |
| V3-B | 11 | 7/11 | -0.348 / -3.387 / +2.154 | +1.365 / -2.330 / +3.600 | +2.372 / +0.812 / +3.670 |
| V3 | 7 | 5/7 | +0.038 / -1.587 / +1.333 | +0.297 / -1.523 / +1.317 | +1.303 / +1.619 / +1.388 |

Rejection and acceptance counts

- V0 — accepted 8, discovery_gain 3, discovery_known_ce 1
- V2 — accepted 6, discovery_gain 1, discovery_known_ce 5
- V3-A — accepted 3, confirmation_gain 3, discovery_gain 1, discovery_known_ce 5
- V3-B — accepted 11, discovery_gain 1
- V3 — accepted 7, confirmation_gain 4, discovery_gain 1

Evaluated 48 actions ({'move': 23, 'abstain': 25}); 0 unfilled positions, 0 batches without eligible MOVE. Fold coverage ranged from 94 to 96 of 96. Sealed coverage: 3072 of 3072. Batch compute: 394.8 s.

[Per-batch results, decisions and provenance](r2_v3_diagnostic.json)
