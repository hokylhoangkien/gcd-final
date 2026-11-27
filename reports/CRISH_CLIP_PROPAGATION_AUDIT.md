# CLIP repair propagation on Stanford Cars

This sealed audit reuses R2-150, the three R2-155 continuation endpoints and all five intervention ledgers. The fixed 588-ID panel was seen before the continuation and excluded throughout the five new epochs. Source cohorts contain actual T1/T2 MOVE IDs, can overlap one another, and are disjoint from the panel.

The initial slot alignment is fitted on the same non-panel reference IDs used in the continuation report. An occurrence is unresolved when either slot has no mapping. Moved mass is 0.2 per occurrence. The batch-normalized contribution to the weighted distillation target is 0.65 × 0.2 / 128 per occurrence, with the two-view mean already accounted for. This is target mass rather than a measurement of gradient or parameter change.

Per-ID margins compare the same frozen CLIP source and destination on one reproducible pair of image augmentations across the four states. Teacher and student share raw logits and use temperatures .04 and .1. Endpoint crops do not reproduce historical batch logits. Confidence, true-slot rank and slot changes use center views. Ranks require a mapped true slot. Feature drift is one minus normalized CLS cosine. Effective classifier weights use frozen-scale weight normalization and are compared to R2-150 without permuting native slots.

Intervals are pointwise 95% paired query bootstraps, stratified by Old/New. Ledger intervals resample IDs with all repeated occurrences together. Inference conditions on the fixed checkpoints, mapping and cohorts. Source cohorts are selected by observed exposure, so their changes do not establish a causal effect on those IDs.

| Group / arm | Edits / distinct IDs / repeated IDs | Repairs / harmful / wrong-to-wrong / unresolved | Moved mass |
|---|---|---|---:|
| All / T1 | 952 / 269 / 238 | 325 / 46 / 500 / 81 | 190.4 |
| All / T2 | 952 / 873 / 74 | 114 / 522 / 282 / 34 | 190.4 |
| Old / T1 | 277 / 81 / 72 | 101 / 23 / 135 / 18 | 55.4 |
| Old / T2 | 319 / 280 / 36 | 18 / 237 / 58 / 6 | 63.8 |
| New / T1 | 675 / 188 / 166 | 224 / 23 / 365 / 63 | 135.0 |
| New / T2 | 633 / 593 / 38 | 96 / 285 / 224 / 28 | 126.6 |

| Cohort / snapshot | Accuracy All / Old / New (%) | Mean teacher margin (All) | True-slot rank (All) | CLS drift (All) |
|---|---|---:|---:|---:|
| panel / R150 | 58.5034 / 69.7279 / 47.2789 | — | 7.4728 | 0.0000 |
| panel / T0 | 58.5034 / 69.7279 / 47.2789 | — | 7.7029 | 0.0057 |
| panel / T1 | 58.5034 / 69.7279 / 47.2789 | — | 7.3207 | 0.0067 |
| panel / T2 | 58.6735 / 69.7279 / 47.6190 | — | 7.5851 | 0.0067 |
| T1_sources / R150 | 5.2045 / 9.8765 / 3.1915 | -14.0855 | 33.5820 | 0.0000 |
| T1_sources / T0 | 5.2045 / 9.8765 / 3.1915 | -14.0365 | 34.6484 | 0.0069 |
| T1_sources / T1 | 5.5762 / 9.8765 / 3.7234 | -11.5192 | 28.6211 | 0.0178 |
| T1_sources / T2 | 5.2045 / 9.8765 / 3.1915 | -13.7870 | 33.6172 | 0.0078 |
| T2_sources / R150 | 54.8683 / 73.9286 / 45.8685 | -12.4530 | 7.8819 | 0.0000 |
| T2_sources / T0 | 54.9828 / 74.2857 / 45.8685 | -12.4182 | 8.0037 | 0.0058 |
| T2_sources / T1 | 54.9828 / 74.2857 / 45.8685 | -12.1285 | 7.2595 | 0.0074 |
| T2_sources / T2 | 54.9828 / 74.2857 / 45.8685 | -11.7899 | 7.5141 | 0.0072 |

| Snapshot | Relative effective weight L2 | Mean cosine drift All / Old / New |
|---|---:|---|
| R150 | 0.000000 | 0.0000 / 0.0000 / 0.0000 |
| T0 | 0.014595 | 0.0001 / 0.0000 / 0.0002 |
| T1 | 0.021191 | 0.0002 / 0.0001 / 0.0004 |
| T2 | 0.016916 | 0.0001 / 0.0001 / 0.0002 |

| Feature / head | Fixed panel All / Old / New (%) |
|---|---|
| T0 / T0 | 58.50 / 69.73 / 47.28 |
| T0 / T1 | 58.50 / 69.73 / 47.28 |
| T0 / T2 | 58.50 / 69.73 / 47.28 |
| T1 / T0 | 58.50 / 69.73 / 47.28 |
| T1 / T1 | 58.50 / 69.73 / 47.28 |
| T1 / T2 | 58.50 / 69.73 / 47.28 |
| T2 / T0 | 58.67 / 69.73 / 47.62 |
| T2 / T1 | 58.67 / 69.73 / 47.62 |
| T2 / T2 | 58.67 / 69.73 / 47.62 |

Swaps hold the initial semantic alignment fixed across all nine combinations. The previous readout protocol fits spherical and semisupervised K-means on all non-panel IDs using seeds 1/7/19. Oracle centroid and unit-ridge linear readouts use privileged fit-set labels as diagnostic references and cannot be deployed under GCD label access. Panel labels are used only for scoring and optional descriptive rematching.

| Snapshot / readout | Fit-aligned All / Old / New (%) | Rematched All / Old / New (%) |
|---|---|---|
| R150 / spherical | 51.53 / 66.21 / 36.85 | 58.84 / 70.29 / 47.39 |
| R150 / semi | 57.09 / 72.90 / 41.27 | 62.36 / 75.51 / 49.21 |
| R150 / centroid | 64.80 / 76.87 / 52.72 | 67.52 / 77.21 / 57.82 |
| R150 / linear | 64.97 / 79.59 / 50.34 | 66.33 / 79.93 / 52.72 |
| T0 / spherical | 51.19 / 64.74 / 37.64 | 58.39 / 69.61 / 47.17 |
| T0 / semi | 55.84 / 71.54 / 40.14 | 61.62 / 74.83 / 48.41 |
| T0 / centroid | 65.31 / 77.55 / 53.06 | 67.86 / 77.89 / 57.82 |
| T0 / linear | 65.14 / 80.61 / 49.66 | 66.33 / 80.95 / 51.70 |
| T1 / spherical | 51.81 / 65.99 / 37.64 | 58.56 / 70.75 / 46.37 |
| T1 / semi | 55.78 / 71.54 / 40.02 | 61.39 / 74.94 / 47.85 |
| T1 / centroid | 65.14 / 77.55 / 52.72 | 67.69 / 77.89 / 57.48 |
| T1 / linear | 65.48 / 80.27 / 50.68 | 66.67 / 80.61 / 52.72 |
| T2 / spherical | 51.19 / 64.97 / 37.41 | 58.67 / 69.95 / 47.39 |
| T2 / semi | 55.73 / 70.86 / 40.59 | 61.56 / 74.94 / 48.19 |
| T2 / centroid | 64.80 / 76.87 / 52.72 | 67.18 / 77.21 / 57.14 |
| T2 / linear | 65.31 / 80.27 / 50.34 | 66.50 / 80.61 / 52.38 |

Elapsed 257.3 seconds. Per-ID predictions, ranks and margins are in the local per-id.csv.

[Paired intervals, subgroup details and digests](r2_clip_propagation_audit.json)
