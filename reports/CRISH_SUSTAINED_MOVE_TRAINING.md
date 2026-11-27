# Sustained early CLIP MOVE through epoch 105

E0/E1/E2 resume the exact saved optimizer, scheduler and RNG at epoch 102. The original R2-100 CLIP proposal file is read unchanged. E1 uses MOVE mass .8 and E2 uses the same retriever and mass with random eligible sources. Every batch matches the number of applied edits. Targeted edits are truncated when random supply is insufficient. The continuation extends the original checkpoint-derived shared sampler and crop streams. All arms keep the full SimGCD objective with R2 relational loss disabled, as in epochs 101–102.

The fixed 588-ID panel is excluded from all continuation draws and the original proposal and reference banks. It belongs to the original transductive split. Source cohorts consist of IDs ever edited in E1 or E2 by epoch 105, evaluated retrospectively at each endpoint. The cohorts can overlap each other and are disjoint from the panel. True-slot ranks require an aligned class. Panel improvements measure transfer under this protocol.

| Epoch / arm | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure | Cumulative edits / exposure |
|---|---|---|---:|---:|---|---|
| 100 / shared | 58.16 / 70.75 / 45.58 | 63.95 / 77.21 / 50.68 | 0.2344 | -0.0134 | 0 / 0.000% | 0 / 0.000% |
| 101 / E0 | 58.84 / 71.43 / 46.26 | 64.63 / 77.55 / 51.70 | 0.2357 | -0.0135 | 0 / 0.000% | 0 / 0.000% |
| 101 / E1 | 59.18 / 71.43 / 46.94 | 64.80 / 77.55 / 52.04 | 0.2412 | -0.0000 | 156 / 1.935% | 156 / 1.935% |
| 101 / E2 | 58.33 / 70.75 / 45.92 | 63.95 / 76.87 / 51.02 | 0.2367 | -0.0155 | 156 / 1.935% | 156 / 1.935% |
| 102 / E0 | 58.67 / 71.77 / 45.58 | 64.80 / 78.23 / 51.36 | 0.2350 | -0.0137 | 0 / 0.000% | 0 / 0.000% |
| 102 / E1 | 58.84 / 71.43 / 46.26 | 64.46 / 77.89 / 51.02 | 0.2395 | -0.0046 | 108 / 1.339% | 264 / 1.637% |
| 102 / E2 | 58.33 / 71.09 / 45.58 | 64.12 / 76.53 / 51.70 | 0.2371 | -0.0128 | 108 / 1.339% | 264 / 1.637% |
| 103 / E0 | 58.67 / 71.77 / 45.58 | 64.46 / 78.23 / 50.68 | 0.2398 | -0.0116 | 0 / 0.000% | 0 / 0.000% |
| 103 / E1 | 58.84 / 71.77 / 45.92 | 64.63 / 78.23 / 51.02 | 0.2398 | -0.0040 | 82 / 1.017% | 346 / 1.430% |
| 103 / E2 | 57.99 / 70.75 / 45.24 | 64.12 / 77.21 / 51.02 | 0.2415 | -0.0114 | 82 / 1.017% | 346 / 1.430% |
| 104 / E0 | 58.67 / 71.77 / 45.58 | 64.46 / 78.23 / 50.68 | 0.2371 | -0.0139 | 0 / 0.000% | 0 / 0.000% |
| 104 / E1 | 59.18 / 72.11 / 46.26 | 64.97 / 78.57 / 51.36 | 0.2398 | -0.0065 | 86 / 1.066% | 432 / 1.339% |
| 104 / E2 | 58.33 / 71.09 / 45.58 | 63.95 / 77.55 / 50.34 | 0.2398 | -0.0136 | 86 / 1.066% | 432 / 1.339% |
| 105 / E0 | 59.18 / 71.77 / 46.60 | 64.80 / 78.23 / 51.36 | 0.2401 | -0.0087 | 0 / 0.000% | 0 / 0.000% |
| 105 / E1 | 58.50 / 71.09 / 45.92 | 64.63 / 77.21 / 52.04 | 0.2418 | -0.0043 | 50 / 0.620% | 482 / 1.195% |
| 105 / E2 | 58.84 / 70.75 / 46.94 | 64.46 / 77.21 / 51.70 | 0.2412 | -0.0093 | 50 / 0.620% | 482 / 1.195% |

## Live eligibility

A changed source takes precedence. Low mass counts rows that retain their frozen source but fall below .8 in at least one view. Random supply shortfalls are counted after eligibility.

| Epoch | E1 changed source / low mass | E2 changed source / low mass | E1 live | E2 live | Random supply shortfall |
|---|---|---|---:|---:|---:|
| 101 | 14 / 30 | 36 / 210 | 156 | 3591 | 0 |
| 102 | 48 / 40 | 71 / 188 | 108 | 3562 | 0 |
| 103 | 61 / 42 | 73 / 159 | 82 | 3531 | 0 |
| 104 | 69 / 49 | 87 / 177 | 86 | 3450 | 0 |
| 105 | 85 / 45 | 55 / 157 | 50 | 3494 | 0 |

## Source changes and panel transfer

Each row compares to epoch-matched KEEP under the original fixed mapping.

| Epoch / arm / cohort | IDs | Wrong→correct / correct→wrong | Mean rank change |
|---|---:|---|---:|
| 101 / E1 / panel | 588 | 4 / 2 | -0.573 |
| 101 / E1 / CLIP_sources | 261 | 20 / 1 | -7.056 |
| 101 / E1 / random_sources | 472 | 4 / 2 | -0.714 |
| 101 / E2 / panel | 588 | 1 / 4 | -0.164 |
| 101 / E2 / CLIP_sources | 261 | 1 / 0 | -2.266 |
| 101 / E2 / random_sources | 472 | 6 / 8 | -0.489 |
| 102 / E1 / panel | 588 | 4 / 3 | -0.663 |
| 102 / E1 / CLIP_sources | 261 | 37 / 3 | -9.319 |
| 102 / E1 / random_sources | 472 | 2 / 0 | -1.314 |
| 102 / E2 / panel | 588 | 2 / 4 | -0.632 |
| 102 / E2 / CLIP_sources | 261 | 1 / 0 | -3.149 |
| 102 / E2 / random_sources | 472 | 7 / 10 | -1.150 |
| 103 / E1 / panel | 588 | 2 / 1 | -0.717 |
| 103 / E1 / CLIP_sources | 261 | 43 / 5 | -10.060 |
| 103 / E1 / random_sources | 472 | 3 / 0 | -1.600 |
| 103 / E2 / panel | 588 | 3 / 7 | -0.351 |
| 103 / E2 / CLIP_sources | 261 | 1 / 1 | -3.137 |
| 103 / E2 / random_sources | 472 | 10 / 9 | -1.270 |
| 104 / E1 / panel | 588 | 4 / 1 | -0.647 |
| 104 / E1 / CLIP_sources | 261 | 53 / 6 | -10.605 |
| 104 / E1 / random_sources | 472 | 4 / 1 | -1.516 |
| 104 / E2 / panel | 588 | 1 / 3 | -0.411 |
| 104 / E2 / CLIP_sources | 261 | 1 / 0 | -2.520 |
| 104 / E2 / random_sources | 472 | 7 / 14 | -1.564 |
| 105 / E1 / panel | 588 | 1 / 5 | -0.631 |
| 105 / E1 / CLIP_sources | 261 | 50 / 10 | -9.117 |
| 105 / E1 / random_sources | 472 | 4 / 0 | -1.109 |
| 105 / E2 / panel | 588 | 1 / 3 | -0.382 |
| 105 / E2 / CLIP_sources | 261 | 3 / 0 | -2.548 |
| 105 / E2 / random_sources | 472 | 7 / 17 | -1.582 |

Elapsed this invocation 35.3 minutes. Endpoint predictions were frozen before opening labels. Checkpoints and full per-step ledgers are local.

[All/Old/New transition and geometry metrics, exposure and checkpoint provenance](r2_sustained_move_training.json)
