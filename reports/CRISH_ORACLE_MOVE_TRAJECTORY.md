# Oracle MOVE one-epoch trajectory diagnostic

Seven arms from R2 epoch 100 with saved SGD and cosine state, 63 steps and 8,064 scoring rows per arm. We replay the archived sampler RNG and apply a common, explicit per-batch augmentation seed with a single loader process. Frozen-prefix tokens are reused across arms. Projector, classifier and block 11 all update through original SimGCD, including the contrastive losses, with R2 disabled. Only the diagnostic writes privileged MOVE targets. All moves have mass .2 and preserve the full loss denominator.

Exposure is applied edits divided by all mixed training row occurrences, including labeled and repeated draws. Each oracle arm carries forward a quota deficit and stops at its prespecified cumulative target. An ID needs a wrong frozen source, an aligned true slot, agreement with its live teacher source and at least .2 teacher mass in both crops. Random controls replay the .01 and .10 oracle ledgers by exact step and row, drawing another supported predicted slot uniformly. They skip stale sources or low mass; there is no replacement or forced edit. The table gives actual count and shortfall.

The primary panel has 588 unlabeled train IDs that never occur in the shared training draw, sampled up to three per class. Its Hungarian rematch uses the same panel and is descriptive. Fixed alignment is the sealed initial full-split mapping. Probe geometry reports kNN-5 and leave-one-out own-centroid versus nearest-other margin. The corrected/untouched subsets sample up to 64 training IDs in each observed exposure group at each milestone and report fixed accuracy and step-0 accuracy on the same IDs. This comparison is conditional on selection and supports no causal claim by itself.

Disjoint official test images present: 0/8041. When the split is complete, a fixed class-stratified test panel is reported separately in JSON.

| Arm | Exposure target | Applied / target | Nominal shortfall | Matched shortfall | Endpoint fixed All / Old / New (%) | Δ fixed All vs KEEP (pp) |
|---|---:|---|---:|---:|---|---:|
| O0 | 0% | 0 / 0 | 0 | — | 59.01 / 72.11 / 45.92 | +0.000 |
| O1 | 1% | 81 / 81 | 0 | 0 | 59.01 / 72.11 / 45.92 | +0.000 |
| O2 | 0.1% | 8 / 8 | 0 | — | 59.01 / 72.11 / 45.92 | +0.000 |
| O3 | 1% | 81 / 81 | 0 | — | 58.67 / 71.77 / 45.58 | -0.340 |
| O4 | 5% | 403 / 403 | 0 | — | 59.01 / 72.11 / 45.92 | +0.000 |
| O5 | 10% | 806 / 806 | 0 | — | 58.67 / 71.09 / 46.26 | -0.340 |
| O6 | 10% | 802 / 806 | 4 | 4 | 58.67 / 71.77 / 45.58 | -0.340 |

| Arm / step | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 All | Centroid margin All |
|---|---|---|---:|---:|
| O0 / 0 | 58.84 / 71.77 / 45.92 | 63.95 / 77.21 / 50.68 | 0.2344 | -0.0134 |
| O0 / 20 | 59.01 / 72.45 / 45.58 | 64.46 / 77.89 / 51.02 | 0.2371 | -0.0125 |
| O0 / 50 | 59.35 / 72.45 / 46.26 | 64.46 / 77.89 / 51.02 | 0.2361 | -0.0146 |
| O0 / 63 | 59.01 / 72.11 / 45.92 | 64.46 / 77.55 / 51.36 | 0.2374 | -0.0161 |
| O1 / 20 | 58.84 / 72.11 / 45.58 | 64.29 / 77.55 / 51.02 | 0.2381 | -0.0130 |
| O1 / 50 | 59.35 / 72.45 / 46.26 | 64.46 / 77.89 / 51.02 | 0.2371 | -0.0139 |
| O1 / 63 | 59.01 / 72.11 / 45.92 | 64.46 / 77.55 / 51.36 | 0.2361 | -0.0167 |
| O2 / 20 | 59.01 / 72.45 / 45.58 | 64.46 / 77.89 / 51.02 | 0.2374 | -0.0124 |
| O2 / 50 | 59.35 / 72.45 / 46.26 | 64.46 / 77.89 / 51.02 | 0.2367 | -0.0144 |
| O2 / 63 | 59.01 / 72.11 / 45.92 | 64.46 / 77.55 / 51.36 | 0.2374 | -0.0158 |
| O3 / 20 | 58.84 / 72.11 / 45.58 | 64.29 / 77.55 / 51.02 | 0.2381 | -0.0128 |
| O3 / 50 | 59.52 / 72.79 / 46.26 | 64.63 / 78.23 / 51.02 | 0.2371 | -0.0141 |
| O3 / 63 | 58.67 / 71.77 / 45.58 | 64.29 / 77.55 / 51.02 | 0.2357 | -0.0153 |
| O4 / 20 | 58.84 / 71.77 / 45.92 | 64.29 / 77.21 / 51.36 | 0.2391 | -0.0098 |
| O4 / 50 | 59.69 / 72.79 / 46.60 | 64.63 / 78.23 / 51.02 | 0.2398 | -0.0086 |
| O4 / 63 | 59.01 / 72.11 / 45.92 | 64.46 / 77.55 / 51.36 | 0.2412 | -0.0096 |
| O5 / 20 | 59.52 / 72.79 / 46.26 | 64.63 / 78.23 / 51.02 | 0.2412 | -0.0050 |
| O5 / 50 | 59.52 / 72.45 / 46.60 | 64.46 / 77.89 / 51.02 | 0.2459 | -0.0006 |
| O5 / 63 | 58.67 / 71.09 / 46.26 | 64.12 / 76.87 / 51.36 | 0.2429 | -0.0019 |
| O6 / 20 | 59.18 / 72.45 / 45.92 | 64.63 / 77.89 / 51.36 | 0.2401 | -0.0100 |
| O6 / 50 | 59.35 / 72.45 / 46.26 | 64.29 / 77.89 / 50.68 | 0.2364 | -0.0121 |
| O6 / 63 | 58.67 / 71.77 / 45.58 | 64.12 / 77.21 / 51.02 | 0.2374 | -0.0126 |

Edit dose is logged at the backbone CLS features and classifier weights as the L2 norm of the differential gradient, alongside per-step eligibility, stale sources and matched-dose shortfall. These are single shared-seed trajectories. A gap on one epoch or one split does not establish a general dose-response or long-run effect.

Elapsed 33.0 min. Smoke peak allocated VRAM 683 MiB. Completed arms and milestone recovery checkpoints are local artifacts.

[Per-step ledgers, geometry and paired differences](r2_oracle_move_trajectory.json)
- O1 eligibility divergence — {}.
- O6 eligibility divergence — {'matched_stale_source': 2, 'matched_low_mass': 2}.
