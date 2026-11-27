# Dynamic CLIP proposal refresh through epoch 105

Both dynamic arms resume the original epoch-102 checkpoints. At the start of epochs 103, 104 and 105, current E1 features and classifier assign slots on all non-panel unlabeled reference IDs. CLIP cosine uses the existing image embeddings, excludes self and ranks best alternative support minus current-slot support. The previous .2 initial two-view eligibility screen and top 5% rule are applied anew. Center views and two deterministic refresh crops are fixed across refreshes and cached through the last frozen DINO block.

E1 and E2 use the same refreshed source/destination map. E2 samples random eligible sources and has its own live teacher check. Both require the refreshed source to match their own live prediction and have .8 mass in both shared training crops. Per-batch edits are matched without padding. The fixed E0 and fixed-proposal E1/E2 come from the published run. The excluded panel had prior transductive exposure before continuation.

Panel geometry and native accuracy use the original fixed alignment plus the descriptive panel Hungarian rematch. Same-ID transitions use the fixed alignment. Source cohorts are retrospective sets of ever-edited IDs, including shared epochs 101 and 102. For an ID whose proposal changes, endpoint source/destination probability summaries use its last applied pair. The semantic ledger scores the actual pair on each occurrence.

| Epoch / policy | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure |
|---|---|---|---:|---:|---|
| 103 / fixed E1 | 58.84 / 71.77 / 45.92 | 64.63 / 78.23 / 51.02 | 0.2398 | -0.0040 | 82 / 1.017% |
| 103 / fixed E2 | 57.99 / 70.75 / 45.24 | 64.12 / 77.21 / 51.02 | 0.2415 | -0.0114 | 82 / 1.017% |
| 103 / dynamic E1 | 58.67 / 71.43 / 45.92 | 64.46 / 77.89 / 51.02 | 0.2388 | -0.0095 | 108 / 1.339% |
| 103 / dynamic E2 | 58.67 / 71.77 / 45.58 | 64.46 / 78.23 / 50.68 | 0.2412 | -0.0124 | 108 / 1.339% |
| 103 / reused KEEP | 58.67 / 71.77 / 45.58 | 64.46 / 78.23 / 50.68 | 0.2398 | -0.0116 | 0 / 0.000% |
| 104 / fixed E1 | 59.18 / 72.11 / 46.26 | 64.97 / 78.57 / 51.36 | 0.2398 | -0.0065 | 86 / 1.066% |
| 104 / fixed E2 | 58.33 / 71.09 / 45.58 | 63.95 / 77.55 / 50.34 | 0.2398 | -0.0136 | 86 / 1.066% |
| 104 / dynamic E1 | 58.33 / 71.09 / 45.58 | 64.12 / 77.55 / 50.68 | 0.2439 | -0.0021 | 111 / 1.376% |
| 104 / dynamic E2 | 58.84 / 71.43 / 46.26 | 64.63 / 77.89 / 51.36 | 0.2354 | -0.0172 | 111 / 1.376% |
| 104 / reused KEEP | 58.67 / 71.77 / 45.58 | 64.46 / 78.23 / 50.68 | 0.2371 | -0.0139 | 0 / 0.000% |
| 105 / fixed E1 | 58.50 / 71.09 / 45.92 | 64.63 / 77.21 / 52.04 | 0.2418 | -0.0043 | 50 / 0.620% |
| 105 / fixed E2 | 58.84 / 70.75 / 46.94 | 64.46 / 77.21 / 51.70 | 0.2412 | -0.0093 | 50 / 0.620% |
| 105 / dynamic E1 | 59.35 / 71.77 / 46.94 | 64.63 / 78.23 / 51.02 | 0.2422 | +0.0012 | 105 / 1.302% |
| 105 / dynamic E2 | 58.33 / 70.41 / 46.26 | 63.95 / 76.87 / 51.02 | 0.2378 | -0.0116 | 105 / 1.302% |
| 105 / reused KEEP | 59.18 / 71.77 / 46.60 | 64.80 / 78.23 / 51.36 | 0.2401 | -0.0087 | 0 / 0.000% |

## Turnover and realized dose

| Epoch | Top 5% IDs | Retained / added / dropped | Destination changes among retained | Dynamic / fixed edits per edited arm | Δ exposure (pp) |
|---|---:|---|---:|---|---:|
| 103 | 272 | 174 / 98 / 101 | 4 | 108 / 82 | +0.322 |
| 104 | 271 | 208 / 63 / 64 | 6 | 111 / 86 | +0.310 |
| 105 | 269 | 187 / 82 / 84 | 3 | 105 / 50 | +0.682 |

Dynamic versus fixed changes both the selected rows and the realized intervention dose. Differences in panel performance describe the two trajectories. When exposure differs, these comparisons do not isolate the effect of refreshing proposals.

| Epoch / dynamic arm | Changed source / low mass | New edited IDs / within-epoch repeat edits | Correct / harmful / wrong-to-wrong / unresolved | Panel fixes / regressions vs KEEP |
|---|---|---|---|---|
| 103 / E1 | 22 / 40 | 62 / 13 | 37 / 9 / 51 / 11 | 2 / 2 |
| 103 / E2 | 93 / 167 | 100 / 1 | 9 / 63 / 35 / 1 | 2 / 2 |
| 104 / E1 | 29 / 33 | 59 / 12 | 30 / 10 / 62 / 9 | 2 / 4 |
| 104 / E2 | 121 / 193 | 107 / 1 | 10 / 65 / 34 / 2 | 3 / 2 |
| 105 / E1 | 28 / 32 | 47 / 8 | 22 / 18 / 52 / 13 | 4 / 3 |
| 105 / E2 | 146 / 166 | 92 / 1 | 14 / 56 / 31 / 4 | 1 / 6 |

Runtime this invocation 39.4 minutes.

[All/Old/New outcomes, panel contrasts, ledgers and provenance](r2_dynamic_move_training.json)
