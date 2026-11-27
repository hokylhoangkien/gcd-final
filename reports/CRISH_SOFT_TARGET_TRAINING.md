# Matched CLIP hard versus soft MOVE on Stanford Cars

A and B resume the same E1 checkpoint at epoch 102. C resumes E2. The detector, source slots, top-1 destinations and ranking come from the previously saved dynamic trajectory for each epoch and remain fixed for this experiment. Top-3 retrieval uses the same maximum peer cosine per alternative slot, excludes the source and self, and verifies that the top-1 support and risk match the saved proposal.

A moves .8 to the top slot. B and C divide .8 among the top three using softmax over CLIP support at temperature .07. Each arm uses its own live teacher gate. A and B share exactly the same applied source occurrences. Their shared list is trimmed to C's supply when needed. C chooses uniformly among live eligible source occurrences in the same batch. All arms use full SimGCD with R2 relational loss off, identical draws and crops. KEEP reuses the prior checkpoint at each epoch.

The panel's 588 images were seen before the continuation and excluded from its draws and retriever bank. Source transitions use separate ever-edited cohorts and the same fixed slot alignment. Endpoint source/destination probability summaries refer to the last applied top-1 pair for each ID. The sealed true-class mass analysis uses the complete target mixture from each occurrence and the pre-update teacher probabilities. Unmapped true classes are counted separately. All predictions precede semantic scoring.

| Epoch / arm | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure |
|---|---|---|---:|---:|---|
| 100 / shared | 58.16 / 70.75 / 45.58 | 63.95 / 77.21 / 50.68 | 0.2344 | -0.0134 | — |
| 102 / KEEP | 58.67 / 71.77 / 45.58 | 64.80 / 78.23 / 51.36 | 0.2350 | -0.0137 | — |
| 102 / A | 58.84 / 71.43 / 46.26 | 64.46 / 77.89 / 51.02 | 0.2395 | -0.0046 | — |
| 102 / B | 58.84 / 71.43 / 46.26 | 64.46 / 77.89 / 51.02 | 0.2395 | -0.0046 | — |
| 102 / C | 58.33 / 71.09 / 45.58 | 64.12 / 76.53 / 51.70 | 0.2371 | -0.0128 | — |
| 103 / KEEP | 58.67 / 71.77 / 45.58 | 64.46 / 78.23 / 50.68 | 0.2398 | -0.0116 | 0 / 0.000% |
| 103 / A | 58.84 / 71.77 / 45.92 | 64.63 / 78.23 / 51.02 | 0.2391 | -0.0091 | 106 / 1.314% |
| 103 / B | 58.84 / 71.77 / 45.92 | 64.63 / 78.23 / 51.02 | 0.2395 | -0.0078 | 106 / 1.314% |
| 103 / C | 58.67 / 71.43 / 45.92 | 64.46 / 77.89 / 51.02 | 0.2412 | -0.0104 | 106 / 1.314% |
| 104 / KEEP | 58.67 / 71.77 / 45.58 | 64.46 / 78.23 / 50.68 | 0.2371 | -0.0139 | 0 / 0.000% |
| 104 / A | 58.50 / 71.43 / 45.58 | 64.29 / 77.89 / 50.68 | 0.2446 | -0.0013 | 108 / 1.339% |
| 104 / B | 59.01 / 71.77 / 46.26 | 64.63 / 77.89 / 51.36 | 0.2435 | -0.0044 | 108 / 1.339% |
| 104 / C | 58.33 / 70.41 / 46.26 | 63.95 / 76.19 / 51.70 | 0.2395 | -0.0155 | 108 / 1.339% |
| 105 / KEEP | 59.18 / 71.77 / 46.60 | 64.80 / 78.23 / 51.36 | 0.2401 | -0.0087 | 0 / 0.000% |
| 105 / A | 58.84 / 71.09 / 46.60 | 64.12 / 77.55 / 50.68 | 0.2425 | -0.0006 | 101 / 1.252% |
| 105 / B | 59.18 / 71.77 / 46.60 | 64.63 / 77.89 / 51.36 | 0.2422 | -0.0021 | 101 / 1.252% |
| 105 / C | 59.18 / 71.09 / 47.28 | 64.63 / 77.55 / 51.70 | 0.2418 | -0.0086 | 101 / 1.252% |

## Matched exposure and sealed target mass

| Epoch | A / B / C live | Common A/B / random shortfall | Applied per arm / moved mass | A / B / C mean true-mass Δ | Paired B−A mean Δ |
|---|---|---|---|---|---:|
| 103 | 109 / 112 / 3476 | 106 / 0 | 106 / 84.8 | +0.2275 / +0.1133 / -0.4317 | -0.1141 |
| 104 | 114 / 116 / 3336 | 108 / 0 | 108 / 86.4 | +0.1649 / +0.0961 / -0.4165 | -0.0689 |
| 105 | 109 / 116 / 3366 | 101 / 0 | 101 / 80.8 | +0.0495 / +0.0063 / -0.4221 | -0.0432 |

## Transfer and source transitions

| Epoch / arm | CLIP-source fixes / regressions | Random-source fixes / regressions | Panel fixes / regressions |
|---|---|---|---|
| 103 / A | 15 / 4 | 0 / 0 | 2 / 1 |
| 103 / B | 8 / 3 | 0 / 0 | 2 / 1 |
| 103 / C | 3 / 0 | 3 / 2 | 2 / 2 |
| 104 / A | 23 / 8 | 3 / 1 | 1 / 2 |
| 104 / B | 14 / 7 | 2 / 0 | 4 / 2 |
| 104 / C | 4 / 1 | 3 / 4 | 3 / 5 |
| 105 / A | 32 / 14 | 3 / 0 | 3 / 5 |
| 105 / B | 19 / 7 | 3 / 0 | 3 / 3 |
| 105 / C | 2 / 0 | 2 / 5 | 2 / 2 |

Elapsed 38.4 minutes in this invocation. Paired checkpoints and full ledgers remain local.

[All/Old/New outcomes, eligibility, target mass and checkpoint digests](r2_soft_target_training.json)
