# Early CLIP MOVE continuation on Stanford Cars

E0 keeps supervision. E1 moves 0.8 teacher mass using R2-100 CLIP proposals. E2 samples eligible source occurrences uniformly within each batch and uses the same frozen CLIP retriever and mass. The top 5% ranking is recalculated using R2-100 classifier slots and the shared CLIP image cache, with the previous .2 initial eligibility screen. Each applied edit requires the frozen source to match the live teacher and at least .8 source mass in both views. E1 is capped by the available E2 rows per batch.

All arms use identical conditioned weighted draws, crops and SimGCD losses, with R2 relational loss disabled as in the late strength diagnostic. SGD and cosine scheduling resume from R2-100. The fixed panel has 588 images from the original transductive training split. During continuation, it is excluded from draws, proposals and reference neighbors. Actual edited IDs are reported separately. Labels are first opened after the two checkpoints and endpoint predictions are complete.

| Epoch / arm | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure | Target mass |
|---|---|---|---:|---:|---|---:|
| 100 / shared | 58.16 / 70.75 / 45.58 | 63.95 / 77.21 / 50.68 | 0.2344 | -0.0134 | 0 / 0.000% | 0.0 |
| 101 / E0 | 58.84 / 71.43 / 46.26 | 64.63 / 77.55 / 51.70 | 0.2357 | -0.0135 | 0 / 0.000% | 0.0 |
| 101 / E1 | 59.18 / 71.43 / 46.94 | 64.80 / 77.55 / 52.04 | 0.2412 | -0.0000 | 156 / 1.935% | 124.8 |
| 101 / E2 | 58.33 / 70.75 / 45.92 | 63.95 / 76.87 / 51.02 | 0.2367 | -0.0155 | 156 / 1.935% | 124.8 |
| 102 / E0 | 58.67 / 71.77 / 45.58 | 64.80 / 78.23 / 51.36 | 0.2350 | -0.0137 | 0 / 0.000% | 0.0 |
| 102 / E1 | 58.84 / 71.43 / 46.26 | 64.46 / 77.89 / 51.02 | 0.2395 | -0.0046 | 108 / 1.339% | 86.4 |
| 102 / E2 | 58.33 / 71.09 / 45.58 | 64.12 / 76.53 / 51.70 | 0.2371 | -0.0128 | 108 / 1.339% | 86.4 |

| Epoch | CLIP live | Random live | Random supply shortfall | Matched |
|---|---:|---:|---:|---:|
| 101 | 156 | 3591 | 0 | 156 |
| 102 | 108 | 3562 | 0 | 108 |

| Epoch / arm | Correct / harmful / wrong-to-wrong / unresolved | Pre-step teacher source / destination | CLIP source cohort true-slot rank |
|---|---|---|---:|
| 101 / E1 | 54 / 10 / 81 / 11 | 0.9823 / 0.0023 | 21.9895 |
| 101 / E2 | 22 / 86 / 40 / 8 | 0.9913 / 0.0019 | 28.0105 |
| 102 / E1 | 36 / 6 / 60 / 6 | 0.9797 / 0.0046 | 19.9476 |
| 102 / E2 | 16 / 59 / 30 / 3 | 0.9906 / 0.0026 | 27.5969 |

## Descriptive comparison with R2-150

The late comparison uses S2/S3 at mass .8 and S0 KEEP. It conditions on the late experiment's shared .2/.8 live eligibility. The schedules, initial weights, proposals and realized doses differ between starting checkpoints. These contrasts describe the runs and do not identify an effect of starting earlier.

| Start / completed offset | Matched edits | Fixed All / New Δ vs KEEP (pp) | Fixed All / New Δ vs random (pp) | kNN-5 / margin Δ vs KEEP |
|---|---:|---|---|---|
| R2-100 / 1 | 156 | +0.34 / +0.68 | +0.85 / +1.02 | +0.0054 / +0.0135 |
| R2-100 / 2 | 108 | +0.17 / +0.68 | +0.51 / +0.68 | +0.0044 / +0.0092 |
| R2-150 / 1 | 166 | +0.00 / +0.34 | -0.17 / -0.34 | +0.0054 / +0.0132 |
| R2-150 / 2 | 141 | +0.00 / +0.34 | +0.00 / +0.34 | +0.0078 / +0.0140 |

Elapsed in this invocation 21.8 minutes. Paired checkpoints and per-step ledgers are local.

[Subgroup probabilities, ranks, prediction transitions, and checkpoint digests](r2_early_move_training.json)
