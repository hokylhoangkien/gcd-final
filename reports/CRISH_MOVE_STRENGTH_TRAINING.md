# Matched two-epoch CLIP MOVE strength continuation

All arms start at R2-150 with the saved SGD momentum, cosine scheduler and RNG. The two batch schedules and paired crops match the previous CLIP diagnostic. The shared objective keeps all SimGCD terms and leaves R2 relational loss off. The locked CLIP proposal order, source slot and destination retriever are unchanged.

S1 and S2 use the intersection of same-occurrence live eligibility at masses .2 and .8. S3 samples eligible occurrences uniformly in the same batch at mass .8 and uses the same CLIP retriever. If S3 cannot match the intersection count, S1 and S2 drop the same trailing ranked rows. S0 keeps supervision. Exposure divides applied occurrences by all mixed training rows. Target mass sums the moved amount once per occurrence after the two-view averaging convention.

The original R2-150 checkpoint previously saw the fixed 588-ID panel in transductive training. Through continuation, panel IDs have zero sampler weight and are excluded from proposal and reference banks. Semantic scoring opened only after both paired epoch checkpoints and the endpoint predictions had been saved. The fixed bank-only alignment and descriptive panel rematch follow the prior continuation. CLIP and random source cohorts use distinct IDs that received at least one edit in the corresponding arm and can overlap each other.

| Epoch / arm | Fixed All / Old / New (%) | Rematched All / Old / New (%) | kNN-5 | Centroid margin | Edits / exposure | Target mass |
|---|---|---|---:|---:|---|---:|
| 150 / shared | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2344 | -0.0145 | 0 / 0.000% | 0.0 |
| 151 / S0 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2323 | -0.0183 | 0 / 0.000% | 0.0 |
| 151 / S1 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2327 | -0.0146 | 166 / 2.059% | 33.2 |
| 151 / S2 | 58.50 / 69.39 / 47.62 | 64.46 / 77.21 / 51.70 | 0.2378 | -0.0051 | 166 / 2.059% | 132.8 |
| 151 / S3 | 58.67 / 69.39 / 47.96 | 64.63 / 77.21 / 52.04 | 0.2364 | -0.0135 | 166 / 2.059% | 132.8 |
| 152 / S0 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2299 | -0.0195 | 0 / 0.000% | 0.0 |
| 152 / S1 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2320 | -0.0160 | 141 / 1.749% | 28.2 |
| 152 / S2 | 58.50 / 69.39 / 47.62 | 64.46 / 77.21 / 51.70 | 0.2378 | -0.0055 | 141 / 1.749% | 112.8 |
| 152 / S3 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2330 | -0.0219 | 141 / 1.749% | 112.8 |

## Eligibility and matched supply

| Epoch | S1 live | S2 live | Common | S3 live | S3 supply shortfall | Matched |
|---|---:|---:|---:|---:|---:|---:|
| 151 | 182 | 166 | 166 | 3723 | 0 | 166 |
| 152 | 177 | 141 | 141 | 3718 | 0 | 141 |

## Applied edits and endpoint response

| Epoch / arm | Correct / harmful / wrong-to-wrong / unresolved | Mean pre-step teacher source / destination | CLIP source cohort true-slot rank |
|---|---|---|---:|
| 151 / S1 | 47 / 9 / 94 / 16 | 0.9946 / 0.0009 | 33.8370 |
| 151 / S2 | 47 / 9 / 94 / 16 | 0.9898 / 0.0018 | 29.5598 |
| 151 / S3 | 21 / 94 / 44 / 7 | 0.9930 / 0.0010 | 33.9728 |
| 152 / S1 | 45 / 5 / 74 / 17 | 0.9962 / 0.0006 | 33.4565 |
| 152 / S2 | 45 / 5 / 74 / 17 | 0.9808 / 0.0054 | 26.7337 |
| 152 / S3 | 20 / 75 / 40 / 6 | 0.9933 / 0.0019 | 35.0815 |

Probability summaries use pre-update two-view posteriors for applied rows and center-view posteriors for the fixed cohorts at endpoints. Prediction transitions use the fixed slot alignment. This is one paired continuation from one Cars checkpoint.

Elapsed in this invocation 25.4 minutes. Full per-step ledgers remain in local paired checkpoints.

[All/Old/New transitions, probabilities, matched exposure and checkpoint digests](r2_move_strength_training.json)
