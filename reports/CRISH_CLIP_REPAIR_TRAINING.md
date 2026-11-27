# Five-epoch CLIP direct supervision repair on Stanford Cars

All arms resume R2-150 with the saved SGD momentum, cosine scheduler and RNG. Training keeps the full SimGCD objective, and the R2 relational term is off in all arms to isolate direct supervision repair. The 200-epoch scheduler continues through epoch 155. T0 keeps supervision, T1 applies locked CLIP MOVE and T2 samples eligible source occurrences uniformly from each same batch with the same frozen CLIP destination rule.

The top 5% of the initial teacher-eligible pool are 277 of 5544 IDs, rounded down. CLIP risk is best alternative minus current-slot maximum cosine, with self excluded. The initial source needs agreement with the average cached teacher distribution and at least .2 mass in both views. These checks repeat on each live augmented occurrence. T1 is capped by T2's live supply per batch; any shortfall is logged. The 5% ID budget does not set the training exposure, which counts applied MOVE occurrences divided by all sampled rows, including repeated IDs.

The original checkpoint has already seen every image on the fixed 588-ID panel during transductive training (at least 66 replay-verified draws per ID through R2-150). During continuation its IDs have zero sampler weight and are excluded from proposal selection and the CLIP reference bank. The remaining relative sampler weights, draw length and drop-last behavior are preserved, changing the labeled fraction slightly. All arms share each pair of augmented crops. Sampler and augmentation streams derive from the checkpoint CPU RNG; single-process crop generation differs from the historical loader.

Ground truth is used solely in panel evaluation and fitting the fixed initial slot alignment on non-panel IDs. Panel rematching is descriptive. kNN purity uses five neighbors; centroid margin compares the leave-one-out own centroid with the nearest other class. Results are paired continuations of one trained checkpoint.

| Epoch / arm | Fixed All / Old / New % | Rematched All / Old / New % | kNN-5 | Centroid margin | Moves / exposure | Next LR |
|---|---|---|---:|---:|---|---:|
| 150 / shared | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2344 | -0.0145 | 0 / 0.000% | 0.014730 |
| 151 / T0 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2323 | -0.0183 | 0 / 0.000% | 0.014180 |
| 151 / T1 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2337 | -0.0141 | 182 / 2.257% | 0.014180 |
| 151 / T2 | 58.67 / 69.73 / 47.62 | 64.63 / 77.55 / 51.70 | 0.2347 | -0.0168 | 182 / 2.257% | 0.014180 |
| 152 / T0 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2299 | -0.0195 | 0 / 0.000% | 0.013638 |
| 152 / T1 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2323 | -0.0147 | 175 / 2.170% | 0.013638 |
| 152 / T2 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2306 | -0.0202 | 175 / 2.170% | 0.013638 |
| 153 / T0 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2330 | -0.0157 | 0 / 0.000% | 0.013105 |
| 153 / T1 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2344 | -0.0111 | 211 / 2.617% | 0.013105 |
| 153 / T2 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2337 | -0.0153 | 211 / 2.617% | 0.013105 |
| 154 / T0 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2333 | -0.0186 | 0 / 0.000% | 0.012582 |
| 154 / T1 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2347 | -0.0118 | 187 / 2.319% | 0.012582 |
| 154 / T2 | 58.67 / 69.73 / 47.62 | 64.63 / 77.55 / 51.70 | 0.2333 | -0.0173 | 187 / 2.319% | 0.012582 |
| 155 / T0 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2310 | -0.0182 | 0 / 0.000% | 0.012068 |
| 155 / T1 | 58.50 / 69.73 / 47.28 | 64.46 / 77.55 / 51.36 | 0.2354 | -0.0124 | 197 / 2.443% | 0.012068 |
| 155 / T2 | 58.67 / 69.73 / 47.62 | 64.63 / 77.55 / 51.70 | 0.2327 | -0.0178 | 197 / 2.443% | 0.012068 |

Elapsed 49.7 minutes. Paired epoch checkpoints and per-step ledgers are local.

[Subgroup geometry, checkpoint metrics and matched exposure](r2_clip_repair_training.json)
