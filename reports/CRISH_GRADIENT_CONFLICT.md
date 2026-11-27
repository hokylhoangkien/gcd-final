# Supervision gradient conflict on Stanford Cars

Four label-blind draws from batches with actual T1 MOVE ledger entries are replayed at both R2-150 and T1-155. Each replay uses full batch size 128 and the original paired crops from the saved continuation seeds. Up to two distinct source IDs per batch are ranked by the minimum source mass across both views and both endpoints. The probe is disjoint from all replay batches and reuses 64 frozen-prefix center-crop tokens from the excluded panel.

Losses are the coefficient-weighted SimGCD distillation, MeMax, known CE, InfoNCE and SupCon. R2 loss is disabled. Positive projection means the direction of descent raises the proposed raw destination-minus-source margin to first order, or the logged SGD displacement raises it when applying the same projection. Cosines compare raw gradients to the raw edit delta. MOVE only changes distillation. Its differential gradient is the raw margin gradient times −0.65 × mass / (128 × student temperature 0.1). Gradients are reused across masses exactly. The projector does not affect the classifier margin directly.

Functional SGD reads saved learning rate, momentum and weight decay. Every branch starts at its checkpoint with the same optimizer state. Nonlinear post-step margin changes are measured after recomputing block-11 features and logits. Random selects a fixed represented destination other than source or proposed target for the same ID, dose and replay batch. Oracle uses a different target only when the initial bank alignment supplies the true slot and the source is wrong. The oracle branch runs after non-oracle outputs are frozen.

Tables condition on common four-mass support. JSON separately reports available support and invalid rows at each strength. Proposal correctness is joined with held-out labels after freezing non-oracle scores. The replays are endpoint-conditioned and do not recover historical training gradients. Independent probe transfer includes candidate-slot margin and true-slot log probability gains. These one-step effects do not identify training effects.

| Snapshot / arm / mass | Occurrences | Train margin gain vs KEEP | Probe margin gain vs KEEP | Nearest-probe gain |
|---|---:|---:|---:|---:|
| R150 / MOVE / 0.1 | 8 | +2.490e-03 | +1.518e-04 | +2.942e-04 |
| R150 / MOVE / 0.2 | 8 | +4.989e-03 | +3.037e-04 | +5.886e-04 |
| R150 / MOVE / 0.4 | 8 | +1.001e-02 | +6.077e-04 | +1.178e-03 |
| R150 / MOVE / 0.8 | 8 | +2.012e-02 | +1.216e-03 | +2.360e-03 |
| R150 / Random / 0.1 | 8 | +1.100e-03 | +9.258e-05 | +1.763e-04 |
| R150 / Random / 0.2 | 8 | +2.203e-03 | +1.852e-04 | +3.527e-04 |
| R150 / Random / 0.4 | 8 | +4.417e-03 | +3.703e-04 | +7.054e-04 |
| R150 / Random / 0.8 | 8 | +8.887e-03 | +7.404e-04 | +1.411e-03 |
| R150 / Oracle / 0.1 | 5 | +3.343e-03 | +1.847e-04 | +3.979e-04 |
| R150 / Oracle / 0.2 | 5 | +6.699e-03 | +3.696e-04 | +7.962e-04 |
| R150 / Oracle / 0.4 | 5 | +1.345e-02 | +7.396e-04 | +1.594e-03 |
| R150 / Oracle / 0.8 | 5 | +2.708e-02 | +1.481e-03 | +3.192e-03 |
| T1-155 / MOVE / 0.1 | 8 | +2.228e-03 | +1.393e-04 | +2.705e-04 |
| T1-155 / MOVE / 0.2 | 8 | +4.460e-03 | +2.786e-04 | +5.413e-04 |
| T1-155 / MOVE / 0.4 | 8 | +8.937e-03 | +5.573e-04 | +1.084e-03 |
| T1-155 / MOVE / 0.8 | 8 | +1.794e-02 | +1.115e-03 | +2.171e-03 |
| T1-155 / Random / 0.1 | 8 | +1.235e-03 | +8.689e-05 | +1.611e-04 |
| T1-155 / Random / 0.2 | 8 | +2.477e-03 | +1.738e-04 | +3.222e-04 |
| T1-155 / Random / 0.4 | 8 | +4.978e-03 | +3.475e-04 | +6.444e-04 |
| T1-155 / Random / 0.8 | 8 | +1.003e-02 | +6.948e-04 | +1.289e-03 |
| T1-155 / Oracle / 0.1 | 5 | +2.508e-03 | +1.474e-04 | +3.221e-04 |
| T1-155 / Oracle / 0.2 | 5 | +5.022e-03 | +2.949e-04 | +6.446e-04 |
| T1-155 / Oracle / 0.4 | 5 | +1.006e-02 | +5.900e-04 | +1.290e-03 |
| T1-155 / Oracle / 0.8 | 5 | +2.021e-02 | +1.180e-03 | +2.586e-03 |

## Gradient contribution at MOVE mass 0.2

| Snapshot / group / weighted loss | N | Mean factual norm | Mean cosine to edit | Mean factual descent projection |
|---|---:|---:|---:|---:|
| R150 / classifier / distill | 8 | +1.069e-01 | -5.175e-02 | -5.581e-03 |
| R150 / classifier / memax | 8 | +1.411e-01 | +3.462e-02 | +4.805e-03 |
| R150 / classifier / known_ce | 8 | +1.129e-02 | -5.350e-03 | -6.173e-05 |
| R150 / classifier / infonce | 8 | +0.000e+00 | — | +0.000e+00 |
| R150 / classifier / supcon | 8 | +0.000e+00 | — | +0.000e+00 |
| R150 / classifier / total | 8 | +5.851e-02 | -1.299e-02 | -8.377e-04 |
| R150 / projector / distill | 8 | +0.000e+00 | — | — |
| R150 / projector / memax | 8 | +0.000e+00 | — | — |
| R150 / projector / known_ce | 8 | +0.000e+00 | — | — |
| R150 / projector / infonce | 8 | +4.427e-02 | — | — |
| R150 / projector / supcon | 8 | +2.296e-03 | — | — |
| R150 / projector / total | 8 | +4.443e-02 | — | — |
| R150 / block11 / distill | 8 | +3.681e-01 | -6.023e-02 | -2.081e-02 |
| R150 / block11 / memax | 8 | +4.933e-01 | +4.337e-02 | +2.117e-02 |
| R150 / block11 / known_ce | 8 | +5.236e-02 | -3.151e-02 | -1.652e-03 |
| R150 / block11 / infonce | 8 | +4.204e-02 | -2.483e-02 | -9.350e-04 |
| R150 / block11 / supcon | 8 | +2.440e-03 | +5.687e-03 | -2.611e-05 |
| R150 / block11 / total | 8 | +2.073e-01 | -1.100e-02 | -2.254e-03 |
| T1-155 / classifier / distill | 8 | +1.146e-01 | -6.909e-02 | -7.928e-03 |
| T1-155 / classifier / memax | 8 | +1.442e-01 | +4.075e-02 | +5.841e-03 |
| T1-155 / classifier / known_ce | 8 | +1.255e-02 | -9.849e-03 | -1.229e-04 |
| T1-155 / classifier / infonce | 8 | +0.000e+00 | — | +0.000e+00 |
| T1-155 / classifier / supcon | 8 | +0.000e+00 | — | +0.000e+00 |
| T1-155 / classifier / total | 8 | +5.737e-02 | -3.800e-02 | -2.210e-03 |
| T1-155 / projector / distill | 8 | +0.000e+00 | — | — |
| T1-155 / projector / memax | 8 | +0.000e+00 | — | — |
| T1-155 / projector / known_ce | 8 | +0.000e+00 | — | — |
| T1-155 / projector / infonce | 8 | +4.230e-02 | — | — |
| T1-155 / projector / supcon | 8 | +1.484e-03 | — | — |
| T1-155 / projector / total | 8 | +4.239e-02 | — | — |
| T1-155 / block11 / distill | 8 | +4.837e-01 | -7.848e-02 | -3.625e-02 |
| T1-155 / block11 / memax | 8 | +5.614e-01 | +6.403e-02 | +3.629e-02 |
| T1-155 / block11 / known_ce | 8 | +6.637e-02 | -4.150e-02 | -2.714e-03 |
| T1-155 / block11 / infonce | 8 | +3.461e-02 | -4.723e-02 | -1.536e-03 |
| T1-155 / block11 / supcon | 8 | +1.403e-03 | -6.027e-03 | -2.621e-05 |
| T1-155 / block11 / total | 8 | +2.068e-01 | -2.011e-02 | -4.237e-03 |

Runtime 208.3 seconds.

[Per-loss KEEP/MOVE vectors summarized by group, validity, raw/SGD differences and sealed transfer](r2_gradient_conflict.json)
