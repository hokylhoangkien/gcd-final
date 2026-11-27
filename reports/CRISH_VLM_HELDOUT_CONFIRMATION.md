# Locked held-out G6 confirmation on Stanford Cars

The primary endpoint is CLIP detection plus CLIP retrieval at 10% coverage on R2-150. Epoch 100 and 20%/40% coverage are secondary. The query sample excludes all G1 queries, the Oracle MOVE panel and previous frozen E3 proposal IDs. This is a fresh query partition within the same Cars trajectory. Independent dataset or training-seed replication would require separate artifacts.

The 1,024 queries are disjoint from the retrieval bank, which contains all other unlabeled Cars IDs. Alternative destinations use maximum peer cosine per slot. Each risk score compares best alternative support against current-slot support. D1 means fine-tuned DINOv1. D2 detection routes to equal D1+D2 retrieval. All3 averages fine-tuned D1, frozen D2 and CLIP cosine scores. Exact flags, destinations and random draws were stored before opening labels. Bank-only Hungarian alignment maps slots to classes during evaluation.

Random controls match each policy's number of actionable edits, use its same retriever and average 64 uniform label-blind draws. Net gain is repairs minus harmful edits as a percentage of all queries in the group. The 95% paired query-bootstrap intervals use 2,000 Old/New-stratified resamples with fixed bank, alignment, scores and flags. Intervals are pointwise and conditional on these snapshots and frozen random controls. Shared bank dependencies and training-run variability are outside these intervals.

| Epoch / group / policy (10%) | Edits | Repairs / harmful / wrong-to-wrong | Gain vs KEEP pp [95% CI] | Gain vs random pp [95% CI] | Stable error AUC / recall |
|---|---:|---|---|---|---|
| 100 / All / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — | — / — |
| 100 / All / D1 | 102 | 17 / 41 / 44 | -2.34 [-3.71, -0.88] | +1.52 [+0.18, +2.90] | — / — |
| 100 / All / D2 | 102 | 18 / 21 / 63 | -0.29 [-1.46, +0.88] | +3.34 [+2.21, +4.52] | — / — |
| 100 / All / CLIP | 102 | 26 / 8 / 68 | +1.76 [+0.68, +2.83] | +5.50 [+4.51, +6.55] | — / — |
| 100 / All / All3 | 102 | 22 / 20 / 60 | +0.20 [-1.07, +1.46] | +3.79 [+2.63, +5.03] | — / — |
| 100 / Old / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — | — / — |
| 100 / Old / D1 | 24 | 1 / 11 / 12 | -3.41 [-5.80, -1.37] | +2.60 [+0.25, +4.67] | — / — |
| 100 / Old / D2 | 38 | 7 / 13 / 18 | -2.05 [-5.12, +1.02] | +3.72 [+0.82, +6.58] | — / — |
| 100 / Old / CLIP | 35 | 9 / 6 / 20 | +1.02 [-1.37, +3.42] | +6.86 [+4.51, +9.24] | — / — |
| 100 / Old / All3 | 30 | 6 / 8 / 16 | -0.68 [-3.07, +1.71] | +5.06 [+2.71, +7.35] | — / — |
| 100 / New / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — | — / — |
| 100 / New / D1 | 78 | 16 / 30 / 32 | -1.92 [-3.56, +0.00] | +1.09 [-0.54, +2.82] | — / — |
| 100 / New / D2 | 64 | 11 / 8 / 45 | +0.41 [-0.68, +1.64] | +3.18 [+2.05, +4.41] | — / — |
| 100 / New / CLIP | 67 | 17 / 2 / 48 | +2.05 [+0.96, +3.28] | +4.95 [+3.93, +6.13] | — / — |
| 100 / New / All3 | 72 | 16 / 12 / 44 | +0.55 [-0.82, +2.05] | +3.29 [+1.93, +4.65] | — / — |
| 150 / All / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — | — / — |
| 150 / All / D1 | 102 | 20 / 45 / 37 | -2.44 [-3.91, -0.98] | +1.37 [-0.03, +2.80] | 0.580 / 0.101 |
| 150 / All / D2 | 102 | 21 / 22 / 59 | -0.10 [-1.27, +1.17] | +3.54 [+2.35, +4.77] | 0.678 / 0.153 |
| 150 / All / CLIP | 102 | 27 / 7 / 68 | +1.95 [+0.88, +3.12] | +5.64 [+4.65, +6.71] | 0.732 / 0.182 |
| 150 / All / All3 | 102 | 23 / 21 / 58 | +0.20 [-1.07, +1.46] | +3.75 [+2.48, +4.97] | 0.698 / 0.153 |
| 150 / Old / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — | — / — |
| 150 / Old / D1 | 33 | 6 / 19 / 8 | -4.44 [-7.51, -1.37] | +1.42 [-1.79, +4.50] | 0.675 / 0.131 |
| 150 / Old / D2 | 37 | 9 / 14 / 14 | -1.71 [-4.78, +1.71] | +3.99 [+1.02, +7.01] | 0.709 / 0.232 |
| 150 / Old / CLIP | 35 | 9 / 6 / 20 | +1.02 [-1.71, +3.75] | +6.88 [+4.38, +9.31] | 0.762 / 0.293 |
| 150 / Old / All3 | 31 | 10 / 8 / 13 | +0.68 [-2.05, +3.41] | +6.41 [+3.92, +9.02] | 0.757 / 0.222 |
| 150 / New / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — | — / — |
| 150 / New / D1 | 69 | 14 / 26 / 29 | -1.64 [-3.28, +0.00] | +1.36 [-0.22, +2.92] | 0.516 / 0.094 |
| 150 / New / D2 | 65 | 12 / 8 / 45 | +0.55 [-0.68, +1.78] | +3.35 [+2.14, +4.58] | 0.686 / 0.134 |
| 150 / New / CLIP | 67 | 18 / 1 / 48 | +2.33 [+1.23, +3.56] | +5.14 [+4.10, +6.32] | 0.725 / 0.156 |
| 150 / New / All3 | 71 | 13 / 13 / 45 | +0.00 [-1.37, +1.37] | +2.68 [+1.33, +4.06] | 0.663 / 0.137 |

Primary CLIP difference against D1 was +4.39 [+2.64, +6.25] pp.

## Secondary global net gains

| Epoch / coverage | D1 vs KEEP | D2 vs KEEP | CLIP vs KEEP | All3 vs KEEP |
|---|---|---|---|---|
| 100 / 20% | -4.30 [-6.25, -2.25] | -1.76 [-3.52, +0.00] | +0.78 [-0.98, +2.54] | -0.78 [-2.73, +1.17] |
| 100 / 40% | -10.55 [-13.38, -7.62] | -6.54 [-9.28, -3.91] | -4.20 [-6.84, -1.46] | -4.20 [-6.84, -1.37] |
| 150 / 20% | -4.88 [-6.84, -2.83] | -1.37 [-3.22, +0.39] | +0.49 [-1.27, +2.34] | +0.00 [-1.95, +1.95] |
| 150 / 40% | -10.16 [-13.09, -7.32] | -6.35 [-8.98, -3.71] | -4.98 [-7.52, -2.25] | -3.91 [-6.64, -1.07] |

Stable errors follow G4: both checkpoint assignments are wrong and the assigned slot is unchanged under label-blind bank matching. Epoch 100 has no historical stable-error outcome.

Runtime 3.7 seconds.

[All subgroup and coverage metrics, matched random means and frozen provenance](r2_vlm_heldout_confirmation.json)
