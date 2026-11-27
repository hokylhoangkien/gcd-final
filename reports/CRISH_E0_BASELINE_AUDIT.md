# SimGCD E0 baseline audit on Stanford Cars

E0 epoch 200 measured transductive v2 **55.91 / 72.96 / 47.67% All / Old / New** on the 6,144 unlabeled training rows. The archived metric is 55.91 / 72.96 / 47.67%.

Archive SHA-256 `8aab726c36923f5d97c93d3839685350701356e79e748927cf007720c2f457f8`. The archive has no external SHA-256 sidecar. The recorded digest identifies the local tarball, and all members match the supplied extracted copies. All 11 regular members matched byte-for-byte by digest. The commit `aa7eec0c210f0ae0cc69d4f24d30eed3d3ff944d` resolves locally. Its archived patch changes dataset roots only. The checkpoint marks completed epoch 200 and contains model and SGD state.

The archived run used NVIDIA GeForce RTX 3090, driver 620.73.04, PyTorch 2.5.1+cu124 and torchvision 0.20.1+cu124. The audit ran on NVIDIA GeForce RTX 3050 Ti Laptop GPU with PyTorch 2.6.0+cu124. Recorded launch arguments match the seed-1 Cars SSB protocol, ViT-B/16, batch 128, 200 epochs and v2. The archived class order and the commit's split, preprocessing and evaluation code match the audit. The regenerated local split and ID ordering match the verified R2 split fingerprint and CLIP bank. All frozen backbone tensors, including final LayerNorm, agree across checkpoints. The E0 archive does not include a remote image-ID list or image-content hashes.

All 200 metric lines match the full stdout and trainer log, in strict epoch and timestamp order. E3 has a complete 200-epoch log. R2 has 150 logged epochs. E3-F retains only manually recovered aggregates, so the comparison uses its reported endpoint and gives no epochwise curve. The R2 loss in JSON is the recorded base objective, with its relational term reported separately in the source audit.

| Run | Completed epoch | Validated epoch records | Endpoint All / Old / New (%) | Last-20 mean All / Old / New (%) |
|---|---:|---:|---|---|
| E0 | 200 | 200 | 55.91 / 72.96 / 47.67 | 55.91 / 72.96 / 47.67 |
| E3 | 200 | 200 | 56.51 / 71.51 / 49.26 | 56.51 / 71.51 / 49.26 |
| E3-F | 200 | 0 | 56.49 / 73.36 / 48.35 | — |
| R2 | 150 | 150 | 55.21 / 70.66 / 47.74 | 55.21 / 70.46 / 47.84 |

| Epoch | E0 All / Old / New (%) | E3 All / Old / New (%) | R2 All / Old / New (%) |
|---|---|---|---|
| 1 | 12.76 / 17.09 / 10.67 | 12.76 / 17.09 / 10.67 | 12.61 / 15.39 / 11.27 |
| 20 | 48.71 / 67.62 / 39.58 | 48.67 / 69.62 / 38.55 | 49.38 / 67.62 / 40.57 |
| 50 | 55.34 / 72.51 / 47.04 | 55.86 / 73.56 / 47.31 | 54.52 / 74.31 / 44.97 |
| 100 | 56.27 / 74.21 / 47.60 | 56.71 / 73.16 / 48.76 | 55.29 / 70.31 / 48.03 |
| 150 | 56.04 / 73.26 / 47.72 | 56.53 / 71.61 / 49.24 | 55.21 / 70.66 / 47.74 |
| 200 | 55.91 / 72.96 / 47.67 | 56.51 / 71.51 / 49.26 | — |

## Frozen checkpoint geometry and prediction bias

The v2 rows use the whole transductive split. Geometry uses normalized final CLS, kNN-5 and leave-one-out class centroid contrast. Slot bias uses full-split alignment. Reported classifier misrouting is conditional on the sample's true Old/New group.

| Snapshot | v2 All / Old / New (%) | Matches log | kNN-5 All / Old / New | Margin All / Old / New | Old→New / New→Old | Active / effective slots |
|---|---|---|---|---|---|---|
| E0-200 | 55.91 / 72.96 / 47.67 | Yes | 0.5617 / 0.6357 / 0.5260 | 0.0268 / 0.1503 / -0.0328 | 17.19 / 13.98 | 194 / 171.8 |
| R2-100 | 55.29 / 70.31 / 48.03 | Yes | 0.5781 / 0.6594 / 0.5389 | 0.0276 / 0.1520 / -0.0326 | 21.74 / 11.66 | 191 / 167.2 |
| R2-150 | 55.21 / 70.66 / 47.74 | Yes | 0.5718 / 0.6481 / 0.5350 | 0.0231 / 0.1445 / -0.0355 | 21.79 / 12.12 | 191 / 166.6 |

## Frozen readouts

Readouts follow the earlier fit protocol and seeds 1, 7 and 19. The same 588-ID panel is held out from fitting. All other images form the fit set. Legal-labeled images anchor semisupervised K-means. Oracle centroid and linear ridge use privileged fit labels only inside sealed diagnostics. Fit alignment uses non-panel labels, while panel rematching is descriptive.

| Snapshot / readout | Fit aligned All / Old / New (%) | Panel rematched All / Old / New (%) |
|---|---|---|
| E0-200 / classifier | 57.99 / 72.45 / 43.54 | 61.73 / 73.81 / 49.66 |
| E0-200 / spherical | 51.25 / 63.04 / 39.46 | 57.54 / 67.12 / 47.96 |
| E0-200 / semi | 53.97 / 70.41 / 37.53 | 59.98 / 73.02 / 46.94 |
| E0-200 / centroid | 60.88 / 73.13 / 48.64 | 63.95 / 74.15 / 53.74 |
| E0-200 / linear | 64.63 / 76.53 / 52.72 | 65.65 / 76.53 / 54.76 |
| R2-100 / classifier | 59.18 / 75.51 / 42.86 | 63.95 / 77.21 / 50.68 |
| R2-100 / spherical | 51.47 / 65.53 / 37.41 | 58.67 / 69.73 / 47.62 |
| R2-100 / semi | 56.69 / 72.79 / 40.59 | 62.59 / 76.08 / 49.09 |
| R2-100 / centroid | 64.46 / 77.55 / 51.36 | 67.69 / 77.89 / 57.48 |
| R2-100 / linear | 64.80 / 80.61 / 48.98 | 66.84 / 80.61 / 53.06 |
| R2-150 / classifier | 59.86 / 76.19 / 43.54 | 64.46 / 77.55 / 51.36 |
| R2-150 / spherical | 51.53 / 66.21 / 36.85 | 58.84 / 70.29 / 47.39 |
| R2-150 / semi | 57.09 / 72.90 / 41.27 | 62.36 / 75.51 / 49.21 |
| R2-150 / centroid | 64.80 / 76.87 / 52.72 | 67.52 / 77.21 / 57.82 |
| R2-150 / linear | 64.97 / 79.59 / 50.34 | 66.33 / 79.93 / 52.72 |

## Locked held-out retrieval and correction

DINO and CLIP use the same 1,024 G6-confirm queries, 5,120 bank IDs and fixed R2-150 hard-neighbor set. The bank supplies Hungarian alignment. Risks are alternative-slot minus current-slot support. The confidence baseline flags lowest max teacher probability at temperature .04 and uses the same CLIP destinations as the CLIP arm. Coverage budgets stay at 10, 20 and 40 percent, with ID tie-breaking. Random controls draw from eligible queries with the same retriever and number of edits. Paired pointwise intervals condition on the fixed bank and flags.

| Snapshot / detector | Error AUROC All / Old / New | Destination recall@1/3/5 (All) | Recall@1/3/5 (New) |
|---|---|---|---|
| E0-200 / DINO | 0.628 / 0.739 / 0.569 | 0.178 / 0.379 / 0.462 | 0.163 / 0.367 / 0.448 |
| E0-200 / CLIP | 0.766 / 0.794 / 0.761 | 0.276 / 0.519 / 0.592 | 0.271 / 0.505 / 0.574 |
| E0-200 / Confidence | 0.555 / 0.635 / 0.506 | 0.276 / 0.519 / 0.592 | 0.271 / 0.505 / 0.574 |
| R2-100 / DINO | 0.602 / 0.725 / 0.520 | 0.198 / 0.375 / 0.456 | 0.193 / 0.365 / 0.442 |
| R2-100 / CLIP | 0.740 / 0.769 / 0.732 | 0.220 / 0.446 / 0.554 | 0.212 / 0.430 / 0.542 |
| R2-100 / Confidence | 0.596 / 0.744 / 0.497 | 0.220 / 0.446 / 0.554 | 0.212 / 0.430 / 0.542 |
| R2-150 / DINO | 0.586 / 0.681 / 0.519 | 0.198 / 0.363 / 0.441 | 0.182 / 0.343 / 0.423 |
| R2-150 / CLIP | 0.734 / 0.761 / 0.729 | 0.222 / 0.445 / 0.548 | 0.214 / 0.428 / 0.533 |
| R2-150 / Confidence | 0.591 / 0.676 / 0.524 | 0.222 / 0.445 / 0.548 | 0.214 / 0.428 / 0.533 |

| Snapshot / policy at 10% | Edits | Repairs / harmful / wrong-to-wrong | Δ All vs KEEP (pp) | Δ All vs confidence (pp, 95% CI) | Δ All vs random (pp, 95% CI) |
|---|---:|---|---:|---|---|
| E0-200 / KEEP | 0 | 0 / 0 / 0 | +0.00 | +2.93 [+1.37, +4.49] | — |
| E0-200 / DINO | 102 | 20 / 39 / 43 | -1.86 | +1.07 [-0.59, +2.73] | +2.33 [+0.92, +3.80] |
| E0-200 / CLIP | 102 | 34 / 7 / 61 | +2.64 | +5.57 [+3.81, +7.42] | +6.36 [+5.26, +7.47] |
| E0-200 / Confidence | 102 | 18 / 48 / 36 | -2.93 | +0.00 [+0.00, +0.00] | +0.79 [-0.69, +2.24] |
| R2-100 / KEEP | 0 | 0 / 0 / 0 | +0.00 | +2.05 [+0.68, +3.42] | — |
| R2-100 / DINO | 102 | 17 / 41 / 44 | -2.34 | -0.29 [-1.76, +1.37] | +1.52 [+0.18, +2.90] |
| R2-100 / CLIP | 102 | 26 / 8 / 68 | +1.76 | +3.81 [+2.15, +5.57] | +5.50 [+4.51, +6.55] |
| R2-100 / Confidence | 102 | 16 / 37 / 49 | -2.05 | +0.00 [+0.00, +0.00] | +1.69 [+0.32, +3.05] |
| R2-150 / KEEP | 0 | 0 / 0 / 0 | +0.00 | +2.44 [+0.98, +4.00] | — |
| R2-150 / DINO | 102 | 20 / 45 / 37 | -2.44 | +0.00 [-1.56, +1.66] | +1.37 [-0.03, +2.80] |
| R2-150 / CLIP | 102 | 27 / 7 / 68 | +1.95 | +4.39 [+2.64, +6.35] | +5.64 [+4.65, +6.71] |
| R2-150 / Confidence | 102 | 17 / 42 / 43 | -2.44 | +0.00 [+0.00, +0.00] | +1.24 [-0.26, +2.68] |

The frozen E0-200 and R2-100/150 comparisons are descriptive across different training epochs and trajectories. The held-out queries were reused from a previous Cars analysis. The offline assignment changes use sealed labels for scoring and do not measure training effects. Oracle readouts also require privileged labels.

Audit runtime 74.6 seconds.

[Complete trajectories, package versions, checksums, subgroup metrics and intervals](r2_e0_baseline_audit.json)
