# CRiSH-D1 final audit on Stanford Cars

D1 completed 200 epochs at `94683693ee74ecd0cebcdd32b79d4be903782910`. Its archived final v2 All / Old / New was **60.35 / 72.16 / 54.65%**. The same-epoch D1 minus E0 difference was 4.4416 / -0.7961 / 6.9764 percentage points. Runs are single-seed trajectories and these comparisons do not isolate the effect of CLIP evidence.

The All peak was 60.97% at epoch 116. The final New advantage over E0 coincides with higher New kNN-5 purity (0.627 vs 0.526) and a positive New centroid margin (+0.028 vs -0.033).

The locked D1-200 CLIP offline policy made 19 repairs and 35 harmful edits at 10% coverage (-1.56 pp vs KEEP). The difficulty-adjusted edited-versus-never association was -0.79 pp at epoch 200 in the common-support cohort. These findings leave the contributions of CLIP destination choice and exposure unresolved.

## Provenance and training continuity

Archive SHA-256 receipts were checked before extraction. All analysis checkpoints came from the externally verified epoch-200 TAR. Earlier milestone checkpoint digests and byte-exact journal prefixes were corroborated against that archive. The checkpoint identity, recorded byte boundaries, RNG payloads, populated finite SGD momentum buffers, per-epoch LRs and milestone scheduler position agree. This checks retained state and epoch-level continuity. Individual historical optimizer updates, draws and augmentation RNG cannot be replay-verified from these artifacts.

D1 and E0 report NVIDIA GeForce RTX 3090 and D1 records torch 2.5.1+cu124. Frozen replay used NVIDIA GeForce RTX 3050 Ti Laptop GPU and torch 2.6.0+cu124. The D1/E0/R2 split reconstruction uses the locked Cars SSB seed-1 IDs, verified DINO prefix, archive parameters and compatible v2 protocol. E0 has no external TAR sidecar. Its locally computed SHA-256 and all archived members matched the prior audit. E0 does not contain every historical optimizer/scheduler state, and none of these runs have archived remote image-byte hashes for the full split.

| Completed epochs | Computed TAR SHA-256 | Sidecar | Mismatched members | Momentum entries | Saved next-epoch LR |
|---:|---|---|---|---:|---:|
| 100 | `63c0c0ff7e8614eb78d43c0aec7c5cba9b374d444b69bb107dc7729436ef3e14` | MISMATCH | none | 19 | 0.05005000 |
| 150 | `07e9bc7186cbf997dcc6b6c7965c3f1aa4e04c6d45757591dca6363616cb435e` | match | none | 19 | 0.01473002 |
| 200 | `fca9f6238e5f254213fe7349abee2edcf46081c57cbf17eda145b3839108b55b` | match | none | 19 | 0.00010000 |

Epoch 100 declared SHA-256 `c9521ee2a9ff9439d0d1c3d935ddc354e3a6e33f84d59bf6194dd67c4e2f547e`. The archive and sidecar are intact on disk. The cause of the sidecar discrepancy is unknown.

## Accuracy trajectories

All numbers are percentages, ordered All / Old / New. Last-20 windows end at the marked epoch. In the first ten warmup epochs the largest absolute D1–E0 difference is 0.0049 pp, within E0 stdout rounding. Raw curves and same-epoch differences are in the JSON.

| Epoch | Run | Endpoint | Mean of last 20 |
|---:|---|---|---|
| 100 | D1 | 60.66 / 74.76 / 53.85 | 60.55 / 74.30 / 53.91 |
| 100 | E0 | 56.27 / 74.21 / 47.60 | 56.11 / 73.55 / 47.69 |
| 100 | E3 | 56.71 / 73.16 / 48.76 | 56.58 / 73.58 / 48.37 |
| 100 | R2 | 55.29 / 70.31 / 48.03 | 55.36 / 70.82 / 47.89 |
| 150 | D1 | 60.48 / 72.51 / 54.67 | 60.47 / 73.02 / 54.40 |
| 150 | E0 | 56.04 / 73.26 / 47.72 | 55.94 / 73.14 / 47.63 |
| 150 | E3 | 56.53 / 71.61 / 49.24 | 56.54 / 71.67 / 49.23 |
| 150 | R2 | 55.21 / 70.66 / 47.74 | 55.21 / 70.46 / 47.84 |
| 200 | D1 | 60.35 / 72.16 / 54.65 | 60.36 / 72.22 / 54.63 |
| 200 | E0 | 55.91 / 72.96 / 47.67 | 55.91 / 72.96 / 47.67 |
| 200 | E3 | 56.51 / 71.51 / 49.26 | 56.51 / 71.51 / 49.26 |
| 200 | R2 | — | — |

## Realized exposure

Refresh outcomes {'warmup': 10, 'ok': 190}. There were 57,310 observed applications across 3,649 unique IDs (53,661 additional occurrences), 11,462.0 total moved mass and 1,612,800 mixed rows. The 322-occurrence cap was reached in 131 epochs. Ledger totals and exclusions reconcile to training encounters.

Eligibility and budget counters {'encounters': 76779, 'low_teacher_mass': 1434, 'stale_source': 10415, 'batch_cap_skipped': 2772, 'epoch_budget_skipped': 4848}. Missing alternative-slot detections summed to 0. The JSON contains refresh overlap by ID, native source and destination agreement, slot counts, feature variance, epoch timings and first-edited-batch gradient receipts.

| Window | Applications | Exposure (%) | Median refresh (s) | Median first-edit classifier / block-11 ratio |
|---|---:|---:|---:|---|
| 11–100 | 25,110 | 3.460 | 23.29 | 0.411 / 0.277 |
| 101–150 | 16,100 | 3.993 | 23.24 | 0.572 / 0.433 |
| 151–200 | 16,100 | 3.993 | 23.11 | 0.601 / 0.517 |

## Frozen checkpoint geometry and classifier bias

Geometry uses unit-normalized final CLS, kNN-5 and leave-one-out true-centroid minus hardest-other cosine margin. Bias is class-group misrouting under a full split alignment. Archived E0/R2 rows reuse verified frozen results. The D1 and local DINO source-file digests differ. The replay directly verified identical frozen tensors and the same locked weights, and reports any metric differences below.

| Snapshot | Replayed v2 | kNN-5 All / Old / New | Margin All / Old / New | Active / effective slots | Old→New / New→Old (%) |
|---|---|---|---|---|---|
| D1-100 | 60.66 / 74.76 / 53.85 | 0.6553 / 0.7168 / 0.6256 | 0.0780 / 0.1856 / 0.0260 | 192 / 163.6 | 17.84 / 8.59 |
| D1-150 | 60.48 / 72.51 / 54.67 | 0.6567 / 0.7112 / 0.6304 | 0.0796 / 0.1820 / 0.0302 | 191 / 161.0 | 20.34 / 7.07 |
| D1-200 | 60.35 / 72.16 / 54.65 | 0.6532 / 0.7070 / 0.6272 | 0.0755 / 0.1733 / 0.0282 | 191 / 160.8 | 20.79 / 6.98 |
| E0-200 | 55.91 / 72.96 / 47.67 | 0.5617 / 0.6357 / 0.5260 | 0.0268 / 0.1503 / -0.0328 | 194 / 171.8 | 17.19 / 13.98 |
| R2-100 | 55.29 / 70.31 / 48.03 | 0.5781 / 0.6594 / 0.5389 | 0.0276 / 0.1520 / -0.0326 | 191 / 167.2 | 21.74 / 11.66 |
| R2-150 | 55.21 / 70.66 / 47.74 | 0.5718 / 0.6481 / 0.5350 | 0.0231 / 0.1445 / -0.0355 | 191 / 166.6 | 21.79 / 12.12 |

Frozen v2 discrepancies from the archived D1 metrics (pp): D1-100 [0.0, 0.0, 0.0]; D1-150 [0.0, 0.0, 0.0]; D1-200 [0.0, 0.0, 0.0].

## Frozen readouts and swaps

The 588-ID reference panel and fit split are unchanged. Legal semi-supervised K-means uses only legally labeled anchors. Oracle nearest centroid and linear readouts are privileged diagnostics. Fit-aligned and panel-rematched subgroup scores, ARI and NMI are in JSON. The panel is transductive for D1 and does not constitute independent test data.

| Snapshot | Classifier | Spherical | Legal semi-supervised | Oracle centroid | Oracle linear |
|---|---:|---:|---:|---:|---:|
| D1-100 | 62.76 | 55.27 | 60.83 | 69.56 | 70.92 |
| D1-150 | 61.39 | 55.05 | 60.32 | 68.03 | 70.41 |
| D1-200 | 61.22 | 55.10 | 59.92 | 66.50 | 70.07 |
| E0-200 | 57.99 | 51.25 | 53.97 | 60.88 | 64.63 |
| R2-100 | 59.18 | 51.47 | 56.69 | 64.46 | 64.80 |
| R2-150 | 59.86 | 51.53 | 57.09 | 64.80 | 64.97 |

Swap rows keep the feature snapshot and change only the normalized classifier head. Accuracy is fit aligned. JSON also reports owner-head fixed alignment and slot agreement on the readout fit set. Cross-run swaps are mechanics diagnostics with jointly trained embeddings and classifier weights, so compatibility of dimensions and preprocessing does not guarantee semantic interoperability.

| Feature snapshot | Own head All / Old / New | D1-100 head | D1-150 head | D1-200 head | E0-200 head | R2-100 head | R2-150 head |
|---|---|---|---|---|---|---|---|
| D1-100 | 62.76 / 76.19 / 49.32 | 62.76 / 76.19 / 49.32 | 62.59 / 75.17 / 50.00 | 62.59 / 75.17 / 50.00 | 63.44 / 77.89 / 48.98 | 60.88 / 75.51 / 46.26 | 59.69 / 75.51 / 43.88 |
| D1-150 | 61.39 / 73.81 / 48.98 | 61.39 / 73.81 / 48.98 | 61.39 / 73.81 / 48.98 | 61.39 / 73.81 / 48.98 | 62.41 / 76.19 / 48.64 | 60.03 / 75.51 / 44.56 | 59.52 / 75.51 / 43.54 |
| D1-200 | 61.22 / 73.81 / 48.64 | 61.22 / 74.15 / 48.30 | 61.05 / 73.81 / 48.30 | 61.22 / 73.81 / 48.64 | 62.07 / 75.17 / 48.98 | 61.39 / 73.47 / 49.32 | 59.35 / 74.49 / 44.22 |

## Locked CLIP correction

G6-confirm's fixed, disjoint 1,024 queries and 5,120 retrieval-bank IDs, the common R2-150 hard pairs, bank-only Hungarian mapping, thresholds and matched random draws were reused. These are offline assignment edits. The checkpoints have previously trained transductively on these images.

| Snapshot / detector | Error AUROC All / Old / New | Destination recall@1/3/5 All | Novel recall@1/3/5 |
|---|---|---|---|
| D1-100 / DINO | 0.666 / 0.747 / 0.618 | 0.263 / 0.478 / 0.565 | 0.247 / 0.478 / 0.569 |
| D1-100 / CLIP | 0.698 / 0.706 / 0.701 | 0.261 / 0.502 / 0.618 | 0.250 / 0.505 / 0.624 |
| D1-100 / Confidence | 0.687 / 0.811 / 0.617 | 0.261 / 0.502 / 0.618 | 0.250 / 0.505 / 0.624 |
| D1-150 / DINO | 0.634 / 0.711 / 0.606 | 0.226 / 0.446 / 0.556 | 0.205 / 0.427 / 0.553 |
| D1-150 / CLIP | 0.690 / 0.694 / 0.695 | 0.222 / 0.475 / 0.587 | 0.216 / 0.469 / 0.576 |
| D1-150 / Confidence | 0.660 / 0.769 / 0.601 | 0.222 / 0.475 / 0.587 | 0.216 / 0.469 / 0.576 |
| D1-200 / DINO | 0.630 / 0.714 / 0.596 | 0.228 / 0.420 / 0.535 | 0.204 / 0.395 / 0.528 |
| D1-200 / CLIP | 0.693 / 0.691 / 0.700 | 0.223 / 0.471 / 0.580 | 0.215 / 0.461 / 0.566 |
| D1-200 / Confidence | 0.647 / 0.715 / 0.611 | 0.223 / 0.471 / 0.580 | 0.215 / 0.461 / 0.566 |
| E0-200 / DINO | 0.628 / 0.739 / 0.569 | 0.178 / 0.379 / 0.462 | 0.163 / 0.367 / 0.448 |
| E0-200 / CLIP | 0.766 / 0.794 / 0.761 | 0.276 / 0.519 / 0.592 | 0.271 / 0.505 / 0.574 |
| E0-200 / Confidence | 0.555 / 0.635 / 0.506 | 0.276 / 0.519 / 0.592 | 0.271 / 0.505 / 0.574 |
| R2-100 / DINO | 0.602 / 0.725 / 0.520 | 0.198 / 0.375 / 0.456 | 0.193 / 0.365 / 0.442 |
| R2-100 / CLIP | 0.740 / 0.769 / 0.732 | 0.220 / 0.446 / 0.554 | 0.212 / 0.430 / 0.542 |
| R2-100 / Confidence | 0.596 / 0.744 / 0.497 | 0.220 / 0.446 / 0.554 | 0.212 / 0.430 / 0.542 |
| R2-150 / DINO | 0.586 / 0.681 / 0.519 | 0.198 / 0.363 / 0.441 | 0.182 / 0.343 / 0.423 |
| R2-150 / CLIP | 0.734 / 0.761 / 0.729 | 0.222 / 0.445 / 0.548 | 0.214 / 0.428 / 0.533 |
| R2-150 / Confidence | 0.591 / 0.676 / 0.524 | 0.222 / 0.445 / 0.548 | 0.214 / 0.428 / 0.533 |

| Snapshot / 10% policy | Repairs / Harmful / Wrong-to-wrong | Δ vs KEEP All / Old / New (pp) | Δ vs confidence (All pp) | Δ vs matched random (All pp) |
|---|---|---|---|---|
| D1-100 / DINO | 18 / 39 / 45 | -2.05 / -3.07 / -1.64 | -1.07 | +2.37 |
| D1-100 / CLIP | 20 / 29 / 53 | -0.88 / -2.39 / -0.27 | +0.10 | +3.60 |
| D1-100 / Confidence | 21 / 31 / 50 | -0.98 / -2.05 / -0.55 | +0.00 | +3.50 |
| D1-150 / DINO | 20 / 39 / 43 | -1.86 / -1.71 / -1.92 | -0.29 | +2.88 |
| D1-150 / CLIP | 15 / 38 / 49 | -2.25 / -5.46 / -0.96 | -0.68 | +2.52 |
| D1-150 / Confidence | 18 / 34 / 50 | -1.56 / -2.39 / -1.23 | +0.00 | +3.21 |
| D1-200 / DINO | 22 / 41 / 39 | -1.86 / -3.41 / -1.23 | +0.39 | +2.83 |
| D1-200 / CLIP | 19 / 35 / 48 | -1.56 / -5.46 / +0.00 | +0.68 | +3.14 |
| D1-200 / Confidence | 18 / 41 / 43 | -2.25 / -5.46 / -0.96 | +0.00 | +2.45 |
| E0-200 / DINO | 20 / 39 / 43 | -1.86 / -3.75 / -1.09 | +1.07 | +2.33 |
| E0-200 / CLIP | 34 / 7 / 61 | +2.64 / +1.71 / +3.01 | +5.57 | +6.36 |
| E0-200 / Confidence | 18 / 48 / 36 | -2.93 / -4.78 / -2.19 | +0.00 | +0.79 |
| R2-100 / DINO | 17 / 41 / 44 | -2.34 / -3.41 / -1.92 | -0.29 | +1.52 |
| R2-100 / CLIP | 26 / 8 / 68 | +1.76 / +1.02 / +2.05 | +3.81 | +5.50 |
| R2-100 / Confidence | 16 / 37 / 49 | -2.05 / -2.05 / -2.05 | +0.00 | +1.69 |
| R2-150 / DINO | 20 / 45 / 37 | -2.44 / -4.44 / -1.64 | +0.00 | +1.37 |
| R2-150 / CLIP | 27 / 7 / 68 | +1.95 / +1.02 / +2.33 | +4.39 | +5.64 |
| R2-150 / Confidence | 17 / 42 / 43 | -2.44 / -4.10 / -1.78 | +0.00 | +1.24 |

## Historical applied edits and cohort changes

The reported edit counts use actual ledger occurrences. Repair/harm describes where the mass went under a retrospective fixed snapshot alignment. A full contemporaneous assignment map and live teacher probabilities were not retained, so the exact per-epoch semantic correctness and soft-target effectiveness cannot be verified. JSON includes sensitivity to all three mappings and Old/New breakdowns. The window-end retained snapshot mapping is shown per window.

| Applied window | Alignment snapshot | Repair / harmful / wrong-to-wrong / unmatched | Repair / harm (% of all applied) |
|---|---|---|---|
| 11-100 | D1-100 | 4286 / 6619 / 11967 / 2238 | 17.1 / 26.4 |
| 101-150 | D1-150 | 2301 / 5339 / 6958 / 1502 | 14.3 / 33.2 |
| 151-200 | D1-200 | 2407 / 5573 / 6879 / 1241 | 15.0 / 34.6 |

Among consecutive observed occurrences of the same ID, native slot repeat counts were {'n': 53661, 'same_source': 51684, 'same_destination': 37298, 'later_epoch': 40900}. These are observed edit pairs. Missing non-edited trajectories prevent identifying a continuously stable teacher assignment or mapping intervening proposal drift.

Cohorts use D1-100 mapping for both baseline and endpoint. The exposed cohort has its first edit after epoch 100 and by the endpoint. Comparators were never edited in the retained 200 epochs. The adjusted difference post-stratifies by true class, epoch-100 correctness and epoch-100 max-teacher confidence decile, weighted to edited IDs with shared support. Selection bias, future-defined controls, unrecorded trajectories and transductive exposure preclude a causal propagation interpretation.

| Endpoint / group | Edited / never | Edited / never change (pp) | Matched edited / never | Adjusted difference (pp) |
|---|---:|---:|---:|---:|
| D1-150 / all | 365 / 2495 | -3.01 / +0.08 | 197 / 450 | -2.28 |
| D1-150 / old | 149 / 791 | -4.70 / +0.00 | 73 / 183 | -4.11 |
| D1-150 / new | 216 / 1704 | -1.85 / +0.12 | 124 / 267 | -1.21 |
| D1-200 / all | 564 / 2495 | -1.95 / +0.16 | 315 / 674 | -0.79 |
| D1-200 / old | 242 / 791 | -2.48 / +0.13 | 120 / 265 | +0.00 |
| D1-200 / new | 322 / 1704 | -1.55 / +0.18 | 195 / 409 | -1.28 |

## Evidence limits

- E3 has 200 logged epochs and an exposure ledger. Its checkpoints are unavailable, so no E3 frozen geometry, readout, swap or locked correction was measured.
- E0 retains only epoch 200, R2 retains epochs 100 and 150, and E3-F retains manually recovered aggregate metrics. Missing matrix cells remain unmeasured.
- The ablations need shared realized exposure before attributing differences to destination choice or evidence source. See [ablation design](CRISH_D1_ABLATION_DESIGN.md).

Report replay runtime 24.0 s. [Complete machine-readable audit](CRISH_D1_FINAL_AUDIT.json).
