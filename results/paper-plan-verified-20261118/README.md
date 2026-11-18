# Observed fixed-final registry — 2026-11-18

Single seed per dataset; no main-table mean±std or algorithmic gain claim.
Counts below cover only recorded refreshes. CUB/Aircraft have missing early refresh logs.

| Dataset | Seed | All / Old / New (%) | Move / Abstain | Refresh coverage | Probe sign agreement (same / confirmed) |
|---|---:|---|---|---|---|
| scars | 1 | 78.2064 / 87.3563 / 73.7871 | 65 / 52 | complete_for_current_schedule | 117 / 132 |
| cub | 1 | 83.6744 / 84.8566 / 83.0831 | 1 / 224 | partial | 225 / 232 |
| aircraft | 1 | 78.2244 / 81.2125 / 76.7316 | 62 / 59 | partial | 121 / 145 |

Probe sign agreement is conditioned on candidates reaching confirmation, including failed confirmations; it is not a semantic correctness measure.
Move correctness and Abstain enrichment remain NOT VERIFIED without offline labels/partitions.

| Dataset | Recorded refresh seconds | Mean epoch with refresh (n) | Mean epoch without refresh (n) | Missing scheduled log epochs (zero-based) |
|---|---:|---|---|---|
| scars | 4326.426 | 253.418 (38) | 139.991 (162) | [] |
| cub | 2133.968 | 182.549 (33) | 117.406 (162) | [10, 15, 20, 25, 30] |
| aircraft | 2178.933 | 212.987 (34) | 151.561 (162) | [10, 15, 20, 25] |

Measured epoch times include extraction/refresh/SGD and exclude evaluation. Missing scheduled refresh epochs are excluded from the no-refresh group.
These are descriptive, resumed-history measurements, not causal overhead or one uninterrupted wall-clock run.

Reproduce with `scripts/paper_registry.py` and the three exact source paths in source_manifest.json; use a new output directory.
Curves are available in curves-cars/, curves-cub/, curves-aircraft/ as PDF/SVG/PNG with source data and captions.
