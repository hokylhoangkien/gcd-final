# Frozen readouts after five Oracle MOVE diagnostic epochs

All 7,556 non-panel Cars IDs form the fit bank (2,000 legal labels). All arms use identical center crops and the original 588 held-out IDs. Frozen normalized CLS features feed every readout. Legal refers to readout fitting only: O5 features were trained with oracle edits.

Spherical Lloyd clustering uses 196 slots, seeds 1/7/19, at most 50 iterations, unchanged empty centers, and unweighted seed means. Semi-supervised clustering initializes 98 centers with labeled class means and clamps labeled assignments during fitting. Other initial centers are distinct random unlabeled rows. Oracle centroid uses normalized class sums. Oracle linear uses multiclass least squares with a unit ridge penalty and an unpenalized intercept.

Both accuracy columns use one global Hungarian assignment across Old and New. Fit alignment uses privileged fit labels only during scoring. Panel rematching is descriptive. ARI and NMI use raw slots. Differences pair the same panel and seeds; initialization repeats are not independent training runs.

## Legal readouts

| Arm / readout | Fit All/Old/New (%) | Rematched All/Old/New (%) | ARI / NMI (All) | Δ fit All vs O0 (pp) |
|---|---|---|---|---:|
| O0 / classifier | 59.69 / 77.21 / 42.18 | 64.29 / 77.55 / 51.02 | 0.4104 / 0.8857 | +0.00 |
| O5 / classifier | 60.37 / 76.19 / 44.56 | 63.95 / 76.53 / 51.36 | 0.4053 / 0.8841 | +0.68 |
| O6 / classifier | 59.18 / 76.53 / 41.84 | 63.95 / 76.53 / 51.36 | 0.4051 / 0.8846 | -0.51 |
| O0 / spherical | 52.32 / 66.67 / 37.98 | 58.62 / 69.39 / 47.85 | 0.3608 / 0.8736 | +0.00 |
| O5 / spherical | 53.12 / 66.33 / 39.91 | 60.32 / 71.20 / 49.43 | 0.3775 / 0.8786 | +0.79 |
| O6 / spherical | 51.64 / 67.46 / 35.83 | 58.33 / 70.63 / 46.03 | 0.3550 / 0.8733 | -0.68 |
| O0 / semi | 57.09 / 73.92 / 40.25 | 62.19 / 76.98 / 47.39 | 0.3900 / 0.8822 | +0.00 |
| O5 / semi | 57.20 / 72.34 / 42.06 | 62.64 / 76.30 / 48.98 | 0.3950 / 0.8836 | +0.11 |
| O6 / semi | 56.92 / 73.47 / 40.36 | 61.39 / 75.62 / 47.17 | 0.3773 / 0.8794 | -0.17 |

## Oracle readouts

| Arm / readout | Fit All/Old/New (%) | Rematched All/Old/New (%) | ARI / NMI (All) | Δ fit All vs O0 (pp) |
|---|---|---|---|---:|
| O0 / centroid | 65.31 / 78.23 / 52.38 | 68.20 / 78.57 / 57.82 | 0.4293 / 0.8906 | +0.00 |
| O5 / centroid | 65.31 / 77.21 / 53.40 | 67.52 / 77.89 / 57.14 | 0.4225 / 0.8894 | +0.00 |
| O6 / centroid | 63.95 / 77.21 / 50.68 | 67.18 / 78.23 / 56.12 | 0.4181 / 0.8885 | -1.36 |
| O0 / linear | 65.31 / 80.27 / 50.34 | 66.84 / 80.27 / 53.40 | 0.4190 / 0.8880 | +0.00 |
| O5 / linear | 66.67 / 80.27 / 53.06 | 67.35 / 80.27 / 54.42 | 0.4160 / 0.8894 | +1.36 |
| O6 / linear | 64.80 / 79.59 / 50.00 | 66.50 / 79.59 / 53.40 | 0.4082 / 0.8861 | -0.51 |

[Seed-level metrics, subgroup ARI/NMI and paired arm differences](r2_frozen_readout.json)
