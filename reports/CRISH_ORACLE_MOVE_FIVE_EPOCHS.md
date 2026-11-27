# Five-epoch Oracle MOVE trajectory

Epoch 1 is the published endpoint. Epochs 2–5 set the original sampler weights of all 588 held-out panel IDs to zero. The remaining relative weights, replacement, draw length and drop-last behavior are preserved. This changes the expected labeled fraction. All arms share the exact draw and two crops. Exposure divides applied edits by all 8,064 mixed rows per epoch. O6 replays eligible O5 occurrences with random supported destinations and skips eligibility divergence. The 10% target is nominal.

Fixed accuracy uses the original global mapping. Rematching uses the panel labels. Geometry uses the same held-out panel. All values are single-trajectory diagnostics.

| Epoch | Arm | Fixed All/Old/New (%) | Rematched All/Old/New (%) | kNN-5 | Margin | Applied | Exposure |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | O0 | 59.01 / 72.11 / 45.92 | 64.46 / 77.55 / 51.36 | 0.2374 | -0.0161 | 0 | 0.00% |
| 1 | O5 | 58.67 / 71.09 / 46.26 | 64.12 / 76.87 / 51.36 | 0.2429 | -0.0019 | 806 | 10.00% |
| 1 | O6 | 58.67 / 71.77 / 45.58 | 64.12 / 77.21 / 51.02 | 0.2374 | -0.0126 | 802 | 9.95% |
| 2 | O0 | 59.18 / 72.11 / 46.26 | 64.29 / 77.55 / 51.02 | 0.2381 | -0.0124 | 0 | 0.00% |
| 2 | O5 | 59.86 / 73.13 / 46.60 | 64.63 / 78.57 / 50.68 | 0.2459 | +0.0036 | 806 | 10.00% |
| 2 | O6 | 59.01 / 72.11 / 45.92 | 64.12 / 77.55 / 50.68 | 0.2381 | -0.0112 | 802 | 9.95% |
| 3 | O0 | 59.01 / 72.11 / 45.92 | 64.29 / 77.21 / 51.36 | 0.2361 | -0.0110 | 0 | 0.00% |
| 3 | O5 | 59.86 / 72.45 / 47.28 | 64.29 / 77.89 / 50.68 | 0.2469 | +0.0101 | 806 | 10.00% |
| 3 | O6 | 59.18 / 72.45 / 45.92 | 64.46 / 77.21 / 51.70 | 0.2388 | -0.0086 | 800 | 9.92% |
| 4 | O0 | 59.01 / 72.11 / 45.92 | 64.12 / 77.55 / 50.68 | 0.2361 | -0.0142 | 0 | 0.00% |
| 4 | O5 | 60.71 / 72.79 / 48.64 | 64.63 / 78.57 / 50.68 | 0.2463 | +0.0076 | 806 | 10.00% |
| 4 | O6 | 58.84 / 71.77 / 45.92 | 63.95 / 76.53 / 51.36 | 0.2350 | -0.0129 | 800 | 9.92% |
| 5 | O0 | 59.18 / 72.11 / 46.26 | 64.29 / 77.55 / 51.02 | 0.2361 | -0.0135 | 0 | 0.00% |
| 5 | O5 | 60.03 / 71.09 / 48.98 | 63.95 / 76.53 / 51.36 | 0.2476 | +0.0139 | 806 | 10.00% |
| 5 | O6 | 58.84 / 71.43 / 46.26 | 63.95 / 76.53 / 51.36 | 0.2378 | -0.0113 | 803 | 9.96% |
