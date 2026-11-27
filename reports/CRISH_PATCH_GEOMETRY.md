# G2 patch-token geometry

This exploratory evaluation reuses the G1 Cars query and bank manifests. Neither the backbone nor an auxiliary head is trained. Every token comes from block 11 followed by final LayerNorm. Each patch is L2-normalized before pooling.

Mean pools all 196 normalized patch tokens and normalizes the resulting vector. Spatial averages four fixed 7×7 quadrants, normalizes each quadrant, then concatenates and normalizes the descriptor. The three full-bank similarity matrices use cosine. Each CLS fusion averages its two component similarities with equal fixed weights.

Local matching averages the mean best patch cosine in each direction. It uses the same fixed 64 CLS peers as CLS@64. Similarity uses float16 patch storage and matmul with float32 reductions. Matrix allocation is bounded by 32 peers per chunk. All shortlist variants use the G1 top-64 as their only pair pool; their pair AUC and common hard AUC therefore describe the same restricted discrimination task. Full-bank variants use the original G1 uniform positives and common CLS hard negatives.

Destination recall is conditioned on a wrong source under G1 bank-only Hungarian alignment. The shortlist ceiling is the fraction of such queries whose true slot occurs among the 64 peers. Recovery and regression condition on top-1 failure and success of the CLS comparator from the same scope. Either-correct is the fraction of wrong-source queries with a correct top-1 destination from either scoring rule.

Shortlist error AUROC uses a common subset for all three shortlist rules. Eligible queries have at least one peer in a different slot. Each excluded query has a structural −∞ score for all three rules. All other metrics keep their original denominators.

| Epoch / group / rule | Purity@1/5/10 | Pair / hard AUC | Error AUC | Destination R@1/3/5 | Recovery / regression | Either-correct |
|---|---|---|---:|---|---|---:|
| 100 / All / CLS | 0.640/0.564/0.521 | 0.944 / 0.450 | 0.650 | 0.240/0.398/0.496 | 0.000 / 0.000 | 0.240 |
| 100 / All / Mean | 0.394/0.297/0.242 | 0.781 / 0.489 | 0.611 | 0.115/0.219/0.291 | 0.036 / 0.637 | 0.268 |
| 100 / All / Spatial | 0.337/0.252/0.208 | 0.777 / 0.489 | 0.591 | 0.100/0.232/0.281 | 0.042 / 0.717 | 0.272 |
| 100 / All / CLS+Mean | 0.610/0.563/0.524 | 0.938 / 0.451 | 0.674 | 0.217/0.404/0.502 | 0.028 / 0.186 | 0.262 |
| 100 / All / CLS+Spatial | 0.614/0.562/0.519 | 0.939 / 0.449 | 0.671 | 0.221/0.404/0.498 | 0.042 / 0.212 | 0.272 |
| 100 / All / CLS@64 | 0.640/0.564/0.521 | 0.732 / 0.732 | 0.655 | 0.238/0.381/0.455 | 0.000 / 0.000 | 0.238 |
| 100 / All / Local@64 | 0.551/0.456/0.402 | 0.625 / 0.625 | 0.611 | 0.198/0.385/0.457 | 0.073 / 0.402 | 0.294 |
| 100 / All / CLS+Local@64 | 0.634/0.585/0.536 | 0.732 / 0.732 | 0.663 | 0.230/0.391/0.457 | 0.028 / 0.125 | 0.260 |
| 100 / Old / CLS | 0.716/0.667/0.629 | 0.954 / 0.646 | 0.790 | 0.284/0.395/0.494 | 0.000 / 0.000 | 0.284 |
| 100 / Old / Mean | 0.390/0.291/0.222 | 0.801 / 0.548 | 0.602 | 0.185/0.222/0.272 | 0.052 / 0.478 | 0.321 |
| 100 / Old / Spatial | 0.332/0.241/0.189 | 0.792 / 0.540 | 0.571 | 0.148/0.222/0.235 | 0.069 / 0.652 | 0.333 |
| 100 / Old / CLS+Mean | 0.694/0.664/0.625 | 0.953 / 0.629 | 0.787 | 0.259/0.395/0.506 | 0.017 / 0.130 | 0.296 |
| 100 / Old / CLS+Spatial | 0.690/0.666/0.624 | 0.954 / 0.627 | 0.771 | 0.272/0.395/0.494 | 0.017 / 0.087 | 0.296 |
| 100 / Old / CLS@64 | 0.716/0.667/0.629 | 0.831 / 0.831 | 0.790 | 0.284/0.383/0.432 | 0.000 / 0.000 | 0.284 |
| 100 / Old / Local@64 | 0.555/0.450/0.380 | 0.622 / 0.622 | 0.596 | 0.235/0.370/0.420 | 0.052 / 0.304 | 0.321 |
| 100 / Old / CLS+Local@64 | 0.713/0.683/0.635 | 0.819 / 0.819 | 0.778 | 0.259/0.407/0.432 | 0.000 / 0.087 | 0.284 |
| 100 / New / CLS | 0.606/0.519/0.473 | 0.938 / 0.378 | 0.549 | 0.231/0.398/0.496 | 0.000 / 0.000 | 0.231 |
| 100 / New / Mean | 0.395/0.299/0.251 | 0.772 / 0.460 | 0.630 | 0.100/0.219/0.296 | 0.033 / 0.678 | 0.257 |
| 100 / New / Spatial | 0.339/0.257/0.216 | 0.769 / 0.462 | 0.619 | 0.090/0.234/0.290 | 0.037 / 0.733 | 0.260 |
| 100 / New / CLS+Mean | 0.574/0.519/0.480 | 0.931 / 0.383 | 0.598 | 0.208/0.406/0.501 | 0.030 / 0.200 | 0.254 |
| 100 / New / CLS+Spatial | 0.581/0.517/0.473 | 0.933 / 0.382 | 0.603 | 0.211/0.406/0.499 | 0.047 / 0.244 | 0.267 |
| 100 / New / CLS@64 | 0.606/0.519/0.473 | 0.687 / 0.687 | 0.556 | 0.229/0.380/0.460 | 0.000 / 0.000 | 0.229 |
| 100 / New / Local@64 | 0.549/0.459/0.411 | 0.618 / 0.618 | 0.631 | 0.190/0.388/0.465 | 0.077 / 0.427 | 0.288 |
| 100 / New / CLS+Local@64 | 0.599/0.543/0.493 | 0.692 / 0.692 | 0.584 | 0.224/0.388/0.463 | 0.033 / 0.135 | 0.254 |
| 150 / All / CLS | 0.624/0.552/0.516 | 0.942 / 0.436 | 0.618 | 0.249/0.401/0.501 | 0.000 / 0.000 | 0.249 |
| 150 / All / Mean | 0.354/0.255/0.206 | 0.747 / 0.494 | 0.593 | 0.098/0.215/0.264 | 0.040 / 0.726 | 0.279 |
| 150 / All / Spatial | 0.311/0.213/0.169 | 0.742 / 0.494 | 0.571 | 0.092/0.200/0.245 | 0.031 / 0.726 | 0.273 |
| 150 / All / CLS+Mean | 0.594/0.548/0.514 | 0.936 / 0.440 | 0.637 | 0.235/0.407/0.475 | 0.034 / 0.162 | 0.275 |
| 150 / All / CLS+Spatial | 0.598/0.541/0.504 | 0.937 / 0.439 | 0.635 | 0.226/0.412/0.486 | 0.037 / 0.205 | 0.277 |
| 150 / All / CLS@64 | 0.624/0.552/0.516 | 0.725 / 0.725 | 0.624 | 0.241/0.384/0.437 | 0.000 / 0.000 | 0.241 |
| 150 / All / Local@64 | 0.529/0.444/0.386 | 0.616 / 0.616 | 0.585 | 0.196/0.397/0.431 | 0.062 / 0.381 | 0.288 |
| 150 / All / CLS+Local@64 | 0.625/0.566/0.519 | 0.728 / 0.728 | 0.627 | 0.239/0.397/0.435 | 0.034 / 0.115 | 0.267 |
| 150 / Old / CLS | 0.694/0.643/0.610 | 0.953 / 0.622 | 0.748 | 0.321/0.476/0.548 | 0.000 / 0.000 | 0.321 |
| 150 / Old / Mean | 0.348/0.245/0.185 | 0.761 / 0.542 | 0.555 | 0.143/0.226/0.250 | 0.053 / 0.667 | 0.357 |
| 150 / Old / Spatial | 0.281/0.193/0.145 | 0.750 / 0.534 | 0.522 | 0.095/0.190/0.226 | 0.035 / 0.778 | 0.345 |
| 150 / Old / CLS+Mean | 0.661/0.638/0.608 | 0.953 / 0.610 | 0.719 | 0.321/0.440/0.524 | 0.035 / 0.074 | 0.345 |
| 150 / Old / CLS+Spatial | 0.674/0.637/0.602 | 0.953 / 0.608 | 0.700 | 0.286/0.464/0.536 | 0.053 / 0.222 | 0.357 |
| 150 / Old / CLS@64 | 0.694/0.643/0.610 | 0.814 / 0.814 | 0.748 | 0.321/0.464/0.488 | 0.000 / 0.000 | 0.321 |
| 150 / Old / Local@64 | 0.529/0.431/0.350 | 0.606 / 0.606 | 0.569 | 0.179/0.429/0.476 | 0.018 / 0.481 | 0.333 |
| 150 / Old / CLS+Local@64 | 0.684/0.660/0.610 | 0.806 / 0.806 | 0.711 | 0.321/0.452/0.488 | 0.018 / 0.037 | 0.333 |
| 150 / New / CLS | 0.594/0.512/0.475 | 0.936 / 0.370 | 0.521 | 0.234/0.384/0.491 | 0.000 / 0.000 | 0.234 |
| 150 / New / Mean | 0.357/0.259/0.216 | 0.739 / 0.467 | 0.623 | 0.088/0.213/0.268 | 0.037 / 0.744 | 0.262 |
| 150 / New / Spatial | 0.324/0.221/0.180 | 0.735 / 0.470 | 0.608 | 0.091/0.203/0.249 | 0.031 / 0.711 | 0.257 |
| 150 / New / CLS+Mean | 0.564/0.509/0.473 | 0.929 / 0.377 | 0.578 | 0.216/0.400/0.465 | 0.034 / 0.189 | 0.260 |
| 150 / New / CLS+Spatial | 0.564/0.500/0.462 | 0.930 / 0.376 | 0.586 | 0.213/0.400/0.475 | 0.034 / 0.200 | 0.260 |
| 150 / New / CLS@64 | 0.594/0.512/0.475 | 0.687 / 0.687 | 0.526 | 0.223/0.366/0.426 | 0.000 / 0.000 | 0.223 |
| 150 / New / Local@64 | 0.529/0.450/0.401 | 0.611 / 0.611 | 0.608 | 0.200/0.390/0.421 | 0.070 / 0.349 | 0.278 |
| 150 / New / CLS+Local@64 | 0.599/0.525/0.480 | 0.693 / 0.693 | 0.565 | 0.221/0.384/0.423 | 0.037 / 0.140 | 0.252 |

## Shortlist destination coverage

| Epoch / group | Reachable / wrong sources | Recall ceiling |
|---|---:|---:|
| 100 / All | 225 / 470 | 0.479 |
| 100 / Old | 36 / 81 | 0.444 |
| 100 / New | 189 / 389 | 0.486 |
| 150 / All | 212 / 469 | 0.452 |
| 150 / Old | 43 / 84 | 0.512 |
| 150 / New | 169 / 385 | 0.439 |

## Shortlist error AUROC coverage

| Epoch / group | Eligible / queries | Excluded | Excluded wrong sources | Coverage |
|---|---:|---:|---:|---:|
| 100 / All | 998 / 1024 | 26 | 16 | 0.975 |
| 100 / Old | 310 / 310 | 0 | 0 | 1.000 |
| 100 / New | 688 / 714 | 26 | 16 | 0.964 |
| 150 / All | 983 / 1024 | 41 | 24 | 0.960 |
| 150 / Old | 310 / 310 | 0 | 0 | 1.000 |
| 150 / New | 673 / 714 | 41 | 24 | 0.943 |

Sealed evaluation took 2.6 seconds. The frozen predictions reused exact prefix tokens for 588 images.

[Counts, disagreement metrics and frozen prediction digests](r2_patch_geometry.json)
