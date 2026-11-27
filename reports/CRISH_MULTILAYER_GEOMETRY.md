# G1 multi-layer geometry on Stanford Cars

Blocks 3, 7 and 9 use the L2-normalized post-block CLS token. Block 11 applies the checkpoint's ViT final LayerNorm before L2 normalization. The frozen prefix is identical at epochs 100 and 150. The final CLS cache is reused after exact-protocol forward parity on the first eight unlabeled IDs.

Both checkpoints share 1,024 fresh label-blind queries and 5,120 other unlabeled retrieval IDs. Query IDs exclude the prior 588-ID Oracle MOVE panel and frozen E3 proposal IDs. Earlier split-wide diagnostics included these images, so this is a new held-out query partition within Cars. The fit data consists only of separate legally labeled images, split 78/20 by class.

Equal fusion averages cosine scores. Learned fusion minimizes logistic loss under nonnegative weights summing to one. Known validation AUROC selects the ridge term from 0, .01 and .1. Every scoring rule shares 256 uniform peers per query and the same top-64 B11 neighbors for hard negatives. Hard AUROC uses uniform positives against the wrong-class hard peers. All groups are grouped by query class. Pairs sharing IDs are dependent, so the table is descriptive.

The sealed evaluator maps frozen classifier slots to classes by Hungarian alignment on bank IDs. Candidate destinations are the best alternative slots by maximum peer cosine across that entire bank. Wrong-assignment risk is the best alternative support minus current-slot support. Recall includes all wrong assignments, including ones with no mapped destination. Recovery conditions on B11 top-1 failure on wrong sources. Regression conditions on B11 top-1 success on wrong sources. JSON also reports forced edits on correct sources, mapped-only recall and purity on B11 misses. Coverage and mapping depend on the checkpoint. These metrics do not measure training effects.

| Epoch / group / rule | Purity@1/5/10 | Pair / hard AUC | Wrong AUC | Destination R@1/3/5 | Recovery@1 | Regression@1 |
|---|---|---|---:|---|---:|---:|
| 100 / All / B3 | 0.069/0.042/0.034 | 0.578 / 0.492 | 0.490 | 0.017/0.040/0.068 | 0.014 | 0.973 |
| 100 / All / B7 | 0.178/0.109/0.079 | 0.608 / 0.483 | 0.512 | 0.036/0.077/0.109 | 0.022 | 0.920 |
| 100 / All / B9 | 0.236/0.137/0.104 | 0.632 / 0.478 | 0.537 | 0.053/0.130/0.155 | 0.025 | 0.858 |
| 100 / All / B11 | 0.640/0.564/0.521 | 0.944 / 0.450 | 0.650 | 0.240/0.398/0.496 | 0.000 | 0.000 |
| 100 / All / Equal | 0.638/0.565/0.518 | 0.925 / 0.445 | 0.643 | 0.232/0.396/0.489 | 0.056 | 0.212 |
| 100 / All / Learned | 0.640/0.564/0.521 | 0.944 / 0.450 | 0.650 | 0.240/0.398/0.496 | 0.000 | 0.000 |
| 100 / Old / B3 | 0.055/0.031/0.023 | 0.595 / 0.519 | 0.442 | 0.037/0.049/0.086 | 0.034 | 0.957 |
| 100 / Old / B7 | 0.148/0.085/0.057 | 0.623 / 0.521 | 0.468 | 0.037/0.099/0.123 | 0.000 | 0.870 |
| 100 / Old / B9 | 0.177/0.105/0.077 | 0.631 / 0.509 | 0.510 | 0.074/0.185/0.198 | 0.017 | 0.783 |
| 100 / Old / B11 | 0.716/0.667/0.629 | 0.954 / 0.646 | 0.790 | 0.284/0.395/0.494 | 0.000 | 0.000 |
| 100 / Old / Equal | 0.713/0.660/0.617 | 0.938 / 0.624 | 0.734 | 0.272/0.407/0.506 | 0.017 | 0.087 |
| 100 / Old / Learned | 0.716/0.667/0.629 | 0.954 / 0.646 | 0.790 | 0.284/0.395/0.494 | 0.000 | 0.000 |
| 100 / New / B3 | 0.076/0.046/0.039 | 0.572 / 0.481 | 0.538 | 0.013/0.039/0.064 | 0.010 | 0.978 |
| 100 / New / B7 | 0.190/0.120/0.089 | 0.602 / 0.465 | 0.555 | 0.036/0.072/0.105 | 0.027 | 0.933 |
| 100 / New / B9 | 0.262/0.152/0.115 | 0.629 / 0.460 | 0.578 | 0.049/0.118/0.147 | 0.027 | 0.878 |
| 100 / New / B11 | 0.606/0.519/0.473 | 0.938 / 0.378 | 0.549 | 0.231/0.398/0.496 | 0.000 | 0.000 |
| 100 / New / Equal | 0.605/0.524/0.475 | 0.919 / 0.380 | 0.581 | 0.224/0.393/0.486 | 0.064 | 0.244 |
| 100 / New / Learned | 0.606/0.519/0.473 | 0.938 / 0.378 | 0.549 | 0.231/0.398/0.496 | 0.000 | 0.000 |
| 150 / All / B3 | 0.069/0.042/0.034 | 0.578 / 0.493 | 0.488 | 0.021/0.049/0.077 | 0.014 | 0.957 |
| 150 / All / B7 | 0.178/0.109/0.079 | 0.608 / 0.485 | 0.509 | 0.041/0.087/0.124 | 0.023 | 0.906 |
| 150 / All / B9 | 0.236/0.137/0.104 | 0.632 / 0.481 | 0.532 | 0.060/0.141/0.160 | 0.023 | 0.829 |
| 150 / All / B11 | 0.624/0.552/0.516 | 0.942 / 0.436 | 0.618 | 0.249/0.401/0.501 | 0.000 | 0.000 |
| 150 / All / Equal | 0.617/0.554/0.508 | 0.922 / 0.436 | 0.611 | 0.239/0.409/0.475 | 0.054 | 0.205 |
| 150 / All / Learned | 0.624/0.552/0.516 | 0.942 / 0.436 | 0.618 | 0.249/0.401/0.501 | 0.000 | 0.000 |
| 150 / Old / B3 | 0.055/0.031/0.023 | 0.595 / 0.519 | 0.434 | 0.048/0.060/0.095 | 0.035 | 0.926 |
| 150 / Old / B7 | 0.148/0.085/0.057 | 0.623 / 0.521 | 0.453 | 0.036/0.095/0.119 | 0.000 | 0.889 |
| 150 / Old / B9 | 0.177/0.105/0.077 | 0.631 / 0.510 | 0.499 | 0.071/0.167/0.179 | 0.018 | 0.815 |
| 150 / Old / B11 | 0.694/0.643/0.610 | 0.953 / 0.622 | 0.748 | 0.321/0.476/0.548 | 0.000 | 0.000 |
| 150 / Old / Equal | 0.697/0.647/0.597 | 0.937 / 0.601 | 0.667 | 0.298/0.440/0.500 | 0.053 | 0.185 |
| 150 / Old / Learned | 0.694/0.643/0.610 | 0.953 / 0.622 | 0.748 | 0.321/0.476/0.548 | 0.000 | 0.000 |
| 150 / New / B3 | 0.076/0.046/0.039 | 0.572 / 0.483 | 0.535 | 0.016/0.047/0.073 | 0.010 | 0.967 |
| 150 / New / B7 | 0.190/0.120/0.089 | 0.602 / 0.469 | 0.556 | 0.042/0.086/0.125 | 0.027 | 0.911 |
| 150 / New / B9 | 0.262/0.152/0.115 | 0.629 / 0.463 | 0.573 | 0.057/0.135/0.156 | 0.024 | 0.833 |
| 150 / New / B11 | 0.594/0.512/0.475 | 0.936 / 0.370 | 0.521 | 0.234/0.384/0.491 | 0.000 | 0.000 |
| 150 / New / Equal | 0.583/0.513/0.470 | 0.915 / 0.375 | 0.565 | 0.226/0.403/0.470 | 0.054 | 0.211 |
| 150 / New / Learned | 0.594/0.512/0.475 | 0.936 / 0.370 | 0.521 | 0.234/0.384/0.491 | 0.000 | 0.000 |

Epoch 100 learned weights for B3/B7/B9/B11 were 0.000, 0.000, 0.000, 1.000.

Epoch 150 learned weights for B3/B7/B9/B11 were 0.000, 0.000, 0.000, 1.000.

Runtime 2.8 minutes.

[Counts, subgroup metrics and frozen prediction provenance](r2_multilayer_geometry.json)
