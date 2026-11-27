# G7 dual Euclidean–hyperbolic geometry

The same R2-100/150 checkpoints, G1 query/bank IDs, classifier slots and 78/20 class-disjoint Known split are reused. These are exploratory comparisons on reused queries. All backbone weights stay frozen. Only missing legally labeled center CLS features are encoded.

H0 uses cosine. H1 applies the origin exponential map at curvature c=1, x = tanh(||v||) v / ||v||. Unit input radius makes H1 pair similarity monotonic in cosine. The assignment-error score subtracts two similarities, so it can change despite equivalent ranking.

H2/H3 share a 768–256–128 GELU architecture, initial parameters, Known pair and batch order, AdamW and 20 epochs. Both use a shared smooth tangent radius bound of 2, which preserves radial information. H2 uses Euclidean distance on those vectors and is the radial control. H3 applies the exponential map and uses Poincaré geodesic distance. The equivalent losses are squared distance for positives and squared hinge with distance margin 1 for negatives. Known validation AUROC selects the earliest best epoch per head. H4/H5 average within-query percentile similarities from H0 and the corresponding learned head. Each distance score uses 2/(1+d²)−1 before scoring. The shared radius bound caps the Poincaré norm below tanh(2).

Common hard AUROC uses uniform-pool same-class pairs against negatives in the G1 H0 top-64. Own hard AUROC uses negatives in each rule’s own top-64 and therefore compares different negative pools. Destination metrics use G1 bank-only Hungarian alignment and wrong-source queries. Disagreement accuracy is top-1 accuracy where the rule and H0 propose different slots. JSON includes H0 accuracy on those same disagreements, either-rule recovery, seed results and counts.

| Epoch / group / method | Purity@1/5/10 | Pair / common / own hard AUC | Error AUC | Destination R@1/3/5 | Recovery / regression | Disagreement ACC |
|---|---|---|---:|---|---|---:|
| 100 / All / H0 | 0.640/0.564/0.521 | 0.944/0.450/0.450 | 0.650 | 0.240/0.398/0.496 | 0.000/0.000 | — |
| 100 / All / H1 | 0.640/0.564/0.521 | 0.944/0.450/0.450 | 0.637 | 0.240/0.398/0.496 | 0.000/0.000 | — |
| 100 / All / H2 | 0.557/0.497/0.463 | 0.885/0.480/0.394 | 0.647 | 0.173/0.304/0.383 | 0.060/0.469 | 0.074 |
| 100 / All / H3 | 0.592/0.528/0.489 | 0.900/0.476/0.425 | 0.656 | 0.200/0.376/0.445 | 0.050/0.327 | 0.078 |
| 100 / All / H4 | 0.605/0.542/0.503 | 0.929/0.447/0.407 | 0.530 | 0.209/0.395/0.476 | 0.036/0.245 | 0.073 |
| 100 / All / H5 | 0.614/0.554/0.511 | 0.934/0.437/0.410 | 0.524 | 0.226/0.410/0.484 | 0.033/0.165 | 0.082 |
| 100 / Old / H0 | 0.716/0.667/0.629 | 0.954/0.646/0.646 | 0.790 | 0.284/0.395/0.494 | 0.000/0.000 | — |
| 100 / Old / H1 | 0.716/0.667/0.629 | 0.954/0.646/0.646 | 0.702 | 0.284/0.395/0.494 | 0.000/0.000 | — |
| 100 / Old / H2 | 0.665/0.615/0.578 | 0.919/0.660/0.573 | 0.758 | 0.189/0.284/0.366 | 0.029/0.406 | 0.034 |
| 100 / Old / H3 | 0.674/0.642/0.599 | 0.928/0.662/0.610 | 0.777 | 0.222/0.366/0.416 | 0.023/0.275 | 0.035 |
| 100 / Old / H4 | 0.694/0.649/0.606 | 0.945/0.657/0.623 | 0.409 | 0.222/0.366/0.453 | 0.011/0.246 | 0.022 |
| 100 / Old / H5 | 0.708/0.659/0.616 | 0.949/0.656/0.633 | 0.388 | 0.235/0.370/0.457 | 0.006/0.188 | 0.013 |
| 100 / New / H0 | 0.606/0.519/0.473 | 0.938/0.378/0.378 | 0.549 | 0.231/0.398/0.496 | 0.000/0.000 | — |
| 100 / New / H1 | 0.606/0.519/0.473 | 0.938/0.378/0.378 | 0.625 | 0.231/0.398/0.496 | 0.000/0.000 | — |
| 100 / New / H2 | 0.511/0.446/0.413 | 0.870/0.403/0.328 | 0.563 | 0.170/0.308/0.386 | 0.066/0.485 | 0.082 |
| 100 / New / H3 | 0.556/0.479/0.442 | 0.887/0.397/0.353 | 0.568 | 0.195/0.378/0.451 | 0.056/0.341 | 0.087 |
| 100 / New / H4 | 0.566/0.495/0.458 | 0.925/0.379/0.345 | 0.607 | 0.207/0.401/0.481 | 0.041/0.244 | 0.083 |
| 100 / New / H5 | 0.573/0.508/0.466 | 0.930/0.369/0.346 | 0.610 | 0.224/0.418/0.489 | 0.038/0.159 | 0.097 |
| 150 / All / H0 | 0.624/0.552/0.516 | 0.942/0.436/0.436 | 0.618 | 0.249/0.401/0.501 | 0.000/0.000 | — |
| 150 / All / H1 | 0.624/0.552/0.516 | 0.942/0.436/0.436 | 0.591 | 0.249/0.401/0.501 | 0.000/0.000 | — |
| 150 / All / H2 | 0.554/0.498/0.459 | 0.878/0.465/0.389 | 0.626 | 0.172/0.287/0.349 | 0.053/0.470 | 0.064 |
| 150 / All / H3 | 0.581/0.526/0.489 | 0.897/0.472/0.429 | 0.631 | 0.222/0.367/0.441 | 0.071/0.322 | 0.103 |
| 150 / All / H4 | 0.592/0.537/0.496 | 0.924/0.436/0.400 | 0.490 | 0.220/0.380/0.457 | 0.055/0.282 | 0.104 |
| 150 / All / H5 | 0.604/0.546/0.505 | 0.931/0.431/0.407 | 0.480 | 0.245/0.399/0.478 | 0.053/0.177 | 0.132 |
| 150 / Old / H0 | 0.694/0.643/0.610 | 0.953/0.622/0.622 | 0.748 | 0.321/0.476/0.548 | 0.000/0.000 | — |
| 150 / Old / H1 | 0.694/0.643/0.610 | 0.953/0.622/0.622 | 0.599 | 0.321/0.476/0.548 | 0.000/0.000 | — |
| 150 / Old / H2 | 0.623/0.599/0.555 | 0.913/0.625/0.539 | 0.703 | 0.171/0.274/0.337 | 0.018/0.506 | 0.021 |
| 150 / Old / H3 | 0.633/0.623/0.587 | 0.925/0.634/0.586 | 0.734 | 0.234/0.341/0.433 | 0.029/0.333 | 0.046 |
| 150 / Old / H4 | 0.645/0.632/0.586 | 0.939/0.630/0.596 | 0.282 | 0.238/0.361/0.440 | 0.023/0.309 | 0.048 |
| 150 / Old / H5 | 0.653/0.635/0.597 | 0.944/0.639/0.618 | 0.253 | 0.270/0.401/0.488 | 0.012/0.185 | 0.032 |
| 150 / New / H0 | 0.594/0.512/0.475 | 0.936/0.370/0.370 | 0.521 | 0.234/0.384/0.491 | 0.000/0.000 | — |
| 150 / New / H1 | 0.594/0.512/0.475 | 0.936/0.370/0.370 | 0.604 | 0.234/0.384/0.491 | 0.000/0.000 | — |
| 150 / New / H2 | 0.525/0.454/0.418 | 0.863/0.397/0.333 | 0.558 | 0.172/0.290/0.352 | 0.060/0.459 | 0.072 |
| 150 / New / H3 | 0.559/0.483/0.447 | 0.884/0.400/0.364 | 0.547 | 0.220/0.373/0.442 | 0.079/0.319 | 0.114 |
| 150 / New / H4 | 0.569/0.496/0.456 | 0.920/0.374/0.343 | 0.609 | 0.216/0.384/0.461 | 0.061/0.274 | 0.114 |
| 150 / New / H5 | 0.584/0.507/0.466 | 0.928/0.368/0.347 | 0.612 | 0.240/0.398/0.476 | 0.061/0.174 | 0.149 |

The table reports arithmetic means across seeds. Seeds share queries and are not independent replications.
Runtime 1.1 minutes.

[Seed-level metrics and frozen prediction provenance](r2_hyperbolic_geometry.json)
