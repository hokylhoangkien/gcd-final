# G3 frozen pretrained geometry on Stanford Cars

This exploratory comparison reuses G1 queries, retrieval banks, pair pools and classifier slots. The verified R2 snapshot audit cached DINOv1 center-crop CLS before fine-tuning and after epochs 100 and 150. G3 records the input hashes and reproduces the G1 fine-tuned baseline before scoring.

Pretrained and FT use cosine. Equal averages their scores with fixed weights. DisagreementRisk uses one minus Jaccard overlap of the two top-10 neighborhoods. PretrainedRisk uses the pretrained alternative-minus-current slot support margin. Both risk-only rows retrieve destinations and neighbors using FT. No calibration is fitted. Detection risk is separate from destination ranking.

All rules use the same G1 hard negatives. The bank alone supplies Hungarian alignment. Recovery and regression are measured relative to FT on wrong-source queries. Fixed-coverage flags take the top 10%, 20% or 40% of all queries by risk (rounded down), with ascending sample ID for ties. Old/New precision and recall condition on query class within those same flags. The table gives 20% precision/recall and realized group coverage; JSON includes all levels and counts.

| Epoch / group / method | Purity@1/5/10 | Pair / hard AUROC | Error AUROC | Dest R@1/3/5 | Recovery / regression | P / R at 20% | Group coverage |
|---|---|---|---:|---|---|---|---:|
| 100 / All / FT | 0.640/0.564/0.521 | 0.944 / 0.450 | 0.650 | 0.240/0.398/0.496 | 0.000 / 0.000 | 0.627 / 0.272 | 0.199 |
| 100 / All / Pretrained | 0.227/0.139/0.106 | 0.634 / 0.468 | 0.539 | 0.057/0.115/0.140 | 0.039 / 0.885 | 0.500 / 0.217 | 0.199 |
| 100 / All / Equal | 0.618/0.558/0.514 | 0.926 / 0.438 | 0.644 | 0.228/0.396/0.474 | 0.053 / 0.221 | 0.613 / 0.266 | 0.199 |
| 100 / All / DisagreementRisk | 0.640/0.564/0.521 | 0.944 / 0.450 | 0.531 | 0.240/0.398/0.496 | 0.000 / 0.000 | 0.471 / 0.204 | 0.199 |
| 100 / All / PretrainedRisk | 0.640/0.564/0.521 | 0.944 / 0.450 | 0.539 | 0.240/0.398/0.496 | 0.000 / 0.000 | 0.500 / 0.217 | 0.199 |
| 100 / Old / FT | 0.716/0.667/0.629 | 0.954 / 0.646 | 0.790 | 0.284/0.395/0.494 | 0.000 / 0.000 | 0.561 / 0.284 | 0.132 |
| 100 / Old / Pretrained | 0.184/0.112/0.080 | 0.640 / 0.506 | 0.484 | 0.086/0.123/0.173 | 0.034 / 0.783 | 0.237 / 0.235 | 0.258 |
| 100 / Old / Equal | 0.697/0.658/0.614 | 0.941 / 0.618 | 0.738 | 0.247/0.407/0.469 | 0.034 / 0.217 | 0.474 / 0.222 | 0.123 |
| 100 / Old / DisagreementRisk | 0.716/0.667/0.629 | 0.954 / 0.646 | 0.508 | 0.284/0.395/0.494 | 0.000 / 0.000 | 0.185 / 0.148 | 0.210 |
| 100 / Old / PretrainedRisk | 0.716/0.667/0.629 | 0.954 / 0.646 | 0.484 | 0.284/0.395/0.494 | 0.000 / 0.000 | 0.237 / 0.235 | 0.258 |
| 100 / New / FT | 0.606/0.519/0.473 | 0.938 / 0.378 | 0.549 | 0.231/0.398/0.496 | 0.000 / 0.000 | 0.644 / 0.270 | 0.228 |
| 100 / New / Pretrained | 0.245/0.151/0.117 | 0.631 / 0.450 | 0.586 | 0.051/0.113/0.134 | 0.040 / 0.911 | 0.669 / 0.213 | 0.174 |
| 100 / New / Equal | 0.584/0.515/0.471 | 0.920 / 0.373 | 0.578 | 0.224/0.393/0.476 | 0.057 / 0.222 | 0.645 / 0.275 | 0.232 |
| 100 / New / DisagreementRisk | 0.606/0.519/0.473 | 0.938 / 0.378 | 0.557 | 0.231/0.398/0.496 | 0.000 / 0.000 | 0.604 / 0.216 | 0.195 |
| 100 / New / PretrainedRisk | 0.606/0.519/0.473 | 0.938 / 0.378 | 0.586 | 0.231/0.398/0.496 | 0.000 / 0.000 | 0.669 / 0.213 | 0.174 |
| 150 / All / FT | 0.624/0.552/0.516 | 0.942 / 0.436 | 0.618 | 0.249/0.401/0.501 | 0.000 / 0.000 | 0.529 / 0.230 | 0.199 |
| 150 / All / Pretrained | 0.227/0.139/0.106 | 0.634 / 0.471 | 0.534 | 0.062/0.122/0.147 | 0.034 / 0.855 | 0.500 / 0.217 | 0.199 |
| 150 / All / Equal | 0.610/0.550/0.503 | 0.923 / 0.429 | 0.614 | 0.228/0.392/0.469 | 0.051 / 0.239 | 0.549 / 0.239 | 0.199 |
| 150 / All / DisagreementRisk | 0.624/0.552/0.516 | 0.942 / 0.436 | 0.527 | 0.249/0.401/0.501 | 0.000 / 0.000 | 0.475 / 0.207 | 0.199 |
| 150 / All / PretrainedRisk | 0.624/0.552/0.516 | 0.942 / 0.436 | 0.534 | 0.249/0.401/0.501 | 0.000 / 0.000 | 0.500 / 0.217 | 0.199 |
| 150 / Old / FT | 0.694/0.643/0.610 | 0.953 / 0.622 | 0.748 | 0.321/0.476/0.548 | 0.000 / 0.000 | 0.487 / 0.226 | 0.126 |
| 150 / Old / Pretrained | 0.184/0.112/0.080 | 0.640 / 0.508 | 0.488 | 0.083/0.119/0.167 | 0.035 / 0.815 | 0.268 / 0.262 | 0.265 |
| 150 / Old / Equal | 0.677/0.646/0.593 | 0.939 / 0.595 | 0.686 | 0.286/0.452/0.512 | 0.018 / 0.148 | 0.489 / 0.274 | 0.152 |
| 150 / Old / DisagreementRisk | 0.694/0.643/0.610 | 0.953 / 0.622 | 0.505 | 0.321/0.476/0.548 | 0.000 / 0.000 | 0.203 / 0.167 | 0.223 |
| 150 / Old / PretrainedRisk | 0.694/0.643/0.610 | 0.953 / 0.622 | 0.488 | 0.321/0.476/0.548 | 0.000 / 0.000 | 0.268 / 0.262 | 0.265 |
| 150 / New / FT | 0.594/0.512/0.475 | 0.936 / 0.370 | 0.521 | 0.234/0.384/0.491 | 0.000 / 0.000 | 0.539 / 0.231 | 0.231 |
| 150 / New / Pretrained | 0.245/0.151/0.117 | 0.631 / 0.452 | 0.577 | 0.057/0.122/0.143 | 0.034 / 0.867 | 0.656 / 0.208 | 0.171 |
| 150 / New / Equal | 0.581/0.509/0.464 | 0.915 / 0.369 | 0.560 | 0.216/0.379/0.460 | 0.058 / 0.267 | 0.567 / 0.231 | 0.220 |
| 150 / New / DisagreementRisk | 0.594/0.512/0.475 | 0.936 / 0.370 | 0.552 | 0.234/0.384/0.491 | 0.000 / 0.000 | 0.615 / 0.216 | 0.189 |
| 150 / New / PretrainedRisk | 0.594/0.512/0.475 | 0.936 / 0.370 | 0.577 | 0.234/0.384/0.491 | 0.000 / 0.000 | 0.656 / 0.208 | 0.171 |

## Neighborhood overlap

| Epoch / group | Mean shared neighbors out of 10 | Mean Jaccard |
|---|---:|---:|
| 100 / All | 0.991 | 0.057 |
| 100 / Old | 0.858 | 0.049 |
| 100 / New | 1.049 | 0.060 |
| 150 / All | 0.985 | 0.057 |
| 150 / Old | 0.855 | 0.049 |
| 150 / New | 1.042 | 0.060 |

Runtime 3.3 seconds.

[Coverage counts, disagreement analysis and prediction provenance](r2_pretrained_geometry.json)
