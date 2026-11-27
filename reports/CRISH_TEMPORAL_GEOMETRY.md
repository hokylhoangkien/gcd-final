# G4 temporal geometry on Stanford Cars

Exploratory evaluation on reused G1 query and bank IDs. Epoch 100 is a current-only baseline. The cache supplies snapshot CLS and classifier slots at 100 and 150. No intervening per-ID classifier logits or slot trajectory is available. The verified edge ledger supplies graph history when available.

Current risk is the alternative-minus-current slot similarity margin. Neighborhood is one minus top-10 Jaccard overlap. Drift is one minus same-image cosine. SlotChange uses a bank-only, label-blind maximum-overlap matching of historical and current slots. Persistent measures the fraction of the ten positions occupied by peers that remained neighbors and disagreed with the query's slot at both snapshots. Fusion averages historical and current cosine and ranks current slots. All risk-only rules keep current retrieval. LedgerNeighborhood uses undirected edge-set Jaccard distance from rows 101 and 150 (models 100 and 149), with two empty sets scored zero.

Before opening labels, each rule freezes its global top 10/20/40% flags using ascending sample ID to break ties. Old/New metrics use those same global flags. Error labels use bank-only Hungarian alignment per checkpoint. A stable error is a query wrong under both semantic alignments with the same slot after label-blind matching. Stable AUC treats all other queries as negatives. JSON also gives ordinary error AUROC restricted to unchanged slots and recall of stable errors at each coverage. A single slot match can conceal splits or merges, and binary slot-change scores have large tied groups.

**Slot identity.** 191 of 191 matched slot pairs kept their raw index. Bank agreement was 0.985 raw and 0.985 after matching. Query agreement was 0.981 raw and 0.981 after matching.

| Epoch / group / rule | Error AUC | P/R @10% | P/R @20% | P/R @40% | Dest R@1/3/5 | Stable errors / AUC |
|---|---:|---|---|---|---|---|
| 100 / All / Current | 0.650 | 0.676/0.147 | 0.627/0.272 | 0.553/0.481 | 0.240/0.398/0.496 | — / — |
| 100 / Old / Current | 0.790 | 0.737/0.173 | 0.561/0.284 | 0.526/0.494 | 0.284/0.395/0.494 | — / — |
| 100 / New / Current | 0.549 | 0.663/0.141 | 0.644/0.270 | 0.559/0.478 | 0.231/0.398/0.496 | — / — |
| 150 / All / Current | 0.618 | 0.578/0.126 | 0.529/0.230 | 0.548/0.478 | 0.249/0.401/0.501 | 444 / 0.594 |
| 150 / All / Neighborhood | 0.638 | 0.657/0.143 | 0.618/0.269 | 0.609/0.531 | 0.249/0.401/0.501 | 444 / 0.614 |
| 150 / All / Drift | 0.574 | 0.549/0.119 | 0.534/0.232 | 0.531/0.463 | 0.249/0.401/0.501 | 444 / 0.546 |
| 150 / All / SlotChange | 0.512 | 0.529/0.115 | 0.515/0.224 | 0.479/0.418 | 0.249/0.401/0.501 | 444 / 0.484 |
| 150 / All / Persistent | 0.518 | 0.569/0.124 | 0.515/0.224 | 0.482/0.420 | 0.249/0.401/0.501 | 444 / 0.519 |
| 150 / All / Fusion | 0.636 | 0.647/0.141 | 0.559/0.243 | 0.553/0.482 | 0.249/0.394/0.486 | 444 / 0.603 |
| 150 / All / LedgerNeighborhood | 0.504 | 0.529/0.115 | 0.515/0.224 | 0.460/0.401 | 0.249/0.401/0.501 | 444 / 0.493 |
| 150 / Old / Current | 0.748 | 0.455/0.119 | 0.487/0.226 | 0.525/0.500 | 0.321/0.476/0.548 | 78 / 0.728 |
| 150 / Old / Neighborhood | 0.779 | 0.778/0.167 | 0.692/0.321 | 0.663/0.655 | 0.321/0.476/0.548 | 78 / 0.743 |
| 150 / Old / Drift | 0.630 | 0.382/0.155 | 0.348/0.274 | 0.371/0.548 | 0.321/0.476/0.548 | 78 / 0.586 |
| 150 / Old / SlotChange | 0.536 | 0.379/0.131 | 0.273/0.179 | 0.250/0.345 | 0.321/0.476/0.548 | 78 / 0.487 |
| 150 / Old / Persistent | 0.502 | 0.276/0.095 | 0.246/0.179 | 0.238/0.345 | 0.321/0.476/0.548 | 78 / 0.497 |
| 150 / Old / Fusion | 0.783 | 0.640/0.190 | 0.523/0.274 | 0.526/0.488 | 0.345/0.452/0.548 | 78 / 0.749 |
| 150 / Old / LedgerNeighborhood | 0.538 | 0.500/0.167 | 0.466/0.321 | 0.303/0.393 | 0.321/0.476/0.548 | 78 / 0.527 |
| 150 / New / Current | 0.521 | 0.613/0.127 | 0.539/0.231 | 0.553/0.473 | 0.234/0.384/0.491 | 366 / 0.499 |
| 150 / New / Neighborhood | 0.554 | 0.631/0.138 | 0.600/0.257 | 0.595/0.504 | 0.234/0.384/0.491 | 366 / 0.537 |
| 150 / New / Drift | 0.563 | 0.632/0.112 | 0.623/0.223 | 0.600/0.444 | 0.234/0.384/0.491 | 366 / 0.538 |
| 150 / New / SlotChange | 0.506 | 0.589/0.112 | 0.604/0.234 | 0.570/0.434 | 0.234/0.384/0.491 | 366 / 0.481 |
| 150 / New / Persistent | 0.527 | 0.685/0.130 | 0.629/0.234 | 0.585/0.436 | 0.234/0.384/0.491 | 366 / 0.530 |
| 150 / New / Fusion | 0.536 | 0.649/0.130 | 0.569/0.236 | 0.559/0.481 | 0.229/0.382/0.473 | 366 / 0.506 |
| 150 / New / LedgerNeighborhood | 0.482 | 0.541/0.104 | 0.534/0.203 | 0.517/0.403 | 0.234/0.384/0.491 | 366 / 0.472 |

Runtime 3.4 seconds.

[Counts, coverage, slot matching and frozen digests](r2_temporal_geometry.json)
