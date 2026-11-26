# E3 global evidence and proposal audit

Two checkpoint-conditioned global replays at R2 epochs 100 and 150. We run the production probe reservation, global neighbor shortlist, reciprocal/cross-view evidence, candidate ranking, peer-aware scoring batch and teacher eligibility routines. Config: 512 cached rows, k=5, two trials, 64 boundaries, 256 scoring rows and nine probe folds. MOVE mass is .2 and ABSTAIN is full. V0 is unchanged and is not stepped or scored. No virtual SGD or downstream acceptance is evaluated.

Unlabeled center and first-trial crops come from the existing R2 audit cache. Only missing known centers, known first-trial views, second-trial evidence views and scoring fillers are encoded. This replays the global proposal mechanism on R2 checkpoints. It cannot recreate historical E3 model states, worker-specific random crops or exact historical proposals. A batch-local approximation was unnecessary because the global mechanism was executable.

Global one-to-one Hungarian alignment uses only the sealed full unlabeled split. It maximizes accuracy on the same panel and can favor majority modes, hide splitting and force merging into one assigned class. Zero-support assignments are excluded. An unmatched source counts as wrong and an unmatched destination receives no credit. No alignment is used for evidence or proposals.

### Detection and action precision

| Epoch / subgroup | Global errors / n | Boundaries precision / recall (%) | Eligible precision / recall (%) | Global / cache ROC AUC (%) | Global random precision (%) | ABSTAIN wrong / n · random (%) | MOVE correct / n · random (%) |
|---|---|---|---|---|---|---|---|
| 100 / all | 2474 / 5568 | 53.1 / 1.4 | 56.1 / 1.3 | 48.9 / 53.9 | 44.4 | 32 / 57 · 36.9 | 2 / 16 · 11.8 |
| 100 / old | 536 / 1801 | 40.0 / 1.1 | 38.5 / 0.9 | 49.4 / 53.7 | 29.8 | 5 / 13 · 25.4 | 1 / 3 · 25.0 |
| 100 / new | 1938 / 3767 | 57.1 / 1.4 | 61.4 / 1.4 | 48.5 / 53.1 | 51.4 | 27 / 44 · 41.8 | 1 / 13 · 7.7 |
| 150 / all | 2496 / 5568 | 54.7 / 1.4 | 55.6 / 1.4 | 49.9 / 49.9 | 44.8 | 35 / 63 · 44.0 | 1 / 10 · 7.1 |
| 150 / old | 537 / 1818 | 57.9 / 2.0 | 57.9 / 2.0 | 50.8 / 56.6 | 29.5 | 11 / 19 · 33.6 | 1 / 3 · 16.7 |
| 150 / new | 1959 / 3750 | 53.3 / 1.2 | 54.5 / 1.2 | 49.5 / 47.2 | 52.2 | 24 / 44 · 48.9 | 0 / 7 · 0.0 |

### MOVE destination and evidence ranking

| Epoch / subgroup | Wrong / correct source | Destination accuracy (%) | Recall@1 / 3 / 5 given wrong source (%) | Correct→wrong among MOVE (%) | Random peer accuracy on same wrong-source IDs (%) | Evidence AUC ABSTAIN / MOVE in eligible pool (%) |
|---|---|---|---|---|---|---|
| 100 / all | 8 / 8 | 12.5 | 25.0 / 25.0 / 25.0 | 50.0 | 25.0 | 57.5 / 65.0 |
| 100 / old | 1 / 2 | 33.3 | 100.0 / 100.0 / 100.0 | 66.7 | 100.0 | 57.6 / 100.0 |
| 100 / new | 7 / 6 | 7.7 | 14.3 / 14.3 / 14.3 | 46.2 | 14.3 | 56.6 / 25.0 |
| 150 / all | 5 / 5 | 10.0 | 20.0 / 20.0 / 20.0 | 50.0 | 20.0 | 50.2 / 53.8 |
| 150 / old | 1 / 2 | 33.3 | 100.0 / 100.0 / 100.0 | 66.7 | 100.0 | 56.2 / 60.0 |
| 150 / new | 4 / 3 | 0.0 | 0.0 / 0.0 / 0.0 | 42.9 | 0.0 | 47.7 / — |

Scores for rows without measured support tie below every scored row. Global recall includes non-probe unlabeled errors outside the cache. A low global recall is partly a 64-row budget effect. Random controls are the exact expectations of uniform draws with the same count, conditional on snapshot/group and (for action precision) kind and teacher eligibility. The random destination control holds the MOVE ID fixed and samples from supported alternatives. Recall@k uses only peer-backed targets with sufficient slot support. Many rows have fewer than five. The chosen destination remains the original E3 top-1.

### Bottleneck assessment

- Epoch 100 — Destination generation is the largest directly observed loss among MOVE proposals: a majority of wrong-source candidates lack the aligned true class even in the top five. Detection precision 56.1% against random 44.4%. ABSTAIN evidence AUC 57.5%. Unmatched slots [102, 116, 131, 141, 154, 155, 162, 176, 184, 194]; active unmatched slots [102, 131, 162, 184, 194].
- Epoch 150 — Destination generation is the largest directly observed loss among MOVE proposals: a majority of wrong-source candidates lack the aligned true class even in the top five. Detection precision 55.6% against random 44.8%. ABSTAIN evidence AUC 50.2%. Unmatched slots [102, 116, 131, 141, 154, 155, 162, 176, 191, 194]; active unmatched slots [102, 131, 162, 191, 194].

These are descriptive, dependent samples from two snapshots. A proposal that is correct under Hungarian alignment does not guarantee improved feature geometry or cluster accuracy after an optimizer step.

[Per-action predictions, raw counts and sealed alignment](r2_e3_proposal_audit.json)
