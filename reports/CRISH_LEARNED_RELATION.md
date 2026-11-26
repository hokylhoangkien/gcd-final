# Bounded learned semantic relation audit

R2 checkpoints 100/150. Only missing legal Known center features were encoded. The two heads per checkpoint used the same fixed, class-disjoint 78/20 Known split. All pairs are symmetric and unique within each split, balanced positive/negative, with half of the negative pairs mined from top-eight cosine neighbors of other Known classes. L2 uses fixed L2-regularized logistic regression. L3 uses the 1536→128→1 MLP, at most 25 epochs, with patience 4 on Known validation AUROC. All predictions, training manifests and model hashes were frozen before opening unlabeled labels.

All methods use the exact same frozen teacher-eligible E3 MOVE IDs and source slots, mass .2. Common scoring uses the original peer-backed destination slots and cached unlabeled evidence-pool peers. A frozen slot supported only by an unavailable peer has no common-pool score; the JSON includes the unchanged original E3 top-1 as a separate control. Expanded scoring uses available unlabeled global peers in alternative slots with at least two members. All methods use the same three-or-fewer cosine-shortlisted peers per slot and at most five destination ranks. Learned slot scores are max peer scores.

L0 destination ranking preserves the frozen E3 slot order where available. Expanded slots outside the frozen order receive tied evidence and are ordered by slot ID. Exact original two-trial dyadic scores were not exported, so L0 pairwise results use a one-trial reciprocal/cross-view proxy reconstructed from cached crops in the unlabeled evidence pool. Unsupported pairs score zero. Hard pair evaluation takes up to five highest-cosine different-class test peers per candidate after freezing scores. Pairwise estimates reuse candidate IDs and are descriptive.

| Epoch / mapping / pool / group | Method | Candidates / source-evaluable | Pair n | AUROC / AUPRC (%) | Hard AUROC / AUPRC (%) | Dest. top-1 (%) | R@1 / 3 / 5 on wrong source (%) | MOVE correct / correct→wrong (%) |
|---|---|---|---:|---|---|---:|---|---|
| 100 / hungarian / common / all | L0 | 16 / 16 | 27 | 56.1 / 28.7 | 56.1 / 28.7 | 12.5 | 25.0 / 25.0 / 25.0 | 12.5 / 50.0 |
| 100 / hungarian / common / all | L1 | 16 / 16 | 27 | 67.1 / 60.8 | 67.1 / 60.8 | 12.5 | 25.0 / 25.0 / 25.0 | 12.5 / 50.0 |
| 100 / hungarian / common / all | L2 | 16 / 16 | 27 | 68.6 / 61.6 | 68.6 / 61.6 | 12.5 | 25.0 / 25.0 / 25.0 | 12.5 / 50.0 |
| 100 / hungarian / common / all | L3 | 16 / 16 | 27 | 67.9 / 61.4 | 67.9 / 61.4 | 12.5 | 25.0 / 25.0 / 25.0 | 12.5 / 50.0 |
| 100 / hungarian / common / old | L0 | 3 / 3 | 6 | — / — | — / — | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 100 / hungarian / common / old | L1 | 3 / 3 | 6 | — / — | — / — | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 100 / hungarian / common / old | L2 | 3 / 3 | 6 | — / — | — / — | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 100 / hungarian / common / old | L3 | 3 / 3 | 6 | — / — | — / — | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 100 / hungarian / common / new | L0 | 13 / 13 | 21 | 27.5 / 4.8 | 27.5 / 4.8 | 7.7 | 14.3 / 14.3 / 14.3 | 7.7 / 46.2 |
| 100 / hungarian / common / new | L1 | 13 / 13 | 21 | 35.0 / 7.1 | 35.0 / 7.1 | 7.7 | 14.3 / 14.3 / 14.3 | 7.7 / 46.2 |
| 100 / hungarian / common / new | L2 | 13 / 13 | 21 | 35.0 / 7.1 | 35.0 / 7.1 | 7.7 | 14.3 / 14.3 / 14.3 | 7.7 / 46.2 |
| 100 / hungarian / common / new | L3 | 13 / 13 | 21 | 30.0 / 6.7 | 30.0 / 6.7 | 7.7 | 14.3 / 14.3 / 14.3 | 7.7 / 46.2 |
| 100 / hungarian / expanded / all | L0 | 16 / 16 | 9104 | 53.2 / 2.6 | 47.7 / 42.4 | 12.5 | 25.0 / 25.0 / 25.0 | 12.5 / 50.0 |
| 100 / hungarian / expanded / all | L1 | 16 / 16 | 9104 | 85.4 / 16.7 | 21.8 / 35.8 | 12.5 | 25.0 / 50.0 / 62.5 | 12.5 / 50.0 |
| 100 / hungarian / expanded / all | L2 | 16 / 16 | 9104 | 85.4 / 16.9 | 21.9 / 36.1 | 18.8 | 37.5 / 50.0 / 62.5 | 18.8 / 50.0 |
| 100 / hungarian / expanded / all | L3 | 16 / 16 | 9104 | 86.4 / 16.8 | 21.8 / 35.8 | 12.5 | 25.0 / 50.0 / 62.5 | 12.5 / 50.0 |
| 100 / hungarian / expanded / old | L0 | 3 / 3 | 1707 | 68.2 / 36.8 | 68.2 / 63.3 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 100 / hungarian / expanded / old | L1 | 3 / 3 | 1707 | 87.7 / 67.3 | 63.6 / 77.7 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 100 / hungarian / expanded / old | L2 | 3 / 3 | 1707 | 87.5 / 67.6 | 63.6 / 77.7 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 100 / hungarian / expanded / old | L3 | 3 / 3 | 1707 | 89.6 / 67.4 | 63.6 / 77.7 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 100 / hungarian / expanded / new | L0 | 13 / 13 | 7397 | 49.9 / 0.7 | 43.1 / 43.5 | 7.7 | 14.3 / 14.3 / 14.3 | 7.7 / 46.2 |
| 100 / hungarian / expanded / new | L1 | 13 / 13 | 7397 | 84.5 / 7.2 | 12.2 / 28.0 | 7.7 | 14.3 / 42.9 / 57.1 | 7.7 / 46.2 |
| 100 / hungarian / expanded / new | L2 | 13 / 13 | 7397 | 84.8 / 7.2 | 12.2 / 28.0 | 15.4 | 28.6 / 42.9 / 57.1 | 15.4 / 46.2 |
| 100 / hungarian / expanded / new | L3 | 13 / 13 | 7397 | 85.5 / 7.4 | 12.2 / 28.0 | 7.7 | 14.3 / 42.9 / 57.1 | 7.7 / 46.2 |
| 100 / legal_old / common / old | L0 | 3 / 2 | 6 | — / — | — / — | 0.0 | — / — / — | 0.0 / 100.0 |
| 100 / legal_old / common / old | L1 | 3 / 2 | 6 | — / — | — / — | 0.0 | — / — / — | 0.0 / 100.0 |
| 100 / legal_old / common / old | L2 | 3 / 2 | 6 | — / — | — / — | 0.0 | — / — / — | 0.0 / 100.0 |
| 100 / legal_old / common / old | L3 | 3 / 2 | 6 | — / — | — / — | 0.0 | — / — / — | 0.0 / 100.0 |
| 100 / legal_old / expanded / old | L0 | 3 / 2 | 1707 | 68.2 / 36.8 | 68.2 / 63.3 | 0.0 | — / — / — | 0.0 / 100.0 |
| 100 / legal_old / expanded / old | L1 | 3 / 2 | 1707 | 87.7 / 67.3 | 63.6 / 77.7 | 0.0 | — / — / — | 0.0 / 100.0 |
| 100 / legal_old / expanded / old | L2 | 3 / 2 | 1707 | 87.5 / 67.6 | 63.6 / 77.7 | 0.0 | — / — / — | 0.0 / 100.0 |
| 100 / legal_old / expanded / old | L3 | 3 / 2 | 1707 | 89.6 / 67.4 | 63.6 / 77.7 | 0.0 | — / — / — | 0.0 / 100.0 |
| 150 / hungarian / common / all | L0 | 10 / 10 | 26 | 81.9 / 63.9 | 81.9 / 63.9 | 10.0 | 20.0 / 20.0 / 20.0 | 10.0 / 40.0 |
| 150 / hungarian / common / all | L1 | 10 / 10 | 26 | 56.9 / 38.4 | 56.9 / 38.4 | 10.0 | 20.0 / 20.0 / 20.0 | 10.0 / 40.0 |
| 150 / hungarian / common / all | L2 | 10 / 10 | 26 | 56.9 / 38.4 | 56.9 / 38.4 | 10.0 | 20.0 / 20.0 / 20.0 | 10.0 / 40.0 |
| 150 / hungarian / common / all | L3 | 10 / 10 | 26 | 55.6 / 37.4 | 55.6 / 37.4 | 10.0 | 20.0 / 20.0 / 20.0 | 10.0 / 40.0 |
| 150 / hungarian / common / old | L0 | 3 / 3 | 6 | 75.0 / 83.3 | 75.0 / 83.3 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 33.3 |
| 150 / hungarian / common / old | L1 | 3 / 3 | 6 | 50.0 / 81.7 | 50.0 / 81.7 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 33.3 |
| 150 / hungarian / common / old | L2 | 3 / 3 | 6 | 50.0 / 81.7 | 50.0 / 81.7 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 33.3 |
| 150 / hungarian / common / old | L3 | 3 / 3 | 6 | 50.0 / 81.7 | 50.0 / 81.7 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 33.3 |
| 150 / hungarian / common / new | L0 | 7 / 7 | 20 | 93.8 / 66.7 | 93.8 / 66.7 | 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 42.9 |
| 150 / hungarian / common / new | L1 | 7 / 7 | 20 | 65.6 / 34.3 | 65.6 / 34.3 | 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 42.9 |
| 150 / hungarian / common / new | L2 | 7 / 7 | 20 | 65.6 / 34.3 | 65.6 / 34.3 | 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 42.9 |
| 150 / hungarian / common / new | L3 | 7 / 7 | 20 | 62.5 / 32.5 | 62.5 / 32.5 | 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 42.9 |
| 150 / hungarian / expanded / all | L0 | 10 / 10 | 5690 | 53.3 / 5.4 | 51.3 / 55.4 | 10.0 | 20.0 / 20.0 / 20.0 | 10.0 / 50.0 |
| 150 / hungarian / expanded / all | L1 | 10 / 10 | 5690 | 83.1 / 16.6 | 21.8 / 41.9 | 20.0 | 40.0 / 60.0 / 60.0 | 20.0 / 50.0 |
| 150 / hungarian / expanded / all | L2 | 10 / 10 | 5690 | 82.7 / 16.8 | 21.8 / 41.9 | 20.0 | 40.0 / 60.0 / 60.0 | 20.0 / 50.0 |
| 150 / hungarian / expanded / all | L3 | 10 / 10 | 5690 | 83.7 / 16.8 | 21.9 / 41.9 | 20.0 | 40.0 / 60.0 / 60.0 | 20.0 / 50.0 |
| 150 / hungarian / expanded / old | L0 | 3 / 3 | 1707 | 60.0 / 20.5 | 60.0 / 52.0 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 150 / hungarian / expanded / old | L1 | 3 / 3 | 1707 | 87.3 / 31.1 | 34.0 / 48.8 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 150 / hungarian / expanded / old | L2 | 3 / 3 | 1707 | 85.2 / 31.8 | 35.3 / 49.8 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 150 / hungarian / expanded / old | L3 | 3 / 3 | 1707 | 85.2 / 24.9 | 33.3 / 43.1 | 33.3 | 100.0 / 100.0 / 100.0 | 33.3 / 66.7 |
| 150 / hungarian / expanded / new | L0 | 7 / 7 | 3983 | 52.0 / 3.2 | 49.1 / 58.5 | 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 42.9 |
| 150 / hungarian / expanded / new | L1 | 7 / 7 | 3983 | 82.1 / 15.9 | 17.3 / 44.0 | 14.3 | 25.0 / 50.0 / 50.0 | 14.3 / 42.9 |
| 150 / hungarian / expanded / new | L2 | 7 / 7 | 3983 | 82.0 / 16.2 | 16.6 / 43.8 | 14.3 | 25.0 / 50.0 / 50.0 | 14.3 / 42.9 |
| 150 / hungarian / expanded / new | L3 | 7 / 7 | 3983 | 83.2 / 16.4 | 17.4 / 44.0 | 14.3 | 25.0 / 50.0 / 50.0 | 14.3 / 42.9 |
| 150 / legal_old / common / old | L0 | 3 / 2 | 6 | 75.0 / 83.3 | 75.0 / 83.3 | 0.0 | — / — / — | 0.0 / 50.0 |
| 150 / legal_old / common / old | L1 | 3 / 2 | 6 | 50.0 / 81.7 | 50.0 / 81.7 | 0.0 | — / — / — | 0.0 / 50.0 |
| 150 / legal_old / common / old | L2 | 3 / 2 | 6 | 50.0 / 81.7 | 50.0 / 81.7 | 0.0 | — / — / — | 0.0 / 50.0 |
| 150 / legal_old / common / old | L3 | 3 / 2 | 6 | 50.0 / 81.7 | 50.0 / 81.7 | 0.0 | — / — / — | 0.0 / 50.0 |
| 150 / legal_old / expanded / old | L0 | 3 / 2 | 1707 | 60.0 / 20.5 | 60.0 / 52.0 | 0.0 | — / — / — | 0.0 / 100.0 |
| 150 / legal_old / expanded / old | L1 | 3 / 2 | 1707 | 87.3 / 31.1 | 34.0 / 48.8 | 0.0 | — / — / — | 0.0 / 100.0 |
| 150 / legal_old / expanded / old | L2 | 3 / 2 | 1707 | 85.2 / 31.8 | 35.3 / 49.8 | 0.0 | — / — / — | 0.0 / 100.0 |
| 150 / legal_old / expanded / old | L3 | 3 / 2 | 1707 | 85.2 / 24.9 | 33.3 / 43.1 | 0.0 | — / — / — | 0.0 / 100.0 |

The global Hungarian mapping is a sealed one-to-one alignment on the same unlabeled panel, so class splits, merges and unmatched slots limit interpretation. The Old anchor mapping uses only legal Known labels, drops zero-support assignments and excludes unmapped source slots from source-conditioned Old rates. Unmapped targets receive no correctness credit. Novel is evaluated only with sealed Hungarian. This is a destination-ranking diagnostic, with no virtual update or accuracy claim.

- Epoch 100 — 3124 training pairs, 800 validation pairs, MLP selected epoch 1, 98 legally anchored slots. Actual E3 aligned top-1 2/16.
- Epoch 150 — 3124 training pairs, 800 validation pairs, MLP selected epoch 1, 98 legally anchored slots. Actual E3 aligned top-1 1/10.

[Frozen prediction provenance and full metrics](r2_learned_relation.json)
