# Bounded expanded-destination verifier diagnostic

R2 checkpoints 100/150 and the 26 previously frozen MOVE IDs. D0 uses original E3 target. D1, D2 and D3 use global cosine top 1, 3 and 5. Each ID shares its scoring context, KEEP, and fresh discovery and confirmation folds across arms. MOVE mass .2 and full ABSTAIN retain the original divisor. Unchanged E3 source and two-view source-mass checks are repeated on fresh scoring crops. Other IDs and supported destination sets come from frozen receipts. V0 ranks positive discovery gains with strict known CE, then checks only the winner on the independent confirmation fold. Rejection returns KEEP.

This diagnostic forces the listed candidates into a fresh scoring context. It tests one action per source against shared KEEP and does not apply multiple simultaneous edits. Full joint virtual SGD uses original SimGCD including contrastive gradients, saved optimizer state, and no R2 loss. Prefix tokens are cached within each batch and differential backprop touches only the two edited rows. The unchanged functional V0 step checks that joint classifier parameters agree. Proxy inputs, scores, eligibility and all non-oracle decisions are frozen before any hidden label access. The sealed score uses a fixed center-feature bank and held-out center queries.

D4 is the expected uniform control over destinations sharing each policy's accepted source ID and action kind, preserving intervention count and MOVE mass. In the mixed comparison it therefore also preserves the MOVE/ABSTAIN mix. A second D4 control averages uniformly over eligible source IDs of that kind in the same scoring batch, then over their destinations. D5 chooses the best positive realized semantic gain among semantically valid D3 actions, otherwise KEEP. It uses hidden labels and bypasses the guard, so it is an upper-reference diagnostic rather than a deployable policy. The matched oracle column holds each policy's IDs, action kinds and count fixed, preferring a class-correct alternative when available. A single global Hungarian mapping per checkpoint is fitted only in the sealed stage; unmatched slots receive no correctness credit.

| Mode / policy | MOVE / ABSTAIN accepted | Harmful / accepted | Correct destination in pool among wrong-source eligible IDs | Discovery destination correct / MOVE | Selected destination correct / MOVE | Correct→wrong MOVE | Gain All / Old / New ×10⁶ | D4 same ID | D4 random ID | Matched oracle gap |
|---|---|---|---|---|---|---:|---|---|---|---|
| move / D0 | 2 / 0 | 1 / 2 | 3 / 13 | 2 / 10 | 0 / 2 | 1 | -0.096 / -0.189 / -0.050 | -0.096 / -0.189 / -0.050 | -0.099 / +0.681 / -0.606 | +0.000 / +0.000 / +0.000 |
| move / D1 | 1 / 0 | 1 / 1 | 4 / 13 | 2 / 9 | 0 / 1 | 1 | -0.165 / -0.002 / -0.272 | -0.165 / -0.002 / -0.272 | -0.106 / +0.516 / -0.517 | +0.000 / +0.000 / +0.000 |
| move / D2 | 2 / 0 | 0 / 2 | 7 / 13 | 1 / 14 | 0 / 2 | 1 | +0.435 / +2.119 / -0.598 | +0.064 / +0.142 / +0.020 | -0.223 / +0.292 / -0.559 | -0.357 / -2.328 / +0.825 |
| move / D3 | 2 / 0 | 1 / 2 | 8 / 13 | 2 / 18 | 0 / 2 | 1 | +0.027 / +0.933 / -0.490 | -0.021 / -0.240 / +0.114 | -0.180 / +0.319 / -0.507 | +0.050 / -1.142 / +0.717 |
| move / D5 | 5 / 0 | 0 / 5 | 8 / 13 | 5 / 5 | 5 / 5 | 0 | +1.055 / +3.331 / -0.361 | +0.158 / +2.020 / -0.955 | -0.422 / +0.713 / -1.092 | +0.000 / +0.000 / +0.000 |
| mixed / D0 | 2 / 0 | 1 / 2 | 3 / 13 | 1 / 7 | 0 / 2 | 1 | -0.096 / -0.189 / -0.050 | -0.096 / -0.189 / -0.050 | -0.099 / +0.681 / -0.606 | +0.000 / +0.000 / +0.000 |
| mixed / D1 | 1 / 1 | 2 / 2 | 4 / 13 | 1 / 6 | 0 / 1 | 1 | -0.348 / -1.010 / +0.039 | -0.348 / -1.010 / +0.039 | -0.075 / +0.501 / -0.460 | +0.000 / +0.000 / +0.000 |
| mixed / D2 | 2 / 0 | 0 / 2 | 7 / 13 | 1 / 14 | 0 / 2 | 1 | +0.435 / +2.119 / -0.598 | +0.064 / +0.142 / +0.020 | -0.223 / +0.292 / -0.559 | -0.357 / -2.328 / +0.825 |
| mixed / D3 | 2 / 0 | 1 / 2 | 8 / 13 | 2 / 18 | 0 / 2 | 1 | +0.027 / +0.933 / -0.490 | -0.021 / -0.240 / +0.114 | -0.180 / +0.319 / -0.507 | +0.050 / -1.142 / +0.717 |
| mixed / D5 | 4 / 3 | 0 / 7 | 8 / 13 | 4 / 4 | 4 / 4 | 0 | +2.414 / +3.553 / +1.680 | +1.621 / +3.427 / +0.562 | -0.200 / +0.641 / -0.686 | +0.000 / +0.000 / +0.000 |

Gains average over all 26 source trials, with zero for a no-op. Old/New refer to the semantic probe class. Coverage counts are recorded before V0 screening. The discovery and selection columns show what survives ranking and independent confirmation. One-step semantic margins on two checkpoints do not establish full-training effects.

- Epoch 100 — 16 frozen IDs, fresh eligibility {'eligible': 16}, 96 non-KEEP virtual arms.
- Epoch 150 — 10 frozen IDs, fresh eligibility {'eligible': 10}, 60 non-KEEP virtual arms.
- move D0 — {'discovery': 16, 'confirmation_known_ce': 1, 'confirmation_gain': 7, 'accepted': 2}.
- move D1 — {'discovery': 17, 'confirmation_known_ce': 1, 'confirmation_gain': 7, 'accepted': 1}.
- move D2 — {'discovery': 12, 'confirmation_known_ce': 1, 'confirmation_gain': 11, 'accepted': 2}.
- move D3 — {'confirmation_known_ce': 2, 'discovery': 8, 'confirmation_gain': 14, 'accepted': 2}.
- move D5 — {'accepted': 5, 'oracle_keep': 21}.
- mixed D0 — {'discovery': 15, 'confirmation_known_ce': 1, 'confirmation_gain': 8, 'accepted': 2}.
- mixed D1 — {'discovery': 15, 'confirmation_known_ce': 1, 'confirmation_gain': 8, 'accepted': 2}.
- mixed D2 — {'discovery': 11, 'confirmation_known_ce': 1, 'confirmation_gain': 12, 'accepted': 2}.
- mixed D3 — {'confirmation_known_ce': 2, 'discovery': 8, 'confirmation_gain': 14, 'accepted': 2}.
- mixed D5 — {'accepted': 7, 'oracle_keep': 19}.

[Frozen prediction hashes and per-arm sealed measurements](r2_destination_diagnostic.json)
