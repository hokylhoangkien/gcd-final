# Cross-geometry classifier counterfactual on Stanford Cars

R2-150 only. The 1,024 fresh query IDs exclude G1, held-out G6, Oracle MOVE and frozen E3 proposal IDs. The reference bank, discovery, confirmation, scoring context and independent sealed probe are disjoint from the candidate query pool and each other. Candidate labels are opened only to score the frozen edits. An independent 64-image probe measures transfer of the virtual update.

CLIP locks the top 10% by alternative-minus-source support and fixes the best alternative slot. All policies share the exact pair and move 0.2 teacher mass. Both views must have enough mass and their mean teacher argmax must match the frozen source. Direct retains detection order. CLIPConfidence ranks by the gap to the next alternative. StaticD2 ranks the proposed slot's support minus source support in DINOv2.

V0 reuses the original weight-normalized classifier SGD and opposite-view teacher probe. CrossCF changes only the unlabeled target: the ten nearest DINOv2 reference images contribute frozen SimGCD slot distributions, weighted by softmax cosine at temperature .07. Both require positive total gain and nonincreasing known CE on discovery and confirmation. All KEEP/MOVE steps start from the same saved classifier and SGD momentum. Candidates are evaluated one edit at a time from that common state; ranking and capping do not simulate a sequence of updates. Confirmation is ID-disjoint from discovery, shared across candidates and descriptive under repeated screening.

Net gain counts slot corrections on the full query panel. Pointwise 95% paired query bootstrap intervals condition on the fixed reference and flags. Random filtering averages 64 frozen matched-count draws. Counts below cap are reported without padding. Matched tiers omit policies unable to fill that tier. Candidate ranking uses sealed repair correctness and transfer is mean true-slot log-probability gain on the independent probe. These are offline slot edits and one virtual classifier step, with no measurement of training effects.

CLIP proposed 102 edits; 102 were eligible. V0 passed 2, CrossCF passed 4.

| Tier / group / policy | Edits | Repairs / harmful / wrong→wrong | Net gain pp [95% CI] | Δ vs random pp [95% CI] |
|---|---:|---|---|---|
| cap_25 / All / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — |
| cap_25 / All / Direct | 25 | 14 / 0 / 11 | +1.37 [+0.68, +2.15] | +0.85 [+0.30, +1.45] |
| cap_25 / All / CLIPConfidence | 25 | 11 / 5 / 9 | +0.59 [-0.20, +1.37] | +0.07 [-0.57, +0.68] |
| cap_25 / All / V0 | 2 | 0 / 2 / 0 | -0.20 [-0.49, +0.00] | -0.24 [-0.54, -0.04] |
| cap_25 / All / CrossCF | 4 | 1 / 2 / 1 | -0.10 [-0.39, +0.20] | -0.18 [-0.50, +0.14] |
| cap_25 / All / StaticD2 | 25 | 11 / 2 / 12 | +0.88 [+0.20, +1.66] | +0.36 [-0.17, +0.97] |
| cap_25 / Old / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — |
| cap_25 / Old / Direct | 4 | 3 / 0 / 1 | +0.98 [+0.00, +2.28] | +0.93 [-0.03, +2.01] |
| cap_25 / Old / CLIPConfidence | 10 | 4 / 3 / 3 | +0.33 [-1.30, +1.95] | +0.27 [-1.08, +1.60] |
| cap_25 / Old / V0 | 1 | 0 / 1 / 0 | -0.33 [-0.98, +0.00] | -0.35 [-1.04, +0.03] |
| cap_25 / Old / CrossCF | 2 | 0 / 1 / 1 | -0.33 [-0.98, +0.00] | -0.33 [-0.99, +0.08] |
| cap_25 / Old / StaticD2 | 3 | 1 / 1 / 1 | +0.00 [-0.98, +0.98] | -0.05 [-0.91, +0.83] |
| cap_25 / New / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — |
| cap_25 / New / Direct | 21 | 11 / 0 / 10 | +1.53 [+0.70, +2.51] | +0.82 [+0.15, +1.55] |
| cap_25 / New / CLIPConfidence | 15 | 7 / 2 / 6 | +0.70 [-0.14, +1.53] | -0.02 [-0.70, +0.70] |
| cap_25 / New / V0 | 1 | 0 / 1 / 0 | -0.14 [-0.42, +0.00] | -0.20 [-0.50, -0.04] |
| cap_25 / New / CrossCF | 2 | 1 / 1 / 0 | +0.00 [-0.42, +0.42] | -0.11 [-0.50, +0.28] |
| cap_25 / New / StaticD2 | 22 | 10 / 1 / 11 | +1.26 [+0.42, +2.23] | +0.54 [-0.13, +1.30] |
| cap_50 / All / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — |
| cap_50 / All / Direct | 50 | 23 / 4 / 23 | +1.86 [+0.88, +2.83] | +0.90 [+0.26, +1.56] |
| cap_50 / All / CLIPConfidence | 50 | 22 / 12 / 16 | +0.98 [-0.10, +2.15] | +0.02 [-0.63, +0.67] |
| cap_50 / All / V0 | 2 | 0 / 2 / 0 | -0.20 [-0.49, +0.00] | -0.24 [-0.54, -0.04] |
| cap_50 / All / CrossCF | 4 | 1 / 2 / 1 | -0.10 [-0.39, +0.20] | -0.18 [-0.50, +0.14] |
| cap_50 / All / StaticD2 | 50 | 23 / 5 / 22 | +1.76 [+0.78, +2.73] | +0.80 [+0.16, +1.48] |
| cap_50 / Old / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — |
| cap_50 / Old / Direct | 12 | 5 / 1 / 6 | +1.30 [+0.00, +2.93] | +1.19 [+0.02, +2.41] |
| cap_50 / Old / CLIPConfidence | 17 | 6 / 6 / 5 | +0.00 [-2.28, +1.96] | -0.11 [-1.38, +1.11] |
| cap_50 / Old / V0 | 1 | 0 / 1 / 0 | -0.33 [-0.98, +0.00] | -0.35 [-1.04, +0.03] |
| cap_50 / Old / CrossCF | 2 | 0 / 1 / 1 | -0.33 [-0.98, +0.00] | -0.33 [-0.99, +0.08] |
| cap_50 / Old / StaticD2 | 13 | 7 / 3 / 3 | +1.30 [-0.65, +3.26] | +1.19 [-0.05, +2.43] |
| cap_50 / New / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — |
| cap_50 / New / Direct | 38 | 18 / 3 / 17 | +2.09 [+0.97, +3.35] | +0.78 [+0.04, +1.60] |
| cap_50 / New / CLIPConfidence | 33 | 16 / 6 / 11 | +1.39 [+0.14, +2.79] | +0.08 [-0.65, +0.85] |
| cap_50 / New / V0 | 1 | 0 / 1 / 0 | -0.14 [-0.42, +0.00] | -0.20 [-0.50, -0.04] |
| cap_50 / New / CrossCF | 2 | 1 / 1 / 0 | +0.00 [-0.42, +0.42] | -0.11 [-0.50, +0.28] |
| cap_50 / New / StaticD2 | 37 | 16 / 2 / 19 | +1.95 [+0.84, +3.21] | +0.64 [-0.10, +1.45] |
| matched_2 / All / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — |
| matched_2 / All / Direct | 2 | 2 / 0 / 0 | +0.20 [+0.00, +0.49] | +0.15 [-0.05, +0.45] |
| matched_2 / All / CLIPConfidence | 2 | 2 / 0 / 0 | +0.20 [+0.00, +0.49] | +0.15 [-0.06, +0.45] |
| matched_2 / All / V0 | 2 | 0 / 2 / 0 | -0.20 [-0.49, +0.00] | -0.24 [-0.54, -0.04] |
| matched_2 / All / CrossCF | 2 | 0 / 1 / 1 | -0.10 [-0.29, +0.00] | -0.15 [-0.35, -0.03] |
| matched_2 / All / StaticD2 | 2 | 2 / 0 / 0 | +0.20 [+0.00, +0.49] | +0.15 [-0.05, +0.45] |
| matched_4 / All / KEEP | 0 | 0 / 0 / 0 | +0.00 [+0.00, +0.00] | — |
| matched_4 / All / Direct | 4 | 3 / 0 / 1 | +0.29 [+0.00, +0.68] | +0.22 [-0.07, +0.57] |
| matched_4 / All / CLIPConfidence | 4 | 3 / 1 / 0 | +0.20 [-0.20, +0.59] | +0.12 [-0.24, +0.52] |
| matched_4 / All / CrossCF | 4 | 1 / 2 / 1 | -0.10 [-0.39, +0.20] | -0.18 [-0.50, +0.14] |
| matched_4 / All / StaticD2 | 4 | 3 / 0 / 1 | +0.29 [+0.00, +0.68] | +0.22 [-0.07, +0.57] |

| Ranking rule (All) | Repair AUROC | Spearman net | Spearman independent gain |
|---|---:|---:|---:|
| CLIPDetection | 0.712 | 0.361 | 0.110 |
| CLIPConfidence | 0.645 | 0.086 | 0.078 |
| StaticD2 | 0.662 | 0.258 | -0.008 |
| V0 | 0.550 | 0.040 | -0.010 |
| CrossCF | 0.472 | -0.103 | 0.025 |

Runtime 10.0 seconds, including 1.3 seconds for virtual updates.

[Full counts, ranks and frozen prediction provenance](r2_cross_geometry_cf.json)
