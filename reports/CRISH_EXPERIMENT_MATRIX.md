# CRiSH experiment matrix

| Training run | Seed | Observed endpoint | All / Old / New (%) | Evidence |
|---|---:|---:|---|---|
| D1 | 1 | 200 | 60.35 / 72.16 / 54.65 | [Final archive audit](CRISH_D1_FINAL_AUDIT.md) |
| D1-Random / D1-DINO | 1 | — | — | [Matched-exposure design](CRISH_D1_ABLATION_DESIGN.md) |
| E0 | 1 | 200 | 55.91 / 72.96 / 47.67 | [Archive and frozen audit](CRISH_E0_BASELINE_AUDIT.md) |
| E3 | 1 | 200 | 56.51 / 71.51 / 49.26 | [Global verifier](CRISH_SIMGCD_GLOBAL.md) |
| R2 | 1 | 150 | 55.21 / 70.66 / 47.74 | [R2 mechanistic audit](CRISH_RELATIONAL_MECHANISTIC_AUDIT.md) |
| E3-F | 1 | 200 | 56.49 / 73.36 / 48.35 | [Manual aggregates only](E3F_RECOVERED_RESULTS.md) |

| Diagnostic | Starting checkpoint or feature bank | Comparison |
|---|---|---|
| [D1 final audit](CRISH_D1_FINAL_AUDIT.md) | D1-100/150/200, E0-200, R2-100/150 | Archive continuity, frozen geometry/readouts, locked correction and retrospective exposure cohorts |
| [Frozen E0 benchmark](CRISH_E0_BASELINE_AUDIT.md) | E0-200, R2-100/150 | v2, geometry, readouts, DINO/CLIP and matched offline correction |
| [G6 confirmation](CRISH_VLM_HELDOUT_CONFIRMATION.md) | R2-100/150 | Locked detector/retriever policies and fixed budgets |
| [Oracle trajectory](CRISH_ORACLE_MOVE_FIVE_EPOCHS.md) | R2-100 | Oracle MOVE, KEEP and matched random |
| [CLIP continuation](CRISH_CLIP_REPAIR_TRAINING.md) | R2-150 | Direct MOVE, KEEP and matched random |
| [MOVE strength](CRISH_MOVE_STRENGTH_TRAINING.md) | R2-150 | Paired mass .2/.8 and random |
| [Early continuation](CRISH_SUSTAINED_MOVE_TRAINING.md) | R2-100 | Fixed proposals through epoch 105 |
| [Dynamic refresh](CRISH_DYNAMIC_MOVE_TRAINING.md) | Shared epoch 102 | Dynamic versus fixed with observed exposure |
| [Soft targets](CRISH_SOFT_TARGET_TRAINING.md) | Shared epoch 102 | Matched hard and top-three soft MOVE |

Training endpoints and frozen checkpoints record different comparisons. E3-F lacks raw trajectory files. The frozen benchmark compares checkpoint-conditioned geometry and cannot establish a causal method effect.
