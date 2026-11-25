# Full Stanford Cars experiments

Run from the root of branch `feat/crish-simgcd` on a Linux CUDA host with Bash, util-linux `flock`, and the SimGCD dependencies installed. The default suite uses seed 1. Each mode starts from the same pretrained DINOv1 ViT-B/16 and classifier initialization, runs 200 complete epochs with the Cars batch-128 protocol, evaluates Old/New/All with v2 each epoch, and saves the final epoch checkpoint. E0 is the existing official reproduction. Check its split, preprocessing, seed, evaluation and epoch before comparison.

| Run directory suffix | Policy |
| --- | --- |
| E4-adaptive | Primary verified global method, cap 8. Equivalent policy to the standalone E3 cap-8 Cars launcher. |
| E3-fixed | Same verifier, cap 1. A veto permits zero actions. Same nine disjoint probe reservations as E4 for a controlled comparison. |
| E1-random | Randomize the eligible boundary shortlist, then randomly select an eligible action of the reference type. |
| E2-evidence | Use global evidence ranking and choose the highest-ranked eligible action of the reference type. |

E1/E2 reuse the global snapshot, two evidence trials, peer batch, eligibility, stale-teacher guard, mass and persistence. Their training omits virtual updates and confirmation. Each epoch borrows the number and type of **planned** actions from the completed cap-1 E3 ledger. A separate per-type ceiling limits **applied** occurrences to E3's realized exposure; the sampler, stale checks and skipped optimizer steps can yield fewer. The ledger records both `plan_matched` and `exposure_matched`. The final summary lists unmatched epochs. A run with shortfalls supports a capped-dose comparison; describing it as an exact dose match would overstate the evidence.

## Launch and resume

Configure `SIMGCD_CARS_ROOT` to contain `cars_train/` and `devkit/{cars_train_annos.mat,cars_test_annos_withlabels.mat}`. The repo ships the official Cars SSB split. The evaluation uses held-out labels from the unlabeled training images. By default the trainer loads the official DINOv1 model through torch.hub. For an offline server, set both `SIMGCD_DINO_REPO` (official DINO source checkout) and `SIMGCD_DINO_WEIGHTS` (pretrained ViT-B/16 state dict).

~~~bash
export SIMGCD_CARS_ROOT=/data/stanford-cars
export SIMGCD_RUN_ROOT=/runs/simgcd-crish-cars
bash scripts/run_cars_suite.sh --dry-run
bash scripts/run_cars_suite.sh
~~~

Rerun the same command after an interruption. A per-seed lock prevents two launchers writing the same outputs. Each atomic epoch checkpoint contains model, SGD, scheduler, optional AMP scaler, RNG states and the committed length of the repair ledger. Recovery discards incomplete ledger bytes after that checkpoint and repeats the interrupted epoch. Settings, split IDs, initial backbone weights and the contents of the dose reference are checked for compatibility. Reusing a completed checkpoint skips its training loop. To run another seed, set `SIMGCD_SEED`.

The suite prints final metrics, accepted actions, applied occurrences and unmatched epochs. Reprint after completion with `python scripts/summarize_cars_suite.py "$SIMGCD_RUN_ROOT/simgcd/log" 1`. Per-epoch logs are under each mode's `checkpoints/crish-global.jsonl`; upstream evaluation output is in `log.txt`.

## Validation scope

`python validate_cars_suite.py` exercises the actual training loop for three small synthetic epochs per mode on CPU with an evaluation stub. It checks real selection in each control, shared probe reservations, the applied dose ceiling, rejection of a wrong seed, and checkpoint recovery after a simulated interruption with a partial ledger line. The recovered model must equal the uninterrupted model bit for bit. This fixture and the four-command Bash dry run were executed during implementation. Full Cars training and validation on the destination GPU remain pending.
