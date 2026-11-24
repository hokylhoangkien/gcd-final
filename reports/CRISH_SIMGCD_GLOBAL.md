# Global counterfactual supervision repair on Stanford Cars

## Revisions and run configuration

Primary method E3. Source implementation: hokylhoangkien/gcd-final at `1c6ada810b451e5a1f3b36b930c550489917cf0a` (feat/local-adaptive-acceptance). Previous SimGCD port: `6af77dc3220d2abb8033fc67ab9bc8a5695394db`. Official SimGCD base: `f843f27195701b7c8c5713cfd17499ad67c86247`.

The Cars launcher uses the official DINOv1 ViT-B/16, trainable block 11, two views, 200 epochs, mixed labeled/unlabeled training batches of 128, SGD and the official Cars hyperparameters and evaluation. The repair schedule starts at zero-based epoch 10, then runs once before training in each eligible epoch. The frozen snapshot covers the training split. Each refresh has a 300-second wall-clock budget including snapshot, scoring and confirmation.

| Component | Previous SimGCD port | Current E3 and relationship to source |
| --- | --- | --- |
| Snapshot and shortlist | Current 128-row training batch | Deterministic center-crop normalized backbone feature per training ID, single eval-mode pass. Exclude all probe IDs, sample unlabeled anchors and search the entire remaining snapshot in blocks. Reuse the source global_neighbor_pool, which reserves the anchors, then neighbors, then fills to 512 IDs. |
| Evidence | One two-view trial, batch-local graph | Two independently augmented two-view trials over all 512 cached IDs. Exact Euclidean 5-NN per view. Use the source joint fraction for reciprocal edges that persist across both views, producing weights of 0, 0.5 or 1. The final measured graph has cache-local neighbors. |
| Boundaries and slots | At most 16 boundaries, local class counts | At most 64 unlabeled boundary IDs by measured (cross − within) / degree. Flat global center-crop classifier assignments replace SelEx cluster labels. Offer ABSTAIN and up to one supported MOVE. Destination is the slot with the largest measured reciprocal support; ties use class index. Require at least three source and two destination rows before a MOVE. |
| Peer-aware scoring | Remaining rows of a single batch | Reuse source scoring_ids. Select all candidate IDs first, then their nearest source/destination peers, measured neighbors and fill to 256. Prefer cached views; encode missing peers with the current frozen backbone. Exclude every probe ID. Reserve a legal known anchor if the proposed score batch contains none. |
| Independent probes | Small folds carved out of the live mini-batch | Reserve discovery plus eight unused confirmation folds globally. Every fold has 32 known (16 legally labeled classes × 2 distinct images) and 64 unlabeled IDs. The same known classes recur across folds with disjoint images. Use the source class-balanced sampler. Encode each needed fold once with fresh two-view crops. |
| Counterfactual | Factual and edited virtual classifier SGD | Same weight-normalized classifier parameter, copied live SGD momentum, weight decay and current LR. Evaluate identical frozen probe targets on factual and alternative updates from the same state. Discovery and fresh confirmation must both improve total proxy and preserve known CE, versus both the original and already-accepted plan. |
| Adaptive quantity | E3 ceiling of one | Up to 8 accepted IDs, up to 8 confirmations. Stop on a veto, insufficient data, cap or timeout. Retain confirmed prefix when the deadline expires. A confirmation finishing after the deadline cannot add a new action. |
| Persistence | One scheduled optimizer step | Bind accepted actions to stable IDs for exactly the current epoch. Every sampled occurrence is checked against current two-view teacher probabilities before application. Reject stale source or insufficient transferable mass. Clear plan at epoch close. Repeated sampler IDs have independently checked exposure. |

The snapshot classifier sees only images. Training membership and known targets come from SimGCD's training split; ImageOnly returns a sentinel for unlabeled images. The evaluator remains separate. All snapshot passes run in eval mode with gradient recording disabled on the backbone. Python, NumPy, PyTorch RNG and module modes are restored after the refresh. The verifier uses the frozen functional classifier rather than executing a virtual update on the live network.

For a proposed set A, take one virtual classifier SGD step on the 256 scoring rows. The classifier-dependent objective combines supervised CE, official distillation with teacher edit A and memax. On a probe, J is legal known CE plus opposite-view CE to frozen pre-update teacher predictions. Positive gain means J(factual) − J(counterfactual) exceeds the configured threshold (default zero). No edit changes a probe target or denominator. Novel-class semantic correctness is unobserved. Greedy rounds recompute cumulative gradients, enforce distinct IDs and recheck support against any changed peer slot.

For the actual update, KEEP calls official DistillLoss when there are no active edits. ABSTAIN zeros both cross-view terms for that row with the original denominator. MOVE transfers 0.2 detached probability mass from source to destination in each view. The live source must equal the argmax of the mean of both sharpened teacher distributions. A failed check skips that occurrence; another crop later in the epoch can pass. All other real training loss terms remain the SimGCD losses. No hierarchy or SelEx projector is introduced.

## Entry points and records

- `train.py` starts and closes EpochRepairs for E3. Sampling and training still use the official weighted loader. Historical E1/E2/E4 paths were outside this change; their previous dose format is specific to the previous revision.
- `crish/global_repair.py` implements snapshot extraction, candidate generation, paired discovery/confirmation and per-occurrence application. `crish/sampling.py` and `crish/scoring_batch.py` are carried over from the pinned source. `crish/runtime.py` defines the CLI profile.
- `checkpoints/crish-global.jsonl` writes one refresh record and one exposure record per eligible epoch. Records include score/evidence/probe IDs, support, gain traces, accepted actions, stop reason and synchronized stage timings. Exposure includes encountered rows, rejected stale or low-mass rows, actual optimizer steps, applied occurrences by action type and unique applied IDs. Missing count keys mean zero. Phase boundaries check the deadline between operations; an in-flight encoder batch or worker join can cause wall-clock overshoot.
- `validate_global_crish.py` checks fold separation, unchanged live state and RNG, natural confirmation, retention on a simulated deadline, repeated occurrences, stale assignments, skipped optimizer steps and epoch expiry.
- `benchmark_crish_cars.py` invokes the real trainer for 3–5 refreshes and a bounded number of real joint SGD steps per refresh. It skips accuracy evaluation and checkpoint writes in this benchmark mode. A supplied official checkpoint restores the model, epoch and optimizer. Without one, it uses pretrained DINO and a newly initialized classifier at the warmup-boundary schedule index; this measures compute and mechanics only.

## Reproduction

From the repository root, with an official-layout Cars directory and the SimGCD dependencies installed:

~~~bash
python validate_global_crish.py
python benchmark_crish_cars.py --cars /data/cars --checkpoint /path/to/post-warmup/model.pt \
  --output /runs/cars-budget --refreshes 3 --steps 4
~~~

Supply `--dino-repo` and `--dino-weights` to use a local official backbone checkout and weight file. Use a fresh output directory per run. Four batches observe 512 sampler draws per refresh. For exposure over a complete Cars epoch use `--steps 63`. The future full experiment command is `bash scripts/run_cars_crish.sh` after configuring the official dataset paths. No full experiment is part of this benchmark.
