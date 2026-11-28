# CRiSH-D1 asymmetric dual-backbone repair

D1 trains the original DINOv1 ViT-B/16 SimGCD model and uses a separate, frozen OpenAI CLIP ViT-B/16 image embedding for proposal evidence. The E0 audit established a useful distinction on the reused G6-confirm panel. CLIP had error AUROC .766 versus DINO .628 and wrong-source destination Recall@1 .276 versus .178. At 10% coverage the offline CLIP policy made 34 correct repairs and 7 harmful edits out of 102. These are fixed-checkpoint assignment results. They motivate testing a training intervention without establishing a training benefit. [E0 audit](CRISH_E0_BASELINE_AUDIT.md)

## Locked policy

| Setting | Value |
|---|---|
| Dataset and run | Stanford Cars, seed 1, SSB, labeled fraction .5 |
| Training | Batch 128, full SimGCD objective, SGD LR .1, momentum .9, weight decay .00005, original 200-epoch cosine to .0001 |
| Model | DINOv1 ViT-B/16, block 11 and original SimGCD head trainable |
| Refresh | After 10 completed epochs and before each remaining epoch, 300 seconds per refresh |
| Evidence | Frozen unit CLIP vectors, current center-view classifier slots |
| Candidate budget | At most floor(.10 × 6,144) = 614 IDs with a supported alternative and positive risk |
| Application budget | At most floor(.04 × 8,064) = 322 mixed-row occurrences per epoch and eight occurrences in a 128-row batch |
| MOVE | Transfer .2 teacher mass in each view |
| Selection | Use original weighted replacement sampler, rank eligible occurrences by locked CLIP risk and stable ID, and use row order for repeated occurrences |
| Evaluation | Official transductive v2 every epoch and manual guards at 100 and 150 |

Candidate coverage is exploratory. The local label-blind mass-.8 pilot produced edit gradients larger than KEEP gradients. The locked dose returns to E3's mass .2 and independently caps occurrence exposure. [Smoke measurements](CRISH_DUAL_BACKBONE_SMOKE.md)

At each refresh, a deterministic seed splits the 6,144 unlabeled training IDs in half. In fold 1 the first half queries the second. Fold 2 reverses the roles. Each query and its reference bank are disjoint, including across duplicate checks. Every unlabeled ID gets one query opportunity, so no permanent fold exclusion suppresses sampler exposure. Folds are rebuilt per epoch. A reference image contributes its current classifier slot, never its hidden label.

For query $i$ and slot $s$, support $S_i(s)$ is the maximum CLIP cosine among reference peers assigned to $s$. Take destination $d_i = \arg\max_{s\ne c_i} S_i(s)$ and score $S_i(d_i) - S_i(c_i)$, where $c_i$ is the refreshed current slot. An absent current slot gets the cosine floor of −1. An absent alternative is ineligible. Equal supports choose the smallest slot. Equal risks choose the smallest image ID. A timed-out refresh yields a logged KEEP epoch without a partial plan. This preserves a fixed scope for retrieval comparisons.

For a sampled unlabeled row, both views must have at least .2 teacher probability at the planned source and their mean teacher distribution must still favor that source. For either view, $q' = q + .2(e_d-e_c)$. This conserves probability mass and preserves nonnegativity under the live guard. E3's opposite-view distillation and original batch denominator apply. Only the distillation target changes. Known CE, MeMax, InfoNCE and SupCon use their original coefficients and inputs. Labeled IDs cannot enter the proposal bank or receive edits. A proposed ID can appear multiple times under the original sampler. Exposure is counted only after the optimizer step.

There is no counterfactual update, ABSTAIN, hierarchy, feature fusion, CLIP fine-tuning or text input. The old E3 and R2 training paths remain available. Their only shared operational changes are optional D1 controller selection and inclusion of a final review file when it exists.

## Implementation and label boundary

| Path | Integration |
|---|---|
| `dual/cache.py` | Official CLIP weight digest, center crop metadata, split and ordered image IDs, unit features and cache integrity |
| `dual/policy.py` | Complementary folds, per-slot maximum support, epoch plans, live source check and actual exposure ledger |
| `dual/train.py` | Original sampler and model builder, E3 MOVE distillation, unmodified zero-edit objective, official v2 evaluator, recovery checkpoints |
| `dual/run.py`, `dual/control.py` | Stage boundaries, manual CONTINUE/STOP, interval summaries, sealed geometry subprocess and verified exports |
| `dual/benchmark.py`, `dual/test_d1.py` | Bounded GPU benchmark, parity and focused invariant tests |
| `scripts/run_dual.sh` | Configurable Linux tmux, driver device, timeout and paths |

Training and CLIP encoding consume the image-only unlabeled adapter. The cache cannot contain labels. The bit-verified audited cache binds the official CLIP artifact, deterministic bicubic short-side 256 and center crop 224, CLIP RGB normalization, and exact ordered image IDs to the seed-1 Cars split. That legacy cache has no upstream image-content manifest. When byte-level image provenance is needed, build a local cache using the verified CLIP archive. Locally built caches bind the actual image contents in addition to the same split and preprocessing. Model initialization and plan eligibility use no semantic score from evaluation. The official evaluator gets labels after each training epoch. Milestone geometry runs separately and is never a policy input.

The controller computes All/Old/New, 20-epoch mean and slope, differences versus the audited E0 history at identical epoch windows, actual exposure and gradient ratios. E0 is a historical matched-protocol reference. The archive used RTX 3090 and PyTorch 2.5.1+cu124. Results from another environment, even with seed and data order matched, do not establish exact pairing. E3 and the manually recovered E3-F endpoints are historical context. A human CONTINUE/STOP governs budgets. No performance threshold changes policy settings during a run.

## Generic Linux GPU operations

Use a pinned Python environment with CUDA-enabled PyTorch and torchvision, plus the repository requirements. Put the official Cars dataset, local DINOv1 source, official ViT-B/16 weights and verified CLIP cache on the host. Resume from the same code, environment, model inputs and execution mode. `--micro` uses 8-image frozen-prefix/VJP chunks while leaving the training batch and loss denominator at 128. It is a memory option. The mode is part of checkpoint identity.

```bash
export D1_REPO=/srv/simgcd-crish
export SIMGCD_CARS_ROOT=/datasets/cars
export SIMGCD_DINO_REPO=/models/dino-v1
export SIMGCD_DINO_WEIGHTS=/models/dino_vitbase16_pretrain.pth
export D1_CLIP_CACHE=/models/clip-image.npz
export D1_RUN_ROOT=/runs/crish-dual
export CUDA_VISIBLE_DEVICES=0
export D1_TIMEOUT=72h
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$D1_REPO"

# Verify an existing audited CLIP bank.
python -m dual.benchmark preflight --cache "$D1_CLIP_CACHE"

# Optional alternative when a verified image bank is unavailable. Use a new output path.
python -m dual.benchmark encode --cache "$D1_CLIP_CACHE" --weights /models/ViT-B-16.pt

# Benchmark the target GPU with the verified R2-100 snapshot.
python -m dual.benchmark benchmark --cache "$D1_CLIP_CACHE" \
  --snapshot /models/epoch-100.pt --steps 12 --output "$D1_RUN_ROOT/benchmark.json"
# Repeat with --micro only if that is the execution mode planned for the run.

# Start once authorized. First run uses official DINO initialization.
bash scripts/run_dual.sh
tail -f "$D1_RUN_ROOT/D1/training.log"
tmux attach -t crish-d1

# Optional epoch-boundary control from another SSH session.
python -m dual.control pause --root "$D1_RUN_ROOT"
python -m dual.control stop --root "$D1_RUN_ROOT"

# Resume an interrupted run or release a paused milestone.
bash scripts/run_dual.sh
bash scripts/run_dual.sh --continue-at 100
# Use --continue-at 150 at the second review, or --stop-at 100/150 to finish there.

# Download the archive and sidecar via ordinary SSH and verify both layers.
scp gpu-host:/runs/crish-dual/exports/D1-epoch-100.tar .
scp gpu-host:/runs/crish-dual/exports/D1-epoch-100.tar.sha256 .
python -m relational.export verify D1-epoch-100.tar
```

Use `--auto-continue` only to explicitly bypass manual milestone pauses. The worker saves model, optimizer, original scheduler, RNG and journal offsets every ten epochs and on a graceful operator request. The latest recovery archive rotates every ten epochs. Epoch 100, 150 and 200 checkpoints and milestone archives are retained. An interrupt between committed checkpoints replays from the latest committed epoch. A wall-clock-triggered refresh timeout can change the plan if it occurs on only one replay. The Linux controller does not shut down the GPU server.
