# Standalone CRiSH-R on a Linux GPU server

**R2 can start immediately without R0 or R1.** The default standalone run pauses after completed epochs 100 and 150, writes a review and exports a resumable checkpoint. Missing comparative evidence is reported as **UNVERIFIED**. An explicit decision advances the same optimizer and original 200-epoch cosine schedule. R0, R1 and matched R2 retain the original research guard.

The experiment uses Stanford Cars SSB, seed 1, pretrained DINOv1 ViT-B/16, block 11, batch 128, two crops, FP32, SGD with initial LR .1, momentum .9, regularized decay .00005, ten warmup epochs, and terminal LR .0001. The graph uses exact reciprocal 5-NN, baseline weight .5, signed edit .25 and a shared **relational coefficient of 1.0 for R1 and R2**. Use a fresh output root and run future matched references on the same pinned checkout. The resume manifest includes the coefficient and Git commit. Evaluation uses the official transductive v2 assignment. [Method and objective](CRISH_RELATIONAL_DESIGN.md) · [Pre-training dose decision](CRISH_RELATIONAL_SENSITIVITY.md)

## 1. Set up the server

The examples use Bash, a Linux machine with an NVIDIA driver, Python 3.12, Git, tmux, GNU coreutils and SSH/SCP. Replace the dataset and storage paths with absolute paths on persistent storage. The repo can be installed anywhere. Run the exports again when opening a fresh SSH shell, or keep them in your own environment file.

~~~bash
export REL_REPO="$HOME/src/crish-relational"
export REL_RUN_ROOT="$HOME/runs/crish-relational"
export SIMGCD_CARS_ROOT="/data/stanford-cars"
export SIMGCD_DINO_REPO="$HOME/src/dino"
export SIMGCD_DINO_WEIGHTS="$HOME/weights/dino_vitbase16_pretrain.pth"
export CUDA_VISIBLE_DEVICES=0       # physical GPU index or GPU UUID
export REL_TIMEOUT=48h             # wall time for this controller invocation
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$(dirname "$REL_REPO")" "$(dirname "$SIMGCD_DINO_REPO")" \
  "$(dirname "$SIMGCD_DINO_WEIGHTS")" "$REL_RUN_ROOT"

# First checkout only
git clone --branch feat/crish-relational https://github.com/hokylhoangkien/gcd-final.git "$REL_REPO"
git clone https://github.com/facebookresearch/dino.git "$SIMGCD_DINO_REPO"
git -C "$SIMGCD_DINO_REPO" checkout 7c446df5b9f45747937fb0d72314eb9f7b66930a
test -f "$SIMGCD_DINO_WEIGHTS" || curl -fL \
  https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth \
  -o "$SIMGCD_DINO_WEIGHTS"

cd "$REL_REPO"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
python -m pip install numpy==1.26.4 scipy==1.14.1 scikit-learn==1.5.2 pandas pillow tqdm loguru
git rev-parse HEAD
command -v tmux timeout scp
nvidia-smi
df -h "$REL_RUN_ROOT"
~~~

The approved Cars layout includes cars_train/ and devkit/cars_train_annos.mat and devkit/cars_test_annos_withlabels.mat. The official dataset builder reads the test annotations even though the transductive evaluation uses the unlabeled training images. Keep the checkout and software stack pinned across resumption and matched experiments. Each run records its protocol fingerprint and environment. The launch script forwards the selected GPU and Python executable into tmux; inside the worker the selected GPU is local device zero.

## 2. Preflight and benchmark on the target GPU server

SSH into the intended training server. To keep the benchmark alive across SSH disconnects, open a tmux shell with `tmux new-session -s cars-benchmark`, then set the environment from section 1 in that shell. Use the same GPU and storage mount intended for training. Detach with Ctrl-b then d and reconnect with `tmux attach -t cars-benchmark`.

~~~bash
cd "$REL_REPO"
source .venv/bin/activate
python -m relational.benchmark preflight --output "$REL_RUN_ROOT/preflight.json"
timeout --kill-after=30s "${REL_BENCH_TIMEOUT:-30m}" python -u -m relational.benchmark benchmark \
  --output "$REL_RUN_ROOT/benchmark.json" >"$REL_RUN_ROOT/benchmark.log" 2>&1
cat "$REL_RUN_ROOT/benchmark.json"
~~~

Preflight loads and verifies the official weight digest, builds the real loaders and checks that all training files exist, including 6,144 distinct unlabeled IDs. The default benchmark runs the actual graph refresh with microbatch 16 and eight loader workers. It separately times the center pass, the joint two-crop pass, and each of the three exact kNN searches. GPU synchronization brackets the wall timers. Encoding includes image I/O and augmentation. Refresh time additionally includes edge assembly.

It calibrates time with three warmup training steps, three active R2 steps and three evaluation inference batches on disposable in-memory weights. No training checkpoint, guard input or run trajectory is created. The server defaults use train batch 128 and eval batch 256. If diagnostic memory is limited, pass for example **--batch 16 --eval-batch 16 --micro 8 --workers 0**. This still constructs the full 6,144-node graph. Smaller probes change only the benchmark and are explicitly marked in its JSON. A graph-only intermediate receipt survives a training-probe allocation failure.

The forecast scales the measured first and ordinary batch times to 63 production optimizer steps and 6,144 evaluation rows, then combines ten warmup and 190 active epochs. It excludes checkpoint writes, set-up, Hungarian assignment, guard exports, sealed offline evaluations and manual waiting. Short calibration and small-batch throughput can bias this estimate. Measure on the intended server before budgeting GPU hours. The measured peak is PyTorch allocated VRAM for the chosen probes; a small probe cannot establish that the full training batch fits.

## 3. Launch R2 and inspect progress

~~~bash
cd "$REL_REPO"
source .venv/bin/activate
bash scripts/run_relational.sh R2 --standalone
tmux ls
tmux attach -t crish-rel-R2
# Detach with Ctrl-b followed by d. The session survives SSH disconnection.
~~~

From another shell, with the same path exports:

~~~bash
tail -n 5 "$REL_RUN_ROOT/R2/training.log"
tail -n 1 "$REL_RUN_ROOT/R2/metrics.jsonl"
tail -n 20 "$REL_RUN_ROOT/R2/controller.log"
ls -lh "$REL_RUN_ROOT/R2"/epoch-*.pt
~~~

The first checkpoint appears at epoch 10. Metrics may briefly lead the latest committed checkpoint; on crash recovery the worker truncates uncommitted metrics and edge records. Use status.json and the controller log for the most recent decision, and training.log for live progress. Set REL_SESSION to a distinct tmux session name when running in another output root.

## 4. Manual reviews, pause, resume and stop

At epochs 100 and 150 the worker returns at an epoch boundary. Read review-100.json or review-150.json, status.json and guards.jsonl. The controller prints the review and exports a bundle even in automatic mode. A paused controller exits cleanly and releases the GPU. Select **one** of the following actions at a paused epoch:

~~~bash
python -m json.tool "$REL_RUN_ROOT/R2/review-100.json"
# Continue from checkpoint 100; the manual decision applies only at 100
bash scripts/resume_relational.sh R2 --continue-at 100
# Alternatively, end the exploratory run at checkpoint 100
# bash scripts/resume_relational.sh R2 --stop-at 100
~~~

At the second guard substitute 150. Re-running the resume command with no decision leaves a manual review paused. A STOP is terminal for that standalone output. The requested epoch must match the paused checkpoint to avoid using an old decision at a later guard.

For automatic continuation, launch with **--standalone --auto-continue**. To opt in after a normal guard pause, pass **--auto-continue** to the resume script. This option applies to that controller invocation. It may continue on UNVERIFIED; a NO-GO from an available matched comparison still pauses for an explicit decision. An operator-requested pause needs an explicit --continue-at. A manual CONTINUE can override an unfavorable comparison in standalone exploratory mode and is recorded. Historical E3 **56.51/71.51/49.26** and terminal-recovered E3-F **56.49/73.36/48.35** are final All/Old/New percentages shown for context only. They never enter the delta, trend, grade or automatic decision.

To pause or gracefully terminate a **live** standalone worker:

~~~bash
python -m relational.control pause --root "$REL_RUN_ROOT"
# Alternatively request a terminal stop
# python -m relational.control stop --root "$REL_RUN_ROOT"
tail -f "$REL_RUN_ROOT/R2/controller.log"
~~~

The worker finishes the current epoch, records its evaluation, saves full state even between regular ten-epoch checkpoints and returns for export. Use the resulting completed epoch with --continue-at. For example, after an operator pause at 37 run **bash scripts/resume_relational.sh R2 --continue-at 37**. GNU timeout is a hard job limit. If it fires during training or export, rerun the resume launcher from the latest committed checkpoint. If no checkpoint was reached, use the start command. The scripts do not issue server shutdown commands.

### What the guard reports

- Accuracy means and ordinary least-squares slopes in percentage points per epoch, plus first and last values, for all three splits. Windows are one-based 81–100 and 131–150.
- Mean first-batch block-11 gradient norm ratios versus SimGCD for the entire relational term and the edit-only differential, with the number of measured epochs.
- Total sampled rows, relationship exposures, STRENGTHEN/WEAKEN occurrences, absolute/effective intervention mass and window-wide distinct edited anchors and edges.
- Mean graph edge count, isolated nodes, cross-view reciprocal-edge Jaccard, feature variance and mean resultant length. Sealed endpoint geometry is exported separately in geometry.json and does not select edits or grade the run.
- Matching R0/R1 windows are graded using the original thresholds. Unavailable or incompatible references are named in the report; available individual comparisons remain visible. UNVERIFIED makes no claim about the benefit of R2 over SimGCD or R1.

## 5. Matched runs and artifact download

For the original matched study use a separate run root or use the standalone root to supply its missing R0/R1 trajectories. Launch **bash scripts/run_relational.sh R0**, then **bash scripts/run_relational.sh R1**. A fresh R2 directory without --standalone uses the original matched controller, including BLOCKED handling and NO-GO stops. Existing standalone folders remember their mode. Re-evaluate a saved R2 window with newly completed matched references using:

~~~bash
python -m relational.guard "$REL_RUN_ROOT" R2 100
python -m relational.guard "$REL_RUN_ROOT" R2 150
~~~

The worker saves full checkpoints every ten epochs and at 100/150/200. Retention keeps the latest checkpoint and reached milestones. At regular ten-epoch checkpoints standalone writes a rolling `R2-recovery.tar` and checksum with the latest model, SGD/scheduler/RNG states, committed journal offsets, logs, metrics, edge ledger, guard journal, available reviews, standalone marker, config and environment. Each milestone and operator pause/stop creates a separate epoch-named archive with status, sealed geometry and retained checkpoints. Use an epoch-named bundle for milestone review or a stable rolling bundle for disaster recovery. If a rolling bundle is replaced during SCP, checksum verification can fail; repeat the download. Packaging is allowlisted. The dataset and input pretrained weights are outside the archive.

On the **download machine**, set your actual SSH address, port and absolute remote export directory. Use the epoch and bundle name printed by the controller:

~~~bash
export SSH_HOST=user@gpu.example.org SSH_PORT=22
export REMOTE_EXPORT_DIR=/home/user/runs/crish-relational/exports
export BUNDLE=R2-epoch-100.tar
scp -P "$SSH_PORT" "$SSH_HOST:$REMOTE_EXPORT_DIR/$BUNDLE" \
  "$SSH_HOST:$REMOTE_EXPORT_DIR/$BUNDLE.sha256" .
sha256sum -c "$BUNDLE.sha256"
python -m relational.export verify "$BUNDLE"
~~~

The second verifier checks every declared member as well as the archive digest. For restoration, extract a trusted, verified archive into the same variant directory on the new server and use the same checkout and paths for the data and DINO sources. The sampler and scheduler resume from the committed state. Identical GPU and software settings are needed for bit-for-bit numerical continuation.

## 6. Validation and pending server measurements

Server benchmark pending. Section 2 produces the hostname, GPU identity, protocol, per-view encoding times, exact kNN times, peak allocations, and epoch and run duration estimates in `benchmark.json`. Use a completed receipt with production batch sizes to budget training. Target server memory fit, wall time, live tmux reconnection and SSH/SCP transfer remain unverified.

| Local check | Result |
| --- | --- |
| Standalone controller and epoch-boundary replay, `python validate_standalone.py` | 7 passed. Includes missing and partial references, manual pause/continue/stop, rolling and milestone exports with checksum verification, matched veto and off-cycle checkpoint replay. Milestone flows use fixture journals. |
| Existing relational tests, `python validate_relational.py` | 9 passed. |
| CUDA parity, `python validate_relational.py --cuda-parity` | R0 matched upstream for a real one-step update. Bounded R2 replay reproduced sampled IDs, exposures, parameters, momentum, scheduler and RNG. |
| Production preflight and Bash syntax | Local Cars paths, split, weight digest and evaluation imports passed. Both Linux launch scripts passed `bash -n`. |

The only change in the graph module corrects ledger serialization: the original record constructor supplied the scalar and the edge list under the same keyword, raising an exception at the end of an active epoch. Records now store the count in `edge_count` and individual edges in `edges`. The benchmark and replay do not measure R2 accuracy or the benefit of repair.
