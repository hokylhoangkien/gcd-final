# CRiSH-D1 matched-exposure ablations

Original D1 remains at commit `9468369`. The coordinated experiment lives in [dual/ablation.py](../dual/ablation.py), on branch `feat/crish-dual-ablation`. Its reference arm shares the original CLIP proposal algorithm, then accepts the same number of eligible edits as its controls in every shared batch. Compare these arms with each other. The completed historical D1 used its natural exposure and is a descriptive reference.

| Arm | Candidate and source | Destination |
|---|---|---|
| D1-matched | CLIP maximum peer support difference, positive risk, top 10% ceiling | Highest-supported alternative slot |
| D1-Random | Same candidate IDs and snapshot source slots as D1-matched | Uniform choice over slots occupied in the Random model's opposite fold, excluding the source and original CLIP destination |
| D1-DINO | Same support difference, positive-risk gate, ID tie-break and budget from this arm's current normalized DINOv1 final CLS | Highest-supported alternative slot |

CLIP and DINO each use the existing `dual.policy.scores` and complementary query/reference folds. Refreshes use their own current classifier assignments. Random alternatives are sampled once per refresh using the run seed, epoch, and ID. Missing supported alternatives remove the candidate from the matched CLIP/Random pair. The random arm measures the value of choosing the CLIP destination conditional on a CLIP-selected eligible source and excluding the original destination. DINO versus D1 changes the evidence for both proposal selection and destination retrieval. The DINO IDs need not be the same as the CLIP IDs.

## Matching actual dose

Each batch is drawn once by the original weighted replacement sampler, with its original two-view crop generation. The three arms consume that same batch in the same order. Each arm independently runs the unchanged source-change and both-view teacher-mass eligibility checks. CLIP and Random keep only their common eligible occurrences, with the same batch row, ID and source. DINO uses its own ranked eligible list. Truncate all three lists to the minimum of these support counts, remaining epoch budgets and the original per-batch cap. The moved mass is 0.2 per occurrence in every arm. There is zero editing in all three arms if an arm supplies no eligible row, including when a refresh times out. No padding, fallback or eligibility relaxation.

Both DINO shortages at refresh and shortages of eligible sampled occurrences are recorded. The journal includes candidate quota and eligibility, per-batch desired/available/applied counts, stale source, low mass, Random support failures and the committed edit rows in each arm. Total moved mass matches exactly. Unique exposure and distinct-ID composition can differ between CLIP and DINO. Matching also conditions the CLIP exposure on opportunities for Random and DINO, which affects the population the controlled contrast describes. Count and mass equality do not establish equality of gradient dose or semantic opportunity.

The worker reuses the original D1 optimizer, scheduler, loss, MOVE and guard. It preserves seed 1, Cars SSB, batch 128, 200-epoch cosine schedule, ten warmup epochs, positive alternative support, 300-second refresh timeout per arm, batch cap 8 and mixed-row epoch ceiling 4%. It previews deterministic train-mode logits for eligibility, then verifies bitwise equality on every edited arm before computing the update. The shared RNG stream advances once per batch and once per evaluation loader iteration at the epoch boundary, as in the original single-arm run. Full epoch boundaries atomically save all three models, momentum, scheduler and RNG together. The committed journal offset is restored before continuation. Official transductive evaluation is isolated from proposal selection.

## Linux setup

Use Linux with Bash, GNU coreutils, tmux, Python 3.10–3.12 and a CUDA-capable NVIDIA driver. Place the Cars dataset, local DINOv1 source, official ViT-B/16 weights, audited CLIP bank, and verified R2-100 checkpoint at the paths below. The checkpoint may be copied from the verified R2 archive. The benchmark checks its SHA-256 before loading it. The data and model inputs remain outside Git.

```bash
git clone --branch feat/crish-dual-ablation https://github.com/hokylhoangkien/gcd-final.git "$HOME/simgcd-crish"
cd "$HOME/simgcd-crish"
git rev-parse HEAD
python3 -m venv "$HOME/venvs/crish-d1"
source "$HOME/venvs/crish-d1/bin/activate"
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
python -m pip install numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2 pandas==2.3.3 pillow==12.3.0 tqdm==4.70.1 loguru==0.8.0
python -m pip freeze > "$HOME/venvs/crish-d1/requirements.lock"

export SIMGCD_CARS_ROOT=/datasets/cars
export SIMGCD_DINO_REPO=/models/dino-v1
export SIMGCD_DINO_WEIGHTS=/models/dino_vitbase16_pretrain.pth
export D1_CLIP_CACHE=/models/clip-image.npz
export R2_100=/models/epoch-100.pt
export D1_ABLATION_ROOT="$HOME/runs/d1-ablation-seed1"
export D1_ABLATION_SESSION=crish-d1-ablation
export D1_ABLATION_TIMEOUT=72h
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8

nvidia-smi
python -m unittest dual.test_d1 dual.test_ablation dual.test_ablation_ops -v
```

## Benchmark before renting a full training window

Run on the target GPU with the planned execution mode. The default uses 8 loader workers, the original 128 mixed rows per batch and the original 6,144-ID full retrieval bank. All arms start from the verified R2-100 model, SGD momentum, scheduler clock and RNG. The benchmark runs three actual production refreshes and 12 shared batches using the original matched optimizer step. It does not evaluate or advance a full epoch and produces only a JSON receipt. The configured 300-second refresh deadline still applies to each arm.

```bash
set -o pipefail
mkdir -p "$HOME/runs/d1-ablation-pilots"
BENCH="$HOME/runs/d1-ablation-pilots/$(date -u +%Y%m%dT%H%M%SZ)"
timeout --signal=TERM --kill-after=90s 30m \
  python -u -m dual.ablation_benchmark "$BENCH" "$D1_CLIP_CACHE" "$R2_100" --steps 12 \
  2>&1 | tee "$BENCH.log"
cat "$BENCH/benchmark.json"
```

Run that block under Bash with `set -o pipefail` so a timeout or benchmark failure propagates through `tee`. Add `--micro` only if using that same memory mode in production and use a new output directory for each benchmark. The benchmark reports DINO candidate and live shortfalls, matched MOVE count and actual dose, separate refresh and training costs, CUDA allocated peaks and a core epoch estimate excluding evaluation and archive I/O. Zero matched MOVE or unchanged classifiers fail the pilot explicitly. Do not pad the pilot to achieve nonzero exposure. The benchmark path is reserved and the production worker rejects it. No optimizer, scheduler, RNG or weights are exported from a benchmark.

## Start, stop and resume production

The production harness starts with official DINO initialization, the original SimGCD head initialization and batch protocol, and the original 200-epoch cosine schedule. It has no snapshot option. The benchmark path and the production run root must be distinct. The launcher starts only the next milestone and exits the tmux session when that stage finishes. Wall time is configurable **per stage**. Do not use a moving Git checkout or change the CUDA execution mode, environment, dataset or cached inputs during the run.

```bash
# First launch and optional attach. Detach without interrupting with Ctrl-b then d.
bash scripts/run_d1_ablation.sh 50
tmux attach -t "$D1_ABLATION_SESSION"

# From another SSH session, with the same environment variables.
tmux ls
tail -n 10 "$D1_ABLATION_ROOT"/logs/epoch-*.log
cat "$D1_ABLATION_ROOT/status.json"
tail -n 1 "$D1_ABLATION_ROOT/run/matched.jsonl" |
  python -c 'import json,sys; r=json.load(sys.stdin); print(r["epoch"],r["accuracy"],{k:v["applied"] for k,v in r["arms"].items()})'

# After checking that the previous stage stopped, launch the next stage.
bash scripts/run_d1_ablation.sh 100
# After epoch 100 has stopped
bash scripts/run_d1_ablation.sh 150
# After epoch 150 has stopped
bash scripts/run_d1_ablation.sh 200
```

If using `--micro`, append it to **every** launch. The worker commits one joint recovery checkpoint after each completed epoch. A timeout or Ctrl-C in the tmux pane abandons the in-process epoch. Relaunch the **same** target milestone to resume from the last committed state. For example, rerun `bash scripts/run_d1_ablation.sh 100` after an interruption between 50 and 100. This also retries export after a stop between checkpoint commit and publication. The committed checkpoint owns the ledger prefix. Exact resume includes model weights, optimizer, scheduler and RNG in the recorded execution environment. A repeated, interrupted epoch can see different timeout outcomes because refresh deadlines use wall time. The launcher does not shut down the server.

## Archive and verify

Each milestone is an immutable directory with a TAR and SHA-256 sidecar. The controller verifies per-member and outer archive hashes before publishing both files together. Existing verified archives are checked against their committed state and are never regenerated. Each archive contains the joint checkpoint, committed metric/exposure journal, run configuration and milestone receipt. Mutable tmux logs live under `logs/` and can be copied separately.

```bash
# On the GPU host
mark=050
cd "$D1_ABLATION_ROOT/exports/epoch-$mark"
sha256sum -c "D1-ablation-epoch-$mark.tar.sha256"
cd "$HOME/simgcd-crish"
python -m relational.export verify \
  "$D1_ABLATION_ROOT/exports/epoch-$mark/D1-ablation-epoch-$mark.tar"

# On the receiving host. Replace user@gpu-host and the remote login name.
scp 'user@gpu-host:/home/user/runs/d1-ablation-seed1/exports/epoch-050/D1-ablation-epoch-050.tar*' .
sha256sum -c D1-ablation-epoch-050.tar.sha256
python -m relational.export verify D1-ablation-epoch-050.tar
```

## Validation

The focused suite covers eligibility, probability transfer, deterministic random destinations, DINO shortfalls without padding, zero-edit baseline parity, exact joint resume through a matched MOVE update after a torn journal append, benchmark provenance and isolation, all four stopping milestones, and archive retries without rewriting earlier artifacts. The launcher passed Bash syntax validation. A live tmux training stage on rented Linux hardware was not started.
