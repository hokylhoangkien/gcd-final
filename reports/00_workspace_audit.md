# Workspace audit — 2026-11-27

**Updated implementation:** the initial proposals below were superseded by the
user-approved 40-SKU, 200-epoch / batch-128 workflow on `feat/train-planogram`.
Preparation code and local checks are now implemented; current status and the
remaining GPU gates are in [PLANOGRAM_READINESS.md](PLANOGRAM_READINESS.md), with
paste-ready commands in [PLANOGRAM_GPU.md](../PLANOGRAM_GPU.md). The initial
20-SKU / 20-epoch proposal below is historical, not the active configuration.

## Checkout and preservation

- User requested branch `feat/crish-simgcd` for a lightweight product-recognition experiment for the planogram project; training will run on a rented GPU.
- Fetched origin and created a separate Git worktree at `../gcd-final-simgcd` tracking `origin/feat/crish-simgcd`.
- Initial HEAD: `ed4b388` (resolve full SHA with `git rev-parse HEAD`).
- Existing `gcd-final` checkout has uncommitted research/manuscript work on another branch; preserved.
- Planogram application files were not changed.

## Runtime and existing work

- Local host: Apple arm64, 16 GiB RAM. No matching Python training process was observed during the process audit; this is a point-in-time observation.
- Existing reports in this branch cover Cars/global CRISH and mechanistic audits. No dataset or weight download, training process, or benchmark was started in this session.
- GPU availability, CUDA forward/backward, dependency compatibility and memory consumption on the rental machine remain unverified.

## Implementation findings

- `train.py` uses CUDA directly and DINO `dino_vitb16`, 768-dimensional features and 224-pixel input. Only the final transformer block is trainable by default.
- Existing dataset registry supports CIFAR, Cars, CUB, Aircraft, ImageNet and Herbarium; no retail-product loader exists.
- Most dataset roots in `config.py` contain literal `${DATASET_DIR}` placeholders; these are not automatically expanded by Python. Set actual paths or implement environment-based path configuration before running.
- `requirements.txt` pins torch 1.10.0 / torchvision 0.11.1. Select a compatible Python/CUDA environment and verify imports before paying for a long run.
- Existing CRISH global-suite CLI and dataset digest logic are Cars-specific. Retail support must account for `crish/data.py` and global repair paths, rather than merely add a registry key.
- Planogram uses a DINOv2 embedding recognizer with a versioned embedding contract. A SimGCD DINO checkpoint cannot be loaded into that architecture directly. Integration requires a compatible recognizer, preprocessing/version update and regeneration of product reference embeddings.

## Recommended lightweight experiment (not implemented or run)

- Prefer an RPC subset for SKU discrimination: choose 20 product classes deterministically, retain 100–200 single-product images per class (2,000–4,000 total), and record the class/image manifest and seed.
- Source: https://rpc-dataset.github.io/ — full distribution is approximately 15 GB; training a subset does not guarantee a small initial download. Official license: CC BY-NC-SA 4.0.
- If testing GCD: define 10 known and 10 novel classes; expose labels for 50% of known-class training examples only. Reserve separate evaluation images, audit duplicates, and never expose novel ground truth to repair scoring.
- Proposed starting settings: one NVIDIA GPU with 24 GiB VRAM, batch 32, input 224, final-block tuning, initial 1-epoch verification followed by a finite 20-epoch pilot. These are starting settings, not verified memory or runtime guarantees.
- Begin with baseline E0, then compare E3 only after retail dataset support and label masking are verified. Short pilots are not official benchmark reproductions.
- Freiburg Groceries is a lighter alternative (~5,000 images, 25 broad food classes), but its classes do not evaluate precise SKU identity. Source: https://github.com/PhilJd/freiburg_groceries_dataset

## Next work before remote training

1. Add a retail loader, deterministic subset/split manifest, image-only adapter and data-integrity checks.
2. Prepare a tested remote bootstrap and finite pilot runner with logs and checkpoints; adapt Cars-specific global repair code if E3 is required.
3. Verify imports, real CUDA forward/backward and a bounded training batch on the rented machine.
4. Train and evaluate All/Old/New; record actual timing and memory before selecting a longer budget.
5. Integrate a compatible embedding backbone into planogram and rebuild reference embeddings; evaluate on local shelf photos separately.

No training accuracy, completed run, rental cost or time-to-completion is claimed.
