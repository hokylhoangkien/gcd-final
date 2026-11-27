# RPC planogram readiness — 2026-11-27

## Scope and branch

- New branch `feat/train-planogram`, derived from `ed4b3885fb9100efaf7770d9b49a5e663ecd2a1e`.
- Existing `train.py`, `train_mp.py`, `config.py`, `model.py`, `data/`, `crish/` and original benchmark scripts are unchanged.
- Added isolated `planogram_rpc/`, GPU scripts, CPU protocol tests and an optional inference adapter.
- No planogram application source, local environment or running service was changed.
- User-authorized defaults: 200 epochs, batch 128, last checkpoint, SS-KMeans after training only, reference-SKU retrieval evaluation.

## Implementation

- Prepare RPC v5: download only annotations/selected exemplar files via pinned kagglehub, or read a local official tree; crop bbox with margin, preserve aspect ratio on white pad, detect exact decoded-pixel duplicates and write a deterministic manifest.
- Default subset: 40 randomly ordered SKU classes, 200 images/class, 70/15/15 holdout, 20 known/20 novel, 50% legally labeled known training images. Randomized ordering avoids making known/novel simply the low/high category IDs.
- Training datasets discard hidden targets and expose `-1` for unlabeled rows. Validation/test images do not enter the train loader or CRISH snapshot/probe/scoring pool.
- Reuse the existing global `EpochRepairs` and SimGCD losses; train only DINO block 11 plus head. Guarantee a 50:50 labeled/unlabeled batch, with replacement.
- Atomic `last.pt` includes model, optimizer, cosine scheduler, AMP, RNG, epoch, dataset/source identity and committed log byte offsets. Resume recovers epoch boundaries; this does not promise bitwise equality across different GPU/software environments.
- No accuracy-based selection or SS-KMeans inside the trainer. Final evaluator fits constrained Lloyd SS-KMeans on train-only features with legal labels; reports classifier, transductive/inductive clustering and gallery retrieval. The SS solver is an application implementation, not a claim of exact HypCD evaluator parity.
- Retrieval compares trained and untouched pretrained DINO using identical train-only references and held-out queries. It matches planogram's maximum-over-references score per product and margin between different products.
- Export the last backbone as TorchScript with hash/manifest and a new 768-D contract, plus optional planogram adapter and integration instructions. Regenerate the app's DINOv2 references under the new contract before switching.

## Observed local verification

Environment: macOS arm64 / 16 GiB RAM. CPU checks used the existing isolated research Python 3.9.6 environment, PyTorch 2.4.1, torchvision 0.19.1, NumPy 1.24.4, SciPy 1.10.1 and Pillow 10.4.0. This is not the target GPU environment. The remote bootstrap pins Python 3.10–3.12, torch 2.7.1 / torchvision 0.22.1 CUDA 12.8 and `requirements-rpc.txt`.

Executed checks:

```bash
python -m unittest discover -s tests_rpc -v
python validate_global_crish.py
python -m compileall -q planogram_rpc deployment tests_rpc
bash -n scripts/planogram/bootstrap.sh scripts/planogram/run.sh
ruff check --select E4,E7,E9,F,I planogram_rpc deployment tests_rpc
git diff --check
```

- CPU protocol suite: **10 tests passed**, including end-to-end preparation of a temporary 160-image fixture, corruption rejection, train-label masking, deterministic balanced batches, the actual RPC objective/backward, checkpoint/RNG recovery with bitwise-identical next optimizer update, SS-KMeans fixed known assignments, product-level retrieval margin, dynamic-batch TorchScript and deployment contract/checksum rejection.
- The image fixture also passed through the existing global E3 selector and new training objective, backward/update, exposure accounting and epoch close. This is execution verification, not a learned performance result.
- Existing `validate_global_crish.py` passed: fold/ID separation, frozen state/RNG, natural confirmation, timeout prefix, stale guard and exposure/persistence semantics.
- Downloaded and parsed the **real RPC v5 train annotation** with kagglehub 0.3.13: 53,739 exemplar images, 200 classes, 125 classes with at least 200 candidate images.
- Downloaded **one real exemplar** `038900004095_camera0-13.jpg` (category 112), decoded its official bbox and successfully generated the RGB 256×256 crop. No full subset download was attempted locally.
- Cloned the pinned official DINO source into a temporary folder, instantiated actual ViT-B/16 **with random weights**, and successfully traced/exported it on CPU with finite output and batch-2 numerical parity. No pretrained weights were downloaded locally.

## Remaining verification on rented GPU

1. Run bootstrap, full subset preparation and data checks.
2. Pass a real batch-128 two-view CUDA forward/backward/optimizer preflight.
3. Pass the bounded one-epoch real-image E3 repair check, then pilot and resume to 200 epochs.
4. Run final SS-KMeans/retrieval and export the actually trained last backbone.
5. Evaluate local shelf photos, unknown-product rejection and thresholds before switching the worker.

Unverified: production memory/runtime, target wheel compatibility on the eventual rental image, full-subset readiness, trained artifact quality, real shelf accuracy and any improvement over pretrained DINO. The GPU commands are explicit gates for these unknowns.

RPC holdout is not the official checkout benchmark or a shelf-photo benchmark. Correlated camera views can survive exact duplicate checks. Last-checkpoint use avoids selecting on novel/test labels; post-training novel reference labels assume a catalog at deployment. Cluster IDs are not store ProductIDs.

See [the GPU runbook](../PLANOGRAM_GPU.md) for full commands and recovery.
