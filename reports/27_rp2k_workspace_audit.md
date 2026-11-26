# RP2K integration audit — 2026-11-26

- Source: feat/local-adaptive-acceptance, 1c6ada810b451e5a1f3b36b930c550489917cf0a.
- Isolated checkout: gcd-final-rp2k, branch feat/rp2k-crish-local.
- Original checkout contains manuscript, figures, audit and tooling edits. None are copied, overwritten or staged here.
- Local process inventory found editor Python helpers, no active local training worker. Existing runs/logs are historical artifacts; no process was stopped.
- Host: macOS arm64. Existing local environment: Python 3.9.6. No local RP2K images or DINOv2 weights found. CUDA training is not available on this host.
- Remote YOLO detector has its own environment/job and is outside this implementation. No CRiSH job will be started there.
- Existing code: corrected Euclidean SelEx, DINOv2 ViT-B/14 reg4, local CRiSH Keep/Abstain/Move. RP2K is not a supported dataset. Upstream snapshots and existing benchmark profiles will remain untouched.
- Proposed integration: isolated RP2K index/loader, legal-label-only training rows, deterministic custom known/novel split, offline image-integrity audit, explicit unbalanced clustering and a single CRiSH run profile.
- RP2K paper: https://arxiv.org/abs/2006.12634 . The old project page returned 404 during inspection. An archive endpoint responded HTTP 200 / 6,330,687,649 bytes; published checksum not verified. No archive or pretrained weights downloaded to Mac.
- GPU readiness and actual dataset counts are pending: synthetic CPU tests cannot establish real-data GPU readiness or recognition quality.
- Next safe action: implement and test adapter/preparer without training or modifying upstream.

## Implementation and validation

- Added gcd_runtime/rp2k.py and selex_runtime/data.py: custom manifest-pinned adapter, no hidden-target file opened by the training dataset constructor.
- Added preparation/download/extract/index/audit/plan CLI, native exact-byte dedup and isolated 16-class image smoke subset.
- Added one local CRiSH RP2K profile: batch32, LR0.025, unbalanced clustering, pairwise batch128; no per-epoch test monitor or test-based checkpoint selection.
- Added deployment export of the locked completed backbone, exact architecture and 768-dimensional inference contract. No worker deployment performed.
- Verified archive metadata using bounded HTTP range reads only: ZIP64,389089 entries including directories, layout all/test/<SKU>/<image>. No image dataset downloaded on Mac.
- Initial regression invocation exposed missing worktree submodule/venv and an existing README bootstrap-command assertion. Initialized a separate pinned HypCD submodule in this worktree, reused the local test interpreter, and documented the actual bootstrap command in the new README introduction. No original source checkout changed.
- Selected regression suite:60 existing tests passed; RP2K suite:8 passed. Commands use /Users/phanhoangkien/Projects/gcd-final/.venv/bin/python with SELEX_PYTHON set to that interpreter. Selected combined run before the eighth RP2K case:67 passed in42.23s; final RP2K-only run:8 passed in4.87s. Warnings are existing sklearn/matplotlib deprecations.
- CLI --help, synthetic generated-plan dry-run and diff whitespace checks pass. Upstream/SelEx and historical configs have no changes; original paper edits remain untouched.
- Not run: full RP2K download/index on real images, pretrained backbone forward/backward on NVIDIA, actual GPU smoke, training, held-out retail evaluation or worker integration. User explicitly reserved GPU rental/launch for a later step.
- Remaining resource risk: full-pool clustering on thousands of SKUs has not been measured. A successful tiny smoke is not evidence that the full dataset fits in24GB or meets a cost/time budget.
