# Final local backup verification — 2026-11-01

Backup root: logs/final-backup-20261101/gcd-runs-v2 (gitignored). SCP transcript returned to Mac prompt without reported transfer error. Locally read63 files,3249009026 bytes (~3.25GB decimal /3.03GiB). Preserved preflight, trial, failed parent, successful resume and successful evaluation runs.

Verified on CPU: resume.pt epoch199, config200 epochs/DINOv2; backbone_final.pt tensors exactly equal resume model tensors; backbone/head values finite; architecture patch14 and4 registers; optimizer/scheduler/RNG keys present; runtime code fingerprint matches current local gcd_runtime; final backbone/resume ZIP archives readable; training/evaluation metadata completed; evaluation JSON exact fractions0.8123372395833334,0.8665667166416792,0.7861453053342988. No local forward/training performed.

Full-chain parser combines parent epochs0–24 and resumed epochs25–199, validates config equality and no duplicate epochs. Outputs history-audit/report.md, metrics.csv and diagnostics.json under backup root. All200 metric rows preserved; final row is plain-kmeans and is kept distinct from SS final evaluation. Local checksums record all63 files; these are local integrity hashes, not independently compared to remote checksums.

## Critical files

- Final backbone SHA256: `146bc8cff40d76e91b6836ba16582af1c891cac2f8bd1da70a2b11c0e0055093`
- Epoch199 resume SHA256: `f20d1e2d45f2b90f9d1482c959b72394280dd5a1ed816f68de14c327316f61c0`
- Final evaluation:20261031T172513Z-881c7f46/evaluation.json.
- Verification details: logs/final-backup-20261101/verification.json.
- Manifest: logs/final-backup-20261101/local_sha256_manifest.json.

This copies results/checkpoints, not remote dataset/pretrained weights or complete virtualenv. Dataset/weights are re-downloadable by documented sources/checksums; local source release and current runtime preserved. Dependency empty-cluster repair script and before/after hashes documented in report09; remote dependency sidecar has not been independently downloaded. Do not claim identical untouched PyPI environment. No dataset, checkpoint or credentials committed.

User may now end rental after confirming no other files are needed; deleting container is user action, not performed by agent. Files and source should ideally have a second external backup before deleting the only Mac copy.
