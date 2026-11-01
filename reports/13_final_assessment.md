# Đánh giá thực nghiệm proposed DINOv2 — 2026-11-01

## Kết luận
Đã hoàn tất một thực nghiệm AI có nguồn gốc, dữ liệu, code mở rộng, checkpoint và final evaluation được xác minh. Ứng viên reliability hiện chưa có bằng chứng cải thiện thuật toán: thiếu matched baseline, single seed, chưa ablation và novelty review chưa hoàn chỉnh. Kết quả thấp hơn số tham khảo paper ở cả3 chỉ số nhưng không quy chênh lệch này cho module vì điều kiện chưa khớp. Không nhận SOTA, tính mới đã chứng minh hay paper readiness.

## Final result
Balanced semi-supervised kmeans trên train-unlabeled có labeled anchors, Hungarian-v2, known total class count196, seed1, final200-epoch backbone. Không phải validation accuracy và không trộn với disjoint-test plain-kmeans.

| Result | All % | Old % | New % |
|---|---:|---:|---:|
| Observed proposed final |81.23|86.66|78.61|
| Hyp-SelEx DINOv2, paper-reported |83.8|93.3|79.2|
| Difference (reference only), percentage points |-2.57|-6.64|-0.59|

Nguồn paper Table1: https://arxiv.org/html/2504.06120v1#S4.T1. Essential differences include grad_from_block10 (last2 blocks) vs paper last3 for Cars, smoothing0.5 vs paper1.0, curvature0.1 vs main paper0.05; checkpoint selection final vs best-old-test code path, runtime dependencies/attention fallback and empty-cluster handling. Full discrepancy discussion report01. Baseline under current runtime is needed to attribute an effect.

## History
SS monitoring records are evaluated BEFORE training index e, while checkpoints save AFTER training e. Figure uses completed epochs at measurement; record199 switches to plain-kmeans and is excluded from SS curve. Final SS point is a separate evaluation after200 updates. Clustering RNG state differs between monitor and independent final evaluation; no precise one-step causal delta inferred.

Mean SS history: epochs50–69 All81.64, Old92.58, New76.36; epochs70–89 All82.52, Old91.96, New77.96; epochs170–189 All80.95, Old85.89, New78.57. Old loss of accuracy in late training dominates modest New improvement. Better early peak does not justify hindsight best-checkpoint selection. Reliability held full strength after index29, but baseline angle/distance mixing continues to vary; no isolation of forgetting/overfit/geometry/reliability as cause.

## Module behavior
Final10 available SS monitoring epochs189–198 diagnostics,20 snapshots per level: levels1–3 fallback94.7–98.2% and weights0.987–0.994; much of pseudo supervision stays baseline-like. Level0 pair-weight mean0.913/downweighted37.47%; level6 mean0.793/downweighted23.51% but hierarchy coefficient1/128. Coefficient and raw mass do not measure true gradient influence because per-anchor positive mass is normalized. Only batch0 and50 out of63 train batches sampled. No pair ground-truth correctness is stored. These observations establish activation, not semantic reliability or causal gain.

## What was accomplished
Pinned upstream preserved; novel-idea candidate separated into explicit runtime and reliability module; training-label masks, floor/symmetric fallback/ramp/detachment; local synthetic tests; Cars image/archive/annotation/split checks; real NVIDIA CUDA forward/backward and full-batch active-method trial; Linux training and epoch-boundary resume; NaN kmeans failure documented and explicit installed-dependency empty-center repair; final evaluation;63 files3.25GB backup verified on Mac, checkpoint199/backbone exact parity and finite tensors, runtime fingerprint match. Remote dataset/pretrained files and virtualenv are not backed up; documented download/repair paths remain. Rental deletion is user action, not verified by agent.

## Readiness
Suitable foundation for an AI course project on reproduction, extension and critical evaluation, subject to supervisor rubric. A thesis with empirical method claims still needs controlled baseline/ablation and clear motivation/novelty differentiation. Publication not yet substantiated. Adding the module does not prove originality; operations/checkpoint fixes are engineering, not algorithm novelty. A null/negative result must be preserved, not concealed.

## Next actions on Mac, before more paid compute
1. Confirm rental termination in dashboard, keep original checkpoint/logs/source intact and make second independent backup if available.
2. Use report and actual training figure to discuss evaluation/claim scope with supervisor.
3. Review implementation alignment and offline fixed-checkpoint pair-quality audit plan. It requires model+head+inputs, evaluation labels used only to assess, never to train/score/tune. Coarse-positive correctness needs carefully stated proxy.
4. Only with agreed budget, preregister matched B0 and ablations (same schedule, source, dependency repair, backbone, fraction, batch, epochs, seed and metrics). A shorter pilot must be matched on both sides; do not compare shorter baseline with200-epoch proposed as a fair algorithm test.

## Artifacts
- results/cars_proposed_dinov2_seed1_user_reported_evaluation.json (now raw-file verified)
- reports/figures/proposed_dinov2_seed1/history.png, history.pdf, history.csv
- logs/final-backup-20261101/history-audit and local_sha256_manifest.json
- reports/12_final_backup_verified.md

Plot command: .venv/bin/python scripts/plot_run_history.py --run <failed-parent-run> <resumed-run> --evaluation <eval-run>/evaluation.json --output <new-output-dir>. CPU only, no training/weights loaded.
