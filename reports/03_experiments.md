> Cập nhật hiện hành 2026-11-01: proposed DINOv2 seed1 đã hoàn tất200 epoch theo log user, final SS evaluation81.2337/86.6567/78.6145%; xem phần cuối. Các mục ban đầu bên dưới là lịch sử, không phải trạng thái hiện tại.

# Thực nghiệm — 2026-10-29

## Run đã quan sát
Run ID: integrity-smoke-20261029. Command: `python3 scripts/run_smoke.py`. Môi trường: macOS ARM64, Python3.9.6; không dependency bên ngoài. Upstream SHA: ffc895b7ff38be8502bb7e3bd851f1d2a55dff73; workspace chưa Git. Seed: không áp dụng (deterministic fixture). Config: timeout 30 giây cho mỗi subprocess, 2 subprocess; fixture tạm 2 file rỗng, không phải ảnh thực. Log: `logs/integrity-smoke-20261029.log`.
Kết quả: 3 unittest đạt; checksum và class partition thật đạt. Manifest test chặn duplicate IDs/paths, thiếu file, partition sai, supervised novel labels, labels ngoài range. Không kiểm decode ảnh, content duplicates, coverage/counts toàn dataset hoặc đúng fraction khi chưa có Cars. Không phải smoke model và không có accuracy.

| Variant | All (%) | Old (%) | New (%) | Trạng thái |
|---|---|---|---|---|
| B0 | — | — | — | Chưa chạy |
| B1 | — | — | — | Chưa triển khai |
| B2 | — | — | — | Chưa triển khai |
| B3 | — | — | — | Chưa triển khai |

## Điều kiện tiếp theo
1. Hoàn tất đọc method RCA, CiPR, local hierarchy HypGCD; chốt novelty gate.
2. Tạo environment riêng, không sửa Python hệ thống; kiểm dependencies và chạy bounded import/CPU loss fixture sau khi cô lập CUDA code. Chưa có forward/backward hay Hungarian parity test.
3. Chuẩn bị Cars images+MAT và DINO weights (checksum), export audit-only train membership manifest từ đúng upstream loader. Chạy integrity; không dùng manifest chứa eval ground truth cho training.
4. Giải quyết paper/code discrepancies và selection protocol trong report01 trước đóng băng B0 config.
5. Chuẩn bị CUDA compute, finite epoch/step và runtime guard, resume đầy đủ; đo memory từ bounded run trước benchmark. Chưa có bằng chứng 16GB MPS chạy được unchanged baseline.
6. Sau B0 smoke mới tích hợp module, parity và ablations. Chưa chạy sweep. Không có mean/std, curve, efficiency hay bằng chứng empirical để viết kết luận.

## Cập nhật local runtime smoke (không benchmark)
Environment riêng: Python3.9.6 ARM64, torch2.4.1, torchvision0.19.1; dependency snapshot `environments/requirements-local-macos-py39.observed.txt`. `pip check`: không có dependency lỗi; có cảnh báo cache pip ngoài workspace không ghi được, không ảnh hưởng kết quả.
Đã chạy bounded smoke, 9 unittest đạt, không skip: integrity 3 tests; loss AST unchanged; config validation; launcher dry-run; checkpoint/RNG/optimizer/scheduler khớp update CPU tiếp theo; import và SupCon/USE hyperbolic backward + Hungarian sample-weighted fixture + clustering dependency fixture; synthetic Poincare bounds/finite gradients. AST syntax và `bash -n` cho 3 shell scripts đạt.
Một lần import/loss probe đầu bị NameError vì fixture thiếu global args.c mà upstream SupCon sử dụng; đã sửa fixture, không sửa loss. Log thất bại giữ ở `logs/baseline-cpu-smoke-attempt1-20261029.log`; lần probe sau đạt. Matplotlib có deprecation warnings; không có test failure trong lần suite hoàn tất.
Ép `--device cuda` trên Mac trả exit1 với `CUDA requested but unavailable` đúng mong đợi; không phải CUDA thành công. Chưa chạy ViT pretrained probe vì không tải weights trên Mac. Linux/CUDA/data/full-training/resume thực tế chưa xác minh.
Các run smoke mới tự tạo UUID+UTC trong logs, lưu metadata/config fixture seeds/source hashes và stdout/stderr; xem run ID cập nhật bên dưới. Các giá trị loss synthetic không là kết quả Cars và không được điền bảng accuracy.

Run cuối đã xác minh: `local-smoke-20261029T123012Z-71d79fdf`, timestamp UTC `2026-10-29T12:30:12.813132+00:00`, trạng thái `passed`; metadata `logs/local-smoke-20261029T123012Z-71d79fdf/metadata.json`. Project Git SHA: `uncommitted` (chưa commit); source hashes đầy đủ trong metadata.

## Xác minh source bàn giao
Run `local-smoke-20261029T151313Z-bdb37bc0`, timestamp UTC `2026-10-29T15:13:13.759403+00:00`: **26 tests đạt**, static syntax checks đạt, shell syntax4 scripts đạt; pip check không có dependency lỗi. Log/metadata tại `logs/local-smoke-20261029T151313Z-bdb37bc0`. Python3.9.6 macOS ARM64; không có NVIDIA. Git dự án chưa commit; source fingerprints lưu trong metadata và release manifest. Upstream git status sạch.

## Remote proposed DINOv2 — final evaluation received 2026-11-01

Evidence: user pasted epoch199 checkpoint save, final reports and successful evaluate launcher completion; evaluation JSON received in chat, raw final artifacts pending local backup and verification. Training resumed from epoch24 after failure; preserved parent20261031T035726Z-761b10ee, resumed run20261031T055221Z-03368e3d; evaluation run20261031T172513Z-881c7f46. Device V100-SXM2-32GB, PyTorch2.4.1+cu121, Python3.10.12, seed1, DINOv2 ViT-B/14 reg4,200 epochs, batch128, grad_from_block10, config snapshot in remote run. Same source fingerprints enforced by resume; installed kmeans dependency changed with before/after hashes as documented in report09. Runtime Git revision uncommitted/release snapshot; do not label as unchanged official implementation.

Final checkpoint backbone_final.pt; balanced SS-kmeans with labeled anchors, train-unlabeled pool, Hungarian-v2, unit fraction.

| Variant | Seed | Epochs | All % | Old % | New % | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Proposed B3, DINOv2 |1|200|81.2337|86.6567|78.6145|Observed remote evaluation pasted by user|
| Matched baseline DINOv2 |—|—|—|—|—|Not run|

Raw record: results/cars_proposed_dinov2_seed1_user_reported_evaluation.json. No seed mean/std or algorithmic gain claim.

Train final report51.77/44.68/55.20 is plain-kmeans on train-unlabeled and disjoint test56.24/65.20/47.59 is plain-kmeans on test, both rounded log values; do not replace the SS final result with them or mix into SS curve. SS81.28 printed before final update measures pre-epoch199 state, not final200-epoch checkpoint.

Paper Hyp-SelEx DINOv2 Stanford Cars83.8/93.3/79.2 (paper-reported, Table1, https://arxiv.org/html/2504.06120v1#S4.T1). Current run below that reference by2.57/6.64/0.59 percentage points; unmatched config/selection/dependency differences preclude attribution to proposed module. Record negative/null evidence; no SOTA or novelty claim.

Next: backup both training runs and evaluation plus source/environment/repair provenance, verify checkpoint/hash, then end rental only after user action/authorization. No further training launched by agent.

## Final artifacts verified locally — 2026-11-01
Raw final evaluation and checkpoint backup now copied and verified on Mac; supersedes earlier pending/user-paste-only status.63 files3.25GB. resume epoch199; final backbone equal to resume backbone, finite weights, local runtime fingerprint matches. Full chain parsed with200 unique metric epochs0–199. See reports/12_final_backup_verified.md and ignored verification/checksum artifacts. No new GPU run; rental not deleted by agent.
