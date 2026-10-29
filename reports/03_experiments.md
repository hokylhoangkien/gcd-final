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
