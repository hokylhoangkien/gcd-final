# Bàn giao source để kiểm thử và train trên RunPod

## Phạm vi hoàn tất
Đã thay adapter sửa chuỗi source bằng file trainer Python trực tiếp `gcd_runtime/trainer.py`. Phần tham chiếu kế thừa Hyp-SelEx được ghi provenance và giữ MIT notice trong LICENSES/THIRD_PARTY_NOTICES. Runtime không thực thi source upstream bằng exec/compile. Prototype cũ bảo toàn ở legacy, không dùng hoặc đóng gói.

Có B0–B3, reliability theo hierarchy, masks, floor/ramp/fallback, diagnostics, atomic checkpoint, RNG/optimizer/scheduler resume, final_epoch selection, cấu hình portable và launcher có deadline. Deadline kmeans riêng mặc định120s giữ toán gốc PyPI0.3, chỉ fail sớm nếu không hội tụ. CUDA gate có cả ảnh thật qua transforms/ViT và active-method backward; **chưa chạy các gate này trên NVIDIA**.

Có release tar.gz theo allowlist, SHA256 và manifests nguồn; hỗ trợ Git submodule hoặc upstream snapshot xác minh bằng hash. Release không chứa .git, .venv, logs, data, weights, checkpoints hoặc legacy. Test đóng gói thực sự giải nén, xác minh snapshot và chạy dry-run proposed mà không cần Git metadata.

## Bằng chứng kiểm thử
Unit/integration chạy CPU, không dataset/pretrained weights. Kiểm tra reference loss AST, baseline synthetic model updates so với train loop gốc, disabled-component parity, weighted-loss oracle, finite gradient/radii, duplicate IDs, masks, schedule, zero-neighbor fallback, poisoned unlabeled-target invariance và all-unlabeled batch. Thêm kiểm tra resume proposed qua warmup/ramp có dropout: cập nhật model/head sau ngắt tại epoch0 rồi resume khớp bitwise run synthetic liên tục3 epoch. Đây không phải bằng chứng bitwise CUDA hoặc accuracy.

Các stub metric trong fixture là dữ liệu test, được đánh dấu synthetic_fixture, không dùng cho bảng Cars. Không có benchmark accuracy/efficiency hoặc kết quả ablation thực tế. Kết quả suite và bản release cuối được ghi trong phần xác minh bên dưới.

## Chưa thể xác minh trên Mac
- Dataset Cars, DINO weights và batch128 memory thật.
- Môi trường Linux/driver/kiến trúc NVIDIA và CUDA kernels.
- Full benchmark, multi-seed, efficiency, tính mới và ưu thế phương pháp.
- Paper/code discrepancies và dependency GitHub-vs-PyPI trong report01 vẫn cần công khai khi báo baseline; bản hiện tại chọn reference config từ train script, không tuyên bố tái lập paper đã thành công.

## Trình tự bàn giao
Đọc RUNPOD.md. Đưa archive hoặc Git checkout lên volume, cài môi trường, chuẩn bị đúng files, chạy preflight. Chỉ sau preflight_passed mới train. Bắt đầu B0/B3 từng run, không sweep tự động. Mọi kết quả phải được đánh giá bằng cùng SS-kmeans/Hungarian-v2 và điều kiện so sánh.

“Hoàn tất source bàn giao” nghĩa là mã và kiểm thử local hoàn tất trong phạm vi trên; không đồng nghĩa phương pháp đã có tính mới hoặc môi trường NVIDIA đã được xác nhận.

## Xác minh source bàn giao
Run `local-smoke-20261029T151313Z-bdb37bc0`, timestamp UTC `2026-10-29T15:13:13.759403+00:00`: **26 tests đạt**, static syntax checks đạt, shell syntax4 scripts đạt; pip check không có dependency lỗi. Log/metadata tại `logs/local-smoke-20261029T151313Z-bdb37bc0`. Python3.9.6 macOS ARM64; không có NVIDIA. Git dự án chưa commit; source fingerprints lưu trong metadata và release manifest. Upstream git status sạch.

## Artifact đã bàn giao
Release `20261029T151419Z-a6d09471`: `dist/gcd-final-runpod-20261029T151419Z-a6d09471.tar.gz` (~1.1MB), SHA256 `0b77932f69f613c3c21ce91a7aa1ecd69b3a4ca0dc752368707ce97282b192c8`. Đã giải nén archive vào thư mục tạm độc lập và chạy lại toàn bộ suite: **26 tests đạt**, static checks đạt, exit0. Log giữ tại `logs/release-extracted-smoke.log`; không cần Git metadata trong gói. Xác nhận này được ghi sau khi đóng gói nên không tự sửa nội dung archive đã hash.
