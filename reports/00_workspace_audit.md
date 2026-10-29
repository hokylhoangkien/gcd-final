# Kiểm kê workspace — 2026-10-29 (Asia/Ho_Chi_Minh)

Đã đọc toàn bộ AGENTS.md; đây là hướng dẫn chính thức.

## Hiện trạng ban đầu
- Thư mục: `<PROJECT_ROOT>`; chỉ có AGENTS.md, chưa có code, dataset, log, checkpoint hoặc cấu hình thực nghiệm.
- `git status --short`: thất bại vì chưa có Git repository; chưa có SHA dự án.
- Không có dữ liệu/công việc cũ bị ghi đè hoặc xóa.

## Tiến trình
Lần đầu `ps` bị sandbox chặn; đọc lại với quyền được cấp thành công. Snapshot lúc khoảng 19:13: chỉ thấy Codex app-server PID 15050 và code-mode-host PID 15191; không thấy Python/training/HypCD trong bộ lọc. Đây là snapshot, không đảm bảo không có job khác ngoài bộ lọc. Không dừng hay thay đổi process nào.

## Máy và môi trường
- macOS Darwin 25.6.0, ARM64; MacBook Air Apple M5, CPU 10 lõi, GPU Apple 8 lõi, RAM hợp nhất 16 GB.
- Metal được hỗ trợ; PyTorch/MPS chưa kiểm tra. Không có `nvidia-smi`; không có NVIDIA CUDA trên máy này.
- Python hệ thống 3.9.6, Git 2.50.1; dung lượng trống khoảng 380 GiB.
- Không lưu các mã định danh phần cứng riêng tư vào báo cáo.

## Thiếu và hành động an toàn
Chưa có upstream, split, ảnh Cars, DINO weights, dependency environment hoặc GPU CUDA. Chưa đủ điều kiện benchmark. Tiếp theo tải upstream chính thức vào thư mục riêng, pin SHA; kiểm tra paper và code trước khi chọn điểm chèn. Chỉ chạy kiểm thử nhỏ có timeout, không khởi động training dài hoặc sweep.

## Cập nhật sau audit
Clone đầu tiên trong sandbox lỗi DNS. Clone lại có quyền mạng thành công, pin commit `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`; `git -C upstream/HypCD status --short` rỗng. Upstream có SSB split thật, nhưng chưa có images/MAT/weights. Python hệ thống thiếu torch, numpy, scipy, pytest. Có thêm integrity checker và tests độc lập; xem report03. Chưa thay baseline hoặc khởi động training.

## Cập nhật chiến lược local/remote — phiên tiếp theo 2026-10-29
Máy M5/16GB được xác định là máy phát triển, không phải blocker cho kiến trúc nghiên cứu; benchmark sẽ dùng Linux NVIDIA do chủ dự án chuẩn bị sau. Snapshot mới chỉ thấy Codex và dịch vụ IDE/Python environment, không thấy training. Không dừng process nào.
Đã tạo `.venv` Python3.9.6 riêng và cài torch2.4.1/torchvision0.19.1 cùng dependencies cho CPU tests; không tải dataset/pretrained weights. Đã khởi tạo Git cha và đăng ký upstream thành submodule; chưa commit/push và chưa có remote dự án. Upstream vẫn sạch.
Đã có runtime tách biệt, config di động, bootstrap và launcher Linux; chi tiết `reports/05_local_remote_workflow.md`. CPU synthetic/epoch-resume và import/loss fixture đã chạy; CUDA vẫn chưa kiểm thử trên NVIDIA. Các kết luận thiếu dependencies ở phần audit ban đầu mô tả thời điểm trước khi tạo venv, không còn áp dụng cho venv hiện tại.

## Cập nhật bản bàn giao RunPod
Snapshot tiến trình mới chỉ thấy Codex/IDE services, không thấy training. Không dừng process hoặc tải dataset/weights. Source trainer riêng, module reliability, packaging và test CPU được hoàn thiện; upstream vẫn giữ nguyên. Git cha còn chưa commit và chưa push, nên có thêm archive portable không chứa Git metadata. Xem report07/RUNPOD.md cho trạng thái hiện hành và giới hạn NVIDIA chưa kiểm thử.
