# Dàn ý paper — tài liệu chuẩn bị, chưa có đóng góp xác nhận

1. Bài toán GCD tổng quát; Cars là benchmark đối chiếu, không phải thiết kế đặc thù.
2. Related work: HypCD/SelEx, RCA, LEAH, HypGCD, CiPR, RLCD; hoàn tất method review trước novelty claim.
3. Phương pháp: chỉ viết thuật toán/loss cuối khi implementation và tests thống nhất; đã có công thức và source ứng viên trong report06; chưa có benchmark để viết kết luận.
4. Protocol: split/checksum, pretrained backbone, masks, matching, selection rule, hyperparameters, training seeds, hardware, exact commands và source SHA.
5. Main/ablation/efficiency tables phải sinh từ observed result files. Hiện tất cả B0–B3 chưa chạy; không tạo hình hay điền số dự đoán.
6. Phân tích Old/New trade-off, reliability, failure/negative results và sensitivity; tối thiểu 3 matched seeds khi compute cho phép.
7. Hạn chế: novelty chưa thông qua; discrepancy baseline; chưa có CUDA/data/weights/dependencies; chưa có kiểm chứng dataset thứ hai.

Danh sách nguồn xác minh ban đầu nằm trong report02; chưa xuất BibTeX đầy đủ cho các đầu mục chưa đọc method.
