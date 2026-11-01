# Kiểm tra tác dụng của reliability

## Kết luận hiện tại
Log scalar đã xác nhận thay đổi weight, chưa xác nhận score tương ứng quan hệ đúng và chưa xác nhận gain accuracy do module. Các snapshot user cung cấp cho thấy fallback cao ở level1–3 và Old/New trade-off. Không thể suy ra nhân quả từ sự tăng accuracy sau warm-up.

## Bước A: chỉ đọc log, không GPU
Dùng scripts/analyze_reliability_history.py --run <original-run> <resume-run>. Cần config.json, metrics.jsonl, reliability.jsonl tại mỗi thư mục. Script từ chối duplicate epoch/config khác để không ghép nhầm thực nghiệm. Tách SS và plain-kmeans, tổng hợp weight theo eligible pair count, nêu snapshot coverage và hierarchy coefficients, không coi chúng là gradient magnitude. Có thể chạy script trên Mac sau copy3 file mỗi run; không cần checkpoint/images/torch. Không chỉnh source remote đang chạy.

Lưu ý quan trọng: trainer lấy SS metrics TRƯỚC cập nhật epoch e, rồi ghi vào record epoch e sau khi train xong. Đây là đánh giá tại e completed epochs. Plain disjoint_test được đo SAU cập nhật e. Vì vậy bảng metrics.jsonl trước đây ghi nhãn epoch index, không thể coi là cùng trạng thái model với disjoint_test hoặc diagnostics của chính epoch đó. Cần đối chiếu SS ở record e+1 với model sau e; clustering stochastic vẫn làm before/after không thành đối chứng nhân quả. Final train report epoch199 đổi sang plain-kmeans; evaluate checkpoint cuối riêng. Không sửa log/runtime giữa run.

## Bước B: audit offline quan hệ, cần chuẩn bị riêng
Checkpoint cố định + backbone + head đã lưu; sampleID và transforms cố định; đo precision pseudo-positive theo level, so cặp được giảm weight với cặp giữ weight1, và score bins với pair correctness. Phân tích Old/Old, Old/New, New/New offline; không dùng evaluation labels để tạo graph, tune k/floor, chọn checkpoint hay training. Với coarse hierarchy, same-class là chỉ số hẹp: khác fine class có thể là positive coarse hợp lý, phải thừa nhận hạn chế proxy. Không lưu embedding/pair IDs trong logs hiện tại nên không tính được chỉ từ jsonl. Không gọi score là calibrated probability. Audit toàn bộ module dùng projection head, không chỉ backbone. Đánh giá model cố định nhiều clustering seeds để phân biệt variation clustering với update variation nếu ngân sách cho phép; không chọn best seed. Bước này chưa được hiện thực/chạy và không khởi động GPU audit cạnh tranh job.

## Bước C: kiểm định lợi ích accuracy
Baseline matched backbone/split/label fraction/optimizer/epochs/seed/evaluation/selection và cùng dependency empty-cluster repair. Khởi tạo baseline độc lập, không lấy proposed checkpoint tắt reliability ở inference (reliability chỉ train). Smoke1 epoch cũ không làm baseline200 epoch. Ablation schedule-matched; một graph chung vs graph từng level kiểm tra claim per-level, shuffled-weight control kiểm tra generic regularization. Chỉ multi-seed variance mới hỗ trợ stability; epoch std không phải seed uncertainty. Không chi thêm GPU trước khi có ngân sách cụ thể.

## Evidence
3 tests stdlib pass: timestamp/protocol alignment and duplicate resume epochs, pair-weighted aggregates with empty snapshot, malformed vs partial last JSON record. Chưa phân tích raw logs đầy đủ: chúng ở máy thuê, chưa được copy về trong bước này. Không thay đổi/dừng run hiện tại.
