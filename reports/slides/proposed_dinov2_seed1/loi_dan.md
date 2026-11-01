# Lời dẫn báo cáo thực nghiệm

Bộ 8 slide, khoảng 5–7 phút. Có thể sửa chữ, bảng và hình khối trong PowerPoint. Font: Arial; bản PDF đã nhúng font. Lời dẫn cũng có trong speaker notes của PowerPoint.

## Slide 1 — Reliability weighting cho Hyp-SelEx

Em khảo sát việc bổ sung trọng số độ tin cậy vào Hyp-SelEx cho bài toán GCD. Đây là kết quả một lần chạy với seed 1 trên Stanford Cars, không phải kết luận phương pháp đã tốt hơn baseline. Đã chạy đủ 200 epoch và đánh giá riêng checkpoint cuối bằng semi-supervised k-means. Ba chỉ số cuối là All 81,23%, Old 86,66%, New 78,61%. All là chỉ số chính. Tên hướng nghiên cứu còn là tên làm việc; tính mới chưa được xác lập.

## Slide 2 — Bài toán và phần mở rộng

Trong GCD, tập có nhãn chỉ bao phủ các lớp đã biết, còn tập chưa nhãn chứa cả lớp đã biết và lớp mới. Mục tiêu là phân nhóm đúng toàn bộ ảnh chưa nhãn. DINOv2, biểu diễn hyperbolic và hierarchy là phần kế thừa từ Hyp-SelEx/HypCD. Phần em bổ sung là cách tính trọng số cho một số cặp pseudo-positive trong loss ở từng mức hierarchy. Giả thuyết là quan hệ lân cận ổn định đáng được tin cậy hơn, nhưng cần thực nghiệm để kiểm chứng. Không tạo classifier confidence và không dùng nhãn thật của ảnh chưa nhãn để tính score.

## Slide 3 — Reliability hoạt động như thế nào?

Mỗi ảnh có hai view được biến đổi ngẫu nhiên. Ở mỗi mức hierarchy, module xây lân cận theo khoảng cách hyperbolic. Cue thứ nhất là reciprocal top-k: hai ảnh có chọn nhau hay không. Cue thứ hai đo tính nhất quán của lân cận giữa hai view. Khi bật cả hai, score là tích của hai cue, nằm trong [0,1] và được detach. Trọng số chỉ áp dụng cho pseudo-positive khác sample ID và có ít nhất một đầu chưa nhãn. Khi hoạt động đầy đủ, weight bằng 0,1 + 0,9 × score, trừ các cặp được fallback bảo vệ. Trong ramp: w = 1 − alpha + alpha × (0,1 + 0,9 × score). Loss chuẩn hóa tổng trọng số positive theo từng anchor. Chỉ số epoch bắt đầu từ 0: 0–9 tắt, 10 bắt đầu, 29 đầy đủ. Không có tham số học thêm trong reliability.

## Slide 4 — Thiết lập và cách tính kết quả

Split SSB đã được kiểm tra: 98 lớp known, 98 lớp novel. Có 2.000 ảnh labeled và 6.144 ảnh unlabeled trong train; 50% ảnh train thuộc known được cấp nhãn. DINOv2 ViT-B/14 có bốn register tokens, fine-tune hai block cuối và projection head. Chỉ số chính dùng balanced semi-supervised k-means trên train-unlabeled với labeled anchors, Hungarian-v2, biết trước tổng số 196 lớp. Đây là đánh giá transductive của GCD, không phải validation hay khả năng tổng quát trên ảnh mới. Nhãn thật của ảnh unlabeled chỉ dùng tính accuracy sau dự đoán. All được tính theo số mẫu, không phải trung bình cộng Old và New. Có một disjoint test riêng 8.041 ảnh nhưng báo cáo bằng plain k-means, cần giữ tách biệt.

## Slide 5 — Kết quả cuối: All 81,23%

Đây là kết quả đánh giá riêng checkpoint cuối sau 200 epoch, không chọn epoch tốt nhất sau khi nhìn đường cong. Dòng paper chỉ là mốc tham khảo. Chênh lệch ở All lớn nhất cần quan tâm, còn New gần số paper không chứng minh phương pháp tốt hơn. Chưa thể quy mức giảm cho reliability vì khác điều kiện huấn luyện và chọn checkpoint. Đặc biệt, log cuối có Train All 51,77% và Test All 56,24% do gọi plain k-means. Sau đó đánh giá lại đúng SS-k-means thu được bộ số trong bảng. Các số này thuộc những protocol khác nhau, không có bằng chứng mô hình đột ngột sụp từ 81% xuống 51%. Cần áp dụng cùng bản sửa k-means và cùng quy trình evaluation cho baseline đối chứng.

## Slide 6 — Tăng sớm, sau đó chững và giảm nhẹ

Đường cong lấy trực tiếp từ log hai lần chạy nối tiếp, loại plain-k-means ở record 199. Trục ngang là số epoch đã hoàn tất lúc đo: metrics SS ở index e được đo trước khi train e. Ngôi sao là đánh giá SS riêng sau 200 epoch. Đường đậm là trung bình trượt 10 lần đánh giá; đường mờ là số đo từng lần. Từ nhóm 70–89 sang 170–189, Old giảm rõ trong khi New chỉ cải thiện nhẹ, khiến All giảm. Đây là mô tả một lần chạy, không chứng minh overfitting hay reliability là nguyên nhân. Lịch trộn loss gốc vẫn thay đổi trong quá trình train. Dao động giữa epoch không phải độ bất định giữa các seed, và không được chọn đỉnh đường cong để thay kết quả cuối một cách hậu nghiệm.

## Slide 7 — Reliability có tác động, chưa rõ lợi ích

Các thống kê này lấy từ diagnostics ở epoch index 189–198, 20 snapshots mỗi level. Mean weight được gộp theo số eligible ordered pairs; fallback là trung bình theo snapshot. Level 0 và level 6 có giảm trọng số quan sát được, còn level 1–3 thường gần weight 1 và fallback rất cao. Fallback là cơ chế bảo vệ khi anchor không có eligible positive với score dương; nó giữ weight 1 cho các cặp liên quan. Đây không phải tỷ lệ cặp sai hay xác suất nhãn đúng. Mỗi epoch chỉ log batch 0 và 50 trên 63 batch. Level 6 có hệ số loss 1/128 và loss chuẩn hóa theo anchor: raw weight không thể chuyển trực tiếp thành ảnh hưởng gradient. Muốn biết score chọn đúng cặp nhiễu cần audit offline trên checkpoint cố định; muốn biết accuracy tăng nhờ module cần baseline/ablation cùng điều kiện.

## Slide 8 — Kết luận và bước tiếp theo

Em đã triển khai một module có thể bật tắt, hoàn thành thí nghiệm 200 epoch, lưu checkpoint và đánh giá lại đúng protocol. Kết quả hiện tại chưa chứng minh cải tiến thuật toán: chỉ một seed, chưa có matched baseline, chưa ablation; tính mới cần đối chiếu sâu hơn với các hướng reliability, neighborhood và hierarchy đã có. Bản sửa empty cluster là xử lý kỹ thuật, không phải đóng góp học thuật; cần dùng nhất quán trong đối chứng. Bước ít tốn tài nguyên trước mắt là rà soát phương pháp và audit offline chất lượng cặp trên checkpoint cố định, không dùng nhãn evaluation để tune. Sau đó xin thống nhất với thầy thiết kế baseline và ablation trong ngân sách. Kết quả âm hoặc không cải thiện vẫn cần được báo cáo trung thực.

## Nguồn và phạm vi

- Kết quả đo: `results/cars_proposed_dinov2_seed1_user_reported_evaluation.json`, đối chiếu raw `evaluation.json` cuối trong backup.
- Cấu hình và số mẫu: raw `config.json`, `readiness.json` của run huấn luyện đã resume.
- Đường cong: `reports/figures/proposed_dinov2_seed1/history.csv` và `history.png`; gồm 199 số đo SS trước train và một đánh giá cuối sau 200 epoch. Không trộn record plain-k-means index 199.
- Diagnostics: `logs/final-backup-20261101/history-audit/report.md`, epoch index 189–198; batch 0 và 50. Các mức 1–3 trình bày khoảng min–max giữa level, không phải độ lệch chuẩn.
- Liu, He, Han. *Hyperbolic Category Discovery*, CVPR 2025, [Table 1 và mục 6.2](https://arxiv.org/html/2504.06120v1#S4.T1); số tham khảo DINOv2 trên Cars: 83.8 / 93.3 / 79.2. Đã đối chiếu trang tác giả ngày 2026-11-01.
- [Mã nguồn HypCD chính thức](https://github.com/Visual-AI/HypCD), upstream pin `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`.
- `reports/13_final_assessment.md`, `reports/09_kmeans_failure.md`, `reports/06_implemented_method.md`.

Không có matched baseline hoàn tất, ablation hay thống kê multi-seed. Không kết luận tính mới, SOTA hoặc readiness để công bố. Không chạy thêm huấn luyện để tạo slide.
