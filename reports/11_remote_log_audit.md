# Phân tích log remote đã tải — 2026-10-31

## Nguồn và giới hạn
Snapshot user tải bằng SCP khoảng19:48 giờ Việt Nam. Run20261031T055221Z-03368e3d, method proposed, DINOv2, seed1. Files trong logs/remote-audit/20261031T055221Z-03368e3d (gitignored). Có103 metric records epoch25–127;1456 diagnostic records epoch25–128,7 levels,208 snapshots/level, batch indices0 và50. Không có duplicate key(epoch,batch,level). All/Old/New finite. Không có log epoch0–24 của run trước trong bộ tải; không kết luận full trajectory hoặc so trước/sau warmup. Các file được tải nối tiếp khi job đang ghi, không phải snapshot atomic cùng thời điểm.

Lệnh đã chạy: python3 scripts/analyze_reliability_history.py --run logs/remote-audit/20261031T055221Z-03368e3d --output logs/remote-audit/analysis-20261031-01. Exit0. Kết quả report.md, diagnostics.json, metrics.csv. Không dùng GPU, không SSH, không thay đổi/dừng remote training.

## Metrics: cùng protocol qua thời gian
SS metrics epoch e đo trước update e, dù ghi sau epoch e hoàn tất; plain disjoint_test đo sau update e. Bảng dưới mô tả nhóm record cùng protocol, không ghép để suy ra hiệu quả trực tiếp của diagnostic e. Không chọn best checkpoint từ bảng.

| Record epoch range | SS All mean % | SS Old % | SS New % | Plain disjoint All mean % | Plain Old % | Plain New % |
|---|---:|---:|---:|---:|---:|---:|
|29–48|80.073|92.179|74.226|55.729|62.847|48.863|
|49–68|81.537|92.564|76.212|56.249|62.670|50.055|
|69–88|82.534|92.014|77.956|55.883|62.736|49.273|
|89–108|82.392|91.312|78.084|55.296|61.505|49.307|
|109–127 (19 records)|81.999|89.292|78.476|53.690|59.877|47.722|

SS cuối10 records118–127: All82.282, Old88.826, New79.121. Plain disjoint cùng range: All53.623, Old59.673, New47.786. SS record127: All82.1289, Old88.4558, New79.0731. Đây không phải final200-epoch result.

Nhận định: initial growth trong phần resume rồi plateau All; Old giảm, New tăng chậm. Trong plain disjoint protocol, các nhóm muộn có giảm All/Old/New. Mức tuyệt đối test và SS không trực tiếp so sánh: khác pool và labeled anchors, không coi chênh lệch82 vs54 là generalization gap đã kiểm soát. Trend cùng test protocol là dấu hiệu cần kiểm chứng, không chứng minh overfit hoặc reliability gây hại. Không có baseline matched, không suy ra gain riêng của module.

## Reliability: records118–127
20 snapshots/level. Mean weight và downweighted fraction được weighted theo số eligible ordered pairs; fallback là mean snapshot anchors, không pair fraction. Chỉ2/63 batch/epoch được log, không phải toàn epoch.

|Level|Eligible ordered pairs|Pair-weight mean|Downweighted %|Fallback %|Hierarchy coefficient|
|---|---:|---:|---:|---:|---:|
|0|890|0.921|35.06|72.23|0.5|
|1|2034|0.990|2.75|97.81|0.25|
|2|3984|0.992|2.01|96.52|0.125|
|3|9336|0.986|2.59|93.16|0.0625|
|4|20794|0.960|5.38|85.90|0.03125|
|5|43392|0.914|10.40|76.91|0.015625|
|6|145428|0.733|30.29|53.63|0.0078125|

Level1–3 gần identity trong phần lớn eligible pair mass. Level0 và6 thay đổi rõ; coefficient level6 nhỏ hơn level0 nhưng không suy ra gradient contribution từ coefficient/mean weights. Positive normalization theo anchor nên raw weight giảm không tương đương gradient giảm cùng tỷ lệ. Fallback và raw scores không cho biết true pair correctness. Không đánh đồng labeled/unlabeled với Old/New.

## Kết luận và next action
Đã chứng minh module thay đổi weighting ở sampled batches, chưa chứng minh weighting là đúng hoặc giúp quality. Không tăng budget chỉ vì expectation200 epochs sẽ vượt paper. Không yêu cầu dừng job từ phân tích mô tả này. Giữ logs/checkpoints và final-epoch policy. Trước thí nghiệm tốn thêm tiền, chuẩn bị offline fixed-checkpoint pair-quality audit (cần model+projection head+inputs, labels evaluation-only), phân biệt within-class correctness với coarse-hierarchy semantic compatibility. Nếu đạt, đối chứng B0 matched và ablation schedule-matched. Frozen-model inference tắt reliability không phải baseline counterfactual vì module chỉ ảnh hưởng training. Cùng dependency empty-cluster repair và provenance phải áp dụng/ghi rõ.
