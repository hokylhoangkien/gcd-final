# Phương pháp đã hiện thực: ứng viên reliability-aware Hyp-SelEx

Trạng thái: đã có code và kiểm thử CPU synthetic; chưa có Cars/NVIDIA run, chưa xác nhận cải thiện hoặc tính mới. Tên RAH-GCD chỉ là tên làm việc. Module toán không phụ thuộc Cars; runner hiện kiểm tra Cars + DINO v1.

## Điểm chèn đúng code
`gcd_runtime/trainer.py` là source trainer trực tiếp, kế thừa reference có MIT notice và provenance. Không sửa upstream hoặc rewrite source khi chạy. Prototype adapter cũ chỉ lưu ở legacy, không dùng trong release. `gcd_runtime/reliability.py` và `method_config.py` chứa thuật toán. `EpochRuntime.hierarchy_loss` điều phối mỗi nhánh, cache weights dùng chung để giữ thứ tự tạo graph của baseline.

Trong `train_HypSelEx.py`, baseline có supervised contrastive trên labeled thật; sau đó cộng SupCon trên pseudo labels ở nhiều mức. Chỉ những call pseudo-hierarchy này được weighted. USE, hệ số loss, optimizer, pseudo clustering và manifold giữ nguyên. Labeled–labeled trong pseudo loss và mọi same-ID pair có trọng số1; supervised true-label objective tách riêng không bị weighted. Phép chuẩn hóa positive trong pseudo objective thay đổi tương quan các positive, nên không diễn giải rằng đóng góp labeled trong chính pseudo objective hoàn toàn bất biến.

Hai ngoại lệ số học cho proposed: batch không có labeled thì supervised term là zero thay vì gọi criterion rỗng; confusion normalization chỉ có fallback hữu hạn cho zero-mass columns. Với mass dương giữ nguyên broadcast của upstream, dù code này không tương đương row normalization theo nghĩa thông thường. Baseline vẫn giữ phép tính gốc; mọi loss đều có finite check trước backward.

## Ký hiệu và score
Tại level l (code index bắt đầu0), dùng `z_l = semisup_con_feats[:, :, :floor(D / 2^(l+1))]` theo baseline. z có shape B×2×d_l, đã map Poincare. Không tạo classifier probabilities.

Chọn top-k bằng khoảng cách `hyptorch.pmath.dist_matrix` riêng cho mỗi view. Không có self-neighbor. Nếu sampler lặp ID, chọn occurrence đầu của mỗi unique ID chỉ để tính graph; broadcast score về các occurrence. Các occurrence cùng ID không trở thành neighbor giả, và pair giữa chúng luôn có weight1. Tie break ổn định theo unique ID đã sort. K hiệu dụng min(k, số unique IDs−1).

Gọi A¹,A² là adjacency directed hai view, Rᵛ=Aᵛ∩(Aᵛ)ᵀ. Score reciprocal:

`R = (R¹ + R²) / 2`.

Agreement từng node là Jaccard `J_i=|N¹_i∩N²_i| / max(|N¹_i∪N²_i|,1)`. Persistent symmetric edge `E=(A¹∩A²) ∪ (A¹∩A²)ᵀ`. Score cross-view:

`V_ij = E_ij * sqrt(J_i * J_j)`.

Nếu bật cả hai: `s_ij=R_ij*V_ij`; nếu bật một chỉ dùng thành phần đó. Graph không dùng true labels hoặc known/novel identities. Pseudo labels chỉ xác định tập positive P_l sau khi score feature-only đã có. Score và weight detach, đối xứng, trong [0,1]. Không có trainable reliability head.

## Membership, fallback và schedule
Eligible pair: cùng pseudo label ở level l, khác sample ID, và ít nhất một đầu unlabeled (`either_unlabeled`, mặc định). Có thể đổi `both_unlabeled`. Labeled membership là thông tin training được phép, không phải oracle old/novel của unlabeled.

Node không có eligible positive với score>0 được fallback. Bảo vệ đối xứng mọi incident pair khi một trong hai node fallback, tránh loss zero hoặc vô tình phá symmetry. Những pair khác không eligible có weight1.

`alpha(e)=0` khi e<warmup; sau đó `min(1,(e−warmup+1)/ramp)`; ramp0 tương ứng bật ngay. Với pair active:

`w_ij(e)=1−alpha(e)+alpha(e)*(floor+(1−floor)*s_ij)`.

Mặc định B3: k5, floor0.1, warmup10, ramp20, epoch0-based; epoch0–9 giữ baseline, epoch10 strength.05, epoch29 strength1. Đây là **giá trị khởi đầu chưa tune**, không có bằng chứng tối ưu. Floor giữ tín hiệu nonzero; fallback tránh loại sạch neighborhood. Chưa chứng minh giải quyết confirmation bias/known domination, phải xem diagnostics và Old/New thực tế.

## Loss hiện thực
Giữ logits a_ij^g = −d_g(z_i,z_j)/temperature và mẫu số contrastive trên mọi sample view khác chính anchor. Nhánh được upstream gọi angle thực tế dùng Euclidean distance trên features hiện tại; không tự đổi thành cosine. Nhánh hyperbolic dùng đúng dist_matrix.

Với positive P_l(i) của pseudo labels (bao gồm other view cùng sample), thay trung bình positive bằng:

`L_hier^g = mean_i [sum_(j in P_l(i)) w_ij * (−log p_ij^g) / sum_(j in P_l(i)) w_ij] * temperature/base_temperature`.

Weights tile sang thứ tự view-major đúng SupCon gốc. Self-view positive giữ1 nên mass luôn dương. Bản weighted dùng logsumexp để tránh underflow. Khi mọi weight1, gọi criterion gốc để có exact value/gradient parity. Tắt hai thành phần, floor1 hoặc warmup cũng đi đường criterion gốc.

Giữ hệ số hierarchy `2^−(l+1)`, sup_con_weight và lịch trộn angle/hyperbolic. Không normalize hoặc thay cấu trúc supervised loss gốc. Code source loss gốc không sửa, chỉ dispatch call hierarchy.

## Diagnostics và rủi ro
`reliability.jsonl` lấy mẫu mỗi50 batch tại mỗi level khi reliability active. Có eligible ordered-pair count, effective weight mass, mean/histogram weights, raw-score mean, downweighted/zero fraction, node fallback fraction, neighbor agreement, unique IDs, means theo labeled/unlabeled anchor. Không gọi downweighted là excluded; chỉ weight0 mới bị loại trong positive averaging. Không có oracle novel diagnostics trong training score.

Chi phí bổ sung hai B_unique×B_unique distance matrices mỗi level và graph/score tensors; chưa đo VRAM/thời gian trên NVIDIA. K nhỏ tại hierarchy coarse có thể hạ weight quá nhiều hoặc gây floor domination; fallback có thể làm phương pháp gần baseline. Cross-view overlap cao không đảm bảo semantic correctness. Batch composition và first duplicate occurrence có thể ảnh hưởng score. Đây là các yếu tố phải ablate, không phải bảo đảm tăng All accuracy.

## So sánh và chạy
B0 `cars_baseline_dino_v1.json`: reference loss gốc, final_epoch selection chung.
B1 `cars_neighborhood_dino_v1.json`: reciprocal only, bật ngay.
B2 `cars_crossview_dino_v1.json`: cross-view only, bật ngay.
B3 `cars_proposed_dino_v1.json`: cả hai + warmup/ramp.

B3 vs B1/B2 còn khác schedule theo thiết kế minimum AGENTS; cần thêm combined-no-ramp hoặc schedule-matched components trước khi quy gain riêng cho tương tác hai cue. `scripts/ablation_plan.py` chỉ chuẩn bị config/command, không khởi chạy sweep.

Trên NVIDIA Linux đã có data/weights/environment:

```bash
.venv/bin/python -m gcd_runtime.launch preflight --config configs/cars_proposed_dino_v1.json
bash scripts/train_proposed_linux.sh
bash scripts/evaluate_linux.sh --config configs/cars_proposed_dino_v1.json --checkpoint runs/<RUN_ID>/backbone_final.pt
```

Preflight kiểm branch reliability ở strength đầy đủ bằng tensor synthetic (trained_epochs0), không đợi10 epoch. Toàn bộ NVIDIA path vẫn chưa được kiểm thử trong phiên Mac này.
