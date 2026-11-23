# CRiSH-G v2 preregistration

Verbatim configuration thresholds (section 6) and decision rules (section 8)
from owner-supplied finetuneok.md; source SHA256: 64dc2dad36f499a4fb95d11f727c2e705fbcb7c4648357926656ea424520c218

## 6. Khóa cấu hình mới

Mọi khóa dưới đây có mặc định giữ nguyên hành vi cũ; cột cuối là giá trị dùng cho profile `g2_`. Khóa nằm trong `population_policy`, trừ khi cột "Nơi" ghi khác.

| Khóa | Nơi | Mặc định | Giá trị g2 | WP |
| --- | --- | --- | --- | --- |
| `probe_view_reduction` | `crish` | `mean_all` | `per_row_sum` | WP1 |
| `scorer` |  | `exact_sgd` | `first_order_fp64` | WP2 |
| `selector` |  | `legacy` | `normalized_v1` | WP3 |
| `dose_mode` |  | `threshold` (hành vi hiện tại) | `rank_quantile` | WP4 |
| `dose_quantiles` |  | – | \[0; 0,10; 0,25; 0,40\] | WP3 |
| `fixed_dose` |  | – | 0,25 (`g2_fixed`, `g2_random`, `g2_oracle_dose`) | WP7 |
| `abstain_ceiling` |  | 1,0 (không trần) | 0,5 | WP4 |
| `min_boundary_size` |  | 0 | 20 | WP4 |
| `hierarchical_abstain` |  | false | true | WP4 |
| `tie_break` |  | `id` | `hash` | WP4 |
| `t_min` |  | – | 1,0 | WP3 |
| `t_min_confirm` |  | – | 0,0 | WP3 |
| `guard_se` |  | – | 2,0 | WP3 |
| `sigma_floor_ratio` |  | – | 0,25 | WP3 |
| `scoring_batches` (đã có) |  | 1 | 8 | WP3 |
| `confirmation_batches` |  | 0 | 8 | WP3 |
| `confirm_fresh_chunks` |  | false | true | WP3 |
| `scoring_sampling` (đã có) |  | `uniform` | `train_sampler` | WP3 |
| `reliability_ema.enabled` / `beta` / `reset_jaccard` |  | false / 0,7 / 0,5 | false; chỉ bật theo luật Phase 0 | WP4 |
| `evidence_views` |  | 1 | 1; chỉ lên 2 theo luật Phase 0 | WP6 |
| `random_control` |  | `none` | `count_matched` (chỉ `g2_random`) | WP7 |
| `diagnostic_oracle` / `oracle_labels_csv` |  | false / null | true / đường dẫn CSV (chỉ `g2_oracle_dose`) | WP7 |
| `move_gate_asymmetric` |  | false | false | WP5 |
| `emergency_deadline_seconds` |  | – | 900 | WP8 |
| `diagnostics.use_saturation` / `diagnostics.exposure` |  | false / false | true / true | WP9 |
| `ss_eval_last_epochs` | top-level | 0 | 10 (Phase 3) | WP10 |

Khóa đã có, giữ giá trị sau cho mọi profile `g2_`: `knn_k` = 10, `nonreciprocal_weight` = 0,5, `max_cluster_fraction` = 0,5, `min_cluster_kept` = 2, `min_retained_mass` = 0,7, `move_min_share` = 0,60, `move_target_cap` = 0,25, `warmup_epochs` = 10, `refresh_every` = 5, `save_arrays` = true, `persist` = true (false ở `g2_cf_nopersist`), `actions` = `abstain` (`abstain_move` ở `g2_cf_move`).

Thêm khóa `confirm` (mặc định true, giữ bước xác nhận như hiện tại). Riêng `g2_cf_noconfirm` đặt `confirm: false` để bỏ hẳn bước xác nhận. Với `confirm_fresh_chunks: false` (mặc định), bước xác nhận dùng lại chunk của discovery như code cũ.

## 8. Lộ trình chạy và luật quyết định

Agent chỉ làm Phase A; các phase sau do nhóm chạy trên GPU theo đúng luật dưới đây. File đăng ký trước chứa nguyên văn mục này phải được commit trước khi chạy G1.

| Phase | Ai làm | Việc | Chi phí | Đầu ra |
| --- | --- | --- | --- | --- |
| A | Agent | WP1–WP11, test, smoke 2 epoch | CPU, tối đa khoảng 15 phút GPU | Code, test đạt, báo cáo |
| 0 | Nhóm | Hậu kiểm offline, đánh giá 5 seed k-means, hook bão hòa | Vài phút GPU | Quyết định EMA và hai view; ghi nhận về USE và bộ đánh giá |
| G1 | Nhóm | Cars, seed 1, 100 epoch, 5 run | ≈ 20 GPU-giờ | Go, vùng xám hoặc No-Go |
| 3 | Nhóm | Ma trận nhiều seed | ≈ 190–210 GPU-giờ | Bảng chính, ablation, số liệu cơ chế |

### Phase 0: hậu kiểm trên dữ liệu có sẵn

Đầu vào:

- Run gốc của 3 run CRiSH local, chứa `crish.jsonl`, `partitions_epoch_*.npz` và `supervision_epoch_*.npz`: Cars `btl-dl/cars-paper-light-20261115T112441Z-0ec6f014.tar.gz`, CUB `runs/downloads/cub-results-20261116T002137Z-056eafad.zip`, Aircraft `runs/downloads/aircraft-results-20261116T002008Z-ce74b2d9.zip`. Thư mục `results/paper-plan-verified-20261118/` chỉ có bản tổng hợp, không có `crish.jsonl`.
- Thư mục run CRiSH-G trên Cars đã có, kèm mảng đã lưu.
- File CSV nhãn offline do `scripts/export_offline_labels.py` sinh ra.
- Checkpoint cuối `backbone_final.pt` của các run. Checkpoint epoch 198 và 199 nhiều khả năng đã bị ghi đè.

Luật:

1. **EMA** chỉ bật cho G1 khi AUROC(1 − r̃⁰) ≥ AUROC(1 − r⁰) + 0,02 trên mảng CRiSH-G.
2. **Hai view** chỉ bật khi AUROC hai view ≥ AUROC một view + 0,03, tính trên một checkpoint (vài phút GPU). Nếu cả hai đều dưới 0,60, vẫn chạy G1 nhưng ghi trước rằng cổng chọn ảnh nhiều khả năng trượt.
3. **Bộ đánh giá:** nếu biên độ All qua 5 seed k-means của một checkpoint cuối từ 2,0 điểm trở lên, ghi "xác nhận bộ đánh giá bất ổn". Median qua 5 seed vẫn là chỉ số cuối chính trong mọi trường hợp.
4. **USE:** nếu frac\_F1 > 0,99 ở mọi level trong smoke, ghi "USE bão hòa" để đưa vào paper.

Không luật nào ở Phase 0 làm đổi thiết kế lõi.

### G1: sàng lọc trên Cars

- Các run: `matched_selex`, `g2_fixed`, `g2_random`, `g2_cf`, `g2_oracle_dose`; Cars, seed 1, 100 epoch trên lịch cosine 200 epoch, cùng commit.
- Đánh giá SS mỗi epoch (một seed k-means). Dừng ở epoch 100 bằng `scripts/stop_after_checkpoint.py` để có thể chạy tiếp tới 200.
- Cửa sổ: epoch\_number 61–100. Δ(X) là trung bình cửa sổ của X trừ trung bình cửa sổ của `matched_selex`.

Luật, xét theo thứ tự:

1. **Trần hiệu quả.** Nếu Δ\_All(`g2_oracle_dose`) < +1,0 thì dừng CRiSH-G v2: ở liều này, không chính sách không dùng nhãn nào có thể kỳ vọng đạt +1.
2. **Vùng quyết định**, áp riêng cho `g2_cf` và `g2_fixed`:
   - **Go:** Δ\_All ≥ +1,0, Δ\_Old ≥ −1,0 và Δ\_New ≥ 0.
   - **Vùng xám:** +0,3 ≤ Δ\_All < +1,0. Chạy thêm seed 2 cho `matched_selex` và nhánh đó tới epoch 100, rồi Go nếu trung bình 2 seed cho Δ\_All ≥ +0,7 với cùng điều kiện Old và New.
   - **No-Go:** Δ\_All < +0,3.
3. **Phương pháp cuối.** Nếu cả hai nhánh Go: chọn `g2_cf` khi `g2_cf` − `g2_fixed` ≥ +0,5 (trung bình cửa sổ), ngược lại chọn `g2_fixed` và đưa bộ chọn vào ablation. Chỉ một nhánh Go thì chọn nhánh đó. Không nhánh nào Go thì dừng.
4. **Cổng chọn ảnh** (hậu kiểm offline trên mảng của G1): tỷ số mức giàu lỗi của cặp bị gỡ ở level 0, `g2_fixed` / `g2_random`, phải ≥ 1,5 và có cận dưới khoảng tin cậy bootstrap 95% > 1,0. Trượt thì paper không được claim việc chọn ảnh theo độ tin cậy; dừng trước Phase 3 và báo nhóm.
5. **Kiểm hợp lý** cho nhánh được chọn:
   - tỷ lệ D\_U bị Abstain ở level 0, trung bình theo epoch, nằm trong 3–35%;
   - với `g2_cf`, q\* > 0 ở ít nhất 50% số refresh;
   - `deadline_triggered` bằng 0 ở mọi refresh.

Chỉ sang Phase 3 khi cả năm luật đều đạt.

### Phase 3: ma trận nhiều seed

| Nhóm | Run | Seed | Ghi chú |
| --- | --- | --- | --- |
| Chính | `matched_selex` và phương pháp cuối, trên Cars, CUB, Aircraft | 1, 2, 3 | Seed 1 trên Cars chạy tiếp từ G1 |
| Đối chứng (Cars) | `g2_fixed`, `g2_random`; thêm `g2_cf` nếu phương pháp cuối là `g2_fixed` | 1, 2, 3 | Seed 1 chạy tiếp từ G1 |
| Ablation (Cars) | `g2_cf_nopersist`, `g2_cf_noconfirm`, `g2_cf_move` | 1 | `g2_cf_move` chỉ thêm seed 2 và 3 nếu hơn `g2_cf` từ +0,5 trở lên ở seed 1 |

Đánh giá trong Phase 3: `ss_eval_every_epochs = 5`, `ss_eval_last_epochs = 10`; chỉ số cuối là median qua 5 seed k-means trên checkpoint cuối (WP10).

### Phân tích và báo cáo

- **Chỉ số chính:** All ở checkpoint cuối (median 5 seed k-means). Lấy hiệu ghép cặp (phương pháp cuối − `matched_selex`), gộp 9 cặp (dataset, seed), kiểm bằng paired t-test hai phía với α = 0,05. Báo thêm Wilcoxon signed-rank và sign test.
- **Chỉ số phụ:** theo từng dataset, mean ± std và khoảng tin cậy 95% ghép cặp; Old và New; trung bình 10 epoch cuối; chỉ số theo protocol một seed chuẩn.
- **Đối chứng trên Cars (3 seed):** việc chọn ảnh (`g2_fixed` so với `g2_random`) và việc chọn liều (`g2_cf` so với `g2_fixed`), kèm số liệu cơ chế từ WP11.
- **Không dùng làm chỉ số chính:** số best-epoch (chỉ để ở phụ lục, ghi rõ là chọn bằng nhãn đánh giá). Số SelEx đã công bố chỉ đặt ở một khối riêng, ghi "reported, different protocol".

### Ngân sách GPU

Theo thời gian mỗi epoch đã log, một run 200 epoch tốn khoảng 7,8 giờ trên Cars, 6,5 giờ trên CUB và 8,4 giờ trên Aircraft, chưa kể thời gian đánh giá.

| Hạng mục | GPU-giờ |
| --- | --- |
| G1: 5 run × 100 epoch | ≈ 20 |
| Vùng xám của G1, nếu có | ≈ 8–12 |
| Phase 3, nhóm chính: 18 run, tái dùng run G1 | ≈ 128 |
| Phase 3, đối chứng trên Cars | ≈ 39 (thêm ≈ 20 nếu phương pháp cuối là `g2_fixed`) |
| Phase 3, ablation | ≈ 23 |
| Tổng Phase 3 | ≈ 190–210 |

