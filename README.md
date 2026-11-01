# Reliability-Aware GCD — code và kết quả thực nghiệm

Nghiên cứu GCD tổng quát trên nền Hyp-SelEx; Stanford Cars là benchmark chung. Phát triển trên Mac, huấn luyện trên Linux NVIDIA.

## Trạng thái ngày 01/11/2026

Đã hoàn thành một lần huấn luyện proposed với DINOv2 trên Stanford Cars SSB, **200 epoch, seed 1**, có checkpoint và final evaluation được xác minh. Kết quả cuối theo **balanced semi-supervised k-means trên train-unlabeled có labeled anchors**, Hungarian-v2:

| Kết quả quan sát | All (%) | Old (%) | New (%) |
|---|---:|---:|---:|
| Proposed DINOv2, checkpoint cuối | 81.23 | 86.66 | 78.61 |

Đây không phải validation accuracy. **Chưa có baseline 200 epoch cùng điều kiện, chưa hoàn tất ablation, chưa xác nhận tính mới hoặc lợi ích của reliability.** Không diễn giải số paper hoặc khác biệt backbone/cấu hình thành cải thiện thuật toán.

- [Đánh giá cuối và giới hạn](reports/13_final_assessment.md), [kết quả JSON](results/cars_proposed_dinov2_seed1_user_reported_evaluation.json).
- [Biểu đồ quá trình huấn luyện](reports/figures/proposed_dinov2_seed1/history.png), [CSV số đo](reports/figures/proposed_dinov2_seed1/history.csv).
- [Slide PowerPoint](reports/slides/proposed_dinov2_seed1/bao_cao_thuc_nghiem.pptx), [PDF](reports/slides/proposed_dinov2_seed1/bao_cao_thuc_nghiem.pdf), [lời dẫn](reports/slides/proposed_dinov2_seed1/loi_dan.md).
- [Xác minh bản sao lưu](reports/12_final_backup_verified.md), [bản sửa k-means khi resume](reports/09_kmeans_failure.md).

Git lưu source/config, báo cáo, số đo nhỏ và slide. Dataset, pretrained weights, `model.pt`, `resume.pt`, `backbone_final.pt` và raw logs nằm ngoài Git; vẫn cần giữ riêng bản sao lưu checkpoint. Bản sửa k-means phải áp dụng giống nhau cho các lần chạy đối chứng.

## Phát triển và tái lập

Bắt đầu từ [RUNPOD.md](RUNPOD.md): chuyển code, cài môi trường, chuẩn bị Cars/DINO, chạy CUDA preflight rồi mới train. Với DINOv2 xem [hướng dẫn chuyển backbone](reports/08_dinov2_migration.md); các lệnh DINO v1 dưới đây là ví dụ ban đầu, không phải cấu hình của kết quả cuối phía trên.

```bash
# Trên máy phát triển, environment đã cài:
.venv/bin/python scripts/run_smoke.py
.venv/bin/python -m gcd_runtime.launch train --config configs/cars_proposed_dino_v1.json --dry-run

# Trên Linux NVIDIA, sau khi chuẩn bị data/weights:
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
.venv/bin/python -m gcd_runtime.launch preflight --config configs/cars_proposed_dino_v1.json
bash scripts/train_proposed_linux.sh
```

Source chính:

- `gcd_runtime/trainer.py`: trainer Python trực tiếp, kế thừa reference Hyp-SelEx có ghi nguồn; không runtime source rewriting/exec.
- `gcd_runtime/reliability.py`: reciprocal hyperbolic neighbors, cross-view agreement và weighted hierarchy loss.
- `gcd_runtime/method_config.py`: floor, warm-up/ramp-up và switches.
- `gcd_runtime/checkpoint.py`: dispatch baseline/proposed, diagnostics và epoch resume.
- `configs/cars_*_dino_v1.json`: B0, B1, B2, B3, cùng thiết lập tối ưu hóa chính.
- `configs/cars_baseline_dino_v2.json`, `configs/cars_proposed_dino_v2.json`: cấu hình baseline/proposed DINOv2; kết quả full-run hiện chỉ có proposed seed 1.
- `upstream/HypCD`: reference pin `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`, không sửa.

Bản prototype adapter cũ lưu ở `legacy/`, không dùng trong entrypoints và không có trong release. Giấy phép và nguồn kế thừa tại [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Đóng gói không cần Git remote:

```bash
.venv/bin/python scripts/build_release.py
```

Gói trong `dist/` kèm SHA256; chỉ chứa source/config/tests/docs và upstream snapshot được kiểm hash. Không chứa dataset, weights, checkpoint, log, environment hoặc Git history. Mọi run có deadline/log/metadata; không tự chạy sweep hoặc thuê Pod.

Công thức, điểm chèn và giới hạn: [reports/06_implemented_method.md](reports/06_implemented_method.md). Báo cáo [07](reports/07_release_readiness.md) ghi trạng thái bàn giao ban đầu; báo cáo [13](reports/13_final_assessment.md) ghi đánh giá thực nghiệm hiện tại. Hướng dẫn chính thức vẫn là [AGENTS.md](AGENTS.md).
