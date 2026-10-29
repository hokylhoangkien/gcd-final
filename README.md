# Reliability-Aware GCD — bản bàn giao source

Nghiên cứu GCD tổng quát trên nền Hyp-SelEx; Stanford Cars là benchmark chung. Phát triển trên Mac, huấn luyện trên Linux NVIDIA. **Chưa có benchmark Cars, chưa xác minh CUDA và chưa xác nhận tính mới.**

Bắt đầu từ [RUNPOD.md](RUNPOD.md): chuyển code, cài môi trường, chuẩn bị Cars/DINO, chạy CUDA preflight rồi mới train.

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
- `upstream/HypCD`: reference pin `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`, không sửa.

Bản prototype adapter cũ lưu ở `legacy/`, không dùng trong entrypoints và không có trong release. Giấy phép và nguồn kế thừa tại [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Đóng gói không cần Git remote:

```bash
.venv/bin/python scripts/build_release.py
```

Gói trong `dist/` kèm SHA256; chỉ chứa source/config/tests/docs và upstream snapshot được kiểm hash. Không chứa dataset, weights, checkpoint, log, environment hoặc Git history. Mọi run có deadline/log/metadata; không tự chạy sweep hoặc thuê Pod.

Công thức, điểm chèn và giới hạn: [reports/06_implemented_method.md](reports/06_implemented_method.md). Trạng thái bàn giao và kết quả kiểm thử: [reports/07_release_readiness.md](reports/07_release_readiness.md). Hướng dẫn chính thức vẫn là [AGENTS.md](AGENTS.md).
