# EzyCloudX: thuê máy và chạy GCD

Console đăng nhập không đọc được bằng công cụ web; chưa xác minh giá, GPU còn trống hoặc tên các nút. Hướng dẫn này dùng launcher thật của repository, không giả định EzyCloudX có layout RunPod.

## Chọn máy

Ưu tiên một NVIDIA RTX 3090/4090 24GB nếu có; đây là cấu hình khởi đầu để kiểm thử, chưa bảo đảm batch128 vừa VRAM. RAM host nên từ32GB, khoảng8 vCPU và100GB disk là mức dự trù ban đầu, không phải số đo. Chọn Linux Docker có terminal/SSH, Python3.10 hoặc3.11 và NVIDIA driver phù hợp CUDA12.1. Không chọn Windows/RDP. Không ưu tiên GPU Blackwell với profile torch2.4.1 chưa kiểm thử. Xác minh nơi lưu bền vững, phí storage khi stop và thao tác kết thúc tính phí. Chưa có dự toán giờ train; bắt đầu bằng phiên kiểm tra ngắn theo điều kiện thanh toán hiển thị.

## Đưa code và dữ liệu lên

Dùng Git private hoặc archive theo RUNPOD.md; các lệnh bootstrap/train dùng chung cho Linux. Thay /workspace bên dưới bằng mount bền vững thực tế EzyCloudX cung cấp; không tự giả định thư mục này tồn tại.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git
cd gcd-final
python3 scripts/bootstrap_upstream.py
python3 --version
nvidia-smi
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
export GCD_DATA_ROOT=/workspace/gcd-data
export GCD_WEIGHTS_ROOT=/workspace/gcd-weights
export GCD_RUNS_ROOT=/workspace/gcd-runs
```

Private Git yêu cầu đăng nhập tài khoản có quyền trước clone; không đặt token trong URL. Nếu có checkout sẵn, kiểm tra git status và tiến trình trước, không clone đè.

Cần Cars images và annotations thật:

```text
$GCD_DATA_ROOT/cars/cars_train/*.jpg
$GCD_DATA_ROOT/cars/cars_test/*.jpg
$GCD_DATA_ROOT/cars/devkit/cars_train_annos.mat
$GCD_DATA_ROOT/cars/devkit/cars_test_annos_withlabels.mat
$GCD_WEIGHTS_ROOT/dino_vitb16_pretrain.pth
```

## Preflight và trial có giới hạn

```bash
.venv/bin/python -m gcd_runtime.launch preflight --config configs/cars_baseline_dino_v1.json
```

Chỉ sau preflight_passed, tạo config trial mới, không sửa cấu hình benchmark:

```bash
.venv/bin/python - <<'PYCONFIG'
import json
from pathlib import Path
p = Path('configs/cars_baseline_ezy_trial.json')
if p.exists():
    raise SystemExit('Trial config already exists; inspect before reuse')
cfg = json.loads(Path('configs/cars_baseline_dino_v1.json').read_text())
cfg.update(epochs=1, max_runtime_seconds=1800)
p.write_text(json.dumps(cfg, indent=2) + '\n')
PYCONFIG
bash scripts/train_linux.sh --config configs/cars_baseline_ezy_trial.json
```

Trial tối đa1800s cho worker cộng preflight; có thể hết giờ trước epoch đầu và chưa có checkpoint. Đây là kiểm tra vận hành, không phải benchmark. Launcher in PID, log path và run ID. Không chạy job khác cạnh tranh. Sau khi kiểm tra VRAM, thời gian epoch, dữ liệu và ngân sách mới dùng full config200 epoch/max86400s; không tự khởi chạy từ tài liệu này.

```bash
bash scripts/train_linux.sh --config configs/cars_baseline_dino_v1.json
```

Đọc RUNPOD.md để resume, evaluate và backup. Chỉ chạy proposed sau khi baseline vận hành ổn, với điều kiện so sánh khớp; tính mới chưa xác nhận. Trước kết thúc thuê, tải toàn bộ run logs/metadata/checkpoints về và xác minh file; kiểm tra trạng thái tính phí trong console. Chưa có CUDA hoặc Cars benchmark thực tế trong phiên chuẩn bị này.
