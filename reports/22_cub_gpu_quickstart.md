# CUB adaptive local — từ máy GPU mới đến training

Máy riêng Linux NVIDIA, Ubuntu22.04, Python3.10; terminal root. Không dùng
checkout/GPU đang chạy Cars. Dataset CUB-200-2011 gồm 200 loài chim;
dùng SSB100 known/100 novel của repo, không tự chia class.
Phương pháp: SelEx + CRiSH local adaptive0–8 action, 200 epoch, SS mỗi epoch.
Batch training128; chưa bật batch chủ động A/B. Chưa xác minh full adaptive
benchmark CUB; CUDA/preflight/smoke thật vẫn bắt buộc trên GPU thuê.

Chạy lần lượt từng khối. Có lỗi thì dừng và gửi log, không cài lại từ đầu.

## 1. Công cụ và tmux

```bash
nvidia-smi
apt-get update
apt-get install -y git curl tar tmux coreutils ca-certificates build-essential python3.10 python3.10-venv python3.10-dev
mkdir -p /workspace
cd /workspace
tmux new -s cub-adaptive
```

Từ đây làm bên trong tmux. Không cài lại NVIDIA driver. Nếu tài khoản không
phải root thì dùng sudo cho apt-get. /workspace là ví dụ nơi lưu project;
chọn volume được nhà cung cấp giữ lại nếu vị trí thực tế khác.

## 2. Code và môi trường ghim

```bash
git clone --branch feat/local-adaptive-acceptance --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git gcd-final-cub-adaptive
cd gcd-final-cub-adaptive

/usr/bin/python3.10 -m venv .venv
.venv/bin/python -m pip install pip==24.3.1
.venv/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r environments/requirements-common.txt -r environments/requirements-research-tests.txt
.venv/bin/python -m pip install -r environments/requirements-selector-cuda.txt
.venv/bin/python -m pip install --force-reinstall --no-deps -r environments/requirements-selector-cuda.txt
.venv/bin/python -m pip install --only-binary=:all: --no-deps xformers==0.0.28.post1
.venv/bin/python -m pip check
.venv/bin/python scripts/bootstrap_upstream.py
```

Nếu thư mục repo đã tồn tại, Git từ chối clone để bảo vệ dữ liệu; không xóa
hoặc ghi đè. Hai lệnh selector cài dependencies trước, rồi ghim chính backend
Git mà không đổi Torch. Cùng các phiên bản như bootstrap CUDA của repo.
Không gọi lại bootstrap wrapper, vì nó chạy full CPU test với timeout60s.

Chạy CPU suite đúng một lần với timeout300s, tắt CUDA chỉ cho lệnh này:

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 300s .venv/bin/python -m pytest tests -q --tb=short --junitxml=logs/cub-cpu-tests.xml
```

Phải có passed, không fail/timeout. CUDA được kiểm thật bằng pretrained
image forward/backward trong preflight trước training, không phải test CPU.

## 3. Tải CUB-200-2011 và kiểm tra dữ liệu

Nguồn chính thức: https://www.vision.caltech.edu/datasets/cub_200_2011/
Archive images + annotations và MD5: https://data.caltech.edu/records/65de6-vp158
Dùng bản 2011, không dùng CUB-200 bản 2010 hoặc archive chỉ có segmentation.

```bash
mkdir -p downloads datasets/cub weights logs
export GCD_DATA_ROOT="$PWD/datasets"
export GCD_WEIGHTS_ROOT="$PWD/weights"
export GCD_RUNS_ROOT="$PWD/runs"

curl -fL -C - --connect-timeout 20 --max-time 3600 -o downloads/CUB_200_2011.tgz 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
echo '97eceeb196236b17998738112f37df78  downloads/CUB_200_2011.tgz' | md5sum -c -
```

Chỉ giải nén sau checksum OK. Archive khoảng1.2GB; giải nén trực tiếp trong
dataset root, không tạo symlink ảnh sang downloads. MD5 đối chiếu bản phát hành.

```bash
tar -xzf downloads/CUB_200_2011.tgz -C datasets/cub --keep-old-files

.venv/bin/python - <<'PY'
from pathlib import Path
from collections import Counter
from gcd_runtime.datasets import validate_cub_metadata
root = Path('datasets/cub')
validate_cub_metadata(root)
base = root / 'CUB_200_2011'
images = (base / 'images.txt').read_text().splitlines()
classes = {int(x.split()[1]) for x in (base / 'image_class_labels.txt').read_text().splitlines()}
pools = Counter(x.split()[1] for x in (base / 'train_test_split.txt').read_text().splitlines())
assert len(images) == 11788, len(images)
assert classes == set(range(1, 201)), len(classes)
assert pools == {'1': 5994, '0': 5794}, pools
attrs = base / 'attributes/image_attribute_labels.txt'
assert attrs.stat().st_size > 0, attrs
print('CUB: 11788 images; 200 classes; native train=5994, test=5794; paths/attributes: OK')
PY
```

--keep-old-files bảo vệ file đã có. Nếu giải nén lại báo file exists, chạy
riêng đoạn validator; đừng xóa dataset để tải lại. Loader upstream còn đọc
attributes/image_attribute_labels.txt: thiếu file này thì dừng, gửi lỗi và
kiểm tra archive/layout, không tự tạo attributes giả. Chưa tải archive trên
Mac để kiểm tra nội dung; validator trên GPU là kiểm tra thực tế.

```text
datasets/cub/CUB_200_2011/
  images/<species>/*.jpg
  images.txt
  image_class_labels.txt
  train_test_split.txt
  attributes/image_attribute_labels.txt
```

5994/5794 là hai pool native. Preflight tiếp tục chia labeled/unlabeled
trong native train bằng SSB và labeled fraction của config, ghi counts thật;
không lấy native test vào training. Attributes được loader đọc theo upstream,
không thêm chúng làm tín hiệu chấm điểm repair.

## 4. Weights DINOv2 reg4

```bash
curl -fL -C - --connect-timeout 20 --max-time 1800 -o weights/dinov2_vitb14_reg4_pretrain.pth https://huggingface.co/cubert-gmbh/dinov2/resolve/e2c8060a74112f7537484ed50097b1497d5f032c/dinov2_vitb14_reg4_pretrain.pth
echo '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71  weights/dinov2_vitb14_reg4_pretrain.pth' | sha256sum -c -
```

Phải OK. Có thể dùng weights đã có ở volume khác qua GCD_WEIGHTS_ROOT;
không cần tải lại hoặc tạo symlink ảnh ra ngoài dataset root.

## 5. Sinh profile adaptive và train với SS mỗi epoch

```bash
.venv/bin/python scripts/prepare_local_adaptive.py --base configs/selex_cub_crish_200.json --output runs/plans/cub-adaptive-200

bash scripts/train_selex.sh --config runs/plans/cub-adaptive-200/adaptive.json --variant local --eval-every 1 --dry-run
```

Plan phải có dataset_name=cub, epochs=200, discrete_repair=true,
max_accepted=8, boundaries=16, train_batch=128, confirmation_budget=8 và
population_policy.enabled=false. Output plan folder phải chưa tồn tại.

Không cần gọi check/smoke thủ công rồi lặp lại: launcher train tự chạy
preflight CUDA/data, bounded smoke và đánh giá smoke trước khi mở full run.

```bash
set -o pipefail
timeout --signal=TERM --kill-after=60s 48h bash scripts/train_selex.sh --config runs/plans/cub-adaptive-200/adaptive.json --variant local --eval-every 1 2>&1 | tee "logs/cub-adaptive-$(date -u +%Y%m%dT%H%M%SZ).log"
```

48h là giới hạn dừng, không phải ETA. Đặt theo thời gian thuê còn lại.
Lúc đầu epochs=1/max_steps_per_epoch=2 là smoke. Sau đó phải thấy run mới
epochs=200/max_steps_per_epoch=0, rồi TRAIN epoch=0 step=0. Chưa có bước SGD
thì chưa tuyên bố full training đã chạy. Smoke không phải kết quả benchmark.
Nếu OOM/thiếu support/fail preflight, gửi run.log; không tự nới guard, giảm
batch, cài lại Torch hoặc chạy job cạnh tranh. Không hứa trước VRAM của full
CUB adaptive chỉ từ kết quả Cars trên cùng loại GPU.

## 6. Thoát, xem log, can thiệp và lưu kết quả

Ctrl+B rồi D để rời tmux, không dừng job. Vào lại:

```bash
tmux attach -t cub-adaptive
```

Terminal SSH khác, đọc log run mới nhất:

```bash
cd /workspace/gcd-final-cub-adaptive
tail -n 40 -f "$(ls -t runs/20*/run.log | head -n 1)"
```

Tail không tự đổi file khi smoke chuyển sang full. Ctrl+C chỉ dừng tail;
chạy lại lệnh để chọn log mới nhất. Giữ máy thuê hoạt động, không pull source
hay sửa config đang chạy. Checkpoints/evaluation/logs nằm trong runs/RUN_ID.

Đọc can thiệp (thay RUN_ID bằng thư mục full thật):

```bash
.venv/bin/python scripts/analyze_crish_interventions.py runs/RUN_ID --output "logs/cub-interventions-$(date -u +%Y%m%dT%H%M%SZ).csv"
```

Refresh đầu ở epoch hiển thị11, rồi16,21... SS đánh giá mỗi epoch.
JSON epoch bắt đầu0; epoch:10 tương ứng hiển thị11. intervention_diagnostics
ghi supervision thật sau repair, không đo trực tiếp số cặp gặp trong SGD.
Không suy ra tác động nhân quả từ SS trước/sau một action.

Trước khi trả máy, tải về cả thư mục full run, manifest plan và log launcher.
Resume dùng đúng config snapshot/source, không bootstrap hay đổi hyperparameter.
Các helper chuyển cadence không cần dùng: run này đã SS mỗi epoch từ đầu.

Baseline đối chứng (job riêng, GPU riêng hoặc sau khi adaptive kết thúc):

```bash
timeout --signal=TERM --kill-after=60s 48h bash scripts/train_selex.sh --config runs/plans/cub-adaptive-200/baseline.json --baseline --eval-every 1
```

Baseline cùng optimization/evaluation và seed; báo fixed-final All/Old/New,
per-seed và variability khi có compute, không chỉ best. 200 epoch là lịch
mở rộng, không đồng nhất lịch50 epoch của bảng corrected SelEx.

## Kiểm chứng hướng dẫn

Đã đọc CUB loader, metadata validator, config và profile generator thật;
URL và MD5 đối chiếu CaltechDATA. Kiểm cú pháp shell, generated-config
và CLI dry-run trên Mac. Không tải dataset/weights, cài CUDA hoặc chạy GPU
training. Chưa xác nhận download, CUDA preflight hay full benchmark CUB thật.

Kiểm chứng2026-11-16: 12 khối Bash qua bash -n; sinh profile CUB và CLI
dry-run đạt (200 epoch, batch128, cap8, SS mỗi epoch); git diff --check đạt.
