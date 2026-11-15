# Aircraft adaptive local — từ máy GPU mới đến training

Máy riêng Linux NVIDIA, Ubuntu22.04, Python3.10; terminal root. Không dùng
checkout/GPU đang chạy Cars. Hướng dẫn này chạy FGVC-Aircraft2013b,
100 variant, SSB50 known/50 novel, trainval để train và test để đánh giá riêng.
Phương pháp: SelEx + CRiSH local adaptive0–8 action, 200 epoch, SS mỗi epoch.
Batch training128; chưa bật cách xếp batch chủ động A/B. Không hứa accuracy
hay thời gian hoàn thành. Bộ dữ liệu này chưa có full adaptive benchmark
được agent xác minh; CUDA/preflight/smoke thật vẫn bắt buộc trên máy thuê.

Chạy lần lượt từng khối. Có lỗi thì dừng và gửi log, không cài lại từ đầu.

## 1. Công cụ và tmux

```bash
nvidia-smi
apt-get update
apt-get install -y git curl tar tmux coreutils ca-certificates build-essential python3.10 python3.10-venv python3.10-dev
mkdir -p /workspace
cd /workspace
tmux new -s aircraft-adaptive
```

Từ đây làm bên trong tmux. Không cài lại NVIDIA driver. Nếu tài khoản không
phải root thì dùng sudo cho apt-get. /workspace là ví dụ nơi lưu project;
chọn volume được nhà cung cấp giữ lại nếu vị trí thực tế khác.

## 2. Code và môi trường ghim

```bash
git clone --branch feat/local-adaptive-acceptance --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git gcd-final-aircraft-adaptive
cd gcd-final-aircraft-adaptive

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
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 300s .venv/bin/python -m pytest tests -q --tb=short --junitxml=logs/aircraft-cpu-tests.xml
```

Phải có passed, không fail/timeout. CUDA được kiểm thật bằng pretrained
image forward/backward trong preflight trước training, không phải test CPU.

## 3. Tải đúng dataset, giải nén trực tiếp vào dataset root

Nguồn chính thức:
https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

Full2013b archive có test annotations. MD5 được trang phát hành công bố:
https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.html

```bash
mkdir -p downloads datasets/aircraft weights logs
export GCD_DATA_ROOT="$PWD/datasets"
export GCD_WEIGHTS_ROOT="$PWD/weights"
export GCD_RUNS_ROOT="$PWD/runs"

curl -fL -C - --connect-timeout 20 --max-time 3600 -o downloads/fgvc-aircraft-2013b.tar.gz https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
echo '32eca553f897e747706144f3bb444df0  downloads/fgvc-aircraft-2013b.tar.gz' | md5sum -c -
```

Chỉ giải nén nếu checksum OK. Không tạo symlink sang downloads:

```bash
tar -xzf downloads/fgvc-aircraft-2013b.tar.gz -C datasets/aircraft --keep-old-files

.venv/bin/python - <<'PY'
from pathlib import Path
from gcd_runtime.datasets import validate_aircraft_metadata
root = Path('datasets/aircraft/fgvc-aircraft-2013b')
validate_aircraft_metadata(root)
for pool, expected in [('trainval', 6667), ('test', 3333)]:
    rows = (root / f'data/images_variant_{pool}.txt').read_text().splitlines()
    assert len(rows) == expected, (pool, len(rows))
    print(pool, len(rows), 'images: OK')
print('Aircraft metadata, classes, image paths, disjoint pools: OK')
PY
```

--keep-old-files bảo vệ file cũ; chạy lại trên thư mục đã giải nén có thể báo
file exists. Đừng xóa dữ liệu rồi tải lại; chạy riêng đoạn kiểm tra layout.
MD5 ở đây dùng đối chiếu archive phát hành, không phải bảo đảm an toàn mật mã.
Full archive nằm khoảng2.75GB theo trang phát hành; không dùng annotation-only.

Layout sau giải nén:

```text
datasets/aircraft/fgvc-aircraft-2013b/data/
  images/*.jpg
  images_variant_trainval.txt
  images_variant_test.txt
```

## 4. Weights DINOv2 reg4

```bash
curl -fL -C - --connect-timeout 20 --max-time 1800 -o weights/dinov2_vitb14_reg4_pretrain.pth https://huggingface.co/cubert-gmbh/dinov2/resolve/e2c8060a74112f7537484ed50097b1497d5f032c/dinov2_vitb14_reg4_pretrain.pth
echo '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71  weights/dinov2_vitb14_reg4_pretrain.pth' | sha256sum -c -
```

Phải OK. Có thể dùng weights đã có ở volume khác qua GCD_WEIGHTS_ROOT;
không cần tải lại hoặc tạo symlink ảnh ra ngoài dataset root.

## 5. Sinh profile adaptive và train với SS mỗi epoch

```bash
.venv/bin/python scripts/prepare_local_adaptive.py --base configs/selex_aircraft_crish_200.json --output runs/plans/aircraft-adaptive-200

bash scripts/train_selex.sh --config runs/plans/aircraft-adaptive-200/adaptive.json --variant local --eval-every 1 --dry-run
```

Plan phải có dataset_name=aircraft, epochs=200, discrete_repair=true,
max_accepted=8, boundaries=16, train_batch=128, confirmation_budget=8 và
population_policy.enabled=false. Output plan folder phải chưa tồn tại.

Không cần gọi check/smoke thủ công rồi lặp lại: launcher train tự chạy
preflight CUDA/data, bounded smoke và đánh giá smoke trước khi mở full run.

```bash
set -o pipefail
timeout --signal=TERM --kill-after=60s 48h bash scripts/train_selex.sh --config runs/plans/aircraft-adaptive-200/adaptive.json --variant local --eval-every 1 2>&1 | tee "logs/aircraft-adaptive-$(date -u +%Y%m%dT%H%M%SZ).log"
```

48h là giới hạn dừng, không phải ETA. Đặt theo thời gian thuê còn lại.
Lúc đầu epochs=1/max_steps_per_epoch=2 là smoke. Sau đó phải thấy run mới
epochs=200/max_steps_per_epoch=0, rồi TRAIN epoch=0 step=0. Chưa có bước SGD
thì chưa tuyên bố full training đã chạy. Smoke không phải kết quả benchmark.
Nếu OOM/thiếu support/fail preflight, gửi run.log; không tự nới guard, giảm
batch, cài lại Torch hoặc chạy job cạnh tranh. Không hứa trước VRAM của full
Aircraft adaptive chỉ từ kết quả Cars trên cùng loại GPU.

## 6. Thoát, xem log, can thiệp và lưu kết quả

Ctrl+B rồi D để rời tmux, không dừng job. Vào lại:

```bash
tmux attach -t aircraft-adaptive
```

Terminal SSH khác, đọc log run mới nhất:

```bash
cd /workspace/gcd-final-aircraft-adaptive
tail -n 40 -f "$(ls -t runs/20*/run.log | head -n 1)"
```

Tail không tự đổi file khi smoke chuyển sang full. Ctrl+C chỉ dừng tail;
chạy lại lệnh để chọn log mới nhất. Giữ máy thuê hoạt động, không pull source
hay sửa config đang chạy. Checkpoints/evaluation/logs nằm trong runs/RUN_ID.

Đọc can thiệp (thay RUN_ID bằng thư mục full thật):

```bash
.venv/bin/python scripts/analyze_crish_interventions.py runs/RUN_ID --output "logs/aircraft-interventions-$(date -u +%Y%m%dT%H%M%SZ).csv"
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
timeout --signal=TERM --kill-after=60s 48h bash scripts/train_selex.sh --config runs/plans/aircraft-adaptive-200/baseline.json --baseline --eval-every 1
```

Baseline cùng optimization/evaluation và seed; báo fixed-final All/Old/New,
per-seed và variability khi có compute, không chỉ best. 200 epoch là lịch
mở rộng, không đồng nhất lịch50 epoch của bảng corrected SelEx.

## Kiểm chứng hướng dẫn

Đã đọc loader/config/preparer thật, đối chiếu archive URL và MD5 với trang
phát hành. Chỉ kiểm cú pháp shell và generated-config dry-run trên Mac; không
tải2.75GB, không cài CUDA, không truy cập máy thuê, không khởi chạy GPU job.
Chưa xác nhận hoàn thành download, preflight hay benchmark Aircraft thật.

Kiểm chứng2026-11-16: 12 khối Bash qua bash -n; generated Aircraft adaptive
config và CLI dry-run với --eval-every1 đạt; tests/test_layout.py 3 passed
in0.02s; git diff --check đạt. Không thay runtime/config lịch sử.
