# RP2K → CRiSH local → backbone cho planogram

## Trạng thái và phạm vi

Chuẩn bị từ branch feat/local-adaptive-acceptance commit 1c6ada8, trong branch
riêng feat/rp2k-crish-local. Một run SelEx có bật CRiSH local; không tự chạy thêm
baseline, ablation hay sweep. Chưa train RP2K và chưa xác nhận chất lượng retail.
Mac chỉ dùng phát triển/test; dừng tại đây cho đến khi chủ project thuê GPU.
Không thay đổi hoặc chạy thêm job trên máy YOLO detector đang thuê.

Backbone: DINOv2 ViT-B/14 reg4, vector768, input224. Projector65536 chỉ dùng khi
train. Local Keep/Abstain/Move giữ scoring, confirmation và guard của adaptive
profile. Population policy/online pair weighting tắt. Không dùng branch CRiSH-G v2.

## 1. Nguồn và protocol dữ liệu

- Paper tác giả: https://arxiv.org/abs/2006.12634 .
- Trang dự án cũ https://www.pinlandata.com/rp2k_dataset trả404 khi kiểm tra.
- Endpoint archive đã kiểm tra HTTP200, 6,330,687,649 bytes, ZIP64; chỉ đọc
  metadata/range nhỏ để xác nhận layout, chưa tải ảnh thật lên Mac:
  https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip
- Archive này là bản phân phối cần kiểm tra, không giả định số ảnh/lớp bằng đúng
  con số trong abstract. Layout thực: all/train/<SKU>/<image>, all/test/<SKU>/<image>.
  Số lượng chính xác được in sau index. Không dùng bản retrieval subset từ MTEB
  thay thế bản classification mà không thiết kế lại protocol.
- Chưa xác minh được checksum do tác giả công bố. Download lưu SHA256 và receipt
  của bytes thực nhận; đây là fingerprint tái lập, không tự chứng minh nguồn gốc.

Preparer kiểm tra Pillow, hash từng ảnh và loại duplicate **byte-identical** trong
  và giữa train/test, ưu tiên giữ train. Không chuyển ảnh test sang train.
  Ảnh giống hệt nhưng nhãn SKU khác làm preparation thất bại. Chưa tự phát hiện
  ảnh gần trùng hoặc cùng cảnh được encode lại; nếu có scene IDs, cần group split
  theo cảnh trước khi dùng số đo để kết luận chất lượng thực tế.

Class cần >=4 ảnh train khác nhau và >=1 test sau lọc; các class bị loại được ghi
trong manifest. SKU đủ điều kiện được xếp bằng SHA256(seed0:SKU): nửa đầu known,
nửa sau novel; mỗi known cấp nhãn cho floor(50%) ảnh train, tối thiểu2 anchors.
Split tùy chỉnh của project, **không phải SSB/benchmark split chính thức RP2K**.
Giả định biết tổng K từ manifest, không giải bài toán tự ước lượng số SKU.

Training chỉ đọc train.jsonl; target unlabeled=-1. Nhãn ẩn ở train_evaluation.jsonl
và test.jsonl chỉ dùng ở preparer/audit/evaluator độc lập. Final SS là điểm GCD
transductive trên train pool, không phải độ chính xác SKU end-to-end của app.
Không dùng test để chọn checkpoint, không có plain-test monitor từng epoch.

## 2. Máy GPU sẽ thuê

Ứng viên khởi đầu: Linux, một NVIDIA24GB, RAM64GB, disk150GB, Python3.10.
Profile dùng batch32 và clustering pairwise batch128 để giảm workspace. Các mức
này chưa được đo trên RP2K thật. Nhiều SKU làm ma trận khoảng cách và thời gian
clustering tăng; smoke nhỏ không chứng minh full-pool VRAM đã đủ. Chỉ quyết định
batch/GPU cuối sau preflight, smoke và đo epoch đầu. Có thể cần VRAM lớn hơn.

Mỗi epoch extract lại toàn pool và dựng hierarchy; 200epoch có thể tốn đáng kể
thời gian. Không có ETA/chi phí được bảo đảm. Giữ checkpoint và giới hạn worker
7ngày/run; evaluator2giờ. Nếu cần đổi batch/epoch/LR sau khi đo, tạo config/run
mới trước full; không sửa config/source của job đang chạy hoặc bypass resume.

Khác cấu hình Cars: batch128→32, LR0.1→0.025 theo tỉ lệ batch, unbalanced=true,
evaluation_rng=isolated, không best-test selection. Đây là lựa chọn triển khai
ban đầu, chưa được tuning/benchmark; không báo thành kết quả so sánh Cars.

## 3. Trên máy mới: source và môi trường

Chỉ làm phần này sau khi đã thuê GPU. Đặt /workspace trên storage được giữ lại
theo chính sách nhà cung cấp. Không xóa instance/volume trước khi tải kết quả về.

```bash
cd /workspace
git clone --branch feat/rp2k-crish-local --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git gcd-rp2k
cd gcd-rp2k
git status --short
nvidia-smi
python3 --version
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
```

Nếu image trống: cài python3/python3-venv/python3-dev, git, curl, build-essential,
tmux và ca-certificates bằng apt theo quyền máy. Bootstrap cần Python3.9–3.11 và
ghim torch2.4.1/torchvision0.19.1 CUDA12.1; không cài requirements của AI worker
hoặc dùng chung venv với YOLO. Không cần tự cài lại NVIDIA driver/CUDA toolkit
khi driver hoạt động. CUDA chỉ được công nhận sau preflight forward/backward thật.

Nếu GitHub private, dùng SSH key hoặc login; không đặt token trong lệnh/URL.
Nếu môi trường không truy cập được repo, có thể chuyển Git bundle từ Mac như
workflow detector. Giữ source commit và generated config cố định sau khi bắt đầu.

## 4. Download dataset và weights (một lần, chưa train)

```bash
export GCD_DATA_ROOT=/workspace/gcd-rp2k/datasets
export GCD_WEIGHTS_ROOT=/workspace/gcd-rp2k/weights
export GCD_RUNS_ROOT=/workspace/gcd-rp2k/runs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

.venv/bin/python scripts/prepare_rp2k.py download
.venv/bin/python scripts/prepare_rp2k.py extract
.venv/bin/python scripts/prepare_rp2k.py index
.venv/bin/python scripts/prepare_rp2k.py smoke-index
.venv/bin/python scripts/prepare_rp2k.py audit
.venv/bin/python scripts/prepare_rp2k.py plan --output runs/plans/rp2k/crish.json
```

Download có retry, deadline và resume archive. Extraction/index không ghi đè
thư mục/manifest cũ. Nếu preparation lỗi, xem thông báo và giữ dữ liệu để kiểm tra;
không chạy rm/reset đại trà. Audit mặc định kiểm checksum index, membership, nhãn,
path và kích thước ảnh. Thêm --rehash nếu ảnh đã được di chuyển/sửa và cần kiểm lại
toàn bộ bytes. Manifest/config sinh ở máy GPU, không commit ảnh hoặc nhãn vào Git.

Smoke có16 SKU ×48 train và tối đa2 test/SKU, được copy thành dataset nhỏ riêng;
known8 lớp,24 anchors/lớp đủ dự trữ9fold ×2 anchors/lớp. Đây là readiness subset,
không dùng điểm smoke làm benchmark. Không bỏ smoke khi nó không exercise policy.

Lấy weights DINOv2 reg4 đúng bản đã ghim trong repo:

```bash
mkdir -p weights downloads
curl --fail --location --retry 5 --retry-all-errors --max-time 1800 --continue-at - \
  --output downloads/dinov2_vitb14_reg4_pretrain.pth \
  https://huggingface.co/cubert-gmbh/dinov2/resolve/e2c8060a74112f7537484ed50097b1497d5f032c/dinov2_vitb14_reg4_pretrain.pth
printf '%s\n' '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71  downloads/dinov2_vitb14_reg4_pretrain.pth' | sha256sum -c -
```

Chỉ khi checksum OK, đặt vào vị trí runtime. Lệnh dưới từ chối ghi đè file cũ:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
source=Path('downloads/dinov2_vitb14_reg4_pretrain.pth')
target=Path('weights/dinov2_vitb14_reg4_pretrain.pth')
with source.open('rb') as src, target.open('xb') as dst:
    import shutil
    shutil.copyfileobj(src,dst)
PY
```

## 5. Check, dry-run và train một run CRiSH

```bash
GCD_CONFIG=runs/plans/rp2k/crish.json
bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local --dry-run
bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant local
```

Dry-run phải có dataset=rp2k, dino=v2, discrete_repair=true, unbalanced=true,
population/online=false. Snapshot smoke dùng rp2k/smoke, full dùng rp2k; hai hash
manifest khác nhau. Template configs/selex_rp2k_crish_200.json chưa pin dữ liệu,
không dùng template đó để train; luôn dùng generated plan.

Sau check đạt, mở tmux:

```bash
tmux new -s crish-rp2k
```

Trong tmux, đặt lại các biến đường dẫn nếu là shell mới, rồi chạy từ gốc repo:

```bash
mkdir -p logs
set -o pipefail
bash scripts/train_selex.sh --config runs/plans/rp2k/crish.json --variant local \
  2>&1 | tee logs/rp2k-launch.log
```

Launcher tự chạy bounded smoke trước rồi mới full; mỗi stage có run ID/log riêng
và preflight riêng. Không cần chạy thêm baseline. Smoke một epoch,2minibatch tối
đa nhưng vẫn extract/clustering toàn smoke subset và exercise CRiSH. Full200epoch,
batch32, input224, local refresh sau warmup10 rồi mỗi5epoch, cap0–8 edits/refresh.
Không bắt buộc có edit accepted; phải có scoring hợp lệ, không fallback/deadline.

Ctrl+B rồi D để detach. Theo dõi launcher và log RUN_ID thật được in ra:

```bash
tail -f logs/rp2k-launch.log
tail -f runs/RUN_ID/run.log
```

Ctrl+C ở tail chỉ dừng xem log. Ghi lại RUN_ID full, không lấy nhầm run smoke.

## 6. Resume và đánh giá

```bash
bash scripts/train_selex.sh --config runs/FULL_RUN_ID/config.json \
  --resume runs/FULL_RUN_ID/resume.pt
```

Source, environment, split và config phải khớp. Resume tạo run mới, dùng state
SGD/RNG từ run cũ; ghi lại ID mới. Không resume bằng backbone_final.pt.
Final SS chạy trên checkpoint cố định và nhãn đánh giá train-pool trong worker
riêng. Xem evaluation_last.json, metrics.jsonl, crish.jsonl, readiness.json,
metadata.json và environment.txt; status completed mới là toàn bộ run hoàn tất.

Nếu muốn plain clustering evaluation **một lần** trên native held-out test sau
khi đã khóa checkpoint cuối (không dùng làm chọn model):

```bash
.venv/bin/python -m gcd_runtime.worker \
  runs/FULL_RUN_ID/config.json runs/FULL_RUN_ID select \
  runs/FULL_RUN_ID/backbone_final.pt 199
```

Lệnh này dành cho run200epoch, dùng nhãn test chỉ để Hungarian metric; không đo
khả năng matching catalog hay reject UNKNOWN của app. Chạy có deadline bên ngoài
nếu dùng thủ công, ví dụ `timeout 7200 ...` trên Linux.

## 7. Export và tải về sau khi train xong

```bash
.venv/bin/python scripts/export_retail_backbone.py \
  --run runs/FULL_RUN_ID --output runs/exports/rp2k-v1
tar -czf runs/exports/rp2k-v1.tar.gz -C runs/exports rp2k-v1
```

Export từ chối run chưa completed hoặc smoke. Gói gồm backbone.pt, định nghĩa
kiến trúc đã ghim, contract.json, training_config.json và evaluation_last.json
nếu có. Contract: CLS768, L2-normalize, RGB, bicubic resize256/center crop224,
ImageNet normalization. Version là SHA256 weights, không dùng nhãn `main` chung.

Tải export cùng full-run logs/config/checkpoint về Mac qua scp/download của nhà
cung cấp trước khi kết thúc thuê. Giữ resume.pt nếu cần tiếp tục; không push
dataset/weights/run artifacts lên GitHub.

Tích hợp worker là bước tiếp theo: load đúng kiến trúc/checkpoint, dùng cùng
encoder cho ảnh catalog và crop từ detector, tạo lại embedding catalog768chiều,
đồng bộ contract backend và hiệu chỉnh threshold trên validation. Cluster ID
không tự là ProductID; chưa có độ chính xác retail/end-to-end được xác nhận.
