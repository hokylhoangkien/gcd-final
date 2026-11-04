# CRiSH-GCD v2.1

Source nghiên cứu GCD với **một pipeline huấn luyện**: CRiSH mặc định, B0 dùng cùng pipeline và tắt sửa phân hoạch để đối chứng. Backbone mặc định: DINOv2 ViT-B/14 reg4; benchmark: Stanford Cars SSB.

Code đã có kiểm thử CPU tổng hợp. **Chưa xác nhận chạy CUDA/dữ liệu thật và chưa có kết quả benchmark CRiSH.** Đây là bản bàn giao source; hiệu quả và tính mới của phương pháp vẫn cần thực nghiệm.

## 1. Cài trên máy train

Linux + NVIDIA CUDA, Python 3.9–3.11. Dung lượng VRAM và thời gian thực tế phải đo bằng smoke test; cấu hình đầy đủ dùng batch 128, projector 8192 chiều, 200 epoch.

Giải nén gói source rồi mở terminal trong thư mục vừa giải nén. Gói đã chứa upstream HypCD đúng phiên bản, không cần clone lại.

```bash
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
```

Lệnh tạo `.venv` nếu chưa có, cài PyTorch 2.4.1/CUDA 12.1 và dependencies, xác minh upstream rồi chạy test CPU có timeout. Cần Internet để cài dependencies. Nếu driver không chạy được bộ CUDA này, xử lý môi trường trước khi train.

## 2. Đặt dataset và pretrained weights

**Gói source không kèm ảnh dataset hoặc pretrained weights.** Tải chúng trực tiếp trên máy Linux NVIDIA sau bước cài môi trường ở mục 1; không cần tải bộ dữ liệu lớn về Mac. Lệnh train không tải data/weights tự động. Các lệnh dưới chạy từ thư mục gốc project, dùng đường dẫn mặc định. Nếu đã có đủ file ở ổ khác, bỏ qua bước tải và đặt biến môi trường ở cuối mục này.

### 2.1. Tải Stanford Cars

Nguồn ảnh và annotations dạng gốc: [mirror cộng đồng trên Hugging Face](https://huggingface.co/datasets/iharabukhouski/stanford_cars/tree/main), khoảng 2 GB theo danh sách file. Nguồn Stanford cũ hiện không còn tải được theo [Torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html). Mirror chưa được kiểm chứng toàn bộ trên workspace này; preflight sẽ kiểm tra ảnh, annotation và membership trước khi train.

Cài thêm công cụ tải rồi tải nguyên tên file/thư mục, giữ nguyên train/test gốc. Cú pháp dùng [Hugging Face CLI 0.26.2](https://huggingface.co/docs/huggingface_hub/v0.26.2/guides/cli):

```bash
.venv/bin/python -m pip install 'huggingface_hub[cli]==0.26.2'
timeout 3600 .venv/bin/huggingface-cli download iharabukhouski/stanford_cars \
  --repo-type dataset --local-dir datasets/cars \
  --include 'cars_train/**' 'cars_test/**' 'devkit/**' 'cars_test_annos_withlabels.mat'
```

`timeout` là lệnh trên Linux; giới hạn tải một giờ, không phải ước lượng thời gian hoàn tất. Nếu tải bị ngắt, chạy lại lệnh download để tiếp tục các file còn thiếu. Nếu nhóm đã có bản Cars dùng trong thí nghiệm trước, có thể dùng lại bản đó và chạy preflight.

Mirror đặt annotation test ở thư mục gốc, còn loader của project đọc trong `devkit/`. Chép file vào đúng vị trí, giữ file đã có:

```bash
mkdir -p datasets/cars/devkit
cp -n datasets/cars/cars_test_annos_withlabels.mat datasets/cars/devkit/
```

Không đổi tên ảnh, tự chia lại train/test hoặc tự sắp ảnh theo tên lớp. Split Cars SSB đã đi kèm upstream: 98 lớp known, 98 lớp novel; 50% ảnh của các lớp known được gán nhãn theo loader gốc. Nhãn của tập unlabeled chỉ được dùng trong evaluator riêng sau khi khóa checkpoint.

### 2.2. Tải pretrained DINOv2

Dùng **DINOv2 ViT-B/14 với 4 register tokens**, file `dinov2_vitb14_reg4_pretrain.pth`. [Weights tải trực tiếp từ Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) được liệt kê trong [repo DINOv2 chính thức](https://github.com/facebookresearch/dinov2). Kiến trúc model đã nằm trong upstream, chỉ cần bổ sung weights.

Nếu chưa có weights, chạy:

```bash
mkdir -p weights
curl --fail --location --continue-at - --connect-timeout 20 --max-time 1800 \
  --output weights/dinov2_vitb14_reg4_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
```

Lệnh giới hạn 30 phút và có thể chạy lại để tiếp tục tải dở. Preflight nạp state dict vào đúng kiến trúc, ghi SHA256 weights và thử forward/backward CUDA; tải thành công chưa phải xác nhận CUDA.

### 2.3. Kiểm tra vị trí file

```text
datasets/cars/
├── cars_train/                         # ảnh train gốc
├── cars_test/                          # ảnh test gốc
└── devkit/
    ├── cars_train_annos.mat
    └── cars_test_annos_withlabels.mat
weights/
└── dinov2_vitb14_reg4_pretrain.pth
```

Nếu data/weights đã ở ổ khác, đặt biến môi trường trỏ tới **thư mục cha** tương ứng. Ví dụ `GCD_DATA_ROOT` chứa thư mục `cars/`, còn `GCD_WEIGHTS_ROOT` chứa trực tiếp file `.pth`:

```bash
export GCD_DATA_ROOT=/path/to/datasets
export GCD_WEIGHTS_ROOT=/path/to/weights
export GCD_RUNS_ROOT=/path/to/runs
```

Sau khi đặt đủ file và biến môi trường, kiểm tra trên máy CUDA:

```bash
bash scripts/train_crish.sh check
```

Chỉ tiếp tục mục 3 khi preflight thành công. Nếu báo thiếu file, sửa vị trí theo thông báo; không tự tạo annotation hoặc split thay thế.

## 3. Chạy train

Kiểm tra lệnh và cấu hình, không dùng GPU:

```bash
bash scripts/train_crish.sh --dry-run
```

Train CRiSH:

```bash
bash scripts/train_crish.sh
```

Lệnh này chạy **preflight → smoke 1 epoch/tối đa 2 batch → một run mới 200 epoch**. Mỗi stage chỉ tiếp tục khi stage trước thành công. Preflight kiểm tra split/ảnh/weights/import, forward/backward CUDA thật, backbone và loss/scorer. Smoke không tiếp tục thành checkpoint của run chính.

Preflight giới hạn 180 giây; smoke tối đa 1.800 giây; training đầy đủ tối đa 86.400 giây; final evaluation tối đa 1.800 giây. Đây là giới hạn dừng, không phải dự báo thời gian hoàn thành. Chương trình chạy foreground, in PID và đường dẫn log, lưu resume sau mỗi epoch. Nếu đã có tiến trình GPU, lệnh dừng để người vận hành kiểm tra; không tự dừng tiến trình khác.

Các lệnh bổ sung:

```bash
# Chỉ kiểm tra độ sẵn sàng trên máy CUDA
bash scripts/train_crish.sh check

# Chỉ smoke dữ liệu thật, chưa chạy đủ 200 epoch
bash scripts/train_crish.sh smoke

# Đối chứng B0: cùng objective, backbone, split và lịch tối ưu
bash scripts/train_crish.sh --baseline

# Tiếp tục từ cuối epoch đã lưu; dùng đúng source/config của run đó
bash scripts/train_crish.sh --resume runs/RUN_ID/resume.pt

# Đánh giá lại backbone đã khóa ở epoch cuối
bash scripts/train_crish.sh evaluate --checkpoint runs/RUN_ID/backbone_final.pt
```

Thêm `--baseline` khi resume/evaluate checkpoint B0. Checkpoint resume có optimizer/RNG nên chỉ nạp file tin cậy. Resume đòi đúng config và fingerprint source; checkpoint từ source cũ cần chạy bằng bản source cũ đã lưu.

## 4. Cấu hình và kết quả

`configs/crish.json` là cấu hình mặc định; `configs/baseline.json` chỉ khác trường `method`. Hai file `_smoke.json` dùng riêng cho kiểm tra ngắn. Các siêu tham số CRiSH hiện là lựa chọn kỹ thuật chưa được xác nhận bằng benchmark. M2 stability mặc định tắt; lookahead chỉ cập nhật bản sao projector trong một bước SGD.

Mỗi lần chạy tạo `runs/RUN_ID/` với `config.json`, `environment.txt`, `metadata.json`, `readiness.json`, `run.log`, checkpoint và diagnostics. Theo dõi đường dẫn log được in ra:

```bash
tail -f runs/RUN_ID/run.log
```

`metrics.jsonl` ghi loss/thời gian; All/Old/New để `null` trong training. Sau khi khóa backbone cuối, worker đánh giá riêng ghi `evaluation.json`: `all_acc`, `old_acc`, `new_acc` ở đơn vị **fraction**, nhân 100 để lấy %. Protocol là balanced semi-supervised k-means trên train-unlabeled, anchored bằng nhãn hợp lệ, Hungarian-v2 toàn cục. Không chọn checkpoint theo nhãn unlabeled/test; B0 ở đây là đối chứng runtime đã chuẩn hóa protocol, cần đối chiếu riêng với baseline công bố.

## 5. Kiểm thử và bố cục

Trên Mac/CPU, chỉ chạy:

```bash
.venv/bin/python scripts/run_smoke.py
```

Nếu máy chưa có môi trường local: `GCD_PYTHON=python3 bash scripts/bootstrap.sh local`. Không chạy train dài trên Mac. Biên nhận test nằm ở `results/local_validation.json`; không phải số liệu benchmark.

```text
crish/          thuật toán, objective adapter, repair/probe/lookahead và vòng train chung
gcd_runtime/   data boundary, CLI, preflight, checkpoint và evaluation
configs/        CRiSH, B0 và smoke
scripts/        bootstrap, integrity check, smoke, train và đóng gói
tests/         unit, protocol và synthetic integration tests
audit/         oracle đối chiếu loss với source baseline
upstream/      HypCD giữ nguyên, pin commit
presentation/  một bản trình bày phương pháp + protocol + giới hạn (Markdown/PDF)
results/       biên nhận kiểm tra; không có accuracy CRiSH giả lập
```

[Bản trình bày PDF](presentation/CRISH_GCD.pdf) · [Nguồn bản trình bày](presentation/CRISH_GCD.md) · [Nguồn gốc mã](NOTICE) · [Giấy phép HypCD](LICENSES/HypCD-MIT.txt).

`AGENTS.md` giữ nguyên. Tài liệu và code lịch sử đã lưu ngoài source hoạt động, không nằm trong gói bàn giao. Đóng gói lại bằng `.venv/bin/python scripts/build_release.py`; gói trong `dist/` có manifest và SHA256, không kèm data/weights/logs/lịch sử.
