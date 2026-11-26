# SelEx + CRiSH — hướng dẫn chạy 5 run đã chốt

**Retail RP2K:** branch `feat/rp2k-crish-local` thêm adapter, split GCD tùy chỉnh,
GPU smoke nhỏ và một run CRiSH local cho tích hợp planogram. Xem
[hướng dẫn RP2K](reports/28_rp2k_gpu_quickstart.md). Chưa có kết quả GPU/retail;
không thay thế các thí nghiệm Cars/CUB/Aircraft bên dưới. Bootstrap môi trường:
`GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda` trên Linux NVIDIA,
Python 3.9–3.11. Chỉ chạy sau khi có máy GPU; không bootstrap CUDA trên Mac.

**Cập nhật kế hoạch Cars ngày21/11:** chạy `evidence_only`, `random_proposal`
và `no_abstain`, cùng seed1; tạm bỏ job FULL. Các config cũ vẫn được giữ nguyên.
Config mới: `configs/requested_run5/scars_no_abstain_seed1.json`, chỉ khác
`scars_full_seed1.json` ở `crish_ablation=no_abstain`. Policy loại proposal
Abstain, vẫn giữ Move, counterfactual scoring, confirmation và mọi budget/guard.

Chỉ chạy khi một GPU đã hết job cũ. Không pull/update checkout đang training;
để đưa riêng config mới vào máy đang dùng commit92e0e9e, sau khi GPU rảnh có thể
copy file mới từ GitHub mà không cập nhật runtime. Đường dẫn config phải chưa
tồn tại; dùng `curl --fail --location --output` với URL của commit bổ sung.
Không resume checkpoint của policy khác. Run mới luôn có RUN_ID riêng.

```bash
bash scripts/train_selex.sh --config configs/requested_run5/scars_no_abstain_seed1.json --variant local --dry-run
```

Khi GPU đã rảnh và setup/data/weights hoàn tất, dùng launcher như hai ablation
còn lại (preflight -> bounded intervention smoke -> fresh200 epochs):

```bash
(
  set -euo pipefail
  mkdir -p logs
  GCD_CONFIG=configs/requested_run5/scars_no_abstain_seed1.json
  GCD_JOB_LOG="logs/scars-no_abstain-seed1-$(date -u +%Y%m%dT%H%M%SZ).log"
  bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant local 2>&1 | tee "$GCD_JOB_LOG"
  timeout --signal=INT --kill-after=120s 48h bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local 2>&1 | tee -a "$GCD_JOB_LOG"
)
```

Bảng5 run và các lệnh FULL bên dưới ghi kế hoạch ban đầu, không phải yêu cầu
khởi lại FULL trong đợt ablation hiện tại. CUB/Aircraft full seed2 giữ nguyên.

Đây là hướng dẫn chính của repository: từ chuẩn bị máy GPU đến chạy adaptive,
ablation, đọc kết quả và bàn giao paper. Không cần mở các báo cáo để chạy.

Phương pháp hiện có là **corrected SelEx + CRiSH local Keep / Abstain / Move**,
dùng hình học Euclidean. Nhánh Hyp-SelEx lịch sử được giữ riêng; không mô tả
các run SelEx hiện tại là hyperbolic CRiSH. Linux NVIDIA dùng cho training;
Mac chỉ phát triển và kiểm thử CPU nhỏ.

**Kiểm tra ngày 21/11/2026:** code có đủ 5 cấu hình bên dưới, lấy từ config
của các run seed1 đã quan sát. Ba ablation Cars dùng chung **seed1**;
Aircraft và CUB chạy **full, seed=2**, khởi tạo mới từ pretrained weights.
Không chạy tiếp optimizer/checkpoint seed1. Các cấu hình được commit trong
`configs/requested_run5/`, nên máy GPU mới không cần archive hoặc plan ở Mac.

**Sẵn sàng về code/config; CUDA, dữ liệu và VRAM của máy thuê chưa được xác minh.**
Chỉ bắt đầu full sau preflight và smoke thực trên Linux NVIDIA. Mac16GB dùng
kiểm thử CPU; không có dataset/weights mặc định hoặc CUDA ở máy audit.
Chưa chạy 5 benchmark này, chưa chứng minh cải thiện hay novelty.

| Run | Dataset | Policy | Seed | Config trong `configs/requested_run5/` |
|---|---|---|---:|---|
| 1 | Stanford Cars | `full` | 1 | `scars_full_seed1.json` |
| 2 | Stanford Cars | `evidence_only` | 1 | `scars_evidence_only_seed1.json` |
| 3 | Stanford Cars | `random_proposal` | 1 | `scars_random_proposal_seed1.json` |
| 4 | FGVC-Aircraft | `full` | 2 | `aircraft_full_seed2.json` |
| 5 | CUB-200-2011 | `full` | 2 | `cub_full_seed2.json` |

| Cần làm | Đọc phần |
|---|---|
| Máy GPU mới | 1 → 2 → 3 → 4 |
| Tiếp tục seed của run đã có | 4.2 |
| Chạy đúng 5 run | 4 → 5 |
| Xem log, điểm, checkpoint | 6 → 7 |
| Xuất bảng, curves, hậu kiểm | 8 |
| Dừng/resume | 9 |
| Kiểm thử và checklist bàn giao | 10 → 11 |

## 1. Code và môi trường

Nếu máy đã có repo/môi trường/data/weights thì dùng lại. Không cập nhật source
hoặc đổi config trong checkout có job đang chạy; fingerprint mới có thể chặn
resume. Giữ checkout và config gốc cho những job đó.

Máy mới: Ubuntu22.04 + Python3.10 là môi trường ví dụ; cần Linux NVIDIA,
driver tương thích CUDA12.1 và Python3.9–3.11. Không cài lại NVIDIA driver.
Cài công cụ trước (dùng sudo nếu tài khoản yêu cầu):

```bash
nvidia-smi
apt-get update
apt-get install -y python3.10-venv python3.10-dev build-essential git curl tar tmux unzip coreutils ca-certificates
```

Checkout mới, thư mục đích chưa tồn tại:

```bash
git clone --branch feat/local-adaptive-acceptance --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git gcd-final
cd gcd-final
python3.10 -m venv .venv
.venv/bin/python -m pip install pip==24.3.1
.venv/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r environments/requirements-common.txt -r environments/requirements-research-tests.txt
.venv/bin/python -m pip install -r environments/requirements-selector-cuda.txt
.venv/bin/python -m pip install --force-reinstall --no-deps -r environments/requirements-selector-cuda.txt
.venv/bin/python -m pip install --only-binary=:all: --no-deps xformers==0.0.28.post1
.venv/bin/python -m pip check
.venv/bin/python scripts/bootstrap_upstream.py
```

Cài selector dependencies trước rồi ghim backend Git bằng `--no-deps` để giữ
Torch. Không chạy lại các bước install nếu môi trường đã sẵn sàng. Nếu clone
báo thư mục đã tồn tại, kiểm tra checkout hiện có; không xóa hoặc ghi đè.
Mọi lệnh dưới đây chạy từ gốc repo; đặt `SELEX_PYTHON` nếu dùng interpreter
đã chuẩn bị ở nơi khác. Không cập nhật source khi có job đang chạy.

Kiểm thử CPU có giới hạn trước khi chuẩn bị GPU run; không tải data/weights:

```bash
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 300s .venv/bin/python -m pytest tests -q --tb=short \
  --junitxml=logs/run5-cpu-tests.xml 2>&1 | tee logs/run5-cpu-tests.log
```

Fail/timeout thì dừng đọc log; không gọi timeout là pass hoặc tự đổi Torch.
`bootstrap.sh cuda` vẫn có sẵn, nhưng hướng dẫn này dùng các bước tương đương
ở trên để CPU suite có deadline300s thay giới hạn60s của wrapper.

## 2. Data, weights và đường dẫn

Giữ metadata, train/test pools và tên ảnh gốc; không tự chia lại known/novel.
Runtime dùng SSB splits đã ghim, không dùng class lists hoặc nhãn unlabeled
để chấm reliability/action. Chỉ phần known được chọn labeled mới có nhãn train.

```text
datasets/
  cars/
    cars_train/                              # 8144 ảnh
    cars_test/                               # 8041 ảnh
    devkit/cars_train_annos.mat
    devkit/cars_test_annos_withlabels.mat
  cub/CUB_200_2011/
    images/
    images.txt
    image_class_labels.txt
    train_test_split.txt
    attributes/image_attribute_labels.txt
  aircraft/fgvc-aircraft-2013b/
    data/images/
    data/images_variant_trainval.txt
    data/images_variant_test.txt
weights/dinov2_vitb14_reg4_pretrain.pth
```

| Dataset | Tên config/CLI | Known / Novel | Labeled trong known | Pool training |
|---|---|---|---|---|
| Stanford Cars | `scars` | 98 / 98 | 50% | Native train |
| CUB-200-2011 | `cub` | 100 / 100 | 50% | Native train |
| FGVC-Aircraft | `aircraft` | 50 / 50 | 50% | Native trainval, 100 variant |

Cars: dùng archive mirror [stanford_cars.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/tree/ceb3112337fb6e1bf71534e355ca5569b63f19a6).
Bản ghim có SHA256 từ LFS metadata; đã kiểm danh mục ZIP từ xa bằng HTTP Range,
chưa tải/kiểm ảnh tại máy audit. Archive đặt test labels ở root, nên cần copy
annotation vào `devkit/`. Không dùng split Zhou trong archive thay SSB.

**Chỉ chạy khối này khi chưa có Cars**, hai thư mục đích phải chưa tồn tại:

```bash
(
  set -euo pipefail
  mkdir -p downloads datasets
  test ! -e datasets/cars
  test ! -e datasets/stanford_cars
  curl -fL -C - --connect-timeout 20 --max-time 3600 \
    -o downloads/stanford_cars.zip \
    https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/stanford_cars.zip
  echo '50812812f33534ef2906b992762f96e8a5bb57b0eedc53800e91b7ed599abf8e  downloads/stanford_cars.zip' | sha256sum -c -
  unzip -n -q downloads/stanford_cars.zip -d datasets
  mv -n datasets/stanford_cars datasets/cars
  cp -n datasets/cars/cars_test_annos_withlabels.mat datasets/cars/devkit/cars_test_annos_withlabels.mat
)
```

Nếu đã có Cars, giữ nguyên và đối chiếu layout ở trên; thiếu annotation phải
lấy từ nguồn đúng, không tự tạo nhãn. Download/checksum/extraction lỗi thì dừng,
kiểm thư mục đã tạo thay vì chạy lại hoặc xóa dataset. Preflight phần4 kiểm tra
membership/counts/split và image paths trên dữ liệu thật trước training.

Nếu chưa có CUB/Aircraft, chọn đúng **một** khối tải bên dưới trên máy GPU.
Không tải lại dữ liệu đã đủ. Chỉ giải nén khi checksum OK;
`--keep-old-files` bảo vệ file cũ và có thể báo lỗi nếu đã giải nén trước đó.
MD5 dùng đối chiếu bản phát hành, không phải bảo đảm an toàn mật mã.

CUB: [CaltechDATA, images + annotations bản2011](https://data.caltech.edu/records/65de6-vp158).
Cần cả attributes; không thay bằng CUB2010 hoặc archive chỉ có segmentation.

```bash
mkdir -p downloads datasets/cub
curl -fL -C - --connect-timeout 20 --max-time 3600 \
  -o downloads/CUB_200_2011.tgz \
  'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
echo '97eceeb196236b17998738112f37df78  downloads/CUB_200_2011.tgz' | md5sum -c -
# Chỉ tiếp tục sau checksum OK:
tar -xzf downloads/CUB_200_2011.tgz -C datasets/cub --keep-old-files
```

Aircraft: [bản phát hành FGVC-Aircraft2013b đầy đủ](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.html),
gồm test annotations; không dùng annotation-only archive.

```bash
mkdir -p downloads datasets/aircraft
curl -fL -C - --connect-timeout 20 --max-time 3600 \
  -o downloads/fgvc-aircraft-2013b.tar.gz \
  https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
echo '32eca553f897e747706144f3bb444df0  downloads/fgvc-aircraft-2013b.tar.gz' | md5sum -c -
# Chỉ tiếp tục sau checksum OK:
tar -xzf downloads/fgvc-aircraft-2013b.tar.gz -C datasets/aircraft --keep-old-files
```

Cả ba dùng **DINOv2 ViT-B/14 với4 register tokens**. Nếu chưa có weights:

```bash
mkdir -p weights
curl -fL -C - --connect-timeout 20 --max-time 1800 \
  -o weights/dinov2_vitb14_reg4_pretrain.pth \
  https://huggingface.co/cubert-gmbh/dinov2/resolve/e2c8060a74112f7537484ed50097b1497d5f032c/dinov2_vitb14_reg4_pretrain.pth
echo '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71  weights/dinov2_vitb14_reg4_pretrain.pth' | sha256sum -c -
```

Đường dẫn mặc định là `datasets/`, `weights/`, `runs/`. Nếu dùng volume khác:

```bash
export GCD_DATA_ROOT=/path/to/datasets
export GCD_WEIGHTS_ROOT=/path/to/weights
export GCD_RUNS_ROOT=/path/to/runs
```

Data root chứa `cars/`, `cub/`, `aircraft/`; weights root chứa trực tiếp `.pth`.
Không commit ảnh, weights, checkpoints hoặc credentials. Preflight ở phần4
kiểm tra metadata/image paths/split/weights và CUDA thật; không tự tạo file thiếu.

Herbarium19 là hỗ trợ bổ sung: dùng [Herbarium2019 FGVC6](https://www.kaggle.com/c/herbarium-2019-fgvc6)
với `datasets/herbarium_19/small-train/<class>/<image>` và
`small-validation/<class>/<image>`,683 lớp và cùng mapping ở hai pool.
SSB341/342, labeled50%, `unbalanced=true` ở hierarchy và SS evaluation.
Config `configs/selex_herbarium_19_crish_200.json` là local cap1; chưa có
benchmark dự án cho dataset này. Không suy rộng kết quả ba dataset sang Herbarium.

## 3. Cấu hình adaptive full đang dùng

Luôn ghi **`--variant local`** khi chạy CRiSH. `local` là cơ chế sửa supervision,
không có nghĩa chạy CPU/Mac. Một số config lịch sử mặc định population policy;
chỉ nhìn `method=crish` chưa phân biệt được local với policy.

| Thành phần | Adaptive full |
|---|---|
| Training | 200 epoch, batch128, input224, projector65536, DINOv2 reg4 |
| Cars/Aircraft | SGD lr0.1, momentum0.9, wd0.0001, grad_from_block9, temperature0.5 |
| CUB | Giữ temperature/block theo config CUB, không copy thông số Cars |
| Loss | SupCon weight0.35, smoothing0.75, USE `selex_legacy`, cosine schedule |
| Lịch refresh | warmup10, mỗi5 epoch: log10,15,…,195 = epoch hiển thị11,16,…,196 |
| Evidence | Global normalized-backbone neighbor shortlist; reciprocal/two-view verification trong cache |
| Evidence budget | cache128, trials2, k5, reference block1024 |
| Candidate/scoring budget | ≤16 boundary IDs, ≤1 Move target/boundary, scoring batch128 |
| Support guards | min_support2, min_edges1, reliability≥0.5, protected shift≤0.25 |
| Probe | Mỗi fold16 labeled (8 classes×2) +32 unlabeled; các fold/cache disjoint |
| Confirmation | 1 discovery +8 reserved independent confirmation folds |
| Acceptance | **0–8 edits/refresh**, gain>0, known increase≤0; không ép đủ8 |
| Thời gian | ≤120s/refresh, worker training limit604800s, evaluator limit7200s |
| Controller khác | Population policy và online pair weighting đều tắt |

Discovery xếp hạng gain sau SGD thử trên projector với state SGD/momentum được
copy từ optimizer thật. Chỉ winner được confirmation bằng fold mới; failure
thì dừng. Sau commit, rebuild và re-score candidates còn lại. Labeled assignments
được bảo vệ. Lỗi validation cuối có thể rollback toàn refresh; log phân biệt
proposal từng được nhận với `applied_actions` thực áp.

Mỗi epoch extract và dựng lại hierarchy; edit chỉ sống trong supervision epoch
đó. Model/optimizer tiếp tục học, không reset weights sau refresh. Gain là
chênh lệch probe loss, không phải phần trăm accuracy tăng. Evidence không dùng
classifier confidence giả hoặc nhãn thật của unlabeled.

Template Cars adaptive: `configs/selex_scars_local_adaptive_200.json`.
Profile local lịch sử có cap1/scoring32, khác adaptive cap8/boundaries16/scoring128.
Đối chứng `cap1_context` do preparer tạo chỉ đổi cap so với adaptive, vẫn giữ
8 confirmation folds và cùng context/budgets. Manuscript cũ còn khác cấu hình
quan sát; không dùng nội dung draft để suy ra thông số run.

## 4. Chuẩn bị và kiểm tra đúng 5 run

Dùng trực tiếp năm file trong bảng đầu README. **Không lấy thẳng
`selex_cub_crish_200.json` / `selex_aircraft_crish_200.json`**: các template
đó mặc định population policy và không khớp adaptive full đã chạy.
Năm config mới giữ local cap8/boundaries16/scoring128/confirmation8,
SS mỗi epoch, population và online weighting tắt. Giữ cấu hình riêng của CUB.

Nguồn đối chiếu là `results/paper-plan-verified-20261118/registry.json`, trường
`config_json`; manifest trong `configs/requested_run5/` ghi run ID gốc,
config hash và các trường đổi. Chỉ đổi seed theo yêu cầu và khai báo
`crish_ablation`; Cars chỉ khác policy. Mọi run: 200 epoch, batch128,
DINOv2 reg4, fixed-final SS-v2. Không cần sinh lại config hoặc sửa source.
Source hiện tại khác source seed1 lịch sử; ghi provenance riêng, chưa gộp
seed1/seed2 thành nhóm matched chỉ vì hyperparameters giống nhau.

Chạy từ gốc repo sau phần1–2. Nếu mở tmux/SSH mới, đặt lại các biến môi trường
trỏ volume. Trong checkout đã có job, giữ nguyên source và dùng checkout riêng.

```bash
mkdir -p logs
export GCD_DATA_ROOT="$PWD/datasets"
export GCD_WEIGHTS_ROOT="$PWD/weights"
export GCD_RUNS_ROOT="$PWD/runs"
# Nếu dữ liệu nằm trên volume khác, thay ba đường dẫn trên bằng đường dẫn thật.
nvidia-smi
ps -axo pid,ppid,etime,%cpu,%mem,command | grep -E 'python|train_selex' | head -n 40
git rev-parse HEAD
git status --short
```

GPU cần đủ VRAM cho DINOv2 ViT-B/14, projector65536, batch128, feature cache
và scoring/SS clustering. Chưa đo peak của năm job mới nên không hứa mức VRAM
hay ETA; dùng smoke batch thật làm gate. Mỗi job có hai minibatch smoke rồi
full200 epoch; không chạy đồng thời trên cùng GPU. Đảm bảo volume còn đủ chỗ
cho data, weights, checkpoints và partition snapshots; lưu cả run folder.

Dry-run cả năm cấu hình, không train/download hay kiểm CUDA:

```bash
set -e
for config in \
  configs/requested_run5/scars_full_seed1.json \
  configs/requested_run5/scars_evidence_only_seed1.json \
  configs/requested_run5/scars_random_proposal_seed1.json \
  configs/requested_run5/aircraft_full_seed2.json \
  configs/requested_run5/cub_full_seed2.json
do
  bash scripts/train_selex.sh --config "$config" --variant local --dry-run
done
```

Phải thấy đúng dataset/seed/policy, `epochs=200`, `head_dimension=65536`,
`discrete_repair=true`, cap8/boundaries16/train_batch128/confirmation8,
population và pair reliability đều false. Full snapshot phải có
`ss_eval_every_epochs=1`, `max_steps_per_epoch=0`; smoke được launcher tạo riêng.
Dry-run vẫn tạo các config snapshot nhỏ trong `runs/plans/`.

Mở tmux trước khi chạy job đầu; các bước còn lại làm trong tmux:

```bash
tmux new -s crish-run5
```

Mỗi khối ở phần5: **check CUDA/data/weights/import → launcher tự smoke → full**.
`check` có pretrained image forward/backward và optimizer step thật với head;
launcher kiểm lại trước mỗi stage. Không thêm `--eval-every` vì cadence1 đã
nằm trong config; flag này chỉ hợp lệ ở train, không ở check/smoke.

Smoke1 epoch, tối đa2 minibatch, warmup0, extraction/clustering toàn pool và
SS cuối. Full/random phải score CF candidate; evidence-only phải thực thi
policy không qua CF. Không bắt buộc nhận edit; fallback hoặc chưa exercise
policy sẽ chặn full. Smoke không phải kết quả benchmark.

## 5. Năm lệnh chạy đầy đủ, từng job một

Chỉ chạy khối kế tiếp sau khi job trước kết thúc thành công. Mỗi khối được bọc
trong subshell `set -euo pipefail`: check lỗi thì không train; train lỗi không
bị `tee` che. Không dán thành sweep hoặc bỏ qua lỗi. `timeout48h` là giới hạn
thời gian thuê, không phải ETA; điều chỉnh trước khi chạy theo ngân sách thật.
Runtime còn có preflight180s, smoke1800s, worker604800s, SS evaluator7200s;
giới hạn ngoài48h chặn toàn lệnh, kể cả smoke và evaluation. Nếu timeout/SSH
lỗi, kiểm PID/GPU/log trước khi chạy lại, không khởi job trùng.

### Run 1 — Cars full, seed1

```bash
(
  set -euo pipefail
  GCD_CONFIG=configs/requested_run5/scars_full_seed1.json
  GCD_JOB_LOG="logs/scars-full-seed1-$(date -u +%Y%m%dT%H%M%SZ).log"
  bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant local 2>&1 | tee "$GCD_JOB_LOG"
  timeout --signal=INT --kill-after=120s 48h bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local 2>&1 | tee -a "$GCD_JOB_LOG"
)
```

### Run 2 — Cars evidence_only, seed1

```bash
(
  set -euo pipefail
  GCD_CONFIG=configs/requested_run5/scars_evidence_only_seed1.json
  GCD_JOB_LOG="logs/scars-evidence_only-seed1-$(date -u +%Y%m%dT%H%M%SZ).log"
  bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant local 2>&1 | tee "$GCD_JOB_LOG"
  timeout --signal=INT --kill-after=120s 48h bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local 2>&1 | tee -a "$GCD_JOB_LOG"
)
```

### Run 3 — Cars random_proposal, seed1

```bash
(
  set -euo pipefail
  GCD_CONFIG=configs/requested_run5/scars_random_proposal_seed1.json
  GCD_JOB_LOG="logs/scars-random_proposal-seed1-$(date -u +%Y%m%dT%H%M%SZ).log"
  bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant local 2>&1 | tee "$GCD_JOB_LOG"
  timeout --signal=INT --kill-after=120s 48h bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local 2>&1 | tee -a "$GCD_JOB_LOG"
)
```

### Run 4 — Aircraft full, seed2

```bash
(
  set -euo pipefail
  GCD_CONFIG=configs/requested_run5/aircraft_full_seed2.json
  GCD_JOB_LOG="logs/aircraft-full-seed2-$(date -u +%Y%m%dT%H%M%SZ).log"
  bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant local 2>&1 | tee "$GCD_JOB_LOG"
  timeout --signal=INT --kill-after=120s 48h bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local 2>&1 | tee -a "$GCD_JOB_LOG"
)
```

### Run 5 — CUB full, seed2

```bash
(
  set -euo pipefail
  GCD_CONFIG=configs/requested_run5/cub_full_seed2.json
  GCD_JOB_LOG="logs/cub-full-seed2-$(date -u +%Y%m%dT%H%M%SZ).log"
  bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant local 2>&1 | tee "$GCD_JOB_LOG"
  timeout --signal=INT --kill-after=120s 48h bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local 2>&1 | tee -a "$GCD_JOB_LOG"
)
```

Full dùng reciprocal evidence + discovery counterfactual SGD + independent
confirmation; tối đa8 edits/refresh, có thể0. Evidence-only giữ proposal,
actions và structural/known-label/protected-shift/drift guards nhưng bỏ
CF/confirmation; ưu tiên Move đủ support, nếu không có thì Abstain, Move trượt
guard thì Keep. Random-proposal chỉ shuffle eligible boundaries trước cap bằng
RNG riêng theo seed, giữ action types, CF, confirmation và budgets của full.
Không dùng nhãn unlabeled để chọn action. Các switch khác vẫn có trong code
nhưng không thuộc năm job được yêu cầu.

Không thay seed, batch, LR, cadence hoặc guard khi một run thất bại rồi báo
như cùng nhóm. Giữ log lỗi và xử lý nguyên nhân trước khi lên lịch lại.
Sau mỗi run ghi lại **full RUN_ID** và đọc `evaluation_last.json` ở phần7;
metadata status phải completed, epoch cuối199. Check/smoke có run ID riêng.

## 6. Theo dõi log và action

Launcher in đường dẫn run. Dùng đúng full RUN_ID, không chọn nhầm smoke:

```bash
RUN_DIR=runs/RUN_ID
tail -n 100 -f "$RUN_DIR/run.log"
```

`tail` không tự chuyển file khi smoke sang full. `TRAIN plan epochs=200` mới
cho biết vào full, chưa chứng minh đã chạy xong. Ở tab khác có thể dùng
`watch -n 2 nvidia-smi`; utilization0% tức thời không chứng minh job treo.
Evaluator/trainer có thể là hai PID khác nhau; đọc command/log trước khi kết luận.

Ctrl+B rồi D để rời tmux, `tmux attach -t crish-train` để quay lại. Ctrl+C ở
`tail` chỉ dừng xem log; Ctrl+C ở launcher chính dừng job. Không tự kill PID.

| File | Dùng để làm gì |
|---|---|
| `run.log`, `metrics.jsonl` | Tiến độ, loss, steps, epoch seconds và peak CUDA memory |
| `crish.jsonl` | Proposal, gain, decision/reason, action counts, stop reason và refresh seconds |
| `intervention_diagnostics.jsonl` | Số ID/positive relations thực đổi trong supervision epoch |
| `partitions_epoch_NNNN.npz` | Partition trước refresh, nếu bật lưu |
| `supervision_epoch_NNNN.npz` | Supervision sau can thiệp, nếu có thay đổi và bật lưu |
| `config.json`, `metadata.json`, `environment.txt`, `readiness.json` | Seed/version/fingerprint/command/environment/preflight |

Adaptive nhận0–8 edits/refresh; cap1 chỉ áp dụng profile lịch sử/đối chứng.
`accepted`/`applied_actions` đếm edit thực áp, không tính rolled-back proposal.
`proposed` là action sau shortlist/type filter; `not_applied` gồm rejection,
alternative cùng ảnh hoặc chưa score vì cap/deadline. Trace có thể lặp qua
nhiều vòng, không dùng số dòng trace như số candidate độc lập.

Keep đếm unique shortlisted sample IDs không có applied edit, không phải toàn
unlabeled pool. Touched fraction là unique accepted IDs/n_unlabeled. Registry
có cả union IDs toàn run; union không bằng tổng image-epoch exposure. Thiếu
log ghi MISSING/null, không ghi0. `crish_p.jsonl` thuộc population policy riêng.

Đối chiếu action với SS trước/trong/sau, chỉ đọc log:

```bash
.venv/bin/python scripts/analyze_crish_interventions.py "$RUN_DIR" --output /tmp/crish_action_ss.csv
```

Log epoch10 là can thiệp trong epoch thứ11. SS delta theo thời gian không đo
tác động nhân quả của từng action; nhiều action cùng epoch dùng chung một delta.
Missing SS để NA, không thay bằng best hoặc plain report.

## 7. Evaluation và checkpoint dùng cho paper

Primary là **`evaluation_last.json`**, purpose=final, epoch=epochs−1, fixed-final
SS-v2 trên unlabeled train pool. Balanced semi-supervised k-means dùng legal
known labels làm anchors; ground truth unlabeled chỉ được dùng sau prediction
để global Hungarian matching và báo All/Old/New. JSON lưu fraction, bảng báo%.

`All = (n_old × Old + n_new × New) / (n_old + n_new)`, không phải trung bình
cộng Old/New. Cars quan sát có labeled2000, unlabeled6144 (Old2001/New4143).
SSB Cars checksum:
`28d471f0fb9e2c773e1255b8b447e542974f6617952203056a0635c638513d39`.

| File/dòng log | Ý nghĩa |
|---|---|
| `epoch_metrics.jsonl`, `evaluations/epoch_NNNN.json` | SS định kỳ; `--eval-every 1` chấm epoch1–199, final chấm epoch200 |
| `evaluation_last.json`, `evaluation.json` | Điểm fixed-final; dùng cho bảng chính |
| `ss_best.json`, `backbone_best_ss_{all,old,new}.pt` | Best SS thứ cấp, chọn bằng nhãn đánh giá |
| `evaluation_best_old_test.json` | SS của checkpoint chọn theo plain-test Old-v1 |
| `test_metrics.jsonl`, `plain_kmeans_disjoint_test` | Plain evaluation khác SS, không thế chỗ SS |
| `backbone_final.pt` | Backbone cuối để evaluation |
| `resume.pt` | Model/head/optimizer/scheduler/RNG/config để tiếp tục train |

Không ghép cực đại All/Old/New từ ba epoch vào một hàng. Không dùng best-All
chọn bằng unlabeled truth làm số chính. Registry không đọc best reports.
Bảng chính cần final mean±sample std trên ≥3 matched seeds, cùng backbone,
split, label fraction, class-count assumption và protocol. Không gộp seed
khác code/config/weights/provenance. Với một seed ghi rõ preliminary.

```bash
.venv/bin/python scripts/show_metrics.py "$RUN_DIR"
```

Upstream SelEx sửa báo Cars/CUB/Aircraft **50 epoch**, last checkpoint, seeds1–3;
CRiSH đang chạy200 epoch. Dù cùng metric, vẫn khác training budget, không coi
chênh lệch là algorithm-only gain. Kế hoạch không yêu cầu GPU chạy lại baseline.

## 8. Xuất registry, curves và hậu kiểm offline

### 8.1. Registry final và thống kê action/timing

Nhận run directories hoặc ZIP/tar.gz text export, mỗi archive đúng một run
config. Source chỉ đọc; output phải là thư mục mới:

```bash
.venv/bin/python scripts/paper_registry.py \
  runs/CARS_RUN_ID /path/to/cub-results.zip /path/to/aircraft-results.zip \
  --output results/paper-registry-NEW
```

Output `runs.csv`, `registry.json`, `refreshes.csv`, `actions.csv`: provenance,
final metrics, exact per-seed/group mean±std khi hợp lệ, accepted edits, touched
fraction, refresh seconds và mean epoch time có/không refresh. `paper_ready`
cần ≥3 unique matched seeds. Duplicate imports/refresh epochs bị từ chối.
Thiếu provenance thì không aggregate; thiếu action log không được hiểu là0.

Epoch time gồm extraction/refresh/SGD, không gồm evaluation; lịch resume có
thể chứa record kế thừa. Hiệu hai nhóm mean không phải causal overhead.
Epoch scheduled refresh mà thiếu log được loại khỏi nhóm “without refresh”.
Sign agreement đếm mọi candidate đã confirmation, kể cả trượt; có selection
bias, không phải semantic correctness.

### 8.2. Curves All/Old/New có refresh markers

Dependency chỉ cho phân tích offline, không cần đổi môi trường training:

```bash
.venv/bin/python -m pip install -r environments/requirements-figures.txt
.venv/bin/python scripts/plot_paper_curves.py "$RUN_DIR" --output results/curves-NEW
```

Input cũng nhận archive. Output PDF/SVG/PNG, caption, source data và hash
manifest; giữ gap khi thiếu measurement, gray markers ở log_epoch+1 và dots
ở fixed-final. Curves một seed không thay bảng mean±std.

### 8.3. K5: Move đúng lớp và Abstain error, chỉ phân tích sau run

Cần partitions **trước refresh** và CSV nhãn thật `sample_id,true_label`, cùng
ID hệ thống và bao phủ mọi snapshot ID. Không dùng các file nhãn này cho train,
selection hoặc action. Text export thường không chứa NPZ nên chưa đủ để tính:

```bash
.venv/bin/python scripts/prepare_posthoc_snapshots.py "$RUN_DIR" \
  --labels /path/to/offline_sample_labels.csv --output /tmp/posthoc-RUN_ID.jsonl
.venv/bin/python scripts/paper_posthoc.py "$RUN_DIR" \
  --snapshots /tmp/posthoc-RUN_ID.jsonl --output results/posthoc-RUN_ID.json
```

Join NPZ dùng allow_pickle=False. Snapshot JSONL có
`run_id,epoch,level,sample_ids,true_labels,clusters,labeled`; kiểm run ID,
source assignment, membership và full unlabeled coverage. Missing snapshot
báo PARTIAL. Mỗi chỉ số có numerator/denominator; mẫu số0 trả null.

Move diagnostic so true label với unique majority của target cluster; Abstain
so với source-cluster error và error chung ở snapshot. Loại chính ảnh đang
xét khỏi majority, loại ties khỏi mẫu số; cluster ID không phải semantic class.
Đây là offline purity diagnostic, khác global Hungarian accuracy. Chưa có
nhãn/partitions phù hợp thì ghi NOT VERIFIED, không điền ước lượng.

## 9. Dừng, resume và giữ artifacts

Nếu cần dừng, attach đúng tmux, Ctrl+C một lần, chờ prompt và kiểm log/GPU.
Checkpoint chỉ bảo đảm tới epoch lưu xong; minibatch đang chạy dở có thể chưa lưu.
Giữ toàn run folder, config và source version. Resume từ đúng checkout cũ:

```bash
bash scripts/train_selex.sh --config "$RUN_DIR/config.json" --resume "$RUN_DIR/resume.pt"
```

Baseline thêm `--baseline`. Không override seed/variant/epochs/cadence hoặc bỏ
qua fingerprint để resume. Best backbone thiếu optimizer nên không thay được
resume.pt. Run resume có ID/log mới; không xóa run nguồn chứa history.
Trước khi trả GPU, copy toàn bộ full run, plan manifest và launcher log;
text-only export đủ phân tích một phần, không đủ resume/evaluate lại weights.

## 10. Kết quả đang có và kiểm thử

Kiểm chứng cho đúng5 run ngày21/11: **94 tests passed** trong29.46s;
15 dry-runs (check/smoke/train ×5) đạt, config khớp snapshot gốc ngoài seed và
policy, upstream pins/hashes đạt. Các khối Bash trong README được kiểm bằng
`bash -n`. Receipts local: `logs/requested-run5-tests.xml`,
`logs/requested-run5-tests.log`, `logs/requested-run5-readiness.json`.
Không xác minh CUDA/VRAM/data trên máy thuê và không khởi chạy benchmark.


Các số sau là **observed fixed-final seed1**, không phải best-All trong kế hoạch:

| Dataset | All (%) | Old (%) | New (%) | Recorded Move / Abstain | Action-log coverage |
|---|---:|---:|---:|---|---|
| Stanford Cars | 78.2064 | 87.3563 | 73.7871 | 65 / 52 | Đủ38 scheduled refreshes |
| CUB-200 | 83.6744 | 84.8566 | 83.0831 | 1 / 224 | Thiếu log epoch10,15,20,25,30 |
| FGVC-Aircraft | 78.2244 | 81.2125 | 76.7316 | 62 / 59 | Thiếu log epoch10,15,20,25 |

Epoch ở cột coverage đếm từ0. Counts CUB/Aircraft chỉ là phần ghi lại, không
khẳng định tổng toàn run. Model checkpoints không có trong text imports.
Dữ liệu phân tích, tables/curves và receipts đã commit nằm trong
`results/paper-plan-verified-20261118/`; raw archives ở máy riêng, không trên Git.
Mỗi source archive có hash lưu trong `source_manifest.json`.

Kiểm chứng CPU ngày18/11:337 distinct test cases được phủ qua các đợt hữu hạn,
bao gồm full trace parity, independent RNG, known-label protection, actual
objective backward/CF/rebuild, resume, registry và release archive chạy ngoài
Git. Final targeted55 passed; launcher dry-run cả6 variants đạt. Các bộ có
overlap, không cộng chúng thành số thí nghiệm. Receipt:
`results/paper-plan-verified-20261118/validation.json`. Đây không phải GPU
verification hay benchmark mới của code ablation.

Chạy lại focused CPU tests, không train:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_paper_ablation.py tests/test_paper_registry.py tests/test_local_adaptive.py -q
```

Code/protocol smoke runner có timeout mỗi subprocess:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py --code-only
```

Nếu full suite vượt60s, chia batch và giữ timeout riêng; không gọi timeout là
pass. OpenMP SHM failure là vấn đề sandbox/runtime cần xử lý, không phải bằng
chứng loss sai. Split check không có image manifest thì chưa kiểm toàn bộ ảnh.

## 11. Checklist bàn giao và giới hạn claim

| Mã | Phần kỹ thuật đã có | Còn cần người/GPU |
|---|---|---|
| K1 | Cấu hình, pipeline, budget và khác biệt draft ở phần3 | Chốt version/config G1 |
| K2 | Cars1-seed launcher, final-only registry | NOT RUN tối ưu; không tự đổi method |
| K3 | Full +5 switches, guards/RNG/regression tests | GPU smoke và ablation benchmark NOT RUN |
| K4 | Per-refresh proposals/outcomes/reasons, counts/touched exports | Bổ sung missing early logs; không coi thiếu là0 |
| K5 | Offline join/semantic audit, sign agreement có mẫu số | BLOCKED semantic metrics: thiếu labels/partitions |
| K6 | Measured refresh seconds và hai nhóm epoch time | Chỉ descriptive timing; chưa có matched causal overhead |
| A1 | Registry3 dataset seed1, provenance và final | Tiếp tục cập nhật các run mới |
| A2 | Preparer full seed2–3 từ config/archive thực | NOT RUN trên GPU |
| A3 | Matched-seed grouping, mean±sample std, ≥3-seed gate | Hiện mỗi dataset1 seed |
| A4 | Six variants/seed, config/commands không sửa source | Chạy Cars1–2 seeds theo nguồn lực, sau G1 |
| A5 | Registry nhận baseline artifacts đã có | Thiếu đủ baseline logs nhóm; không chạy thêm theo kế hoạch |
| A6 | Curves3 dataset, refresh/final markers, hashes | Vẽ lại trên config chốt G1 khi có thêm seeds |
| G1/G2 | Công cụ kiểm chứng/bàn giao | Nhóm chốt cấu hình và kiểm chéo số độc lập |

Bản triển khai gắn với `CRiSH-GCD_Ke_hoach_Codex.md`; chưa có quyết định thay
phương pháp tại G1 hay xác nhận main-table claim tại G2. CPU tests không chứng
minh accuracy hoặc novelty. Gates so sánh full-method RCA/LEAH/HypGCD vẫn mở.

Upstream SelEx ghim `d8107b5458f3337b2d5342f79345a976037a1265`, HypCD ghim
`ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`; không sửa upstream.
[SelEx Issue11](https://github.com/SarahRastegar/SelEx/issues/11) xác nhận lỗi
row ordering; trang ghi merge/close08/11/2026. Chưa xác minh commit0cc628d
hoặc account người báo thuộc nhóm, nên không dùng các chi tiết đó như claim
đóng góp. Không coi sửa ordering hay reciprocal neighbors là novelty đã xác lập.

`reports/` lưu audit/lịch sử nghiên cứu, không phải các bước hướng dẫn phải
đọc thêm. Hai tài liệu bàn giao23/24 đã gộp vào README này; bản cũ còn trong
lịch sử Git. Các sửa manuscript/PDF cũ ở workspace không nằm trong đợt sửa
hướng dẫn này. Giấy phép và nguồn gốc: [NOTICE](NOTICE),
[HypCD MIT license](LICENSES/HypCD-MIT.txt).
