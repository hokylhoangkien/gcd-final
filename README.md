# SelEx + CRiSH — cài đặt, chạy và phân tích thí nghiệm

Đây là hướng dẫn chính của repository: từ chuẩn bị máy GPU đến chạy adaptive,
ablation, đọc kết quả và bàn giao paper. Không cần mở các báo cáo để chạy.

Phương pháp hiện có là **corrected SelEx + CRiSH local Keep / Abstain / Move**,
dùng hình học Euclidean. Nhánh Hyp-SelEx lịch sử được giữ riêng; không mô tả
các run SelEx hiện tại là hyperbolic CRiSH. Linux NVIDIA dùng cho training;
Mac chỉ phát triển và kiểm thử CPU nhỏ.

**Trạng thái ngày 18/11/2026:** có ba run adaptive seed1, 200 epoch được nhập
log từ GPU; có full +5 ablations và công cụ phân tích, chưa có kết quả GPU
cho các ablation mới hoặc bảng chính ≥3 seed. Chưa chứng minh cải thiện hay novelty.

| Cần làm | Đọc phần |
|---|---|
| Máy GPU mới | 1 → 2 → 3 → 4 |
| Tiếp tục seed của run đã có | 4.2 |
| Chạy ablation | 5 |
| Xem log, điểm, checkpoint | 6 → 7 |
| Xuất bảng, curves, hậu kiểm | 8 |
| Dừng/resume | 9 |
| Kiểm thử và checklist bàn giao | 10 → 11 |

## 1. Code và môi trường

Nếu máy đã có repo/môi trường/data/weights thì dùng lại. Không cập nhật source
hoặc đổi config trong checkout có job đang chạy; fingerprint mới có thể chặn
resume. Giữ checkout và config gốc cho những job đó.

Checkout mới, thư mục đích chưa tồn tại:

```bash
git clone --branch feat/local-adaptive-acceptance --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git gcd-final
cd gcd-final
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
```

Cần Python **3.9–3.11**, Linux và NVIDIA driver tương thích CUDA12.1.
Bootstrap tạo `.venv`, ghim Torch2.4.1/cu121, torchvision0.19.1 và dependencies,
kiểm source/CPU; không tải data/weights hoặc train dài. Không cài lại NVIDIA
driver khi `nvidia-smi` đang hoạt động. Nếu suite chạm timeout thì chưa phải
pass; xem log thay vì tự đổi Torch/cài lại môi trường.

Ubuntu trống cần các công cụ sau (dùng sudo nếu tài khoản yêu cầu):

```bash
apt-get update
apt-get install -y python3-venv python3-dev build-essential git curl tar tmux unzip ca-certificates
```

Đặt `SELEX_PYTHON` nếu dùng interpreter đã chuẩn bị ở nơi khác. Mọi lệnh bên
dưới chạy từ gốc repo; `.venv/bin/python` là interpreter ví dụ của bootstrap.

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

Nguồn Cars có thể dùng [stanford_cars.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/tree/main).
Đặt ảnh và annotation theo cây trên; annotation test cần ở `devkit/`.
Không dùng split Zhou trong archive thay SSB.

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

## 4. Chạy full và seed tiếp theo

### 4.1. Thí nghiệm adaptive mới

Chọn một dataset, rồi tạo profile. Output phải là thư mục **mới**:

```bash
GCD_DATASET=scars
# Đổi thành cub hoặc aircraft khi cần.
.venv/bin/python scripts/prepare_local_adaptive.py \
  --base "configs/selex_${GCD_DATASET}_crish_200.json" \
  --output "runs/plans/${GCD_DATASET}-adaptive-NEW"
GCD_CONFIG="runs/plans/${GCD_DATASET}-adaptive-NEW/adaptive.json"
bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local --eval-every 1 --dry-run
```

Preparer chỉ ghi `adaptive.json`, `legacy.json`, `cap1_context.json`,
`baseline.json`, manifest và commands. Nó không download/train.
Dry-run phải hiện `discrete_repair=true`, cap8/boundaries16/scoring128,
confirmation_budget8, population/online đều false và epochs200.

`--eval-every 1` ghi SS mỗi epoch. Template có thể để cadence0; snapshot config
thực trong run sẽ ghi override1. Giữ cùng cadence cho mọi seed/ablation:
`evaluation_rng=upstream` khiến thay cadence có thể làm đổi RNG train.

Kiểm data/weights/import và **CUDA image forward/backward với head thật**:

```bash
nvidia-smi
bash scripts/train_selex.sh check --config "$GCD_CONFIG" --variant local
```

Có thể chọn GPU bằng `CUDA_VISIBLE_DEVICES`. Launcher cho phép GPU có process
khác nhưng preflight/smoke vẫn bắt buộc; tự kiểm VRAM và job trước khi chạy.
Không suy ra CUDA đã đạt chỉ từ `nvidia-smi`.

Chạy một job trong tmux, sau khi check đạt:

```bash
tmux new -s crish-train
```

Trong tmux, đặt lại `GCD_CONFIG` nếu dùng shell mới, rồi chạy:

```bash
bash scripts/train_selex.sh --config "$GCD_CONFIG" --variant local --eval-every 1
```

Launcher tự thực hiện **preflight → bounded smoke → full200 epoch** khi không
resume. Không cần chạy smoke riêng trước lệnh full; nếu đã chạy riêng, launcher
vẫn lặp smoke. Chỉ muốn smoke:

```bash
bash scripts/train_selex.sh smoke --config "$GCD_CONFIG" --variant local
```

Smoke1 epoch, ≤2 minibatch, warmup0, có extraction/clustering toàn pool và SS
cuối; vì vậy không phải phép kiểm tức thời. Full/ablation CF phải thực sự score
candidate; evidence-only phải exercise policy không qua CF. Không yêu cầu có
edit được nhận, nhưng fallback hoặc không exercise được policy sẽ chặn full.
Smoke score không dùng làm benchmark. Smoke/full có run ID và log riêng.
Các deadline là điều kiện dừng, không phải ETA hay hứa đủ runtime thuê GPU.

### 4.2. Tiếp nối seed của cấu hình đã quan sát

Dùng **`config.json` của đúng full run**, không suy ra từ config template hoặc
đổi cadence giữa các seed. Trên máy có run, thay RUN_ID bằng ID thật:

```bash
RUN_DIR=runs/RUN_ID
.venv/bin/python scripts/prepare_paper_plan.py \
  --base "$RUN_DIR/config.json" --seeds 2 3 --variants full \
  --output runs/plans/full-seed23-NEW
bash scripts/train_selex.sh --config runs/plans/full-seed23-NEW/full_seed2.json --variant local --dry-run
```

Sau check/smoke, chạy **từng seed** bằng cùng launcher; không dán cả danh sách
commands như sweep. Preparer cũng nhận ZIP/tar.gz chứa đúng một run config:

```bash
.venv/bin/python scripts/prepare_paper_plan.py \
  --base /path/to/cub-results-RUN_ID.zip --seeds 2 3 --variants full \
  --output runs/plans/cub-seed23-NEW
```

Archive gốc, `runs/` và các plans cá nhân không nằm trên Git. Source hiện tại
được triển khai tại commit `a9640c4`; runtime fingerprint đã khác source của
các run cũ. Full default có regression parity nhưng không tự gộp các seed
khác source/provenance thành cùng nhóm. Chốt version/config trước G1 và giữ
checkout cũ cho resume; không bypass fingerprint.

## 5. Năm ablation và cách chạy

Trường top-level `crish_ablation` chọn policy; không có trường tương đương
`full`. Không sửa source giữa các run. Chỉ hỗ trợ local SelEx controller,
population/online weighting phải tắt.

| Table II | Giá trị | Proposal | Abstain | Move | CF | Confirmation |
|---|---|---|---|---|---|---|
| (a) | SelEx công bố | — | — | — | — | — |
| (b) | `evidence_only` | Reciprocal | Có | Có | Không | Không |
| (c) | `no_confirmation` | Reciprocal | Có | Có | Có | Không |
| (d) | `full` | Reciprocal | Có | Có | Có | Có |
| (e) | `no_abstain` | Reciprocal | Không | Có | Có | Có |
| (f) | `no_move` | Reciprocal | Có | Không | Có | Có |
| (g) | `random_proposal` | Random | Có | Có | Có | Có |

- Evidence-only gộp evidence + actions: ưu tiên boundary, có Move target đủ
  support thì chọn Move, không có thì Abstain. Giữ structural/known-label,
  protected-shift và nonzero-SSE-drift guards; không tính CF/confirmation gain.
  Move trượt guard thì Keep, không thử Abstain để lách guard.
- No-confirmation giữ discovery CF và các guard, chỉ bỏ independent confirmation.
- No-abstain bỏ proposal Abstain; trường hợp chỉ có Abstain trở thành Keep,
  không biến thành Move. No-move chỉ bỏ Move; không ép dùng hết action budget.
- Random xáo thứ tự cùng eligible boundary pool **trước shortlist cap**, không
  random action type hay cố tình hạ quota. Dùng `random.Random(seed)` riêng,
  không tiêu thụ Python/NumPy/Torch training RNG. Tập được chọn sau cap có thể khác.
- Tất cả giữ cùng reserved context/folds, budgets, split, augmentation,
  dataloader, seed,200 epoch và lịch refresh/SS. Số edit thực có thể khác do guards.

Tạo đủ6 variants từ **config thực của full** đã chọn:

```bash
.venv/bin/python scripts/prepare_paper_plan.py \
  --base "$RUN_DIR/config.json" --seeds 1 2 3 \
  --output runs/plans/cars-ablation-NEW
GCD_ABLATION_CONFIG=runs/plans/cars-ablation-NEW/no_confirmation_seed1.json
bash scripts/train_selex.sh --config "$GCD_ABLATION_CONFIG" --variant local --dry-run
```

Thay tên `no_confirmation` bằng giá trị trong bảng để chọn variant. Có
`manifest.json` với Git SHA/runtime fingerprint/config hashes và `commands.txt`.
Để chuẩn bị trước khi có full run, có thể dùng template adaptive Cars làm
`--base`; nếu muốn SS mỗi epoch thì thêm `--eval-every 1` **như nhau** ở mọi
lệnh train. `--eval-every` chỉ hợp lệ ở mode train; check/smoke dùng lệnh
không có override này. Snapshot hiệu lực trong mỗi run là nguồn đối chiếu.

Trên Linux/NVIDIA đã sẵn sàng, check rồi chọn lệnh train:

```bash
bash scripts/train_selex.sh check --config "$GCD_ABLATION_CONFIG" --variant local
bash scripts/train_selex.sh --config "$GCD_ABLATION_CONFIG" --variant local
```

Không chạy lại baseline theo kế hoạch paper hiện tại: hàng(a) dùng corrected
SelEx published last/3 seeds; hàng(d) dùng full seeds. Chỉ cần chạy mới5 biến
thể. Ưu tiên Cars1–2 seed theo GPU; full main table cần ≥3 seed khi đủ compute.
Nếu cần control riêng ngoài kế hoạch, dùng generated `baseline.json` với
`--baseline`; đây là job bổ sung, không tự khởi chạy.

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
