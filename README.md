# SelEx + CRiSH — hướng dẫn train

**Đọc file này là đủ để chạy Stanford Cars, CUB và FGVC-Aircraft.**
Baseline là **SelEx**; phương pháp cần kiểm chứng là **SelEx + CRiSH local
Keep / Abstain / Move**. Train trên Linux NVIDIA; Mac chỉ phát triển/test nhỏ.

| Bạn muốn chạy | Flag bắt buộc | Có Move? |
|---|---|---|
| SelEx baseline | `--baseline` | Không |
| Thuật toán CRiSH local | `--variant local` | Có, nếu ứng viên đạt điều kiện |
| CRiSH-P thử nghiệm riêng | `--variant policy` | Không, chỉ Keep/Abstain toàn tập |

**Luôn ghi `--variant local` khi train thuật toán của bạn.** JSON 200 epoch
hiện vẫn mặc định policy để bảo toàn config lịch sử; bỏ flag sẽ chọn nhầm.
`local` chỉ cơ chế sửa giám sát, không có nghĩa chạy trên CPU/Mac.
Các biến thể online/hybrid và nhánh Hyp-SelEx cũ không thuộc quickstart này.

## 1. Cài môi trường một lần

Nếu máy đã có repo, môi trường và data/weights thì bỏ qua phần cài/tải.
Không clone đè, cập nhật runtime hoặc đổi config dưới job đang chạy.

```bash
git clone --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git
cd gcd-final
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
```

Cần Python **3.9–3.11**. Bootstrap tạo `.venv`, cài **Torch 2.4.1/cu121,
torchvision 0.19.1** và dependencies ghim, kiểm tra source và code CPU;
không tải data/weights hoặc train dài. Nếu suite CPU timeout 60 giây, đó chưa
phải test pass; không cài lại cả môi trường hay đổi Torch chỉ vì timeout.
Preflight/smoke thực ở dưới vẫn cần đạt trước full train.

Trên Ubuntu trống, công cụ hệ thống tối thiểu:

```bash
apt-get update && apt-get install -y python3-venv python3-dev build-essential git curl tmux unzip ca-certificates
```

Không cài lại NVIDIA driver nếu `nvidia-smi` đang nhận GPU bình thường.
Nếu dùng interpreter đã chuẩn bị ở nơi khác, đặt `SELEX_PYTHON` trỏ tới nó.

## 2. Chuẩn bị data và weights

Cả ba dùng chung **DINOv2 ViT-B/14 với 4 register tokens**. Không commit ảnh,
weights, checkpoint hoặc credentials. Layout mặc định tính từ gốc repo:

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

**Giữ train/test và metadata gốc; không tự chia lại lớp hay đổi tên ảnh.**
SSB split đã ghim trong upstream. Aircraft dùng bài toán **100 variant**,
`trainval` để train, `test` riêng để theo dõi. CUB cần cả file attributes.

| Dataset | Flag | Known / Novel | Tỷ lệ labeled trong known |
|---|---|---|---|
| Stanford Cars | `scars` | 98 / 98 | 50% |
| CUB-200-2011 | `cub` | 100 / 100 | 50% |
| FGVC-Aircraft | `aircraft` | 50 / 50 | 50% |

Có thể dùng data gốc nhóm đã có. Nguồn Cars tải gộp:
[stanford_cars.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/tree/main).
Giải nén archive đó vào thư mục tạm, xác định `cars_train`, `cars_test`,
`devkit` rồi đặt theo cây trên; annotation test có thể ở thư mục gốc archive,
cần đặt bản sao vào `devkit/`. Không dùng split Zhou trong archive thay SSB.
Nguồn Aircraft: [trang phát hành FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).
CUB cần bản phát hành CUB-200-2011 đầy đủ đúng cây thư mục trên, không chỉ ảnh.
Runtime không tự tải/giải nén dataset; preflight sẽ kiểm tra file và protocol.

Nếu chưa có weights, tải mirror công khai và kiểm tra SHA256 trước khi dùng:

```bash
mkdir -p weights
curl -fL -C - --connect-timeout 20 --max-time 1800 \
  -o weights/dinov2_vitb14_reg4_pretrain.pth \
  https://huggingface.co/cubert-gmbh/dinov2/resolve/e2c8060a74112f7537484ed50097b1497d5f032c/dinov2_vitb14_reg4_pretrain.pth

echo '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71  weights/dinov2_vitb14_reg4_pretrain.pth' | sha256sum -c -
```

Nếu data ở ổ khác, đặt các root trước mọi lệnh (thay bằng đường dẫn của bạn):

```bash
export GCD_DATA_ROOT=/path/to/datasets
export GCD_WEIGHTS_ROOT=/path/to/weights
export GCD_RUNS_ROOT=/path/to/runs
```

Root data là thư mục **chứa** `cars/`, `cub/`, `aircraft/`; root weights chứa
trực tiếp file `.pth`. Mặc định là `datasets/`, `weights/`, `runs/` trong repo.

## 3. Chọn dataset và kiểm tra trước train

Từ thư mục gốc repo, chọn **một** dataset, dùng xuyên suốt phiên shell:

```bash
GCD_DATASET=scars
# Đổi thành cub hoặc aircraft cho hai dataset còn lại.
```

Xem plan, không dùng GPU:

```bash
bash scripts/train_selex.sh --dataset "$GCD_DATASET" --variant local --eval-every 1 --dry-run
```

Plan local phải có `baseline_family=selex`, `method=crish`,
`discrete_repair=true`, `population_policy.enabled=false`, online weighting
false, `epochs=200`, head 65536. Baseline dùng `method=baseline` và bypass
mọi controller. `TRAIN method=crish` đơn lẻ không phân biệt local với policy.

Kiểm tra GPU trống bằng `nvidia-smi`. Nếu có job đang chạy, xem chủ sở hữu và
log trước; không tự kill hoặc mở job cạnh tranh. Sau đó:

```bash
bash scripts/train_selex.sh check --dataset "$GCD_DATASET" --variant local
```

Preflight xác minh data/split/weights/import và **CUDA forward/backward ảnh
thật với head thật**. Chỉ nhận GPU qua `nvidia-smi` chưa phải CUDA test đạt.
Nếu muốn kiểm tra smoke riêng trước quyết định train dài:

```bash
bash scripts/train_selex.sh smoke --dataset "$GCD_DATASET" --variant local
```

Smoke local là 1 epoch, tối đa 2 minibatch; warmup 0, SS định kỳ tắt nhưng vẫn
có đánh giá SS cuối. Nó trích/clustering toàn bộ pool. Gate yêu cầu thực sự
chấm được counterfactual; Keep hợp lệ, fallback/no-scoring sẽ chặn full train.
Smoke hoàn tất báo **`Status=completed`**. Điểm smoke không phải benchmark.

## 4. Train 200 epoch trong tmux

```bash
tmux new -s train
```

Nếu session đã có: `tmux attach -t train`. Kiểm tra job cũ đã kết thúc trước
khi chạy job mới. Trong tmux, chọn **đúng một lệnh** từ các bảng dưới;
không dán cả bảng để chạy tuần tự ngoài ý muốn, không chạy đồng thời.

**Phương pháp: SelEx + CRiSH Keep/Abstain/Move, SS mỗi epoch**

| Dataset | Lệnh |
|---|---|
| Cars | `bash scripts/train_selex.sh --dataset scars --variant local --eval-every 1` |
| CUB | `bash scripts/train_selex.sh --dataset cub --variant local --eval-every 1` |
| Aircraft | `bash scripts/train_selex.sh --dataset aircraft --variant local --eval-every 1` |

**Baseline SelEx, cùng lịch SS**

| Dataset | Lệnh |
|---|---|
| Cars | `bash scripts/train_selex.sh --dataset scars --baseline --eval-every 1` |
| CUB | `bash scripts/train_selex.sh --dataset cub --baseline --eval-every 1` |
| Aircraft | `bash scripts/train_selex.sh --dataset aircraft --baseline --eval-every 1` |

Full command tự chạy **preflight → smoke → full 200 epoch**. Không cần smoke
riêng nếu đã chọn full command; nếu vừa smoke riêng thì full vẫn lặp smoke.
Smoke và full có **run ID/log khác nhau**. `TRAIN plan ... epochs=200` xác nhận
đã vào full run, không có nghĩa đã hoàn thành. Mỗi dataset có config riêng
trong `configs/selex_{scars,cub,aircraft}_{baseline,crish}_200.json`.

Cả ba dùng DINOv2 reg4, batch 128, head 65536, seed mặc định 1. Thêm `--seed N`
cho cả hai arm để so seed khác. Không tự mở multi-seed sweep. Đo VRAM và thời
gian thực, nhất là controller và các mốc bật tầng loss; không hứa thời gian
hoàn tất. Deadline full 604800 giây, mỗi evaluator 7200 giây là **giới hạn**.

- Rời tmux mà giữ train: **Ctrl+B, thả ra, D**.
- Vào lại: `tmux attach -t train`.
- Tmux tự nó không dùng VRAM; Python/CUDA bên trong mới dùng.
- Ctrl+C ở pane chạy chính dừng launcher. Ctrl+C ở `tail`/`watch` chỉ dừng xem.

## 5. Theo dõi log, GPU và can thiệp

Launcher in đường dẫn run full. Ở tab SSH phụ, từ gốc repo đặt đúng thư mục
full (thay `RUN_ID` bằng ID được in, hoặc dùng đường dẫn `GCD_RUNS_ROOT`):

```bash
RUN_DIR=runs/RUN_ID
tail -n 100 -f "$RUN_DIR/run.log"
```

`-n 100` chỉ số dòng cũ hiện ban đầu; `-f` hiện **mọi dòng mới liên tục**.
Nó không tự chuyển file từ smoke sang full: lấy đúng log full mà launcher in.

GPU cập nhật mỗi 2 giây, chạy ở tab khác:

```bash
watch -n 2 nvidia-smi
```

Đọc `Memory-Usage`, không coi utilization 0% tức thời là treo. VRAM không tăng
đơn giản theo số epoch; trainer/evaluator có thể đồng thời hiện hai PID.
Kiểm tra command/PID trước khi kết luận có job người khác. Không tự kill PID.

Xem tất cả action local thực sự được nhận, không in các mảng prototype dài:

```bash
.venv/bin/python -c 'import json,sys; [(print("epoch:",r["epoch"],"accepted:",r["accepted"]), print([(d["kind"],d["sample_id"],d.get("source"),d.get("target")) for d in r["decisions"] if d["status"]=="accepted"])) for r in map(json.loads,open(sys.argv[1]))]' "$RUN_DIR/crish.jsonl"
```

Local warmup 10: lần đầu ở **`epoch=10`, tức epoch thứ 11**, rồi 15,20,25,…
Mỗi lần tối đa **1 action được chấp nhận**, có thể Keep. `crish.jsonl` chứa
candidate, discovery/confirmation gain, quyết định và thời gian. Policy dùng
`crish_p.jsonl`/`CRISH_P`, không phải local. `intervention_diagnostics.jsonl`
ghi số ảnh và quan hệ positive thực sự thay đổi.

Đầu mỗi epoch dựng lại hierarchy; weights/optimizer vẫn tiếp tục, không học
lại từ đầu. Mask/assignment sửa chỉ sống trong epoch đó, ảnh hưởng đã học
vào weights vẫn giữ. Probe gain là chênh lệch loss sau SGD thử, **không phải
% accuracy tăng**. Accepted nhiều hay đổi nhiều quan hệ không chứng minh tốt hơn.

Đối chiếu action đã nhận với SS trước–trong–sau epoch, chỉ đọc log:

```bash
.venv/bin/python scripts/analyze_crish_interventions.py "$RUN_DIR" --output /tmp/crish_action_ss.csv
```

Xuất CSV và JSON cùng tên, đủ action/gain/dose và bộ All/Old/New trước, trong,
sau can thiệp. `epoch=10` là can thiệp trong epoch thứ 11, so SS epoch 10 → 11
→ 12. Báo thiếu SS là `NA`, không lấy best hoặc plain thay thế. Danh sách Move
đi kèm SS giảm chỉ là liên hệ theo thời gian, không chứng minh Move gây giảm.
Nếu nhiều action cùng epoch thì chúng chia sẻ một SS delta, không cộng delta
mỗi action. Câu hỏi “không Move thì sao” vẫn cần baseline matched/replay.
Script chỉ dùng stdlib; trên máy đang train có thể copy riêng script vào `/tmp`
rồi chạy bằng `.venv/bin/python`, không cập nhật source runtime hoặc dừng job.

## 6. Đọc SS, best All/Old/New và thời gian

| Dòng log | Ý nghĩa |
|---|---|
| `TRAIN ... loss=...` | Loss minibatch |
| `EVAL ... plain_kmeans_disjoint_test` | Accuracy plain theo dõi mỗi epoch |
| `SELEX_SELECTION ... Old_v1=...` | Chọn checkpoint phụ theo Old-v1 disjoint-test |
| `EVAL ... balanced_ss_kmeans_train_unlabeled` | **SS All/Old/New cần so sánh** |
| `SS_BEST metric=all/old/new` | Best SS riêng, epoch và full triplet |
| `RUNTIME epoch_complete=0` | Đã hoàn tất epoch đầu (log đếm từ 0) |
| `EVAL worker_seconds=...` | Thời gian evaluator SS, không phải toàn epoch |

SS dùng **Balanced Semi-Supervised K-means**, legal known labels làm anchors,
chấm phần unlabeled của train pool bằng global Hungarian-v2, tương ứng
`Train ACC Unlabelled_v2` của SelEx. Ground truth unlabeled chỉ ở evaluator,
không đưa vào training/controller. All là trung bình theo số mẫu Old/New,
không phải trung bình cộng hai tỷ lệ. JSON lưu fraction; log in phần trăm.

`--eval-every 1` chấm SS epoch 1–199; epoch 200 dùng final-last evaluator để
không chấm trùng. Trong run full:

| File | Nội dung |
|---|---|
| `evaluations/epoch_NNNN.json`, `epoch_metrics.jsonl` | SS từng epoch |
| `ss_best.json` | Best All, Old, New riêng, full triplet, epoch và hash/path |
| `backbone_best_ss_all.pt` + `evaluation_best_ss_all.json` | Weights và điểm best All |
| `backbone_best_ss_old.pt` + `evaluation_best_ss_old.json` | Weights và điểm best Old |
| `backbone_best_ss_new.pt` + `evaluation_best_ss_new.json` | Weights và điểm best New |
| `evaluation_last.json`, `evaluation.json` | SS của fixed-last |
| `evaluation_best_old_test.json` | SS của checkpoint chọn theo plain-test Old-v1 |
| `test_metrics.jsonl`, `test_evaluations/` | Lịch sử plain, khác SS |
| `config.json`, `metadata.json`, `environment.txt`, `readiness.json` | Provenance/preflight |
| `resume.pt` | Model/head/optimizer/scheduler/RNG/config để resume |

Ba SS best giữ weights riêng, không lưu 200 bản backbone; hòa điểm giữ epoch
sớm hơn. **Không ghép cực đại All/Old/New ở ba epoch thành một hàng.** Best SS
được chọn bằng nhãn đánh giá, không phải held-out độc lập. So baseline bằng
cùng split/backbone/seed/epoch/ngân sách chấm/chính sách chọn; giữ SS-last riêng.
Bộ chọn best SS chỉ xét SS định kỳ và fixed-last, không thêm lần
chấm checkpoint phụ Old-test vào tập ứng viên. Baseline phải bật SS cùng lịch.

Xem bảng điểm đã lưu, không chạy evaluation mới:

```bash
.venv/bin/python scripts/show_metrics.py "$RUN_DIR"
cat "$RUN_DIR/ss_best.json"
```

`ss_best.json` là nguồn rõ ràng cho ba weights best SS. `show_metrics.py`
in lịch sử và các peaks theo protocol, không thay chính sách chọn checkpoint.
Để ước lượng thời gian, lấy timestamp hai SS report cách vài epoch chia số
epoch chênh lệch; đừng nhân riêng `worker_seconds` thành thời gian train.
Đo cả giai đoạn sau warmup/bật tầng loss. Ổ đĩa: `df -h .` và `du -sh "$RUN_DIR"`.

## 7. Dừng và resume

Nếu chủ động dừng: attach đúng tmux, Ctrl+C một lần, chờ dấu nhắc, kiểm tra
`nvidia-smi`. Giữ nguyên **toàn bộ run folder**, source commit và config.
Checkpoint chỉ bảo đảm tới epoch lưu xong; batch đang chạy dở có thể chưa lưu.
Không cập nhật source/config để resume rồi bỏ qua fingerprint. Best backbone
không có optimizer; resume phải dùng `resume.pt`.

Local, từ đúng phiên bản source của run:

```bash
bash scripts/train_selex.sh --config "$RUN_DIR/config.json" --resume "$RUN_DIR/resume.pt"
```

Baseline thêm `--baseline`. Không thêm override seed/variant/epoch/SS vì resume
cần config gốc. Không chuyển policy sang local hoặc dataset khác bằng resume.
Run resume có ID mới; phải lấy log mới. Không xóa run nguồn chứa best/history.

## 8. Trạng thái nghiên cứu và tài liệu tham khảo

Code giữ upstream SelEx snapshot `d8107b5` nguyên vẹn, có sửa adapter được ghi
nhận. Đây là **corrected SelEx, lịch kéo dài 200 epoch**, không tự coi là tái
lập nguyên xi số paper 2024 hoặc lịch 50 epoch của README upstream sửa 2026.
Manuscript cũ còn mô tả nền Hyp-SelEx; chưa phải mô tả hoàn toàn khớp run local
trên SelEx. Stability tùy chọn đang tắt. Không hứa tăng accuracy, SOTA hay novelty.

Log CUDA Cars do người dùng cung cấp đã có full local run và action Move/
Abstain, SS/best theo epoch; chưa có completed benchmark 200 epoch hoặc baseline
matched để kết luận hiệu quả. CUB/Aircraft có wiring/config/test, chưa xác nhận
full CUDA run từ phiên này. Bản sửa scoring batch local đã qua 74 test CPU;
đó là kiểm chứng triển khai, không phải bằng chứng hiệu quả thực nghiệm.

Chạy kiểm tra code nhỏ, không train trên Mac:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py --code-only
```

Các báo cáo dưới đây chỉ để tra cứu thêm, **không cần đọc để chạy các lệnh trên**:
[protocol/paper comparability](reports/10_paper_comparability_audit.md),
[algorithm audit](reports/11_source_algorithm_review.md),
[SS-best](reports/13_ss_epoch_best.md),
[local scoring-support fix](reports/15_local_scoring_support.md).
Nhánh Hyp-SelEx cũ được lưu trong lịch sử Git và các báo cáo baseline lịch sử;
`scripts/train_crish.sh` không dùng cho cặp SelEx/local trong README này.
Giấy phép và nguồn gốc: [NOTICE](NOTICE), [LICENSES](LICENSES/).
