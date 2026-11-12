# Train SelEx baseline và CRiSH Keep/Abstain/Move

Ngày 12/11/2026. Đây là hướng dẫn chính cho **baseline SelEx** và **SelEx +
CRiSH local**. Không dùng `scripts/train_crish.sh` (đường Hyp-SelEx cũ) cho
cặp này. Không coi `policy` là phiên bản đầy đủ thay thế local.

## 1. Chọn đúng thuật toán trước khi thuê/chạy GPU

| Flag | Phương pháp | Local repair | Population policy | Online weights |
|---|---|---|---|---|
| `--baseline` | SelEx đối chứng | Bỏ qua | Bỏ qua | Bỏ qua |
| `--variant local` | CRiSH Keep/Abstain/Move trên SelEx | Bật | Tắt | Tắt |
| `--variant policy` | CRiSH-P, Keep/Abstain theo tỷ lệ toàn tập | Tắt | Bật | Tắt |
| `--variant online` | Pair weighting | Tắt | Tắt | Bật |
| `--variant hybrid` | Policy + pair weighting | Tắt | Bật | Bật |

Các dòng variant giả định không thêm override `--pair-reliability`.
JSON 200 epoch hiện mặc định **policy**, không phải local. Luôn chỉ định
`--variant local` khi muốn Keep/Abstain/Move. Không đổi raw config/default
âm thầm: checkpoint cũ cần source/config ban đầu để resume.

Code local: `crish/engine.py` điều phối; `candidates.py` tạo ứng viên;
`reliability.py` cung cấp evidence; `rebuild.py` dựng lại giám sát;
`lookahead.py`/`probe.py` chấm SGD thử; `policy.py` quyết định; tích hợp trong
`crish/training.py` và `selex_runtime/`. Move có thể bị loại vì thiếu support
hoặc probe không đạt. Chọn local không bảo đảm sẽ có action được accept.
Partition stability là tùy chọn và đang tắt (`stability_trials=0`); không gọi
đây là thí nghiệm đầy đủ của mọi thành phần tùy chọn trong manuscript.
Nền hiện tại là corrected Euclidean SelEx; manuscript Hyp-SelEx cũ không phải
bản mô tả hoàn toàn khớp runtime này. Không tự sửa paper để che khác biệt.

## 2. Môi trường và data

Mac chỉ phát triển/test nhỏ; train trên Linux NVIDIA. Môi trường compatibility:
Python 3.9–3.11, Torch 2.4.1/cu121, torchvision 0.19.1; bootstrap cài dependency
ghim và upstream snapshot, không tải data/weights hay train dài:

```bash
git clone --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git
cd gcd-final
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
```

Nếu đã clone/cài/data đầy đủ thì dùng lại, không clone đè hay cài lại.
Bootstrap có suite CPU timeout 60 giây; timeout không chứng minh test pass,
không phải lý do tự đổi Torch/CUDA. Preflight/smoke thực vẫn bắt buộc trước train.

Layout Cars:

```text
datasets/cars/cars_train/                         # 8144 ảnh
datasets/cars/cars_test/                          # 8041 ảnh
datasets/cars/devkit/cars_train_annos.mat
datasets/cars/devkit/cars_test_annos_withlabels.mat
weights/dinov2_vitb14_reg4_pretrain.pth
```

Cách đặt data/weights và biến môi trường được ghi ở
[README, mục data](../README.md#2-đặt-dataset-và-pretrained-weights).
Đây là hướng dẫn layout/download dùng chung; các lệnh train Hyp-SelEx ở phần
dưới README là lịch sử, không dùng cho SelEx. Các root portable có thể đổi bằng
`GCD_DATA_ROOT`, `GCD_WEIGHTS_ROOT`, `GCD_RUNS_ROOT`. Không tự chia lại SSB.
Đủ số ảnh chưa phải đủ protocol: preflight xác minh annotation, split,
pretrained state dict và CUDA forward/backward ảnh thật với head thật.

## 3. Xem kế hoạch và kiểm tra

Không dùng GPU, không bắt đầu train:

```bash
bash scripts/train_selex.sh --dataset scars --variant local --eval-every 1 --dry-run
bash scripts/train_selex.sh --dataset scars --baseline --eval-every 1 --dry-run
```

Ở plan local phải thấy `baseline_family=selex`, `method=crish`,
`discrete_repair=true`, `population_policy.enabled=false`,
`pair_reliability.enabled=false`, `epochs=200`, head 65536.
Plan baseline có `method=baseline`, `discrete_repair=false`; method baseline
bypass tất cả controller dù JSON giữ các setting để đối chứng.

Chỉ khi GPU không có job cạnh tranh:

```bash
bash scripts/train_selex.sh check --dataset scars --variant local
bash scripts/train_selex.sh smoke --dataset scars --variant local
```

Smoke là 1 epoch, tối đa 2 minibatch, warmup 0, vẫn dùng extraction/clustering
toàn bộ pool. Nó phải thực sự chấm ứng viên; Keep hợp lệ, fallback/no scoring
không phải smoke thành công. Không dùng kết quả smoke làm benchmark.
Nếu báo `No pseudo-SSE change in the scoring batch`, kiểm tra
`crish.jsonl`/`scoring_batch` và [audit batch chấm local](15_local_scoring_support.md).
Không đổi sang `policy` hoặc nới guard để bỏ qua lỗi coverage của local.

## 4. Train từng arm, trong tmux

```bash
tmux new -s train
```

Nếu session đã tồn tại, dùng `tmux attach -t train`. Không tạo job mới khi
session cũ còn đang train. Trong tmux, **chọn đúng một** lệnh:

```bash
# Phương pháp của bạn: local Keep/Abstain/Move, SS mỗi epoch
bash scripts/train_selex.sh --dataset scars --variant local --eval-every 1
```

Baseline chạy ở lần độc lập, cùng seed/lịch SS và ngân sách:

```bash
bash scripts/train_selex.sh --dataset scars --baseline --eval-every 1
```

Full command tự lặp preflight/bounded smoke trước run mới; không cần smoke
riêng lần nữa nếu đã chọn full command. DINOv2 reg4, batch 128, 200 epoch,
seed mặc định 1; thêm `--seed N` cho cả hai arm nếu cần đối chứng seed khác.
Không tự chạy sweep. Dung lượng VRAM/thời gian phải đo; SS mỗi epoch tăng
chi phí. Deadline 604800 giây full và 7200 giây mỗi evaluator là giới hạn,
không phải thời gian dự đoán. Logs có run ID, PID, config/metadata và checkpoint.

Rời tmux giữ chạy: **Ctrl+B, thả ra, D**. Vào lại: `tmux attach -t train`.
Ctrl+C ở pane chạy chính dừng launcher; chỉ làm khi chủ động dừng run đó.
Ctrl+C ở tab `tail -f` chỉ dừng xem log. Không `kill` PID lạ/không rõ chủ sở hữu.

## 5. Theo dõi và kết quả

Launcher tạo một run smoke rồi một run full, mỗi run có ID/log riêng. Xem
đường dẫn full mới trong tmux; `TRAIN plan ... epochs=200` xác nhận full run.
Ở tab SSH phụ, dùng đúng đường dẫn launcher in (thay `RUN_ID`):

```bash
tail -n 25 -f runs/RUN_ID/run.log
```

VRAM: `watch -n 2 nvidia-smi`. Không coi GPU utilization 0% tức thời là treo.
Local controller ghi `crish.jsonl`; population controller mới ghi
`CRISH_P`/`crish_p.jsonl`. `intervention_diagnostics.jsonl` ghi dose thực nhận.
Không kết luận local đã chạy từ dòng `TRAIN method=crish` đơn lẻ: kiểm tra
plan/config và các log controller. Warmup 10, refresh mỗi 5 epoch, chỉ áp dụng
nếu ứng viên đủ điều kiện. Mask/assignment sửa sống trong epoch; weights học
được tiếp tục qua epoch sau, hierarchy được dựng lại.

`EVAL ... plain_kmeans_disjoint_test` là theo dõi plain; `SELEX_SELECTION`
chọn best phụ bằng Old-v1. Kết quả SS có protocol
`balanced_ss_kmeans_train_unlabeled`, All/Old/New global Hungarian-v2 trên
unlabeled training pool; chỉ legal known labels dùng làm anchors.

Với `--eval-every 1`: epoch 1–199 chấm SS định kỳ, epoch 200 dùng final-last
evaluator. `EVAL worker_seconds` là thời gian evaluator, không phải cả epoch.
Mỗi lần in `SS_BEST metric=all|old|new`, epoch và bộ ba điểm cùng checkpoint.

| Artifact trong run full | Nội dung |
|---|---|
| `evaluations/epoch_NNNN.json`, `epoch_metrics.jsonl` | SS từng epoch |
| `ss_best.json` | Ba best độc lập, epoch, full triplet, path/hash weights |
| `backbone_best_ss_{all,old,new}.pt` | Ba backbone best, không phải optimizer resume |
| `evaluation_best_ss_{all,old,new}.json` | Metadata/điểm tương ứng |
| `evaluation_last.json`, `evaluation.json` | SS fixed last, giữ riêng |
| `evaluation_best_old_test.json` | SS của best chọn theo plain-test Old-v1 |
| `resume.pt` | Backbone/head/optimizer/scheduler/RNG/config cho resume |

Best SS dùng nhãn đánh giá để chọn checkpoint: báo rõ, không gọi là kết quả
held-out độc lập. Baseline phải có cùng lịch/số lần chọn. Không ghép cực đại
All/Old/New của ba epoch thành một hàng. Không coi best plain là best SS.

## 6. Dừng, giữ artifacts và resume

Không cập nhật runtime/config dưới job đang chạy. Giữ nguyên thư mục run,
source commit, config, mọi JSON/log và checkpoint. Nếu chủ động dừng thì vào
đúng pane launcher, Ctrl+C một lần, chờ dấu nhắc và kiểm tra GPU/processes.
Không chuyển `policy` sang `local` bằng resume: đó là hai thí nghiệm khác.

Resume đúng source/config của local run (không thêm override variant/seed/SS):

```bash
bash scripts/train_selex.sh --config runs/RUN_ID/config.json --resume runs/RUN_ID/resume.pt
```

Với baseline, thêm `--baseline`. Source/config fingerprint phải khớp; lưu
nguyên run folder để khôi phục best SS/plain và history. Không dùng checkpoint
cũ sau khi thay runtime rồi bỏ qua fingerprint. Mỗi run mới có ID riêng.

## Verification boundary

58 CPU tests đã qua ở commit thêm SS best; không đồng nghĩa local đã chạy
được full trên CUDA. Log remote người dùng cung cấp xác nhận Cars/DINOv2,
RTX 3090 và SS mỗi epoch cho **policy**, gồm ba lần Keep 10/15/20; không coi
đó là benchmark local hoặc bằng chứng Move có ích. Bản hướng dẫn này chỉ sửa
tài liệu, không đổi thuật toán/default/config hay gián đoạn remote run.

Hai lệnh dry-run ở mục 3 đã được thực thi và kiểm tra JSON plan: local có
`method=crish`, discrete repair bật, population/online tắt; baseline có
`method=baseline`, discrete repair tắt; cả hai SelEx/200 epoch/head 65536.
`git diff --check` đạt. Không chạy train/test GPU để kiểm tra thay đổi tài liệu.
