# Train RPC cho planogram

Nhánh riêng: **`feat/train-planogram`**, bắt đầu từ `feat/crish-simgcd` tại
`ed4b3885fb9100efaf7770d9b49a5e663ecd2a1e`. Không sửa trainer hoặc phương pháp
Cars/CUB/Aircraft của nhánh gốc. Mọi lệnh bên dưới chạy từ root repo trên máy GPU.

## Cấu hình đã chốt

| Thành phần | Mặc định |
|---|---|
| Dataset | RPC v5, 40 SKU chọn bằng seed 1, 200 ảnh/SKU |
| Tổng ảnh | 8.000 crop sản phẩm, sau loại ảnh trùng pixel |
| Chia dữ liệu | Mỗi SKU: 140 train / 30 validation / 30 test |
| GCD | 20 known / 20 novel; 50% ảnh train của known được lộ nhãn |
| Training | SimGCD + CRISH E3, **200 epoch, batch 128 ảnh gốc × 2 view** |
| Backbone | DINO v1 ViT-B/16 chính thức, chỉ fine-tune block 11 và head |
| Ảnh | Crop bbox + lề 5%, pad trắng 256², crop 224²; không lật ngang chữ bao bì |
| Optimizer | SGD, LR 0.1, momentum 0.9, weight decay 5e-5; cosine tới 1e-4 |
| Loss | SimGCD: supervised weight 0.35; memax 1; hai contrastive terms |
| Teacher | 0.07 → 0.04 trong 30 epoch, sau đó 0.04 |
| CRISH | Warm-up 10 epoch; refresh mỗi epoch sau đó, cap 8, budget 300 giây/refresh |
| Checkpoint | **Chỉ `last.pt`**, ghi atomic sau mỗi epoch; có optimizer/scheduler/AMP/RNG |
| Trong train | Log loss, LR, thời gian, VRAM, CRISH; **không SS-KMeans hoặc chọn best** |
| Sau train | Classifier All/Old/New, **một lần đánh giá SS-KMeans cuối**, retrieval và so DINO gốc |

Sampler RPC đảm bảo 64 known-labeled + 64 unlabeled trong mỗi batch, lấy mẫu có
hoàn lại. Nó giữ tỷ lệ kỳ vọng 50:50 của weighted sampler upstream nhưng không
phải cùng chuỗi mẫu. Đây là cấu hình ứng dụng RPC riêng, không phải benchmark
Cars/SSB hay phép so trực tiếp với số đã train trước đó.

## 1. Chọn máy và lấy code

Chọn Linux x86_64, NVIDIA 24 GB VRAM làm điểm bắt đầu (ví dụ RTX 3090/4090),
RAM khoảng 32 GB, ổ đĩa trống khoảng 50 GB, Python **3.10–3.12** và driver tương
thích wheel CUDA 12.8. Bộ cài kiểm tra CUDA forward/backward thật rồi mới tiếp tục.
Batch 128 vẫn phải qua preflight trên máy thuê; chưa có phép đo production VRAM
hay thời gian hoàn thành. Nếu đã có thư mục cùng tên, dùng thư mục mới thay vì
ghi đè hoặc đổi branch của người khác.

```bash
git clone --branch feat/train-planogram --single-branch https://github.com/hokylhoangkien/gcd-final.git gcd-planogram
cd gcd-planogram
bash scripts/planogram/bootstrap.sh
```

Bootstrap tạo `.venv-rpc`, cài PyTorch 2.7.1 / torchvision 0.22.1 CUDA 12.8 và
các dependency riêng, chạy tests, rồi tải code/weights DINO đã pin SHA và checksum.
Không dùng `pip install -r requirements.txt` cũ cho runner này.

## 2. Chuẩn bị dataset

```bash
bash scripts/planogram/run.sh prepare
```

Lệnh tải annotation và **chỉ các ảnh exemplar cần lấy**, không tải toàn bộ archive
15 GB, không tải checkout test scenes. Cache ảnh gốc của Kaggle vẫn chiếm ổ đĩa;
crop PNG được lưu riêng. Mỗi file có tối đa ba lần thử tải. Nếu mạng bị ngắt, chạy
lại lệnh; cache và crop hoàn tất được giữ lại. Public download đã kiểm tra với
annotation và một ảnh thật; việc tải toàn bộ subset sẽ chạy trên máy GPU.

Nếu một ảnh trả 404, downloader thử cả đường dẫn nested và root của RPC v5.
Nếu cả hai đều 404, builder ghi `download_errors.jsonl`, lấy ứng viên tiếp theo
cùng SKU và vẫn yêu cầu đủ 200 ảnh/SKU. Năm ảnh liên tiếp không truy cập được sẽ
dừng để kiểm tra nguồn. Lỗi xác thực/quyền truy cập và lỗi mạng không bị coi là
ảnh thiếu. Ảnh crop đã có được dùng lại khi chạy lại `prepare`.

Kaggle đôi khi yêu cầu đăng nhập/xác thực: cấu hình tài khoản bằng cơ chế của
[kagglehub](https://github.com/Kaggle/kagglehub#authenticate) nếu có lỗi 401/403;
không đưa token vào Git. Nếu đã có RPC trên ổ đĩa, dùng:

```bash
bash scripts/planogram/run.sh prepare --source /data/retail_product_checkout
```

Thư mục nguồn phải có `instances_train2019.json` và `train2019/`. Không cần cả hai
cách. Kết quả ở `datasets/rpc40/manifest.json`; mỗi ảnh có ID, split, nhãn legal và
checksum pixel. Chạy lại sẽ kiểm tra manifest/ảnh, không chia lại ngẫu nhiên.

Nguồn và giấy phép: [RPC](https://rpc-dataset.github.io/),
[bản Kaggle v5](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset/versions/5),
CC BY-NC-SA 4.0. Đây là **custom exemplar holdout**, không phải bộ kiểm thử checkout
chính thức. Loại ảnh trùng pixel không loại hết các góc chụp rất giống nhau của cùng
gói hàng; kết quả holdout này có thể thuận lợi hơn ảnh kệ thực tế.

## 3. Kiểm tra GPU rồi train

Nên chạy trong terminal `tmux` hoặc console persistent của nhà cung cấp để mất
kết nối SSH không làm ngắt job. Nếu có tmux: chạy `tmux new -s planogram`, sau đó
paste các lệnh dưới đây trong session đó.

```bash
bash scripts/planogram/run.sh preflight
bash scripts/planogram/run.sh check-repair
bash scripts/planogram/run.sh pilot
```

- `preflight`: kiểm tra checksum dữ liệu và **batch 128 × 2 view thật**, forward,
  loss, backward, optimizer trên GPU, in peak VRAM. Giới hạn 20 phút.
- `check-repair`: một epoch riêng với CRISH bật ngay từ epoch đầu để phát hiện lỗi
  trước khi trả tiền train dài. Nó có thư mục riêng `runs/rpc40-e3-s1-repair-check`,
  lịch 200 epoch nhưng dừng sau một epoch; **không dùng checkpoint này làm kết quả**.
  Giới hạn một giờ; log refresh có thể từ chối toàn bộ sửa đổi hợp lệ.
- `pilot`: epoch đầu tiên của **run thật với lịch 200 epoch**, lưu `last.pt`.
  Đây không phải lịch cosine một epoch. Xem thời gian/VRAM trong log rồi tiếp tục:

```bash
bash scripts/planogram/run.sh resume
```

Lệnh resume tiếp từ epoch đã lưu đến epoch 200. Không chạy `train` sau `pilot`,
vì đó là bắt đầu mới và sẽ bị chặn do output đã tồn tại. Nếu muốn chạy thẳng từ đầu
sau preflight/check-repair, có thể dùng `train` thay cho `pilot` + `resume`.

Mỗi lần gọi train/resume giới hạn 24 giờ; trainer xin dừng ở ranh giới epoch sau
23,5 giờ. Nếu chưa đủ 200 epoch, gọi lại `resume`. Ctrl+C/SIGTERM yêu cầu dừng sau
epoch hiện tại; nếu máy bị tắt cưỡng bức thì resume từ epoch đã commit gần nhất.
Không hứa thời gian hoàn thành: CRISH còn có snapshot/verification mỗi refresh.

Log dễ tìm trong `logs_rpc/`; metadata ở `runs/rpc40-e3-s1/run.json`, loss/timing ở
`metrics.jsonl`, repair/exposure ở `crish-global.jsonl`. Resume kiểm tra cấu hình,
manifest, source runner và phục hồi prefix log tương ứng checkpoint, không tiếp
tục dùng repair records của epoch chưa commit.

### Nếu batch 128 bị OOM

Runner không âm thầm đổi batch. Chọn **run mới**, ví dụ batch 64 với LR 0.05, rồi
giữ đúng các tham số đó khi resume. Đây là quy tắc scale LR khởi đầu, chưa phải
kết quả tuning. Nếu đổi batch thì phép so với recipe 128 cũng đã đổi.

```bash
export RPC_RUN=runs/rpc40-e3-b64-s1
bash scripts/planogram/run.sh preflight --batch-size 64 --lr 0.05
bash scripts/planogram/run.sh check-repair --batch-size 64 --lr 0.05
bash scripts/planogram/run.sh pilot --batch-size 64 --lr 0.05
bash scripts/planogram/run.sh resume --batch-size 64 --lr 0.05
```

Giữ `RPC_RUN` đó cho evaluate/export. Sau khi SSH lại, đặt lại biến `RPC_RUN` và
các tham số batch/LR trước resume. Không đổi batch/LR giữa chừng trên checkpoint
cũ. Không dùng gradient accumulation rồi coi nó tương đương batch contrastive 128.

## 4. Sau khi đủ 200 epoch

```bash
bash scripts/planogram/run.sh evaluate
bash scripts/planogram/run.sh export
tar -czf planogram-rpc-result.tar.gz runs/rpc40-e3-s1
```

Nếu dùng tên run khác thì thay đường dẫn ở lệnh tar. Giữ file tar trên storage
bền vững hoặc tải về trước khi trả máy. `last.pt` chứa cả trạng thái resume nên
lớn hơn artifact inference; nếu chỉ cần inference, lấy thư mục `export/`.

Evaluator từ chối checkpoint chưa đủ epoch, trừ khi chủ động thêm `--allow-partial`
để chẩn đoán. Kết quả thực ở `evaluation/metrics.json`:

- **Classifier**: All/Old/New với Hungarian matching trên từng tập đánh giá.
- **SS-KMeans cuối**: fit trên embedding train, chỉ clamp các nhãn known đã được
  phép dùng khi train; tuyệt đối không fit bằng ảnh test hoặc nhãn novel. Một job
  cuối có 10 initialization, tối đa 100 Lloyd iterations; chọn bằng inertia train.
  Báo cả train-unlabeled transductive và val/test inductive. Solver Euclidean
  constrained Lloyd riêng trên feature L2-normalized; chưa tuyên bố đồng nhất
  từng phép toán với evaluator HypCD/CRiSH-GCD cũ.
- **Retrieval**: gallery cố định 5 ảnh train/SKU, query là val/test khác ảnh. Dùng
  backbone 768-D L2-normalized, cosine lấy ảnh tham chiếu tốt nhất mỗi SKU; margin
  tính giữa hai **SKU khác nhau**, không phải hai ảnh cùng SKU. Có top-1, macro
  top-1, Old/New, coverage và accepted accuracy tại threshold 0.35 / margin 0.05.
  Các ngưỡng này là điểm xuất phát từ app, chưa được hiệu chuẩn cho model mới.
- **DINO gốc**: cùng gallery/query để so retrieval trước/sau fine-tune. Gallery có
  nhãn tham chiếu cho cả novel SKU là giả định của hệ thống catalog sau train;
  nó chỉ dùng trong đánh giá retrieval, không cung cấp nhãn novel cho training.

Không chọn best bằng test/novel labels. Không chạy SS-KMeans trong train. Kết quả
SS-KMeans không thay thế phép đo retrieval, và bộ query này chưa kiểm tra sản phẩm
ngoài catalog. Chỉ công bố các metric thực sự có trong file sau khi chạy.

## 5. Đưa model về planogram sau khi đánh giá

Export tạo:

```text
runs/rpc40-e3-s1/export/
  embedding.ts          Backbone TorchScript, không cần code DINO lúc inference
  manifest.json         SHA-256 artifact/checkpoint và contract embedding mới
  worker.env.example    Giá trị cần đồng bộ AI worker
  backend.env.example   Giá trị cần đồng bộ Spring backend
```

Adapter có sẵn tại `deployment/simgcd_rpc_recognizer.py`, cùng interface
`BaseRecognizer` của planogram. **Repo planogram và cấu hình đang chạy chưa bị sửa**;
chỉ chuyển sang model mới sau khi kiểm tra metric và ảnh kệ của mình:

1. Copy thư mục export thành `backend/ai-worker/models/simgcd_rpc/` của planogram.
2. Copy adapter vào `backend/ai-worker/recognizer/simgcd_rpc_recognizer.py`.
3. Trong `worker.py`, thay import bằng
   `from recognizer.simgcd_rpc_recognizer import SimGCDRPCRecognizer as DinoV2Recognizer`.
   Alias giữ nguyên lời gọi khởi tạo hiện có; detector/comparator không cần đổi.
4. Đồng bộ các giá trị từ hai file env example vào env worker/backend tương ứng,
   giữ nguyên secrets và các setting khác. Model mới là **768 chiều**, model DINOv2
   cũ là 384 chiều; model name/version/preprocessing cũng khác và cần khớp cả hai bên.
5. Restart backend/worker và tạo lại embedding tham chiếu cho ảnh sản phẩm theo
   contract mới bằng luồng embedding của app. **Không trộn vector mới với vector
   DINOv2 cũ**, không gán cluster index của RPC thành ProductID của cửa hàng.
6. Kiểm thử ảnh kệ có nhãn thực, gồm các bao bì dễ nhầm và sản phẩm ngoài catalog,
   rồi hiệu chuẩn threshold/margin trên tập validation thực tế.

RPC là bước học domain hàng hóa. Nó không bảo đảm model nhận đúng tất cả SKU Việt
Nam mà chưa có ảnh tham chiếu. SS-KMeans cuối chỉ đánh giá GCD; app nhận diện bằng
embedding + ảnh catalog, không cần chạy lại K-means cho mỗi ảnh kệ.

## Tình trạng kiểm chứng trước GPU

Xem [reports/PLANOGRAM_READINESS.md](reports/PLANOGRAM_READINESS.md) để biết lệnh
đã chạy và giới hạn kiểm chứng. Chưa chạy full subset, DINO real-image training,
CUDA preflight, benchmark 200 epoch hoặc test ảnh kệ trong session chuẩn bị này.

Nguồn kỹ thuật: [PyTorch wheels](https://pytorch.org/get-started/previous-versions/#v2-7-1),
[DINO chính thức](https://github.com/facebookresearch/dino),
[SimGCD](https://github.com/CVMI-Lab/SimGCD).
