# Một epoch A/B: Move có thực sự gặp các cặp SSE bị đổi?

2026-11-15. Branch: `feat/local-adaptive-acceptance`. Base: `a9985b5`.

## Kết quả thực tế và giới hạn

Đã triển khai bộ xếp batch và runner A/B riêng, mặc định không tác động
trainer. Đã chạy test, smoke foreground và một job nền có timeout trên Mac.
**Chỉ có kết quả synthetic fixture; chưa có so sánh CUB/Cars thật.**

Không tìm thấy `.pt` hoặc `crish.jsonl` trong data/weights/runs roots đang cấu
hình. Không có checkpoint, dataset/weights hay CUDA thật để restore, chạy
HSSK và chốt một Move được CRiSH chấp nhận. Không truy cập hoặc dừng remote
job. Move fixture dưới đây là phép can thiệp định trước để kiểm cơ chế,
**không được báo là CRiSH đã accept**.

| Arm | Move stable ID | Level | Steps | Batch đổi mask | Cặp mất positive | Cặp thêm positive | Train loop (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: plan gốc fixture | 91 | 0 | 2 | 0 | 0 | 0 | 0,021029 |
| B: hoán đổi để gặp peer | 91 | 0 | 2 | 1 | 1 | 1 | 0,002283 |

Cùng Move 91 từ cụm 0 sang 2; cùng supervision sau Move, model, head, SGD,
scheduler và RNG khởi đầu. Chỉ kế hoạch batch khác. Hai batch/arm chính là
một epoch của fixture, không phải một epoch của CUB/Cars. Plan A được thiết
kế thiếu exposure để kiểm wiring; bảng không đo tần suất thiếu exposure của
sampler thực tế. Thời gian nhỏ, A chịu warmup; không suy ra B nhanh hơn.
All/New không chạy, để null. Không kết luận accuracy từ loss hay số cặp.

## Cách triển khai

`scripts/move_batch_experiment.py` cung cấp:

- `active_plan(..., enabled=False)`: giữ nguyên plan nếu không bật. Arm B
  bật riêng trong runner A/B; không thay sampler/trainer mặc định.
- Dùng `pseudo_masks(...)[1]` thật trước/sau Move tại đúng level để xác định
  peer mất/thêm positive, bao gồm quy tắc bảo vệ same-image/labeled pair.
  Dò peer bằng các mask block hữu hạn, không dựng ma trận toàn dataset.
- Chỉ hoán đổi slot của sample có cùng labeled membership. Kiểm lại số
  batch, từng batch size, multiplicity của từng stable ID và quota labeled
  **từng batch**. Không thêm sample/optimizer step, không tạo duplicate mới
  trong batch. Nếu plan từ WeightedRandomSampler đã có lặp do replacement,
  giữ multiplicity ban đầu; không thể vừa xóa lặp đó vừa giữ multiplicity.
- Nếu A không được sampler rút hoặc không có peer/slot hợp lệ trong plan,
  giữ nguyên và báo `no_feasible_placement_within_budget`; không ép chèn A.
  Search có ngân sách 10.000 hoán đổi thử, chưa hứa tìm được mọi nghiệm khả thi.
- `run_ab(...)`: nhận duy nhất before/after/action đã khóa, kiểm after đúng
  một Move bằng `rebuild`, không cho từng arm tự repair. Lưu input snapshot
  và hai plan vào output mới. Restore model/head/optimizer/scheduler/RNG
  trước mỗi arm; mỗi arm chạy toàn plan đúng một lần.
- `seeded_views`: cùng một draw hai view theo seed/epoch/stable ID, độc lập
  batch order và không tiêu thụ RNG của SGD. Dùng dataset training đã sanitize;
  không dùng nhãn unlabeled. Với transforms dựa RNG riêng ngoài Python/NumPy/
  Torch cần adapter thêm; không hứa bitwise parity trên CUDA khác nền tảng.
- Trên các batch đã train, đếm cặp unordered theo stable ID, bỏ self và gộp
  set cả epoch để không đếm lại vì hai view hoặc sampler lặp. Mỗi arm so
  before/after trên **cùng IDs của từng batch arm đó**.

Không sửa `crish/`, `gcd_runtime/`, `selex_runtime/` cho thử nghiệm này;
fingerprint runtime a9985b5 được giữ. Checkpoint gốc không bị ghi đè.

## Chạy lại fixture

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_move_batch_experiment.py -q

# Chọn output chưa tồn tại; command chỉ chạy fixture CPU.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/move_batch_experiment.py --fixture --output logs/move-batch-new-fixture --max-seconds 30
```

Kiểm chứng: 8 test mới đạt; cùng regression SelEx là **19 passed in 2.90s**.
Một subprocess bị OpenMP SHM trong sandbox; rerun có quyền SHM đạt. Smoke
đầu tiên phát hiện geometry fixture và model khởi tạo không khớp, gây loss
rất lớn dù hữu hạn; đã khởi tạo backbone identity để khớp frozen geometry,
test lại và dùng kết quả sau sửa cho bảng trên.

## Job nền đã chạy

- Run ID: `move-batch-background-20261115T100825Z-04a2d2`.
- PID wrapper khi khởi chạy: **82558**. Trạng thái tại lần kiểm tra duy nhất
  sau launch: **completed**. Không còn job dài cần theo dõi.
- Lệnh con: `.venv/bin/python scripts/move_batch_experiment.py --fixture --output logs/move-batch-background-20261115T100825Z-04a2d2 --max-seconds 30`.
- Log: `logs/move-batch-background-20261115T100825Z-04a2d2.log`.
- Launch metadata: cùng prefix `.launch.json`; output chứa `manifest.json`,
  `result.json`, `locked_input.pt`, `A_after_epoch.pt`, `B_after_epoch.pt`.
- Wrapper Python `subprocess.run(..., timeout=60)` trong session riêng;
  timeout bên trong epoch 30s. ETA lúc launch khoảng 5s dựa trên smoke khoảng
  1s startup + 0,03s train A/B; không áp ETA này cho dữ liệu thật.
- Chạy macOS cục bộ, không phải Runpod; không có tmux. Log/checkpoints fixture
  gitignored; receipt chia sẻ ở `results/move_batch_fixture.json`.

## Điều kiện để chạy một epoch thật

CLI hiện chỉ hỗ trợ fixture, **chưa có CLI tự restore dataset/model thật từ
checkpoint**. `run_ab` là adapter nhận đầu vào thật đã chuẩn bị; phần nối với
checkpoint/HSSK/repair thật cần làm trên máy có dữ liệu và source tương thích.

1. Kiểm `config.json`, fingerprint, epoch và log accepted Move. Checkpoint
   ghi epoch=e là **sau** SGD epoch zero-based e; epoch kế là e+1. Chọn
   checkpoint sao cho `CrishSettings.due(e+1)` đúng, không nhầm display e+2.
2. Restore đúng model/head/optimizer/scheduler/RNG, kể cả pha evaluation RNG
   còn pending nếu có. Dùng source của checkpoint; không bỏ guard mismatch.
3. Chạy HSSK và repair đúng một lần. Muốn chỉ một Move, lưu snapshot ngay
   trước/sau commit đó trong chuỗi repair; không lấy toàn refresh nhiều edits
   làm “một Move”. Cần log/evidence của chính Move, không ép accepted=True.
4. Materialize sampler batch plan **một lần** theo batch_sampler/drop_last,
   ánh xạ dataset indices sang stable IDs. Khóa initial train state sau các
   bước chuẩn bị để A/B bắt đầu giống hệt nhau, không lấy mẫu lại từng arm.
5. Truyền snapshots/action/plan và sanitized dataset vào `run_ab`, provenance
   chứa checkpoint hash, config/source hash và bằng chứng accepted Move. Giữ
   `max_seconds` hữu hạn; chạy real smoke trước. Evaluation optional ở worker
   riêng chỉ báo exploratory All/Old/New, không dùng để chọn batch.
6. Nếu Runpod: launch qua tmux và external `timeout` cho cả pipeline, log
   riêng và PID/session; ETA từ smoke thật. Không tự launch remote ở phiên này.

Chưa thể báo “thử nghiệm dữ liệu thật xong” cho đến khi hoàn thành các bước
trên. Kết quả hiện tại chỉ xác nhận batch permutation có thể làm một Move
được thể hiện trong SSE mà không thay số lần xuất hiện/steps.
