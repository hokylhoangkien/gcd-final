# CRiSH Local adaptive — triển khai và bàn giao

Ngày: 2026-11-15. Git branch: `feat/local-adaptive-acceptance`.
Phạm vi: SelEx + CRiSH **local Keep / Abstain / Move**. Baseline/upstream và
config lịch sử được giữ nguyên. Chưa có benchmark, chưa xác minh CUDA cho
profile mới; không suy diễn độ chính xác từ test CPU.

## Trạng thái workspace khi bắt đầu

Base commit `4d54cca9603c57f4833babf7279578b4820f65bd`, trước đó ở `main`.
Workspace có manuscript/PDF/build receipt, báo cáo và script kiểm manuscript
đang sửa từ công việc trước. Các thay đổi đó được giữ nguyên và không đưa
vào commit tính năng. Audit đầy đủ tiếp tục ở `00_workspace_audit.md` trong
workspace; tài liệu này ghi phần audit dành riêng cho branch bàn giao.

Máy phát triển Apple M5/16GB, macOS arm64, không có nvidia-smi. Đọc tiến trình
thấy Codex và IDE helpers, không thấy Python training cục bộ. Không can thiệp
process hoặc kết nối điều khiển job từ xa. `runs/` có plans; log chia sẻ cũ
không phải bộ log SelEx gần đây. Chưa có dataset/weights/checkpoint benchmark
được xác minh cục bộ. Chỉ chạy kiểm thử CPU hữu hạn, không tải weights lớn.

## Hành vi đã triển khai

- Profile mới: `max_accepted=8`, `boundaries=16`, scoring batch 128,
  evidence cache 128, 2 trials, k=5, refresh tối đa 120 giây với kiểm deadline
  giữa phép tính. Không đổi warmup=10/refresh_every=5 của profile Cars.
- Discovery xếp hạng các action; chỉ winner được xác nhận bằng fold riêng.
  Sau mỗi commit, rebuild/rescore trên supervision mới. Confirmation trượt
  thì dừng, không thử runner-up trên fold đó. Có thể nhận 0 action.
- `confirmation_budget=8` dành trước 8 confirmation folds + 1 discovery fold.
  Mỗi fold có 16 labeled (8 lớp × 2 ảnh) và 32 unlabeled ảnh trong profile
  thật. Cần ít nhất 18 labeled ảnh/lớp cho 8 lớp đủ điều kiện; nếu thiếu thì
  Keep với lỗi hỗ trợ, không lặp ảnh để lấp quota.
- Đối chứng cap=1 giữ nguyên `confirmation_budget=8`, nên không thay số
  probe dành trước khi chỉ đổi cap. Giá trị mặc định 0 giữ cách tính số fold
  lịch sử cho các profile khác.
- Deadline trong vòng sửa giữ các commit đã xác nhận trước đó. Lỗi cuối
  khiến toàn refresh rollback thì log `applied_actions=accepted=0`, còn
  `accepted_proposals` ghi số proposal từng qua xác nhận. Decisions bị
  rollback không còn mang status `accepted`.
- Giữ supervised labels, protected pair, Move support và drift guards;
  lookahead vẫn projector-only. Action vẫn hết hiệu lực trực tiếp sau epoch.

Đây là số lượng được chọn bằng tín hiệu model dưới một trần tài nguyên,
không phải controller đã học quota tối ưu. Cap=8 là giá trị thử đầu tiên.

## Lấy code trên máy GPU

Repo mới, dùng thư mục khác với checkout đang có job chạy:

```bash
git clone --branch feat/local-adaptive-acceptance --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git gcd-final-local-adaptive
cd gcd-final-local-adaptive
```

Nếu đã có một checkout **không có job đang chạy**, kiểm tra thay đổi trước:

```bash
git status --short
git fetch origin feat/local-adaptive-acceptance
git switch --track origin/feat/local-adaptive-acceptance
```

Lệnh switch dành cho máy chưa có local branch đó. Không reset/stash/ghi đè
thay đổi nếu Git từ chối switch. Môi trường/dataset/weights dùng hướng dẫn
README; có thể đặt `SELEX_PYTHON` trỏ interpreter đã chuẩn bị, cùng
`GCD_DATA_ROOT`, `GCD_WEIGHTS_ROOT`, `GCD_RUNS_ROOT`. Không cần tải lại dữ liệu
đã có. Chỉ bootstrap CUDA theo README khi thực sự thiếu môi trường.

## Cars 200 epoch: xem config, preflight và chạy

```bash
# Chỉ xem kế hoạch, không train, dùng được trên Mac.
bash scripts/train_selex.sh --config configs/selex_scars_local_adaptive_200.json --variant local --dry-run

# Máy Linux/NVIDIA đã có dataset/weights và GPU rảnh.
bash scripts/train_selex.sh check --config configs/selex_scars_local_adaptive_200.json --variant local

# Launcher chạy bounded smoke trước, rồi full run nếu các gate đạt.
bash scripts/train_selex.sh --config configs/selex_scars_local_adaptive_200.json --variant local
```

Smoke riêng, nếu chỉ muốn kiểm mà chưa bắt đầu full training:

```bash
bash scripts/train_selex.sh smoke --config configs/selex_scars_local_adaptive_200.json --variant local
```

Full profile giữ 200 epoch, batch 128, DINOv2 reg4, source loss/evaluation và
giới hạn runtime của config Cars 200 epoch hiện tại. Không coi giới hạn runtime
là thời gian train dự kiến; đo thực tế trên máy GPU. Gate smoke yêu cầu chấm
được counterfactual thật, không yêu cầu phải accept action. Không dùng code
mới để resume run cũ: fingerprint/config khác thì runtime sẽ từ chối.

Đối chứng cap=1 cùng context:

```bash
bash scripts/train_selex.sh --config configs/selex_scars_local_cap1_context_200.json --variant local --dry-run
```

## Tạo đủ bốn profile cho dataset/lịch khác

Script chỉ tạo config, manifest và commands.txt, không chạy training. Output
phải là thư mục mới để tránh ghi đè kế hoạch.

```bash
# Cars 50 epoch từ template đã có.
.venv/bin/python scripts/prepare_local_adaptive.py --base configs/selex_crish.json --output runs/plans/local-adaptive-cars50

# Hoặc CUB 200 epoch; thay template bằng Aircraft/Herbarium19 nếu cần.
.venv/bin/python scripts/prepare_local_adaptive.py --base configs/selex_cub_crish_200.json --output runs/plans/local-adaptive-cub200
```

Mỗi output có `baseline.json`, `legacy.json`, `cap1_context.json`,
`adaptive.json`, `manifest.json`, `commands.txt`. Legacy giữ context local
của template; cap1_context chỉ khác adaptive ở `max_accepted`. Baseline chỉ
khác adaptive ở `method`, do đó dùng cùng optimization/evaluation settings.
Khi các run phân kỳ, không hứa shortlist thực tế giống nhau ở mọi epoch.

Ví dụ chỉ kiểm profile CUB vừa tạo:

```bash
bash scripts/train_selex.sh --config runs/plans/local-adaptive-cub200/adaptive.json --variant local --dry-run
```

## Đọc log và tiêu chí tiếp tục

Mỗi run có config/source/environment/seed/command metadata theo launcher hiện
có. `crish.jsonl` bổ sung `acceptance_cap`, `stop_reason`, `cap_reached`,
`budget_exhausted`, `accepted_proposals`, `applied_actions`, confirmation folds
reserved/used, scoring counts, selection diagnostics và dose. `accepted` tiếp
tục nghĩa số action thực áp. Lý do hết giờ sau commit có thể đi cùng status
`completed`: những commit đó đã được trả về cho SGD.

```bash
.venv/bin/python scripts/analyze_crish_interventions.py runs/RUN_ID --output runs/RUN_ID/adaptive_action_audit.csv
```

Thay RUN_ID bằng thư mục thật. Dùng `intervention_diagnostics.jsonl` để kiểm
liều thật toàn epoch; số action không tương đương số cặp bị đổi. Các dòng
REFRESH và JSON export của analyzer có lý do dừng/ngân sách mới. SS trước/sau
action chỉ là liên hệ thời gian, không phải đo tác động nhân quả từng action.

Nếu luôn nhận 0, kiểm support/drift/discovery/confirmation trước; không tự nới
guard. Nếu thường hết giờ, báo số action thực áp và tỷ lệ hết ngân sách.
Nếu thường đạt cap, chưa thể nói số lượng hoàn toàn do model quyết định.
Đo All/Old/New trên checkpoint cuối và seed khớp; không chọn thiết kế bằng
nhãn evaluation. Persistence, mọi level và thay proxy là các ablation riêng.

## Kiểm chứng

Lệnh thực tế trên Mac, mỗi subprocess có timeout tối đa 60 giây:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py --code-only --report logs/local-adaptive-validation.json
```

Kết quả: **285 passed, 90 warnings in 42.17s**. Các bước kiểm source/split/
protocol và static syntax đều đạt. Có kiểm bản đóng gói chạy ngoài Git;
25 test adaptive gồm 8 commit với 9 fold thật rời nhau, deadline, rollback,
backward, context control và resume sau nhiều commit. Một regression analyzer
mới kiểm stop reason/applied counts. Acceptance có kiểm soát chỉ kiểm wiring;
cũng có fixture chạy exact lookahead thật không ép acceptance.

Lần chạy trong sandbox trước đó có một subprocess lỗi OpenMP SHM; full suite
trên được chạy với quyền SHM và đạt. Test mới ban đầu có một assertion nhầm
vị trí winner trong trace, đã sửa để kiểm status của đúng quyết định.

Run ID: `crish-v21-20261115T094308Z-b534e520`. Log/config/metadata/XML cục bộ
nằm dưới `logs/` cùng run ID; không commit log hay checkpoint. Receipt chia
sẻ: `results/local_adaptive_validation.json`. Kiểm chứng chạy trên base SHA
với working tree thay đổi; receipt ghi fingerprint của code đã kiểm.
Manuscript gate không chạy, manuscript/PDF không nằm trong thay đổi tính năng.
CPU tests không thay thế NVIDIA preflight/smoke thật hoặc full benchmark.
