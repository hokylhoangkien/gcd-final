# Cấu hình CRiSH adaptive và bàn giao K1

Ngày đối chiếu: 18/11/2026, Asia/Ho_Chi_Minh. Base Git SHA:
`c12dc96754b63dc80fc3e8ab201c3645c33f45c8`;
working tree có thay đổi chưa commit. SHA này **không định danh đủ code mới**.
Manifest do `scripts/prepare_paper_plan.py` tạo ghi fingerprint từng module
runtime cùng SHA/config hash; cần commit phiên bản được nhóm chốt trước run
paper. Không có quyết định G1 hoặc thay cấu hình chính trong lần triển khai này.

## Đường thực thi có thật

`scripts/train_selex.sh` → `selex_runtime.cli` → `gcd_runtime.launch` →
`selex_runtime.trainer` → `crish.training.train_strict` → `crish.engine.CrishEngine`.
Dữ liệu training được sanitize: target unlabeled = −1; chỉ legal labeled targets
được dùng. Extractor xây hierarchy từ normalized backbone features bằng fitter
SelEx đã sửa và lưu `SupervisionState` theo stable ID. Mỗi epoch extract lại;
local edits tác động supervision của epoch đó, không phải persistent relabeling.

Runtime đang quan sát là **SelEx Euclidean + local CRiSH**, khác đường lịch sử
Hyp-SelEx có Poincaré mapping. Không mô tả các run này như hyperbolic CRiSH.
Upstream SelEx pin `d8107b5458f3337b2d5342f79345a976037a1265`, HypCD pin
`ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`; cả hai không sửa trong tác vụ.

## Adaptive full

Template Cars: `configs/selex_scars_local_adaptive_200.json`.

| Thành phần | Cấu hình |
|---|---|
| Training | DINOv2 ViT-B/14 reg4, 224, projector 65536, batch128, 200 epoch |
| Optimization Cars | SGD lr0.1, momentum0.9, wd0.0001, block≥9, cosine |
| Loss Cars | supervised0.35, smoothing0.75, temperature0.5, USE `selex_legacy` |
| Lịch sửa | warmup10, refresh_every5; zero-based 10,15,…,195 (display11,…,196) |
| Evidence | global normalized-backbone neighbor shortlist, local two-view reciprocal evidence |
| Evidence budget | cache128, trials2, k5, reference block1024 |
| Proposal | ≤16 boundary sample IDs, ≤1 Move target/boundary, min_support2, min_edges1, reliability≥0.5 |
| Scoring | batch128, projector-only counterfactual with live SGD momentum copied |
| Probe | 16 labeled (8 classes×2), 32 unlabeled/fold, class-balanced legal labels |
| Confirmation | 1 discovery + 8 reserved fresh independent folds |
| Acceptance | cap8, gain>0, known increase≤0, protected shift≤0.25 |
| Time bound | 120s/refresh, 604800s training worker limit; deadlines between operations |
| Other controllers | population policy and online pair weighting disabled |

Discovery ranks candidate gain; only top candidate receives an unused
confirmation fold. Failure stops refresh. Each accepted edit triggers rebuild
and re-scoring of remaining candidates. Structural support, protected labeled
assignments and finite state checks remain enforced. Deadline retains already
committed actions inside policy; a later refresh-level validation error rolls
back refresh and reports `applied_actions=0`. No accuracy labels enter action
selection.

Cars SSB split SHA256:
`28d471f0fb9e2c773e1255b8b447e542974f6617952203056a0635c638513d39`,
98 known/98 novel, 50% known training rows labeled by official loader. Imported
Cars counts: labeled2000/unlabeled6144; Old2001/New4143 in evaluation pool.
All uses these sample counts, not an unweighted Old/New mean. Dataset images
and weights were not re-inspected locally in this task.

## So draft và các run quan sát

Draft/manuscript settings đang mô tả cap1/scoring batch32; adaptive quan sát là
cap8/boundaries16/scoring batch128, larger legal probe and reserved fold budget8.
Không sửa manuscript tự động. `reports/18_local_adaptive_handoff.md` mô tả quá
trình chuyển đổi. Không suy luận draft chính xác chỉ từ tên file.

Quan trọng: template checked-in có `ss_eval_every_epochs=0`, còn cả ba run được
nhập có giá trị **1**. Với `evaluation_rng=upstream`, tần suất đánh giá có thể
ảnh hưởng RNG train; do đó không đổi cadence giữa các seed/ablation. Để tiếp
nối A2/A3, dùng **config.json thực của run đang chọn**, không mặc định dùng
config template và gọi đó là replication. CUB giữ nhiệt độ/block của CUB;
không copy các giá trị Cars sang CUB.

Primary: `evaluation_last.json`, purpose=final, epoch=epochs−1, SS-v2 global
Hungarian on unlabeled train. Old/New labels chỉ dùng sau predictions. Các
best-All từ unlabeled truth trong kế hoạch không dùng làm số chính. Cấu hình
hiện tại vẫn có thể lưu báo cáo best thứ cấp; registry không đọc chúng.
Baseline công bố SelEx sửa dùng 50 epoch; CRiSH200 khác ngân sách, phải ghi rõ.

## Ablation có thể tái lập

Trường top-level `crish_ablation` là tùy chọn; vắng trường tương đương `full`,
không thêm key mặc định vào config lịch sử. Chỉ hỗ trợ local SelEx controller.

| Giá trị | Thay đổi so full |
|---|---|
| full | Giữ discovery, confirmation, Abstain, Move |
| evidence_only | Boundary priority; supported Move nếu có, nếu không Abstain; không tính CF/confirmation |
| no_confirmation | Giữ discovery CF/known/drift guards; bỏ independent confirmation |
| no_abstain | Bỏ proposal Abstain; tình huống chỉ có Abstain trở thành Keep, không chuyển thành Move |
| no_move | Bỏ proposal Move; giữ Abstain và các guards; budget là trần, không ép đủ action |
| random_proposal | Random thứ tự toàn bộ eligible boundary pool trước shortlist cap; không random action type |

Evidence-only vẫn recheck support, rebuild/validate và guard protected shift /
nonzero pseudo-SSE drift trên cùng scoring batch; không sử dụng gain hoặc gọi
projector CF. Khi Move được chọn nhưng guard thất bại thì Keep; không thử
Abstain để lách guard. Random dùng `random.Random(refresh_seed+7919)` riêng;
không tiêu thụ Python/NumPy/Torch training RNG. Tất cả giữ nguyên context và
reserved folds, kể cả khi không sử dụng confirmation. Random thay selection
order trên cùng tập boundary trước cap; tập sau cap có thể khác theo mục đích.

## Lệnh chuẩn bị và chạy

Chỉ tạo configs/manifest/commands, không train:

```bash
.venv/bin/python scripts/prepare_paper_plan.py \
  --base configs/selex_scars_local_adaptive_200.json \
  --seeds 1 2 3 --output runs/plans/paper-cars-NEW
```

Để match config quan sát, đổi `--base` thành `runs/RUN_ID/config.json` trên
máy có run đã chọn. Tạo tương tự CUB/Aircraft. Mỗi seed có full +5 variants;
không có job baseline. Có thể chỉ chạy full cho A2/A3, dùng 1–2 seed ablation
theo GPU. K2 là thử nghiệm riêng Cars1 seed, không tự đổi cấu hình tại G1.

```bash
bash scripts/train_selex.sh --config runs/plans/paper-cars-NEW/no_confirmation_seed1.json --variant local --dry-run
# Chỉ trên Linux/NVIDIA, với inputs đã sẵn sàng:
bash scripts/train_selex.sh check --config runs/plans/paper-cars-NEW/no_confirmation_seed1.json --variant local
bash scripts/train_selex.sh smoke --config runs/plans/paper-cars-NEW/no_confirmation_seed1.json --variant local
bash scripts/train_selex.sh --config runs/plans/paper-cars-NEW/no_confirmation_seed1.json --variant local
```

Launcher full thực hiện bounded smoke khi không resume. Không chạy tất cả
commands như sweep mặc định. Không resume checkpoint cũ bằng source/variant
mới: runtime fingerprint sẽ từ chối. Full default giữ thuật toán nhưng việc
thêm module cũng thay fingerprint; giữ checkout cũ cho job đang chạy/resume.
Mỗi run mới lưu config, version/fingerprint, seed, environment, command, timestamps,
logs/checkpoints theo launcher hiện có. Các config mới chỉ là prepared/not run.

## Plans chuẩn bị từ cấu hình quan sát trong archive

Đã tạo, chưa chạy: `runs/plans/paper-cars-observed-20261118/` (18 configs),
`runs/plans/paper-cub-observed-20261118/` (full seed2–3),
`runs/plans/paper-aircraft-observed-20261118/` (full seed2–3).
Các manifest chứa code fingerprint hiện tại; không coi chúng là run registry.
Chúng giữ ss_eval_every_epochs=1 của nguồn quan sát. Tái tạo ở output mới:

```bash
.venv/bin/python scripts/prepare_paper_plan.py \
  --base runs/downloads/cub-results-20261116T002137Z-056eafad.zip \
  --seeds 2 3 --variants full --output runs/plans/cub-seed23-NEW
.venv/bin/python scripts/prepare_paper_plan.py \
  --base runs/downloads/aircraft-results-20261116T002008Z-ce74b2d9.zip \
  --seeds 2 3 --variants full --output runs/plans/aircraft-seed23-NEW
.venv/bin/python scripts/prepare_paper_plan.py \
  --base btl-dl/cars-paper-light-20261115T112441Z-0ec6f014.tar.gz \
  --seeds 1 2 3 --output runs/plans/cars-ablation-NEW
```

Sau kiểm tra GPU prerequisites, lệnh kế tiếp có thể dùng là:

```bash
bash scripts/train_selex.sh check --config runs/plans/paper-cub-observed-20261118/full_seed2.json --variant local
```

Lệnh `check` này chỉ chạy trên máy Linux/NVIDIA có dữ liệu/weights, chưa được
thực hiện trên máy Mac. Source mới khác fingerprint các run cũ; nhóm phải
quyết định tính tương thích thực nghiệm tại G1, không gộp seed khác code tùy tiện.
