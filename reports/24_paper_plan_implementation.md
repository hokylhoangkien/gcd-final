# Đối chiếu kế hoạch CRiSH — 18/11/2026

Đọc đầy đủ `CRiSH-GCD_Ke_hoach_Codex.md`. Không training, push, PR, sửa paper,
upstream, dataset hay checkpoint. M5/16GB chỉ dùng bounded CPU verification;
GPU thực tế của nhóm/remote readiness chưa xác minh. Các thay đổi cũ giữ nguyên.
Cấu hình và lệnh bàn giao: [23_adaptive_configuration.md](23_adaptive_configuration.md).

| Mã | Đã có / thay đổi kỹ thuật | Thiếu / sai khác / cần xác minh |
|---|---|---|
| K1 | Đã ghi pipeline, budget, schedule, full/draft differences, base SHA và fingerprint manifest | Nhóm chốt version G1; source mới chưa commit |
| K2 | Có launcher 1 seed Cars, final-only registry | NOT RUN; chưa tối ưu hay quyết định đổi phương pháp |
| K3 | Full + 5 ablations qua config; independent RNG; structural guards; CPU integration tests | GPU smoke/benchmark NOT RUN; không bảo đảm sẽ accept edit |
| K4 | Append-only per-refresh proposals/outcomes/counts/reasons; registry exports old/new logs và touched ratio | Log coverage có gaps do resume; Keep là shortlisted ID không được sửa, không phải mọi ảnh |
| K5 | Offline-only join partitions/labels, denominators, Move purity/Abstain error/sign agreement CLI | BLOCKED số thực: text imports thiếu raw partitions và evaluation labels |
| K6 | Tổng measured refresh seconds, epoch mean có/không refresh, số epoch đo | Đây là observational timing; không coi hiệu hai mean là causal overhead; không đo lại GPU |
| A1 | Registry CSV/JSON từ Cars/CUB/Aircraft archives, config/provenance/final metrics | Khác code SHA; không gộp dataset/config/code khác nhau |
| A2 | Có thể tạo full seed2–3 từ config thực CUB/Aircraft | NOT RUN, cần Linux/NVIDIA/data/weights; không khởi tạo job tự động |
| A3 | Final-only grouping, exact per-seed, sample std (ddof1), ≥3-seed paper-ready gate | Hiện mỗi dataset chỉ seed1; chưa có main table mean±std |
| A4 | Prepared Cars18 configs =6 variants×3 seed; commands giữ same settings ngoài variant/seed | NOT RUN; chỉ cần chọn 1–2 seed theo GPU và sau G1 |
| A5 | Registry hỗ trợ method=baseline và config nguồn, không chạy thêm baseline | Chưa tìm đủ artifacts baseline của nhóm; published baseline không phải observed run |
| A6 | Plot CLI đọc archive: All/Old/New, refresh markers, fixed-final dot, PDF/SVG/PNG, caption/hash | Curves hiện là seed1 cũ; config chốt G1 và các seed sau chưa có |

## Semantics và giới hạn

- `proposed`: các action sau shortlist/type filter; `accepted`: thực áp, không
  tính rolled-back proposal. `not_applied` gồm rejection, alternative cùng ảnh,
  chưa score vì cap/deadline. Trace rejection có thể lặp qua nhiều vòng; không
  gọi số dòng trace là số candidate độc lập.
- Keep: unique shortlisted sample IDs không được áp action; không phải action
  commit. Một ảnh có nhiều alternative thì chỉ được commit tối đa một/refresh.
- Touched fraction/refresh = unique accepted IDs / n_unlabeled. Registry có thêm
  union IDs toàn run; union khác tổng số image-epoch exposures.
- Random proposal xáo boundary priority trong cùng eligible pool, không thay
  source dataloader và không hạ quota. Số edit thực có thể khác do guards/gains.
- No-confirmation vẫn giữ cùng reserved context và discovery CF. Evidence-only
  không có CF gain; trường không đo phải null/vắng, không ghi gain giả.
- Dữ liệu thiếu log thì ghi MISSING/null, không ghi 0 action. Registry từ chối
  duplicate run IDs/refresh epochs để tránh nhân đôi import/resume.
- Nhãn thật chỉ được đọc bởi scripts hậu kiểm; không import vào runtime. Cluster
  number không tương đương semantic label. K5 dùng unique majority của target
  cluster (leave-one-out), tie loại khỏi mẫu số. Đây là diagnostic purity, không
  phải global Hungarian accuracy và không chứng minh tác động nhân quả.
- Published SelEx50 và CRiSH200 vẫn khác training budget; không claim fair
  algorithm-only gain. Tài liệu kế hoạch có best-All cũ; registry chỉ final.

## Công cụ phân tích

Đầu ra hiện có: `results/paper-plan-verified-20261118/registry.json`,
`runs.csv`, `refreshes.csv`, `actions.csv`, các thư mục `curves-*`.
Nguồn được đọc nguyên trạng, không extract/ghi đè archive:

```bash
.venv/bin/python scripts/paper_registry.py \
  btl-dl/cars-paper-light-20261115T112441Z-0ec6f014.tar.gz \
  runs/downloads/cub-results-20261116T002137Z-056eafad.zip \
  runs/downloads/aircraft-results-20261116T002008Z-ce74b2d9.zip \
  --output results/paper-registry-NEW

.venv/bin/python scripts/plot_paper_curves.py \
  runs/downloads/cub-results-20261116T002137Z-056eafad.zip \
  --output results/paper-curves-cub-NEW
```

Registry nhận run directory hoặc ZIP/tar.gz có đúng một config run. Dùng output
mới; không ghi đè kết quả cũ. Không dùng bản tải rời thiếu `crish.jsonl` khi có
archive đầy đủ. Các archive/import gốc vẫn giữ nguyên.

K5, chỉ khi có đầy đủ partitions trước refresh và CSV nhãn offline thực:

```bash
.venv/bin/python scripts/prepare_posthoc_snapshots.py runs/RUN_ID \
  --labels /path/to/offline_sample_labels.csv --output /tmp/posthoc-RUN_ID.jsonl
.venv/bin/python scripts/paper_posthoc.py runs/RUN_ID \
  --snapshots /tmp/posthoc-RUN_ID.jsonl --output results/posthoc-RUN_ID.json
```

CSV: `sample_id,true_label`, cùng ID hệ thống run; bao phủ mọi snapshot ID.
Snapshots: `run_id,epoch,level,sample_ids,true_labels,clusters,labeled`;
trình join đọc NPZ với allow_pickle=False. Thiếu snapshot báo PARTIAL, không
điền nhãn suy đoán. Nhãn chỉ tồn tại trong artifact phân tích tách biệt này.
Không đưa artifact hậu kiểm vào đường train. Sign agreement chỉ đo trên các
candidate thực đã confirmation, có selection bias; không suy rộng toàn pool.

## Kiểm chứng nguồn và novelty

[SelEx issue11](https://github.com/SarahRastegar/SelEx/issues/11) xác minh
BanAnA9205 mở 15/10, tác giả xác nhận 20/10; trang ghi merge/close **08/11**,
commit `d8107b5`. Không dùng mốc 09/11/commit0cc628d trong paper khi chưa xác
minh được trang commit; fetch commit đó hiện thất bại. Chưa xác minh account
thuộc nhóm nên không viết “We identified”. README pin local xác nhận corrected
last/3 seeds và Cars/CUB/Aircraft50 epochs; không sửa published numbers.

Đã đọc lại [HypCD full method](https://arxiv.org/html/2504.06120v1), đặc biệt
phân biệt nonparametric Hyp-SelEx và hyperbolic mapping. CVF trả403. Code local
được đối chiếu với audit/protocol hiện có. [RCA](https://proceedings.mlr.press/v306/yu26l.html)
là overlap lead về certainty/cross-view alignment; full-method novelty gates
RCA/LEAH/HypGCD trong `reports/02_novelty_and_method.md` và
`reports/04_method_and_research_plan.md` **vẫn mở**. Tác vụ này bổ sung công cụ
kiểm chứng phương pháp đang có, không xác nhận novelty hay đổi mục tiêu nghiên cứu.

## Kết quả trích xuất thực và phần còn thiếu

Bảng chi tiết sinh từ archive, không gõ số ước lượng:
`results/paper-plan-verified-20261118/README.md`. Fixed-final All/Old/New (%):
Cars 78.2064/87.3563/73.7871; CUB 83.6744/84.8566/83.0831;
Aircraft 78.2244/81.2125/76.7316. Mỗi dataset seed1 duy nhất.

Recorded edits: Cars117 (Move65/Abstain52), CUB225 (Move1/Abstain224),
Aircraft121 (Move62/Abstain59). Cars có đủ38 scheduled refresh records;
CUB thiếu zero-based10/15/20/25/30, Aircraft thiếu10/15/20/25.
Không suy luận missing records là không có action. Registry đánh dấu partial
và loại những epoch thiếu scheduled log khỏi nhóm timing “without refresh”.
Sign agreement ghi cả candidate đã confirmation nhưng bị reject, không chỉ
accepted actions; Cars117/132, CUB225/232, Aircraft121/145. Không có bằng
chứng semantic correctness chỉ từ agreement này.

Curves PDF/SVG/PNG đã xuất cho cả3 dataset, có dữ liệu nguồn/caption/output
hash. Đã xem trực tiếp CUB PNG; gap và endpoint biểu diễn đúng. Input archive
SHA256 được lưu trong source_manifest.json và kiểm lại không thay đổi. Các
đầu ra không thay thế checkpoint/model-file access cho việc evaluate lại.

## File thay đổi trong tác vụ

- `crish/ablation.py`: selectors, RNG riêng, evidence-only sequential commit.
- `crish/engine.py`, `crish/policy.py`, `crish/training.py`: nối ablation vào
  đúng controller, bỏ confirmation có chủ đích, policy-aware smoke, action log.
- `selex_runtime/config.py`, `selex_runtime/cli.py`: validate và hiển thị config;
  không thêm default key vào snapshot lịch sử.
- `scripts/prepare_paper_plan.py`: six variants/seed từ JSON hoặc archive.
- `scripts/paper_registry.py`: final-only registry, coverage/actions/timing/sign.
- `scripts/paper_posthoc.py`, `scripts/prepare_posthoc_snapshots.py`: offline-only
  semantic diagnostic; không được train import.
- `scripts/plot_paper_curves.py`: observed curves từ directory/archive.
- `tests/test_paper_ablation.py`, `tests/test_paper_registry.py`: regression,
  isolation, real backward/CF/rebuild wiring, archival import, offline guards.
- `tests/test_layout.py`: cho phép đúng file kế hoạch do người dùng thêm;
  tài liệu mới vẫn nằm trong reports như layout canonical.
- `tests/test_release.py`: PDF whitelist cho manuscript hoặc paper/figures;
  tiếp tục chặn weights/datasets/runtime artifacts khỏi archive.
- README, reports00/23/24 và results/paper-plan-verified-20261118: hướng dẫn,
  audit và evidence. Các manuscript/figure/source-checker sửa từ trước giữ nguyên.

## Verification và giới hạn thực thi

Chạy Python subprocess timeout60s, CPU một thread, không weights/data download.
Hai đợt regression:180 passed/23.08s và154 passed/49.19s (334 test; loại riêng
release test). Sau thay đổi cuối, targeted suite55 passed/4.84s, gồm hai test
bổ sung archive cadence và evidence candidate-pool isolation. Các con số này
có overlap, không cộng như các thí nghiệm độc lập. Log/JUnit:
`logs/paper-plan-batch{1,2}-20261118.{log,xml}` và
`logs/paper-plan-final-20261118.xml`.

Sandbox ban đầu gây2 lỗi OpenMP SHM; chạy lại đúng CPU suite với shared-memory
access đạt. Một lần toàn suite chạm timeout60s và lộ layout restriction cũ;
đã sửa canonical placement/allowed plan và chia suite như trên. Không tăng
runtime vô hạn. Source/split checks (`scripts/check_protocol.py`,
`scripts/check_integrity.py`) pass; image integrity NOT CHECKED vì thiếu manifest.
AST parse pass; input archives và curve output hashes kiểm đúng.

Những việc vẫn cần người/GPU: chốt G1, kiểm nguồn/version, check+bounded real
smoke trên NVIDIA, chạy matched seeds/ablations, cung cấp pre-refresh partitions
và offline labels cho K5, bổ sung A5 baseline artifacts, kiểm chéo độc lập G2.
Không có quyết định G1/G2 hay claim cải thiện/novelty trong đầu ra này.

Release regression cũng đã pass:1 test/11.49s, kiểm archive thật chạy ngoài Git
và bounded synthetic smoke từ source package. Log/XML:
`logs/paper-plan-release-20261118.{log,xml}`. Whitelist chỉ mở PDF dưới đúng
paper/figures, không mở rộng quyền đưa weights/data vào package.
Validation receipt: `results/paper-plan-verified-20261118/validation.json`, ghi
337 test cases phân biệt được phủ qua các đợt, source fingerprints, environment,
timeouts và source hashes. Đây là tổng coverage, không phải337 test chạy trong
một subprocess cuối. `git diff --check` pass. Không build lại manuscript/PDF.

Launcher thật đã dry-run thành công cả6 variants bằng config Cars seed1 sinh
từ archive quan sát; mỗi subprocess timeout15s. Không training/GPU. Commands
và status nằm trong validation.json; stdout/stderr ở
`logs/paper-plan-dry-VARIANT-20261118.log`.
