# Hướng chốt: CRiSH Local với số action thích nghi theo từng refresh

Ngày chốt tài liệu: 2026-11-15.
Source đã kiểm tra: `4d54cca9603c57f4833babf7279578b4820f65bd`.

**Quyết định:** triển khai trên nhánh phương pháp `local` (Keep / Abstain /
Move). Số action được accept mỗi refresh do bằng chứng từ model và điều kiện
dừng quyết định. Cho phép 0 đến nhiều action; không ép nhận đủ quota.

**Trạng thái cập nhật:** đã triển khai runtime/profile local adaptive trên
branch `feat/local-adaptive-acceptance`; profile local lịch sử vẫn giữ trần 1.
Suite CPU đạt 285 test; CUDA/full benchmark chưa chạy. Các mục dưới giữ vai
trò đặc tả thiết kế; file thực tế và lệnh lấy/chạy branch nằm trong
[hướng dẫn bàn giao](18_local_adaptive_handoff.md). Chưa có bằng chứng
benchmark rằng hướng này cải thiện accuracy hoặc là đóng góp mới.

## 1. Mục tiêu và phạm vi

Mục tiêu kỹ thuật là bỏ giới hạn thực tế “nhiều nhất một action mỗi refresh”
của profile local đang dùng, đồng thời giữ khả năng từ chối mọi action.
Mục tiêu thực nghiệm là kiểm tra liệu tăng khả năng sửa, với lựa chọn có
xác nhận, có cải thiện **All accuracy** và trade-off Old/New hay không.

“Model quyết định số lượng” được định nghĩa rõ: điểm counterfactual từ model
quyết định accept/stop; số nhận là kết quả của chuỗi quyết định đó. Giai đoạn
này không huấn luyện thêm một mạng dự đoán số nguyên n hoặc một RL controller.

Thay đổi chỉ bật qua một profile local mới. Giữ profile local cap=1 làm đối
chứng và giữ baseline SelEx, upstream, population policy, online weighting,
loss coefficients, lịch refresh và lifetime của action như hiện tại.
Đây là thay đổi thuật toán/config của phương pháp local, không bắt buộc phải
tạo Git branch tên `local`.

## 2. Điều đã biết từ code

- `crish/policy.py::sequential_repair` đã có vòng lặp nhận 0 đến
  `max_accepted`, chấm lại sau mỗi commit, dừng khi confirmation thất bại.
- Profile đang cung cấp đặt `max_accepted=1`, `boundaries=2`,
  `train_batch=32`. Chỉ tăng cap mà giữ shortlist 2 mẫu sẽ không sửa được
  hơn 2 mẫu; tăng shortlist mà thiếu peer trong scoring batch cũng không đủ.
- `engine._draw_context` dành trước `1 + min(max_accepted, boundaries)` probe
  folds: một discovery và các confirmation folds rời nhau. Probe và proposal
  evidence/scoring phải tách sample ID.
- Mỗi sample được commit tối đa một action trong một refresh. Level và Move
  destination là một phần của danh tính action; không nhập nhằng cache điểm.
- Supervision được trích lại mỗi epoch. Sửa ở refresh chỉ sống trong epoch
  đó; ảnh hưởng qua trọng số SGD có thể tồn tại lâu hơn.

Vì vậy phần lõi chọn số lượng đã có. Phần triển khai cần làm là profile có
đủ ứng viên/hỗ trợ, kiểm chứng nhiều commit, xử lý ngân sách và log đầy đủ.
Không thêm một `adaptive` mode chỉ để bọc lại chính vòng lặp hiện có.

## 3. Quy tắc quyết định cuối cùng

Với supervision ban đầu S0 và supervision hiện tại S:

1. Sinh shortlist hữu hạn từ bằng chứng reciprocal/cross-view hiện có.
2. Xây dựng thử từng action hợp lệ trên S, rồi chấm discovery gain tương đối
   với Keep trên **chính S**. Không cộng điểm đã chấm độc lập trên S0.
3. Chọn action có discovery gain cao nhất trong các action vượt guard.
4. Xác nhận duy nhất action đó bằng một confirmation fold chưa sử dụng.
5. Nếu đạt: commit nguyên tử, loại mọi action của sample đó, rồi chấm lại
   các action còn lại trên supervision mới.
6. Nếu không đạt hoặc hết ngân sách: kết thúc refresh. Không thử runner-up
   trên confirmation fold vừa trượt và không ép nhận ít nhất một action.

```text
S = S0
accepted = 0
while accepted < resource_cap:
    if time_or_probe_budget_exhausted: stop
    candidates = rebuild_and_rescore_pending_actions(S)
    winner = best_discovery_candidate_passing_guards(candidates)
    if winner does not exist: stop
    result = confirm_on_next_unused_fold(S, winner)
    if result fails: stop
    S = atomic_commit(winner)
    accepted += 1
    remove_all_actions_of_committed_sample()
return S, accepted, stop_reason
```

Giữ ngưỡng hiện tại để lần thử đầu không đồng thời đổi tiêu chí chọn:
discovery/confirmation gain > 0, known increase ≤ 0, có target drift khác 0
và protected-coefficient shift không vượt 0,25. Những điều kiện này chỉ là
guard trên proxy; không bảo đảm Old accuracy hay SS accuracy sẽ tăng.

Trần tài nguyên là bắt buộc để refresh hữu hạn; nó không phải quota cần lấp
đầy. Nếu thường xuyên chạm trần thì số lượng đang bị ngân sách giới hạn;
phải báo đúng điều đó, không gọi là model đã tìm được số lượng tối ưu.

## 4. Profile thử nghiệm đầu tiên

Các giá trị sau là điểm bắt đầu để kiểm tra chi phí và cơ chế, không phải
hyperparameter đã tối ưu. Chỉ profile mới dùng các giá trị này.

| Trường hiện có trong khối `crish` | Giá trị chốt lần thử đầu |
|---|---:|
| `enabled` | true |
| `max_accepted` | 8 |
| `boundaries` | 16 |
| `train_batch` | 128 |
| `evidence_samples` / `evidence_trials` / `k` | 128 / 2 / 5 |
| `probe_known` / `probe_unlabeled` | 16 / 32 mỗi fold |
| `probe_sampling` / `probe_known_per_class` | class_balanced / 2 |
| `warmup_epochs` / `refresh_every` | 10 / 5 |
| `max_refresh_seconds` | 120 |
| `acceptance_gain` / `known_tolerance` | 0 / 0 |
| `max_protected_shift` | 0,25 |

Tắt `population_policy.enabled` và `pair_reliability.enabled` trong profile
này. Mọi thông số khác kế thừa profile SelEx local của dataset tương ứng.

Cap=8 cần 9 folds, tương ứng 144 labeled và 288 unlabeled ảnh chỉ cho probe;
còn cần đủ lớp và số ảnh/lớp cho quota, cùng evidence/scoring tách probe.
Nếu thiếu hỗ trợ thì Keep và ghi lý do, không nhân bản ảnh hoặc lấy nhãn ẩn
để lấp fold. Scoring batch phải log peer thật và drift từng action; không
giả định batch 128 tự bảo đảm mọi ứng viên đều chấm được.

Deadline được kiểm giữa các phép tính; kernel đang chạy không được ngắt tức
thời. Giữ thêm giới hạn runtime của worker như cơ chế chặn cuối hiện tại.
Không tự tăng timeout hoặc giảm guard để buộc đạt nhiều acceptance.

## 5. Các file cần triển khai

Đây là danh sách công việc sắp làm, không phải danh sách file đã sửa.

| File/khu vực | Thay đổi cần làm |
|---|---|
| `crish/engine.py` | Đủ folds cho cap, log từng vòng, phân biệt stop/fallback và trạng thái commit thực tế khi hết ngân sách. |
| `crish/policy.py` | Giữ rebuild/rescore, confirmation riêng và atomic commit; xuất lý do dừng rõ ràng với API tương thích. |
| `crish/scoring_batch.py` | Kiểm nhiều action có source/target peer; giữ loại trừ probe và không bịa evidence edge. Chỉ sửa nếu test cho thấy thiếu. |
| `crish/config.py` | Dùng các trường đang có; bổ sung validation nếu cần, giữ mặc định legacy. |
| `configs/` | Thêm profile Cars local adaptive và đối chứng cùng pool; không ghi đè config run đang chạy. |
| `selex_runtime/cli.py` | Xác minh `--config` + `--variant local` giữ đúng profile, không bị ghi đè về policy. |
| `tests/` | Test policy, engine nhiều commit, deadline, guards, baseline parity, determinism/resume. |
| README và báo cáo phương pháp | Ghi tên profile, lệnh thật sau khi file đã tồn tại, trạng thái và giới hạn. |

Nếu deadline hết sau một số commit, giữ những commit đã xác nhận khi state
vẫn hợp lệ; log chính xác số đã áp. Lỗi làm mất tính hợp lệ của cả refresh
thì quay về S0 và ghi số applied=0. Không log “accepted=3” trong khi trả về
S0. Phân biệt rõ accepted proposal, applied action và refresh rollback.

Giữ snapshot model/projector và trạng thái SGD dùng cho lookahead theo engine
hiện tại; chấm lại **targets** sau mỗi commit. Không âm thầm chuyển thành
chuỗi cập nhật model ảo nhiều bước vì đó là một thay đổi phương pháp khác.

## 6. Log bắt buộc

Mỗi refresh phải đủ thông tin để trả lời vì sao nhận từng đó action:

- Run ID, epoch, seed, config snapshot và source fingerprint.
- Số sample biên, số action đề xuất/chấm/không đủ hỗ trợ/bị loại.
- Cap, số fold dự kiến/thực dùng, thời gian và `stop_reason`.
- Mỗi vòng: supervision version, action đầy đủ, discovery gain,
  confirmation gain, known increase, fold index và target drift.
- Số action thực áp, số sample thực đổi, liều cặp/coefficient mass theo
  level và phần supervision còn giữ; không chỉ đếm action.
- Cờ `cap_reached`, budget exhausted, rollback/fallback và lý do.

Các lý do dừng tối thiểu: không có ứng viên, không còn gain hợp lệ,
confirmation thất bại, known guard thất bại, hết probe, hết thời gian,
đạt cap hoặc state không hợp lệ. Không suy ra nguyên nhân từ accepted=0.

## 7. Kiểm thử và điều kiện hoàn thành code

- Cùng cap phải có thể nhận 0, 1 và nhiều action tùy evidence.
- Sau commit phải rebuild/rescore; interaction có thể khiến action tốt
  lúc đầu thành xấu. Không dùng một lần top-k để commit đồng loạt.
- Confirmation folds không trùng nhau và không trùng evidence/scoring.
  Thất bại ở vòng sau không tự hủy commit hợp lệ trước đó.
- Deadline/exception phải trả state hợp lệ và log khớp state thực áp.
- Không sửa labeled assignment, không bỏ protected pair và không đọc nhãn
  unlabeled trong scoring, selection hoặc stopping.
- Move cập nhật đúng prototype/radius/USE; Abstain có drift thật trong batch;
  cache phân biệt sample, level, source, destination và supervision version.
- Tắt repair giữ baseline parity. Profile legacy giữ hành vi cũ.
- Config mới qua dry-run, synthetic engine smoke và kiểm tra resume khớp
  source/config; không mặc định resume checkpoint cũ bằng code mới.

Hai regression test đã thêm trong phiên trước chỉ kiểm semantics của policy;
15 test đã đạt chưa thay thế engine integration hoặc CUDA smoke của profile
mới. Chỉ xác nhận CUDA sau forward/backward thật trên NVIDIA.

## 8. Thứ tự chạy và đối chứng

1. Hoàn thành CPU tests, dry-run và synthetic smoke có timeout trên Mac.
2. Trên Linux/NVIDIA: xác minh GPU không có job cạnh tranh, data/weights,
   import và CUDA; chạy bounded real-image smoke trước full training.
   Smoke phải thực sự chấm counterfactual; không ép phải accept action.
3. Đọc log thật để kiểm chi phí và liều. Nếu hết giờ trước khi chấm được
   action, chạm cap liên tục, hoặc luôn nhận 0 thì chẩn đoán trước khi chạy dài.
4. Chạy các đối chứng sau với split/backbone/optimizer/schedule/seed khớp:

| Arm | Vai trò |
|---|---|
| B0: SelEx baseline | Đo lợi ích tổng so baseline. |
| L1: local hiện tại, cap=1, pool nhỏ | Giữ mốc local hiện tại. |
| L2: local cap=1, cùng pool/batch và probe reservation như adaptive | Tách ảnh hưởng mở rộng context khỏi nhận nhiều action. |
| LA: local adaptive, cap=8 | Cho phép model nhận 0 đến 8 action. |

L2 phải có cùng danh sách folds/reserved IDs và evidence/scoring IDs với LA
trong fixture kiểm soát; chỉ đổi cap hiện tại sẽ **không** đạt vì
`_draw_context` phụ thuộc cap. Nếu cần, thêm override nội bộ chỉ dành cho
đối chứng để tách reservation budget khỏi cap, giữ mặc định cũ. Trong full
training, các model sẽ phân kỳ; khớp quy tắc/ngân sách/seed, không hứa cùng
shortlist thực tế ở mọi epoch. Báo rõ L2 và LA có chi phí chấm khác nhau.

Chọn checkpoint cuối theo lịch đã định trước làm số chính, báo All/Old/New,
chi phí, liều, per-seed và mean ± std khi đủ seed. Chọn thiết kế bằng legal
labeled holdout, không theo nhãn evaluation. Mục tiêu ba seed khi có compute;
không tự khởi chạy sweep dài và không dùng một ngưỡng tăng điểm cố định để
tuyên bố ý nghĩa thống kê.

## 9. Các hướng để sau vòng kiểm chứng này

Persistence giữa epoch, abstention mọi level, EMA teacher, đổi reliability,
backbone lookahead và population policy là các ablation riêng. Không gộp
chúng vào adaptive local lần đầu vì sẽ không biết thay đổi nào có tác dụng.

Cap=8 và lifetime một epoch vẫn có thể tạo liều rất nhỏ; hướng này là phép
kiểm có kiểm soát, không phải lời hứa giải quyết toàn bộ CRiSH ≈ baseline.
Nếu tăng số nhận mà liều vẫn thấp, đánh giá persistence hoặc can thiệp theo
tập ở bước kế tiếp. Nếu liều đã tăng nhưng không cải thiện, kiểm chất lượng
probe/hướng sửa thay vì tiếp tục nâng cap. Novelty gate vẫn mở; không gọi
đây là đóng góp mới chỉ vì số lượng action được chọn thích nghi.

Tham chiếu lập luận và rủi ro: [báo cáo đánh giá](16_adaptive_acceptance_review.md).
