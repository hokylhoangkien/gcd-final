# Audit đánh giá HypCD / Hyp-SelEx / SelEx trước retrain

Ngày kiểm tra: 08/11/2026. Scope: tính đúng metric, selection, comparability
và chuẩn bị chạy lại; không có kết quả benchmark mới hoặc tuyên bố novelty.

**Kết luận:** chưa tìm thấy lỗi số học trong All/Old/New-v2 hay sai pool của
final evaluator hiện tại. Rủi ro lớn là so sánh với số công bố của upstream
có lỗi ordering, nhầm selector/metric, nhầm plain-test với anchored SS, và
báo cáo peak sau khi xem nhãn. Có một lỗi provenance metadata đã được tái hiện
và sửa; nó không làm đổi accuracy hay gradient.

## 1. Nguồn được kiểm tra, không mặc định paper = code

- [HypCD paper, methods và supplement](https://arxiv.org/html/2504.06120v1).
- [SelEx paper chính thức](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08926.pdf).
- [HypCD source pin](https://github.com/Visual-AI/HypCD/tree/ffc895b7ff38be8502bb7e3bd851f1d2a55dff73).
  GitHub main tại thời điểm audit vẫn chính là SHA này; không tự cập nhật upstream.
- [SelEx source pin](https://github.com/SarahRastegar/SelEx/tree/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73).
  Đã đọc README, training, metric helper, final clustering và k-means implementation.
- [SelEx issue #8: tác giả xác nhận chọn best checkpoint](https://github.com/SarahRastegar/SelEx/issues/8#issuecomment-2629402874).
- [SelEx issue #11: tác giả xác nhận lỗi ordering](https://github.com/SarahRastegar/SelEx/issues/11#issuecomment-6288646207)
  và cảnh báo ở README tại commit trên. Các kết quả shuffle trong issue là
  quan sát của người báo lỗi, không phải thí nghiệm của project này.

Source snapshots và ngữ cảnh issue được giữ trong thư mục `logs/evaluation-audit-*`.
GitHub chỉ được đọc; không gửi issue/comment hay thay đổi repo ngoài.

## 2. Những gì hiện tại đang đúng

| Hạng mục | Pipeline hiện tại | Đánh giá |
|---|---|---|
| Split | Cars/CUB official SSB, kiểm SHA256; labeled fraction 0.5 trên known training rows | Khớp split đã chọn; counts ảnh thật vẫn cần remote preflight |
| Train labels | Unlabeled targets được materialize thành -1; probe chỉ có legal registry | Chưa thấy hidden GT đi vào SGD/repair trong dataflow/tests đã kiểm |
| Final features | Normalized backbone embeddings, không phải projector logits | Đúng hướng non-parametric Hyp-SelEx |
| Final clustering | Balanced semi-supervised k-means trên original train pool, legal labeled anchors | Đúng loại evaluator của Hyp-SelEx; không thay bằng plain k-means |
| Scored pool | Chỉ unlabeled rows; labeled anchors không được cộng vào denominator accuracy | Đúng GCD train-unlabeled evaluation |
| Metric | Global Hungarian-v2 trước, rồi hạn chế Old/New theo cùng mapping | Đúng metric đang đối chiếu; không optimize mapping riêng cho từng nhóm |
| All | Sample-weighted: `(n_old*Old+n_new*New)/(n_old+n_new)` | Đã kiểm bằng fixture nhóm có kích thước khác nhau |
| Hyperbolic learning / Euclidean final clustering | Training có geometry mới, inference clustering trên backbone Euclidean | Không phải một lỗi; HypCD paper mô tả lựa chọn này |

Hungarian dùng ground truth **sau khi đã có predictions** để chấm clustering
là một phần metric. Việc này khác với dùng nhãn ẩn để tạo pseudo-label,
tuning phương pháp hoặc chọn checkpoint. Không gộp tất cả thành một loại leak.

## 3. Lỗi ordering upstream: đã sửa nhưng phải công bố trong so sánh

`fit_mix` trả labels theo `[anchor rows; other rows]`. Ở tầng thô, anchor
membership là **predicted-known**, có thể xen kẽ trong thứ tự loader. SelEx và
HypCD source đã kiểm lưu thẳng vector đó rồi đọc nó theo thứ tự sample gốc.
Đây là lỗi tác giả SelEx đã xác nhận; ảnh hưởng lên từng kết quả HypCD chưa
được xác minh thực nghiệm ở đây.

Runtime hiện gọi `cluster_labels(..., mask)` ở base, mọi coarse level và
evaluator; helper hoàn nguyên về input order và giữ prototype IDs nguyên vẹn.
Regression mới chạy actual extraction adapter với predicted-known/novel
xen kẽ, stable sparse IDs và hai thứ tự loader, xác nhận coarse targets vẫn
gắn đúng ảnh. Đây là test **ID mapping**, không phải lời hứa toàn bộ k-means
bất biến với mọi permutation, initialization hoặc tie-breaking.

Hệ quả cho submission: B0 và CRiSH phải dùng **cùng bản sửa**. Bảng paper có
thể có hàng paper-reported riêng, nhưng không suy ra algorithmic gain từ
chênh lệch với số cũ khi code/protocol đã khác. Số thấp hơn paper cũng không
tự chứng minh evaluator hiện tại sai. Không bật lại bug để làm đẹp baseline.

## 4. Selector và final metric là hai protocol khác nhau

| Chi tiết source | Hyp-SelEx / runtime hiện tại | SelEx standalone đã kiểm |
|---|---|---|
| Selection pool | Original disjoint test | Original disjoint test |
| Selection clustering | External `kmeans_pytorch`, plain Euclidean k-means | External `kmeans_pytorch`, plain Euclidean k-means |
| Selector statistic | Old-v2 | Old-v1 vì helper trả phần tử đầu trong `['v1','v2']` |
| Final paper-style pool | Train-unlabeled với labeled anchors trong fit | README chỉ rõ final train-unlabeled-v2 reports |
| Final SS restarts trong source mặc định | `n_init=1`, `max_iterations=10` | Standalone `k_means.py`: `k_means_init=10`, `max_kmeans_iter=10` |

Khác biệt số restarts được xác minh ở **source defaults**, không phải chứng
minh cấu hình đã sinh mọi hàng của bảng paper. Runtime hiện theo Hyp-SelEx,
không tự nhận là bản tái lập SelEx nguyên vẹn. Không đổi sang sklearn k-means
hoặc tăng restarts riêng cho proposed rồi so với B0 cũ.

Best-Old-test-v2 có trong HypCD code gốc; việc đang dùng nó không phải một
selector mới tự bịa. Tuy nhiên đó vẫn là test-dependent selection và Old-v2
dùng global matching chứa cả nhãn novel. Phải ghi rõ và báo riêng:

- `evaluation_best_old_test.json` / `evaluation.json`: SS metrics của checkpoint
  được selector Old-test-v2 chọn.
- `evaluation_last.json`: SS metrics ở epoch cuối cố định.
- `test_metrics.jsonl`: plain-test scores phục vụ selector, **không phải** bảng
  final anchored-SS train-unlabeled.
- `results_summary.json`: peak All/Old/New hậu nghiệm theo protocol. Không ghép
  ba maxima từ ba epoch thành một hàng. Không chọn best-All sau khi xem curves
  rồi trình bày là checkpoint đã khóa bằng Old hoặc validation.

Nếu chọn `final_epoch` cho một nghiên cứu độc lập hơn, đổi trước khi xem kết
quả, áp dụng cho cả B0/CRiSH và ghi đó là protocol khác. Không cần bỏ best-test
một cách âm thầm chỉ để gọi pipeline label-safe. Giữ best và last giúp người
đọc đánh giá trade-off. Cars đã được xem kết quả từ hướng cũ nên là exploratory,
không còn pristine held-out; điều này không được sửa bằng đổi tên split.

## 5. Khác biệt training cần khóa khi so sánh

- Reference theo lệnh Cars/DINOv2 đã phát hành: smoothing 0.5, hyper weight
  0.5, curvature 0.1, clipping 2, trainable blocks từ 10, projector 8192.
  Đây là reference code/CLI, không suy ra mọi chi tiết paper trùng tuyệt đối.
- SelEx README yêu cầu Cars smoothing 1 và block từ 9; standalone training
  đặt projector 65536. Vì thế B0 hiện tại là **Hyp-SelEx control**, không phải
  SelEx baseline chỉ đổi tên. Pretraining DINO/DINOv2-reg4 cũng phải phân biệt.
- Upgraded đổi USE normalization và cách lấy evidence/probe. Nếu dùng nó,
  cần B0 upgraded cùng normalization; không gán mọi gain cho CRiSH repair.
- Source fixes, split, exact image membership, weights hash, batch, epochs,
  seed, class-count assumption và clustering budget phải được giữ matched.

## 6. Lỗi metadata đã sửa trong audit này

Trước sửa, `selected_using_evaluation_labels` bằng trạng thái selector của
cả run, nên cả last và monitor report bị gắn True khi run có best-test.
Fixture tái hiện: 2 audit tests pass, 1 fail với `True != False` cho last.
Sửa tại `gcd_runtime/checkpoint.py` để phân biệt:

- `selected_using_evaluation_labels`: chỉ checkpoint best-test thực sự được chọn.
- `checkpoint_selection_uses_evaluation_labels`: policy của run có selector test.
- `matching_uses_evaluation_labels`: True cho Hungarian evaluation.
- `used_for_checkpoint_selection`: report test được selector sử dụng.

Đây là sửa provenance, không đổi predictions, accuracy, loss hoặc optimizer.
Source fingerprint thay đổi nên **không resume checkpoint cũ bằng source mới**;
giữ release cũ để resume, hoặc train mới như kế hoạch. Các artifact cũ không
bị ghi lại để giả vờ đã có metadata mới.

## 7. Kế hoạch train lại đã chuẩn bị, chưa thực thi

```bash
.venv/bin/python scripts/prepare_retraining.py --dataset scars --profile reference --seeds 1 2 3
```

Script chỉ tạo full/smoke config snapshots, source fingerprint, split hash,
manifest và `commands.txt` trong `runs/plans/`; không khởi chạy sweep. Mỗi
seed có B0 rồi CRiSH; hai full configs chỉ khác `method`. Mặc định giữ selector
đang dùng. `--selection final_epoch` chỉ là lựa chọn phải khóa trước thí nghiệm.

Plan được tạo trong audit này:
`runs/plans/matched-scars-reference-20261108T100459Z-b2ca3351/`.
Copy source bằng Git hoặc clean source archive; trên Linux tạo lại plan từ
đúng source trước khi chạy. `runs/` không nằm trong source release.

Thứ tự thực thi **từng command**, không chạy đồng thời:

1. Xem `nvidia-smi`, kiểm tra không có job cạnh tranh; không kill job có sẵn.
2. Preflight B0 seed 1: actual data/images/SSB membership, weights/import,
   NVIDIA tensor forward/backward, HSSK và evaluator backend.
3. Bounded real-data smoke cùng batch 128; kiểm exit/status, finite loss,
   đủ bước, VRAM và final SS. Đã có lịch sử OOM trên RTX 3090/24 GB; smoke
   ngắn pass chưa đảm bảo toàn bộ 200 epoch ổn định.
4. Chỉ khi qua gate, full B0 seed 1; sau đó CRiSH seed 1 cùng inputs/protocol.
   Các seed còn lại khi đủ compute; ghi cả null/negative runs.
5. Khóa model-selection/reporting rule, báo All chính, Old/New và best/last,
   exact per-seed values + mean/std khi có >=3 seeds, thời gian/VRAM thực đo.

Mac local thiếu dataset/weights/CUDA nên dừng ở prepare + CPU verification.
Không có xác nhận remote readiness hay training mới từ lần audit này.

## 8. Kiểm chứng thực tế của lần audit

- Regression trước sửa tái hiện cờ last sai: 2 passed / 1 failed; sau sửa,
  bộ targeted evaluation/selection/alignment có 18 passed.
- Full CPU run `crish-v21-20261108T101031Z-85b7ce2e`: **126 passed**, không
  failure/error/skip; có source-archive execution và manuscript/PDF provenance.
- Code/config protocol, Cars/CUB split và source integrity checks pass.
- Plan gồm 3 matched pairs được kiểm config; 6 stage seed-1 chạy dry-run.
  Không có GPU worker thực sự được khởi động bởi các lệnh đó.
- PDF được dựng lại, 7 trang; log không có overfull box hay undefined reference.
- Local preflight input check xác nhận thiếu Cars annotations, pretrained
  DINOv2-reg4; torch báo CUDA=False. Đây là blocker tài nguyên, không phải
  một benchmark thất bại hay bằng chứng remote đã sẵn sàng.
- Log audit/test/source snapshots: `logs/evaluation-audit-20261108T100159Z/`;
  biên nhận full smoke: `results/local_validation.json`.

## 9. Bản chốt cấu hình sau audit

Theo yêu cầu chốt source ngày 08/11/2026, entrypoint và script prepare mặc định
chọn `--profile final`: USE legacy, evidence global frozen-feature shortlist,
probe legal class-balanced 16 labeled / 32 unlabeled mỗi fold. Profile reference
cũ và upgraded dùng row USE vẫn được giữ. Các quy tắc đánh giá/chọn checkpoint,
thông số loss, điều kiện repair và giới hạn full run không đổi. Đây là quyết
định cấu hình trước thực nghiệm, chưa có kết quả hiệu quả của final.

Tạo lại plan trên checkout Linux khớp source:
`.venv/bin/python scripts/prepare_retraining.py --profile final --seeds 1`.
Chỉ xem plan rồi chạy từng stage sau khi kiểm tra tài nguyên; không tự động
chạy sweep. Các kết quả 126-test phía trên thuộc lần audit trước; biên nhận
`results/local_validation.json` của bản đóng gói ghi lần kiểm tra hiện hành.
