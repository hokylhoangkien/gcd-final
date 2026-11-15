# Đánh giá ghi chú CRiSH và số action thích nghi

Ngày: 2026-11-15. Source: `4d54cca9603c57f4833babf7279578b4820f65bd`.
Phạm vi: đối chiếu ghi chú người dùng với code hiện tại; không có log thật
của các run SelEx được nêu trong ghi chú. Không thay runtime hay config.

## Kết luận áp dụng

Có thể dùng ghi chú làm kế hoạch chẩn đoán, đặc biệt việc đo liều can thiệp,
kiểm tra đúng variant và so baseline khớp. Chưa đủ bằng chứng để khẳng định
liều nhỏ là nguyên nhân của kết quả gần baseline hoặc tăng liều sẽ cải thiện.
Code chứng minh giới hạn cơ chế; log mới xác định cơ chế hoạt động bao nhiêu;
thí nghiệm đối chứng mới xác định hiệu quả.

## Đối chiếu code

| Nhận định | Kết quả kiểm tra |
|---|---|
| Local nhận tối đa 1 action/refresh | Đúng với config đang cung cấp: `crish.max_accepted=1`, `boundaries=2`. |
| Số nhận bị cố định là 1 | Sai: `crish/policy.py::sequential_repair` có thể nhận 0 đến trần; không có ứng viên tốt hoặc confirmation trượt thì dừng. |
| Có thể tăng riêng max_accepted để sửa nhiều mẫu | Chưa đủ: shortlist có tối đa `boundaries` mẫu riêng biệt; sau mỗi commit bỏ mọi action của mẫu đó. Tăng cap mà giữ boundaries=2 vẫn sửa tối đa 2 mẫu. |
| Mỗi action chỉ có hiệu lực 1 epoch | Đúng đối với supervision trực tiếp: `crish/training.py` trích hierarchy mới mỗi epoch. Ảnh hưởng qua trọng số SGD có thể tồn tại lâu hơn. |
| Policy chọn được lượng can thiệp | Đúng: `crish/population.py` chọn q từ grid, có Keep=0 và confirmation. Đây là chọn liều cả tập, khác đếm local action. |
| Policy chỉ sửa level 0 | Đúng; `abstain_fraction` sửa mask base. |
| Probe cho biết accuracy cuối sẽ tăng | Chưa chứng minh: `engine.py` clone projector, không thực hiện bước cập nhật backbone trong lookahead; SS đánh giá backbone. |

## Những điểm cần sửa trước khi dùng ghi chú để quyết định

1. Không xem dao động giữa epoch là sai số giữa seed. Mốc 1,43 điểm không
   phải ngưỡng cải thiện phổ quát: paired analysis dùng độ lệch chuẩn của
   **chênh lệch theo seed**, phụ thuộc tương quan, cỡ mẫu, quy tắc kiểm định
   và power định trước. “Paired” và “hai arm độc lập” trong ghi chú không
   cùng một giả định. Với ba seed, ước lượng biến thiên còn yếu.
2. Cặp sai ở level mịn có thể là cặp đúng ở level thô. Việc cặp đó còn
   positive ở coarse level không tự chứng minh coarse supervision có hại.
   Không chép toàn bộ mask level 0 lên mọi level làm mặc định.
3. Bảo vệ labeled–labeled và known probe loss không bảo đảm Old accuracy
   không giảm. Các thành phần loss dùng chung backbone và vẫn tương tác.
4. H1 giữ q rồi xếp hạng lại mỗi epoch là **giữ chính sách liều**, không phải
   giữ nguyên quyết định đã xác nhận: ảnh bị abstain có thể đổi hoàn toàn.
   Nếu làm phải log tập thay đổi, tái đánh giá hoặc có điều kiện hết hạn;
   Move còn cần giải quyết cluster ID thay đổi giữa lần phân cụm và resume.
5. Tỷ phần chuẩn gradient của pseudo nhỏ không đủ làm quy tắc loại bỏ cứng
   ở 10%: hướng gradient, triệt tiêu và tương tác cũng quan trọng. AUROC
   0,70, ΔAll 0,5 hay q=0,2 là ngưỡng thử nghiệm đề xuất, chưa được xác thực.
6. Prior kNN được đóng băng nên không đổi theo fine-tuning, nhưng vẫn có
   sai lệch từ pretrained backbone. Không gọi nó là bằng chứng độc lập hoàn
   toàn; clustering ensemble cũng chỉ đo một dạng ổn định, không bảo đảm đúng.
   Tỷ lệ gain dương khoảng 50% không tự chứng minh probe là nhiễu.

## Thiết kế phù hợp với mong muốn “model tự quyết định số lượng”

Khuyến nghị đầu tiên là **số lượng phát sinh từ bằng chứng và điều kiện dừng**,
không cần huấn luyện thêm mạng dự đoán số nguyên n. Ở refresh t:

1. Sinh tập ứng viên có hỗ trợ thật, bao phủ đủ mẫu; luôn có Keep.
2. Chấm các ứng viên tương đối với supervision hiện tại bằng legal discovery
   probe, không dùng nhãn unlabeled hay All/Old/New để chọn.
3. Xác nhận ứng viên tốt nhất trên fold chưa sử dụng. Đạt thì commit;
   xây dựng/chấm lại các ứng viên còn lại vì hiệu quả không cộng tuyến tính.
4. Dừng khi hết ứng viên có lợi, confirmation trượt, thiếu bằng chứng hoặc
   hết ngân sách thời gian/probe/liều. Cho phép nhận 0 action.

Khi đó `n_t = số commit trước khi dừng`; trần chỉ giới hạn tài nguyên,
không phải quota buộc phải lấp đầy. Local hiện đã có cấu trúc này nhưng trần
1 giới hạn khả năng thích nghi. Đó là **policy dựa trên tín hiệu model**,
chưa phải learned budget controller. Code hiện có ngưỡng gain=0; nếu muốn
biên an toàn theo nhiễu phải hiệu chỉnh bằng legal data và kiểm chứng riêng.

Tăng trần đòi tăng shortlist/batch hỗ trợ và đủ confirmation folds rời nhau:
`engine._draw_context` dành `1 + min(max_accepted, boundaries)` folds từ đầu.
Chi phí và số ảnh labeled/unlabeled cần giữ lại tăng theo số fold. Mỗi action
phải được chấm lại sau commit; không chọn top-n bằng một lần chấm độc lập.
Vì vậy thay cap bằng “unlimited” không giải quyết được giới hạn tính toán.

Nếu mục tiêu là liều đủ lớn trên toàn tập, có thể thử policy chọn một **tập
can thiệp** hoặc tỷ lệ q, trong đó Keep=0 là ứng viên thật. `population.py`
đã có khung q-grid; không được gọi nó tương đương local Keep/Abstain/Move.
Nên so liều bằng cặp/coefficient mass thực tế và khả năng xuất hiện trong
batch, vì 10 action ở cụm nhỏ không tương đương 10 action ở cụm lớn.

Không bảo đảm adaptive tốt hơn cap=1: proxy yếu có thể nhận nhiều sửa sai;
chọn nhiều ứng viên trên discovery probe gây selection bias. Fresh confirmation
giảm việc dùng lại bằng chứng, không chứng minh tổng rủi ro thống kê đã được
kiểm soát hoặc đảm bảo cải thiện SS. Muốn learned controller cần định nghĩa
state/reward, dữ liệu huấn luyện và kiểm tra tổng quát hóa mà không dùng nhãn
evaluation; chưa có cơ sở chọn hướng phức tạp này trước.

## Thứ tự thực nghiệm đề xuất, chưa chạy

1. Đọc config/fingerprint và log thật của local + baseline: candidate/scored/
   accepted, lý do từ chối, gain, confirmation, changed IDs, target mass và
   thời gian refresh. Không suy diễn variant từ số accuracy.
2. Xác minh probe có tín hiệu trên holdout labeled hợp lệ; nếu giả lập novel,
   giấu lớp **trước khi train**. Checkpoint đã học lớp đó làm đánh giá lạc quan.
3. Tách kiểm tra adaptive cardinality khỏi persistence và all-level masking.
   Dùng cùng pool/budget khi muốn quy lợi ích cho quy tắc chọn số lượng;
   kèm control cùng liều để phân biệt lượng can thiệp với chất lượng lựa chọn.
4. So baseline, local cap=1 và adaptive có trần dưới cùng protocol/seed,
   báo All/Old/New tại checkpoint đã định trước, chi phí và liều. Persistence
   và reliability từng level là ablation riêng, không gộp để kết luận nhân quả.

## Nguồn và giới hạn novelty

- [SelEx official README](https://github.com/SarahRastegar/SelEx): xác minh
  post-fix hierarchy/loader order. Local dùng snapshot pin `d8107b5`; không
  thay bằng HEAD từ thư mục vendor vốn không phải repository riêng.
- [HypCD full paper](https://arxiv.org/html/2504.06120v1), mục 3.4–3.5,
  6.3: baseline hyperbolic và label assignment. Đường đang bàn là SelEx
  Euclidean hiện tại, không được diễn giải như đã sửa Hyp-SelEx hyperbolic.
- [RCA](https://proceedings.mlr.press/v306/yu26l.html) là related-work lead
  về reliability/cross-view confidence; full PDF fetch trong phiên này thất
  bại. Không thay method review bằng abstract. LEAH, HypGCD và các leads ở
  `reports/02_novelty_and_method.md` vẫn cần đối chiếu trước claim novelty.

Kiểm chứng CPU: **15 passed in 1.36s** cho
`tests/crish_v21/test_conflicting_repairs.py`,
`tests/test_crish_candidate_identity.py` và `tests/test_crish_scoring_batch.py`,
trong tiến trình có timeout 60 giây. Hai test mới kiểm số nhận 0/1/2/3 dưới
cùng cap và giữ commit trước khi confirmation tiếp theo trượt.
`verify_reference()` và `git diff --check` đạt. Lệnh thực tế có trong
`reports/00_workspace_audit.md`. Test với evidence kiểm soát chỉ xác nhận
logic dừng/commit; không chứng minh model tự học quota, chất lượng probe
hoặc mức tăng accuracy.
