> Trạng thái hiện hành: ứng viên đã hiện thực và kiểm thử CPU; xem reports/06 và07. Các đoạn “chưa tích hợp” bên dưới ghi lịch sử audit ban đầu. Tính mới vẫn chưa được xác nhận.

# Tính mới và phương pháp — chưa thông qua novelty gate

## Các nguồn đã kiểm tra
- HypCD, Liu / He / Han, CVPR 2025: https://arxiv.org/html/2504.06120v1 §3 và §6.3; code pin trong report 01. Đã có hierarchical pseudo-label, confusion-based smoothing và geometric loss; thêm “hierarchical” hoặc “confidence” không đủ làm đóng góp.
- RCA, Jiawei Yu et al., ICML 2026: https://proceedings.mlr.press/v306/yu26l.html . Trang chính thức mô tả RAC evidential anchors và cross-view confidence alignment trên probability simplex. **Chưa đọc được toàn văn**: PDF raw trả unsupported content-type ở web tool; OpenReview gặp browser challenge. Cần tải và đọc method/equations trước kết luận khác biệt. Không coi abstract là novelty review hoàn chỉnh.
- LEAH, Yu Duan et al., AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/39236/43197 . Đã đọc method pruner/constructor: query chấm điểm token, top-K key objects, Lorentz adaptive entailment cone, self/supervised entailment. Rủi ro trùng động cơ hierarchy; hướng pair reliability không cần token pruning là khác biệt thiết kế dự kiến, chưa chứng minh mới.
- HypGCD, Yu Duan et al., TNNLS 2025: https://duannyu.github.io/assets/papers/hypgcd.pdf . Đã đọc phần III về Euclidean feature/label learning, hyperbolic category loss và khung local hierarchy; cần đối chiếu sâu local hierarchical loss trước implementation. Paper kết hợp instance-class và instance-instance; không được tuyên bố tiên phong hyperbolic GCD.
- RLCD, Duo Liu et al.: https://arxiv.org/html/2506.02334v1 §3.2–3.3. AUX branch distill base predictions có uncertainty weight=max probability đã detach; CDR chỉnh bias. Reciprocal learning này là hai nhánh trao đổi, không đồng nghĩa reciprocal kNN. Weight confidence đã có tiền lệ.
- Tìm kiếm thêm phát hiện CiPR https://arxiv.org/abs/2304.06928 (selective neighbor clustering) và Towards Debiased GCD https://ijcai.org/proceedings/2025/587 . Đây là đầu mục phải đọc method tiếp, chưa coi là so sánh hoàn chỉnh.

## Quyết định hiện tại
Chưa xác nhận tính mới, chưa tích hợp RAH-GCD. Cải tiến đầu tiên là kiểm tra integrity/protocol độc lập, giữ upstream nguyên vẹn. Không triển khai classifier confidence giả trong Hyp-SelEx non-parametric.

## Giả thuyết phân biệt cần kiểm chứng
Định lượng độ ổn định của **quan hệ pseudo-positive theo mức hierarchy** bằng reciprocal hyperbolic neighbors và agreement giữa hai augmentation; dùng stop-gradient và mask chỉ pair chứa ít nhất một unlabeled sample. Giữ supervised true-label term và self-view positive gốc. Cần phân biệt “một đầu unlabeled” với “cả hai unlabeled” qua cấu hình và ablation đã đăng ký.
Dự kiến r_ij = floor + (1-floor)*reciprocal_ij*agreement_ij, ramp theo epoch; floor/ramp chưa được chọn hay tune. Không thêm loss khi chưa kiểm tra normalization trong SupCon: chuẩn hóa theo anchor và sau đó mean để tránh thay đổi scale chỉ do số positive. Nếu không có eligible neighbors, fallback đúng baseline ở anchor đó; không làm zero loss. Mọi tensor reliability phải không nhận evaluation labels.
B0 untouched; B1 reciprocal only; B2 cross-view only; B3 combined+ramp. Phải có tests baseline parity, bounds, diagonal/membership mask, duplicate sample IDs do sampler replacement, finite gradients, zero-neighbor fallback, valid Poincare radii; histogram theo labeled/unlabeled, active pairs và agreement. Known/novel diagnostics chỉ offline evaluation, không phản hồi vào scoring.
Nếu CiPR/HypGCD/RCA đã làm cùng cơ chế, phải bác bỏ hoặc thiết kế khác có bằng chứng, không đổi tên.

## Quyết định hiện thực experimental — 2026-10-29
Chủ dự án yêu cầu hiện thực để đem train remote. Triển khai một **ứng viên kiểm chứng**, không coi novelty gate đã chứng minh đạt hay đã có kết quả. Các mục “chưa tích hợp” phía trên là lịch sử trước phiên này.

Đã đọc thêm method: HypGCD §III, Eq15–21 dùng k-reciprocal trong hyperbolic để tạo triplet, mở rộng bằng supervised same-label set, rồi sample common-parent với Gumbel và tối ưu local hierarchical triplet loss. Vì thế bác bỏ cách diễn giải “reciprocal hyperbolic neighbor là điểm mới”. Nguồn: https://duannyu.github.io/assets/papers/hypgcd.pdf .
CiPR §3.2–3.3 dùng SNC/connected components để tạo hierarchical pseudo positives cho contrastive loss, bao gồm labeled–unlabeled. Không thể gọi các quan hệ này là mới. Nguồn: https://arxiv.org/html/2304.06928v1 .
RCA đã tải PDF thành công từ PMLR và đọc §4.1–4.2 (16 trang; bản cache không đưa Git): evidence Exp(Tanh(logits)), Dirichlet α=e+1, RAC risk+KL trên labeled, CCA truyền certainty qua views trên probability simplex. Không dùng evidence/classifier confidence trong ứng viên non-parametric. Nguồn: https://proceedings.mlr.press/v306/yu26l.html .

**Phân biệt thiết kế cần thực nghiệm:** giữ hierarchy/pseudo-label gốc Hyp-SelEx; tại từng mức embedding bị cắt chiều đúng baseline, dùng reciprocal graph và cross-view neighborhood stability làm trọng số *pseudo-positive*, không tạo triplet/common-parent objective hoặc thay SNC. True supervised contrastive objective tách riêng giữ nguyên. Đây là khác biệt implementation cụ thể so với các cơ chế đã đọc, **chưa là chứng minh tính mới**; vẫn phải tìm kiếm thêm và có thể bác bỏ sau kết quả.
