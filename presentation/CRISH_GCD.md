# CRiSH-GCD v2.1
## Bản bàn giao nghiên cứu và triển khai • 04/11/2026
- Reliability-Guided Counterfactual Correction of Multilevel Pseudo-Supervision for Generalized Category Discovery.
- Một pipeline CRiSH/B0; backbone DINOv2 ViT-B/14 reg4, benchmark đầu tiên Stanford Cars SSB.
- Trạng thái: có triển khai và kiểm thử CPU tổng hợp; chưa chạy benchmark CRiSH, chưa xác nhận CUDA.
- Đây là giả thuyết nghiên cứu. Chưa kết luận có lợi về accuracy, tính mới hay khả năng tổng quát hóa.
---
# 01 · Source hiện tại và giới hạn môi trường
- Kiểm kê workspace, Git, logs và tiến trình trước khi dọn. Không phát hiện training đang chạy; chỉ có dịch vụ IDE/Codex/Pylance. Không dừng tiến trình nào.
- Máy local: macOS, Apple M5/16 GB, Python 3.9.6, PyTorch 2.4.1 CPU; không có CUDA, dataset và pretrained weights tại các đường dẫn cấu hình.
- Giữ nguyên AGENTS.md và upstream HypCD. Nhánh reliability cũ, configs cũ và các báo cáo rời đã ra khỏi source hoạt động.
- Bản lưu trước khi dọn: 177 file, có manifest hash; tên pre-cleanup-20261104T135934Z-5a27a619.tar.gz trong .archive/ của workspace gốc, không phát hành kèm.
- SHA256 bản lưu: 33dedc210348237097c5f7a2f8f3f5473d3459e18ffb7d8241e130fbc750d9c2.
---
# 02 · Câu hỏi nghiên cứu và ranh giới nhãn
- D_L = {(x_i, y_i)}: ảnh và nhãn known hợp lệ. D_U = {x_i}: trộn ảnh known và novel, không được đọc nhãn ẩn khi training.
- Labeled/unlabeled là trạng thái được gán nhãn; không đồng nghĩa old/new. Số lớp K được biết như baseline, không tuyên bố estimated-K.
- Câu hỏi: quyết định sửa hoặc bỏ qua từng quan hệ giả ở nhiều mức có giảm giám sát sai mà vẫn giữ được khả năng khám phá novel không?
- Điểm cần kiểm chứng là chính sách chọn can thiệp có ràng buộc. Reciprocal kNN, view consistency, stability và gradient alignment riêng lẻ đều đã có tiền lệ.
- Không dùng tên lớp xe, prompt theo dataset, classifier confidence không tồn tại, hoặc nhãn D_U trong probe/selection.
---
# 03 · Baseline và protocol có thể đối chiếu
- HypCD/Hyp-SelEx: commit ffc895b7ff38be8502bb7e3bd851f1d2a55dff73; upstream giữ nguyên. Runtime có adapter riêng, không chạy sbatch local.
- Source train/train_HypSelEx.py SHA256: af872b72e0d0068f0492a032299a59decbbc049f6d45a4934fdb57bb5453a1d5.
- Cars SSB file scars_osr_splits.pkl: 98 known, 98 novel (Hard 7, Medium 15, Easy 76); prop_train_labels = 0.5 theo loader.
- Split SHA256: 28d471f0fb9e2c773e1255b8b447e542974f6617952203056a0635c638513d39.
- DINOv2 ViT-B/14 reg4, projector 8192, batch 128, SGD lr 0.1/momentum 0.9/weight decay 5e-5, seed 1, 200 epoch; input 224 và transform imagenet upstream.
- Cần khớp split, fraction, checkpoint backbone, K, transforms, optimizer và matching trước khi gọi là so sánh công bằng. Không quy chênh backbone thành lợi ích thuật toán.
---
# 04 · Hai chế độ của cùng một pipeline
- CRiSH: bật repair sau warm-up. B0: method=baseline, không gọi evidence/probe và không tiêu thụ RNG cho repair.
- Cả hai dùng cùng objective, HSSK mỗi epoch, sửa thứ tự hàng fit_mix, policy checkpoint cuối và evaluator độc lập. B0 không được gắn nhãn là tái lập nguyên vẹn mọi hành vi script công bố.
- HSSK chạy trên normalized backbone features; projector và ToPoincare phục vụ loss. M1 hiện dùng Euclidean backbone, không gọi đó là hyperbolic kNN.
- fit_mix trả labels theo thứ tự labeled trước/unlabeled sau; runtime hoàn nguyên về thứ tự input và ánh xạ bằng sample ID tường minh.
- HSSK trả một stack phân hoạch; không có ràng buộc bảo đảm các mức lồng nhau thành cây. Không triển khai thao tác yêu cầu cây.
---
# 05 · Loss và bất thường source phải giữ chung
- USE dùng logits khoảng cách; SSE gồm true-label SupCon và pseudo-SupCon ở các prefix chiều D/2^(l+1), hệ số 2^(-(l+1)). Góc/hyperbolic trộn theo ramp epoch gốc.
- R_g^l = 0.5 min_(h != g) ||mu_g^l - mu_h^l||. Với zero_one: F_ab = sum_l 2^(-l) 1[d_ab > 2 R_(c_b^l)^l].
- Source min-max scale F rồi viết F / F.sum(dim=1). PyTorch broadcast mẫu số theo cột: kết quả là F_ab / sum_c F_bc; không phải chuẩn hóa tổng hàng.
- Sau đó bỏ đường chéo, đưa positive khác view lên đầu, không renormalize. Target smoothing không được bảo đảm có tổng bằng 1.
- Giữ đúng hành vi này trong cả B0 và CRiSH; không âm thầm sửa riêng một nhánh. Trường hợp mẫu số 0 phải báo lỗi rõ ràng. Muốn sửa cần một ablation chung, tách khỏi hiệu ứng repair.
- LabelSmoothingLoss, SupConLoss và info_nce_logits giữ AST bằng source gốc; oracle độc lập đối chiếu loss và gradients.
---
# 06 · M1: evidence thưa, có trạng thái chưa biết
- Với mỗi cặp augmentation t, tạo kNN cùng tập ID, loại self. R_ij^(t,v) = 1[j thuộc N_k(i)] × 1[i thuộc N_k(j)].
- r_ij = (1/T) sum_t R_ij^(t,1) R_ij^(t,2), chỉ trên các cạnh được quan sát E. Điểm đối xứng, trong [0,1], detached.
- Cạnh không được đo là unknown; cạnh đã đo có r=0 là một trường hợp khác. Không lấy phần bù của đồ thị thưa làm negative evidence.
- Mặc định: 128 ảnh evidence, T=2, k=5; evidence/probe/gradient IDs tách nhau. Không đủ support thì Keep.
- So RN-only, CVS-only và RN+CVS ở cùng support/density. Exact blockwise kNN vẫn O(T N²d) tính toán, dù lưu đồ thị O(Nk).
---
# 07 · M2 và audit phân hoạch
- M2 tùy chọn: nhiễu feature độc lập, recluster và ghép cluster theo maximum-total-IoU, có dummy unmatched. Unmatched nhận IoU 0; báo coverage riêng.
- Mặc định stability_trials=0: M2 tắt. Khi bật, đây là stability trên frozen features với nhiễu/restart, chưa phải bằng chứng semantic robustness.
- Đo mức vi phạm refinement: V(l,l+1)=1 - (1/N) sum_g max_h |{i:c_i^l=g,c_i^(l+1)=h}|. V=0 bất biến theo đổi tên cluster.
- Cần đo V trên checkpoint thật; từ source chỉ kết luận không có bảo đảm nestedness. Stability và M1 có thể dư thừa hoặc cùng sai.
- Nếu M2 không thêm thông tin ngoài M1 theo strata support/density và kiểm định đã khóa, bỏ M2 thay vì giữ để tăng độ phức tạp.
---
# 08 · Không gian hành động có ràng buộc
- Mỗi unlabeled i ở mức l: Keep, Abstain hoặc Move(i:g→h,l). Chỉ chuyển tới cluster hiện hữu có support đo được; giữ K và không để cluster rỗng.
- Giữ mọi labeled assignment ở mọi mức. Mức gốc khớp nhãn known hợp lệ; ở mức thô, nhiều lớp known có thể chung cluster đúng như baseline.
- Keep trả nguyên snapshot và centers. Move tính lại mean source/destination trên frozen normalized features, giữ centers khác nguyên vẹn, rồi tính lại toàn bộ radii mức đó.
- Rebuild SSE masks và USE factors/normalization từ cùng phiên bản. Không giả định sửa một hàng chỉ ảnh hưởng một target; không dùng lối tắt local normalization.
- Contract đã kiểm thử: Euclidean prototypes, cluster_momentum=1, hai views. Chế độ geometry khác bị chặn trước training.
---
# 09 · Abstain: giữ hệ số của positive được bảo vệ
- Abstain chỉ tác động pseudo-SSE: loại tử số positive cross-ID liên quan unlabeled được chọn ở mức đó. Giữ labels, prototypes, radii, USE và true-label loss.
- Với P_a là positives gốc và A_a là tập bị loại: L_a = -(tau/tau_base) sum_(b thuộc P_a trừ A_a) log p_ab / |P_a|.
- Giữ mẫu số positive gốc, contrastive denominator và anchor mean. Không chia lại cho số positive còn sống; làm vậy sẽ tăng hệ số cùng-ID/labeled-labeled.
- Không tuyên bố giữ nguyên chuẩn gradient. Ghi removed mass và so với control matched-mass; USE-Abstain chưa được triển khai.
- Move có thể đổi số pseudo-positive nên đổi hệ số cùng-ID bên trong pseudo-term. Runtime đo protected-coefficient shift và chặn nếu vượt ngưỡng; không đánh đồng bảo đảm của Move với Abstain.
---
# 10 · Probe hợp lệ và ngăn rò rỉ nhãn
- Sanitizer chỉ giữ path, sample ID, labeled membership và nhãn D_L; mọi target D_U thành -1. Không giữ dataset gốc chứa hidden target/mapping trong training runtime.
- Probe known dùng nhãn từ registry hợp lệ; probe unlabeled chỉ dùng hai views và frozen teacher embeddings. Không tạo softmax classifier confidence giả cho Hyp-SelEx.
- J = lambda_K J_known + lambda_V J_view + lambda_C J_variance. J_known cân bằng theo lớp known trong probe; J_view là MSE normalized teacher; J_variance có std floor.
- Discovery và confirmation tách ID; cả hai tách proposal/gradient/evidence pools. Các đầu vào, augmentation và teacher được cố định trong refresh.
- Known guard và variance floor vẫn có thể thiên về known, che novel/tail hoặc giữ teacher sai. Đây là surrogate, không bảo đảm semantic correctness hay All ACC tăng.
---
# 11 · Counterfactual và một bước optimizer thật
- Gọi g_a = grad L(theta;a), g_0 = grad L(theta;Keep), q = grad J(theta). Với SGD thường: gain ≈ eta qᵀ(g_a-g_0), sai số bậc eta².
- Runtime dùng G_step = J(Step(theta,state,g_0)) - J(Step(theta,state,g_a)), clone optimizer với momentum/weight decay thật; gain dương nghĩa là probe tương đối tốt hơn.
- Lookahead chỉ trên projector; backbone features đã detach. Không phải full-backbone lookahead và không mô phỏng 5/10 bước tái tính gradient.
- Clone hỗ trợ WeightNorm của DINOHead gốc, bảo toàn parameters/buffers/optimizer state/mode/gradients/RNG của live model.
- Ghi raw alignment, cosine, exact gain, changed SSE/USE mass và protected shift. Chia score cho mass nhỏ có thể tăng nhiễu; không coi đó là chứng cứ sửa đúng.
---
# 12 · Quyết định tuần tự và ngân sách
- Freeze snapshot → đo M1 → shortlist → rebuild từng phương án → so với Keep hiện tại → xác nhận bằng probe độc lập → commit một phương án → hủy score cũ và tính lại.
- Hai can thiệp tốt riêng lẻ có thể hại khi gộp vì prototypes, radii, target normalization và optimizer tương tác. Không cộng gain để chấp nhận hàng loạt.
- Chấp nhận chỉ khi finite, hợp lệ, support đủ, gain vượt ngưỡng, known-loss deterioration và protected shift trong giới hạn. Rejected Move không tự động thành Abstain.
- Mặc định: warm-up 10 epoch, refresh mỗi 5 epoch; tối đa 2 boundaries, 1 target/boundary, 1 hành động chấp nhận/refresh; giới hạn refresh 120 giây.
- Repair hết hiệu lực khi HSSK chạy lại epoch sau. Deadline kiểm tra giữa các phép tính; không bảo đảm ngắt tức thì một kernel. Worker bên ngoài có timeout hữu hạn.
- Probe có thể bị overfit do truy vấn lặp; cần khóa query budget và kiểm chứng bằng các probe/control độc lập.
---
# 13 · Novelty gate và các đối chiếu bắt buộc
- SelEx: HSSK và self-expertise đã có. HypCD: hyperbolic mapping/loss đã có. Không nhận phần này là đóng góp CRiSH.
- HypGCD: hyperbolic reciprocal neighbors và hierarchical triplets; RCA: reliable evidence và cross-view alignment; LEAH: hierarchy bằng token/cones. Reliability/hierarchy riêng lẻ không mới.
- CiPR/ProtoGCD/SEAL: neighbor/pseudo-label/structured-target refinement. Ren et al.: validation-gradient reweighting. Phải có control heuristic và gradient weighting.
- CURE: structure-guided reconstruction/prototype alignment, có nguy cơ trùng gần. TAR: consistency/token refinement; đối tượng khác không tự chứng minh CRiSH có đóng góp.
- ConGCD là lead cần đọc full method tiếp. Chưa đóng novelty gate; không được gọi phương pháp là novel chỉ vì có tên hoặc module riêng.
- Thesis có thể bác bỏ: constrained sequential action screening có giá trị vượt abstention, matched-mass weighting và prototype heuristic ở chi phí chấp nhận được.
---
# 14 · Đánh giá và checkpoint không nhìn oracle
- Khóa epoch cuối cho cả B0/CRiSH. Training chỉ ghi loss/diagnostics; không chạy oracle mỗi epoch, không chọn seed/threshold/early-stop theo nhãn D_U.
- Sau khi khóa backbone, worker riêng chạy balanced semi-supervised k-means trên train-unlabeled với labeled anchors; metric Hungarian-v2 dùng một global assignment.
- All = (n_old × Old + n_new × New)/(n_old+n_new). Báo All chính, kèm Old/New và số mẫu; không lấy trung bình không trọng số.
- evaluation.json lưu fraction; bảng báo cáo dùng %. Đánh giá plain k-means trên disjoint test là protocol khác, không trộn số liệu.
- Tương lai: ít nhất 3 seed matched nếu tài nguyên cho phép, mean ± std và từng seed; runtime/memory/repair acceptance và negative/null results.
---
# 15 · Exposure register và độ tin cậy của bằng chứng
- Cars đã được xem log/Old-New curves của hướng reliability cũ; là exploratory comparison, không còn pristine held-out. Kết quả cũ không thuộc CRiSH và không có matched B0.
- CUB và CIFAR-100: chưa thấy run trong artifacts đã kiểm kê; điều này không chứng minh chưa từng tiếp xúc. Cần người nghiên cứu xác nhận lịch sử trước khi khóa vai trò dataset.
- Aircraft, Herbarium19, ImageNet-100: chỉ là ứng viên confirmation, chưa xác nhận non-exposure và chưa chạy.
- Mọi lần xem oracle sau lock phải ghi dataset/version/checksum, config/source hash, thời điểm, artifact và quyết định sau đó. Nếu dùng để đổi phương pháp, dataset trở thành exploratory cho phiên bản mới.
- Không có bảng accuracy CRiSH trong bản này. CPU fixtures và lịch sử hướng khác không được thay thế một benchmark đã chạy.
---
# 16 · Ma trận nghiên cứu tiếp theo
- E0: pin source, shared fixes/protocol, no-action loss/gradient parity và mini-batch dữ liệu thật; dừng nếu sai ID/target hoặc baseline nonfinite.
- E1: đo nestedness V và support trên stack HSSK thật. E2: RN-only/CVS-only/both, matched support/density; nhãn ẩn chỉ phân tích sau lock.
- E3: M2 có thêm thông tin ngoài M1 không? E4: Keep/Abstain/Move, random/support-distance controls; rebuild, mass và interactions phải được đo.
- E5: first-order/alignment/one-step ranking so probe thật; 5/10-step recomputed trajectories còn là thí nghiệm tương lai. E6: probe bias, known/novel/tail và stable-but-wrong.
- E7: matched training B0–B7/B0+ khi các gate cho phép; ít nhất 3 paired seeds nếu đủ tài nguyên. Báo All/Old/New, thời gian, memory, queries, per-seed và cả negative/null results.
- Không coi các cạnh kNN phụ thuộc nhau là mẫu độc lập để tính significance; xác định đơn vị resampling theo ảnh/cluster/run và báo giới hạn của số seed ít.
---
# 17 · Ablation và điều kiện đi tiếp
- B0: matched baseline. B0+: baseline có compute allowance đối chiếu ngân sách CRiSH. B1: RN/CVS weighting. B2: stability-only. B3: M1+M2 fixed-target weighting.
- B4: random/support-distance interventions, matched accepted count/mass/cost. B5: gradient/meta-reweighting cùng legal probe/query budget. B6: sequential CRiSH. B7: drop-one/mass/scoring controls.
- Đây là thiết kế thí nghiệm tương lai; source hiện chỉ có configs chạy CRiSH và B0. Không có script train thứ hai cho hướng reliability cũ.
- G0: provenance/protocol và data readiness. G1: no-action/protected/rebuild parity cả synthetic và real batch. G2: evidence thêm thông tin. G3: quyết định cạnh tranh với heuristic.
- G4: lookahead dự đoán hiệu ứng và vượt B4/B5. G5: các gate đạt, tài nguyên và quyền chạy sweep đắt đã được xác nhận. Hiện real-data/decision/novelty gates chưa đạt.
---
# 18 · Bàn giao cho người train
- Đọc README.md. Cài Linux NVIDIA: GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda. Chuẩn bị Cars/annotations và weights/dinov2_vitb14_reg4_pretrain.pth.
- Kiểm tra: bash scripts/train_crish.sh --dry-run. Chạy: bash scripts/train_crish.sh. B0: thêm --baseline.
- Lệnh mặc định: preflight 180 giây → smoke 1 epoch/2 batch tối đa 30 phút → run mới 200 epoch tối đa 24 giờ → final evaluator tối đa 30 phút.
- Log/config/environment/fingerprint/PID/checkpoint nằm ở runs/RUN_ID. Resume từ cuối epoch, đúng source/config. Source cũ phải resume bằng archive cũ.
- Không tải weights/data hoặc chạy GPU sweep trên Mac. Test local: .venv/bin/python scripts/run_smoke.py. Release sạch có manifest/SHA256, không chứa lịch sử/data/weights.
---
# 19 · Kết quả kiểm chứng và việc còn mở
- Biên nhận chạy thực tế: results/local_validation.json; liệt kê số test, trạng thái, source hashes và lệnh. results/cleanup_parity.json ghi so sánh fixture trước/sau dọn.
- Test giữ lại kiểm tra baseline loss/gradient, CRiSH-off parity, label poisoning, ID/cluster alignment, protected pairs, zero mass/nonfinite, rebuild, optimizer clone và epoch resume.
- Archive test giải nén gói vào thư mục tạm, xác minh từng hash, chạy smoke và dry-run cả CRiSH/B0 mà không có .git.
- Chưa có: CUDA thực trên NVIDIA, real-image epoch, VRAM/batch feasibility, final evaluation trên Cars, matched multi-seed baseline/CRiSH hay lợi ích thống kê.
- Bước kế tiếp: chuẩn bị inputs trên Linux → check → smoke; chỉ khi qua mới chạy B0/CRiSH theo protocol đã khóa. Xử lý lỗi runtime công khai; không dùng benchmark labels để chỉnh threshold.
---
# 20 · Nguồn chính và phạm vi sử dụng
- HypCD code: https://github.com/Visual-AI/HypCD ; paper: https://arxiv.org/abs/2504.06120 (Liu, He, Han; CVPR 2025). SelEx: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08926.pdf (ECCV 2024).
- RCA: https://proceedings.mlr.press/v306/yu26l.html ; LEAH: https://doi.org/10.1609/aaai.v40i25.39236 ; HypGCD: https://duannyu.github.io/assets/papers/hypgcd.pdf .
- CiPR: https://arxiv.org/abs/2304.06928 ; ProtoGCD: https://arxiv.org/abs/2504.03755 ; SEAL: https://arxiv.org/abs/2510.18740 .
- Ren et al.: https://proceedings.mlr.press/v80/ren18a.html ; CURE: https://proceedings.mlr.press/v306/bian26d.html . Các mô tả trên kế thừa audit method đã lưu; không thêm tuyên bố accuracy.
- TAR (Yang et al., CVPR 2026) và ConGCD (Tang et al., ICCV 2025): nguồn CVF; ConGCD full-method review còn mở. Không coi lead chưa đọc là bằng chứng khác biệt.
- Nguồn gốc mã và giấy phép: NOTICE và LICENSES/HypCD-MIT.txt. Bản trình bày này thay các báo cáo rời trong source hiện tại; dữ liệu lịch sử đầy đủ được giữ trong bản lưu gốc.
