# CRiSH-GCD — Kế hoạch hoàn thiện paper & hướng dẫn thực hiện cho Codex

> **Nguồn:** Chuyển thể từ tài liệu *CRiSH-GCD — Kế hoạch hoàn thiện paper* (17/11/2026, @tzie, 7 trang).  
> **Mục đích:** Dùng làm đặc tả đầu việc để Codex kiểm tra repository, triển khai các thay đổi kỹ thuật, thêm kiểm thử và chuẩn bị công cụ chạy thí nghiệm.  
> **Lưu ý:** Số liệu trong tài liệu là số liệu **được ghi trong kế hoạch**, không phải số Codex tự xác minh từ checkpoint/log. Không được coi một mục là hoàn thành nếu chưa kiểm chứng trên repository.

## 0. Chỉ dẫn thực thi cho Codex

Bạn đang làm việc trong repository CRiSH-GCD. **Trước tiên hãy audit trạng thái code và đối chiếu toàn bộ yêu cầu bên dưới**, rồi thực hiện các phần kỹ thuật phù hợp với vai trò Kiên (K1–K6), đồng thời cung cấp script, CLI và đầu ra có thể tái lập để An thực hiện A1–A6. Đừng giả định tên file, module, class, flag hoặc đường dẫn run; hãy tìm trong repository.

1. Đọc README, cấu trúc repo, entry point train/evaluate, cách tạo checkpoint, refresh, evidence cache, proposal, counterfactual scoring, confirmation và cách lưu action/log.
2. Lập bảng **Đã có / Thiếu / Sai khác / Cần xác minh**, đối chiếu từng yêu cầu K1–K6, A1–A6 và các biến thể ablation (b), (c), (e), (f), (g).
3. Chỉ sửa phần cần thiết; giữ nguyên default behavior của CRiSH full. Không âm thầm thay thuật toán, benchmark protocol hoặc dữ liệu đánh giá để làm tăng điểm.
4. Tạo flag/config có tên rõ ràng và kiểm thử từng flag độc lập. Cần có cách chạy `full`, `evidence_only`, `no_confirmation`, `no_abstain`, `no_move`, `random_proposal` mà **không sửa code giữa các run**.
5. Bảo đảm cùng dataset, seed, lịch refresh, budget, epochs, data split, augmentation, dataloader và các điều kiện chung khi so ablation; RNG riêng cho proposal random, không làm xáo trộn RNG của train.
6. Đánh giá/report bằng **final checkpoint**, tối thiểu 3 seed cho bảng chính; chỉ dùng nhãn thật của unlabeled cho hậu kiểm phân tích, không dùng chọn best checkpoint hay quyết định action. Ghi rõ mọi kết quả vẫn chưa chạy/kiểm chứng.
7. Viết unit/integration tests nếu khả thi, chạy những test có thể chạy không cần GPU, lưu kết quả và không bịa số liệu. Nếu GPU hoặc checkpoint không có sẵn, **chỉ chuẩn bị script/config** và hướng dẫn lệnh chạy; không tuyên bố đã chạy.
8. Báo cáo cuối gồm: file thay đổi, lý do, cách chạy, test pass/fail, risk, các việc vẫn cần con người/GPU, và checklist theo mã K1–K6/A1–A6.
9. Không push, tạo PR, xoá hoặc ghi đè checkpoint/dataset/log cũ nếu chưa có yêu cầu cụ thể. Ưu tiên thay đổi nhỏ, tái lập được, tương thích resume.

### Thứ tự triển khai ưu tiên

- **P0 — Audit + K1:** chốt mô tả cấu hình adaptive, so với draft và định danh commit/version.
- **P0 — K3:** triển khai 5 công tắc ablation và kiểm thử tính cô lập.
- **P1 — K4/K5/K6:** action logging, hậu kiểm chỉ dùng nhãn ở bước phân tích, overhead measurements.
- **P1 — Hỗ trợ A1/A3/A4/A6:** run registry, launcher/config theo seed, báo cáo final mean ± std, biểu đồ theo epoch + refresh markers.
- **P2 — K2:** hỗ trợ thử nghiệm 1 seed trên Cars; **không tự quyết định đổi phương pháp chính** nếu chưa so checkpoint cuối và có chốt G1.

---

## 1. Bối cảnh và hiện trạng kết quả

**Hạn nộp:** trước 01/12/2026; bản thảo cần đến thầy để review trước hạn. Baseline dùng số SelEx đã công bố, không chi GPU chạy lại. GPU dành cho Kiên và An kiểm tra thành phần CRiSH; Thúy Vi viết paper song song. Kế hoạch này chỉ giao việc cho Kiên và An.

Số CRiSH hiện có là **best-All của một run**, chọn nhờ nhãn thật của ảnh unlabeled. Số SelEx công bố là **final checkpoint, trung bình 3 seed**. Do khác protocol, **chưa được đặt cạnh nhau để claim cải thiện**.

Đơn vị: `All / Old / New (%)`.

| Dataset | CRiSH adaptive best-All (epoch) | Draft cũ | SelEx công bố — bản sửa (50 epoch, last, 3 seed) | Δ All so draft cũ |
|---|---|---|---|---:|
| CUB-200 | 83.94 / 86.39 / 82.72 (ep27) | 84.7 / 85.7 / 84.2 | 83.5 / 85.6 / 82.5 | −0.76 |
| Stanford Cars | 82.80 / 94.40 / 77.19 (ep85) | 81.6 / 91.7 / 76.7 | 79.1 / 93.4 / 72.2 | +1.20 |
| FGVC-Aircraft | 79.28 / 82.95 / 77.45 (ep66) | 79.1 / 81.3 / 78.0 | 77.2 / 82.5 / 74.6 | +0.18 |

Theo kế hoạch, so hai run 1 seed với nhau thì chênh dưới khoảng ±1.7 All (CUB, Cars) hoặc ±0.6 (Aircraft) được coi là có thể thuộc nhiễu giữa seed, ước lượng từ std README. Bản adaptive chưa vượt các ngưỡng này so draft cũ ở dataset nào; **chưa thể khẳng định bản nào tốt hơn**.

### Vì sao Old đạt đỉnh sớm, New đạt đỉnh muộn

Ảnh unlabeled có khoảng 1/3 Old và 2/3 New nên:

$$
\mathrm{All}\approx\frac{\mathrm{Old}+2\,\mathrm{New}}{3}
$$

| Giai đoạn | ΔOld | ΔNew | ΔAll dự đoán | ΔAll thực |
|---|---:|---:|---:|---:|
| CUB ep15 → ep27 | −1.60 | +1.61 | +0.54 | +0.53 |
| CUB ep27 → ep141 | −1.53 | +0.60 | −0.11 | −0.11 |
| Cars ep85 → ep156 | −1.90 | +0.31 | −0.43 | −0.41 |

- Đầu train, New tăng đủ để bù Old giảm nên All tăng. Về sau Old giảm nhanh khoảng 2.5–6 lần mức New tăng, khiến All chững lại/giảm nhẹ.
- CUB nằm trên plateau All khoảng 83.4–83.9 từ ep15 đến ep141; đỉnh ep27 không đủ cho kết luận mạnh.
- Thứ tự đỉnh `Old → All → New`: CUB `15/27/141`, Cars `52/85/156`, Aircraft `42/66/69`.
- **An:** cần lấy **final checkpoint**, có thể thấp hơn best; Cars ep156 đã thấp hơn đỉnh 0.41 điểm.
- **Kiên:** giả thuyết đáng kiểm tra là giữ Old ở giai đoạn cuối train; chưa được coi là kết luận nếu không có log/test xác nhận.

## 2. Quy ước báo cáo số liệu

1. **Baseline:** dùng đúng số trong README SelEx **bản đã sửa**, 50 epoch, last checkpoint, mean từ 3 seed; **không chạy lại baseline**. Lưu số baseline nhóm từng chạy lại (config, seed, kết quả) trong sổ run A5 để trả lời reviewer.
2. **CRiSH:** báo cáo **final checkpoint, mean ± std của ≥3 seed**. Tuyệt đối không dùng best chọn bằng nhãn thật ảnh unlabeled làm số chính trong paper.
3. Tất cả run CRiSH có cùng lịch **200 epoch**. Tài liệu ghi code SelEx 2024 mặc định 200 epoch (`contrastive_training.py`, dòng 662); cần xác minh nếu trích dẫn.
4. Về lỗi SelEx: tài liệu ghi Issue #11 do `BanAnA9205` mở 15/10/2026, tác giả xác nhận 20/10 và sửa 09/11/2026 ở commit `0cc628d`. **Kiểm tra GitHub trước khi dùng trong paper**. Nếu tài khoản thuộc nhóm có thể viết `We identified … (reported as Issue #11)`; nếu không, viết `The official SelEx implementation was recently corrected for … (Issue #11)`.
5. Theo lời người mở issue, HypCD/Hyp-SelEx có thể dùng chung phần code bị lỗi. Nếu bảng so sánh có HypCD, cần chú thích phù hợp; không tự khẳng định ảnh hưởng thực tế nếu chưa được xác minh.

## 3. Ablation — Table II

Table II có **7 hàng (a)–(g)**. (a) dùng SelEx có nguồn, (d) lấy từ các full CRiSH seed ở A3; **chỉ cần chạy mới 5 biến thể**: (b), (c), (e), (f), (g). Chuỗi lũy tiến `(a) → (b) → (c) → (d)`.

| Hàng | Ý nghĩa | Proposal | Abstain | Move | CF scoring | Confirmation | Cần chạy mới? |
|---|---|---|---|---|---|---|---|
| (a) | SelEx | — | ✗ | ✗ | ✗ | ✗ | Không, dùng số công bố, ghi nguồn |
| (b) | Evidence-only = evidence + actions | Reciprocal | ✓ | ✓ | ✗ | ✗ | Có |
| (c) | + counterfactual scoring = bỏ Confirmation | Reciprocal | ✓ | ✓ | ✓ | ✗ | Có |
| (d) | Full CRiSH = + confirmation | Reciprocal | ✓ | ✓ | ✓ | ✓ | Không, lấy A3 |
| (e) | Bỏ Abstain | Reciprocal | ✗ | ✓ | ✓ | ✓ | Có |
| (f) | Bỏ Move | Reciprocal | ✓ | ✗ | ✓ | ✓ | Có |
| (g) | Random proposal | Random | ✓ | ✓ | ✓ | ✓ | Có |

### Điều kiện triển khai và chạy

1. Chỉ thêm Reciprocal evidence mà không có action thì supervision không đổi; do đó hàng (b) **gộp evidence + actions**. Cần giải thích điều này với thầy.
2. Mỗi biến thể **chỉ khác thành phần cần kiểm tra**; budget, refresh schedule, seed và epochs giữ như full.
3. **Evidence-only (b):** commit các đề xuất có boundary priority cao nhất, qua cùng constraint cấu trúc; action cố định: có cụm đích đủ support thì **Move**, không có thì **Abstain**. Không áp dụng CF scoring/confirmation.
4. **Random proposal (g):** random thứ tự trong **cùng tập ứng viên hợp lệ**, không random để số action thấp đi một cách giả tạo.
5. RNG của ablation **không làm đổi thứ tự batch train**. Dùng luồng RNG độc lập cho random proposal và kiểm tra khả năng tái lập.
6. Mỗi hàng cần báo cáo **số edit thực sự được nhận trong toàn run**, tránh hiểu nhầm thắng/thua do khác số lượng edit.
7. **Ưu tiên Cars**, ít nhất 2 seed nếu GPU đủ; **CUB 1 seed** nếu còn thời gian. Nếu thiếu GPU, chạy 1 seed và ghi rõ.
8. Tuỳ chọn còn GPU: hàng (h) `random proposal + commit không qua CF`, để làm đối chứng về liều (không bắt buộc).

### Acceptance criteria kỹ thuật cho Codex

- [ ] Mặc định không bật ablation tái hiện logic CRiSH full hiện có; kiểm tra bằng regression test hoặc trace so sánh cố định seed.
- [ ] Có config/CLI tái lập cho 5 biến thể và full; không sửa tay source để đổi biến thể.
- [ ] `no_confirmation` bỏ đúng confirmation nhưng vẫn giữ CF scoring và các điều kiện khác.
- [ ] `no_abstain` bỏ Abstain nhưng không đổi Move, proposal, CF hay confirmation ngoài các thay đổi logic bắt buộc; mô tả cách xử lý tình huống trước đây sẽ Abstain.
- [ ] `no_move` bỏ Move nhưng giữ các thành phần khác và mô tả hệ quả lên action budget.
- [ ] `evidence_only` giữ structural guards, priority và action policy đã quy định, không lén dùng CF/confirmation.
- [ ] `random_proposal` lấy từ cùng candidate pool đủ điều kiện, dùng RNG độc lập, không thay batch order.
- [ ] Mọi refresh ghi `proposed / accepted / rejected`, lý do từ chối, `Keep / Abstain / Move`, action counts và candidate counts với ý nghĩa rõ ràng.
- [ ] Có thể đối chiếu cùng seed/budget/schedule giữa các variant và tính tổng số edit được nhận.
- [ ] Test không dùng nhãn ground truth unlabeled trong quyết định online/action/training/checkpoint selection.

## 4. Việc cần làm — checklist có chủ sở hữu

**Mốc G1:** 20/11/2026 (chốt cấu hình). **Mốc G2:** 25/11/2026 (chốt số). Trạng thái bên dưới là **trạng thái trong tài liệu ngày 17/11**, không đại diện hiện trạng repository ngày chạy Codex.

### Kiên — K1–K6

- [ ] **K1 — 18/11:** Mô tả bằng lời cấu hình adaptive hiện tại: budget, refresh schedule, khác với bản trong draft; định danh code version/commit cho paper. **Xong khi:** Thúy Vi có thể viết Method và Implementation Details từ mô tả.
- [ ] **K2 — 20/11:** Nếu tối ưu, chỉ thử trên Cars, 1 seed, so **final checkpoint** với cấu hình hiện tại. **Xong khi:** chọn một cấu hình cuối tại G1; không claim từ best-All.
- [ ] **K3 — 19/11:** Cài công tắc cho 5 ablation ở mục 3. Kiểm rằng tắt công tắc thì giống full, bật từng cái chỉ đổi đúng thành phần. **Xong khi:** An chạy được từng variant mà không sửa code.
- [ ] **K4 — 21/11, cập nhật tới G2:** Thống kê action trên **mọi run**: số Keep, Abstain, Move được nhận mỗi refresh, theo dataset, tỷ lệ ảnh unlabeled bị chạm. **Xong khi:** xác định các thành phần thực sự hoạt động.
- [ ] **K5 — 24/11:** Hậu kiểm bằng nhãn thật **chỉ để phân tích**: tỷ lệ Move đúng lớp, tỷ lệ Abstain rơi vào ảnh sai so với tỷ lệ sai chung, tỷ lệ discovery/confirmation đồng dấu. **Xong khi:** có số theo dataset và một câu kết luận cho mỗi chỉ số.
- [ ] **K6 — 24/11:** Đo overhead: thời gian epoch có/không có refresh, thời gian từng refresh. **Xong khi:** có số cho phần cài đặt.

### An — A1–A6

- [ ] **A1 — 18/11:** Sổ run chung gồm dataset, seed, code version, config, final checkpoint metrics, thời gian chạy; điền các run đã có. **Xong khi:** mọi số tra cứu được ở một nơi.
- [ ] **A2 — 20/11:** Trong lúc chờ G1, chạy seed 2–3 của cấu hình hiện tại trên CUB và Aircraft. **Xong khi:** nếu G1 giữ cấu hình, các run dùng cho paper được.
- [ ] **A3 — 25/11:** Sau G1, có **≥3 seed** của cấu hình chốt trên CUB/Cars/Aircraft; nếu config đổi thì chạy lại seed 1. **Xong khi:** mean ± std Table I.
- [ ] **A4 — 25/11:** Sau G1, chạy ablation (b), (c), (e), (f), (g) trên Cars, 2 seed nếu GPU đủ; CUB 1 seed nếu còn thời gian. **Xong khi:** đủ hàng Table II cần chạy.
- [ ] **A5 — 18/11:** Ghi lại baseline nhóm từng chạy (config, seed, số), **không chạy thêm**. **Xong khi:** có bằng chứng nếu reviewer hỏi.
- [ ] **A6 — 25/11:** Vẽ All/Old/New theo epoch của run CRiSH chốt, đánh dấu các refresh. **Xong khi:** có hình cho analysis.

### Chung

- [ ] **G2 — 25/11:** Kiên và An kiểm chéo từng con số với log (một người trích log, người kia tổng hợp độc lập) rồi bàn giao Thúy Vi.

## 5. Phân công & bàn giao

| Thành viên | Vai trò | Mã việc | Bàn giao |
|---|---|---|---|
| Kiên | Tối ưu phương pháp, công tắc ablation, phân tích | K1–K6 | Mô tả config cho Thúy Vi (18/11); công tắc cho An (19/11); phân tích K4–K6 (24/11) |
| An | Chạy thí nghiệm, quản lý sổ run | A1–A6 | Sổ run (18/11); số Table I/II (25/11) |

Sau G1 (20/11), **không đổi phương pháp** trừ khi phát hiện lỗi và báo nhóm trước để cân nhắc chạy lại. Mọi số trong paper phải có kiểm tra chéo. Cách cài flag, scripts và logging do người triển khai quyết định dựa trên repository thực tế.

## 6. Tiến độ, GPU và quyết định

| Mốc | Thời gian | Công việc |
|---|---|---|
| Thử nghiệm / tối ưu | 17–20/11 | Kiên tối ưu, viết ablation switches; An cập nhật run registry và seed 2–3 |
| **G1** | **20/11** | Chốt cấu hình paper |
| Chạy kết quả chính | 21–25/11 | Seed và ablation trên cấu hình đã chốt |
| Phân tích | 21–25/11 | K4–K6, overhead, figure |
| **G2** | **25/11** | Chốt số, giao Thúy Vi |
| Thầy review | 27–29/11 | Thầy review bản thảo |
| Nộp | Trước 01/12 (mốc trên sơ đồ: 30/11) | Nộp paper |

Ước lượng thô từ log adaptive đánh giá mỗi epoch: run 200 epoch tối đa khoảng **CUB 11 giờ, Aircraft 16 giờ, Cars 24 giờ**. Sau G1 dự kiến **50–150 GPU-giờ** cho seed (tuỳ có đổi cấu hình) và **~240 GPU-giờ** cho Cars ablation 2 seed. Lịch 21–25/11 giả định **3 GPU** và khá sát; thiếu GPU thì giảm ablation xuống 1 seed hoặc giảm tần suất đánh giá, phải ghi rõ paper. **Đây là ước lượng trong kế hoạch, không phải benchmark đo trực tiếp bởi Codex.**

### G1 — 20/11

Chỉ thay cấu hình adaptive hiện tại khi phiên bản tối ưu cho kết quả **final checkpoint** tốt hơn rõ ràng trên Cars, 1 seed; nếu không thì giữ cấu hình hiện tại, dùng tiếp A2. Ghi cách chọn để báo cáo trung thực.

### G2 — 25/11

So CRiSH final checkpoint **mean của ≥3 seed** với SelEx công bố; ngưỡng **ước lượng thô** theo kế hoạch: khoảng **1 điểm All** với CUB/Cars, **0.4 điểm** với Aircraft.

- Vượt ngưỡng trên ≥2 dataset: có cơ sở giữ claim cải thiện theo kế hoạch, nhưng vẫn cần cân nhắc thống kê/protocol.
- Chỉ Cars vượt: giới hạn claim vào Cars, dùng K4/K5 giải thích.
- Không dataset nào vượt: báo thầy, **không claim cải thiện**.

## 7. Câu hỏi còn mở

- [ ] Nhóm có bao nhiêu GPU, loại gì? Kế hoạch giả định 3 GPU.
- [ ] GitHub account `BanAnA9205` mở Issue #11 có thuộc nhóm không? Quyết định cách ghi nhận đóng góp trong paper.

## 8. Nguồn cần kiểm chứng (theo PDF)

- SelEx Issue #11: người mở, ngày mở và xác nhận tác giả.
- SelEx commit `0cc628d`: bản sửa Issue #11.
- SelEx README bản sửa: kết quả công bố, 50 epoch, mean 3 seed.
- SelEx 2024 `contrastive_training.py`, commit `c3d3b42`: lịch mặc định 200 epoch.
- CRiSH run logs do nhóm gửi ngày 13–17/11/2026.

> **Không tự tạo URL hoặc trích dẫn không tìm thấy trong repository/tài liệu.** Khi cần trích dẫn chính thức, kiểm tra nguồn gốc trước.

---

## 9. Deliverables Codex cần bàn giao

Các đường dẫn sau **chỉ là ví dụ đặt tên output**, Codex chọn vị trí hợp cấu trúc repo hiện hữu:

1. `docs/adaptive_configuration.md`: cấu hình full hiện tại, lịch/budget, sự khác biệt với draft, commit hash, cách tái lập (K1).
2. Config/CLI + README chạy đủ **5 variant ablation** và full, minh hoạ lệnh với Cars seed cố định (K3/A4).
3. Bộ kiểm thử regression, isolation, RNG, action guard, và kiểm tra không leakage nhãn thật.
4. Run registry xuất CSV/JSON với provenance, seed, final metrics và tổng accepted edits (A1/A3/A5).
5. Export số action per-refresh, per-dataset; hậu kiểm offline (K4/K5) có mẫu số, định nghĩa rõ; timing overhead (K6).
6. Script vẽ All/Old/New theo epoch, refresh markers, output file hình với caption/metadata (A6).
7. `docs/implementation_audit.md` gồm checklist từng mã, những gì có thật trong repo, test evidence và lệnh chạy lại.

**Quy tắc hoàn thành:** Nếu chưa có GPU/log/ground truth phù hợp, đánh dấu `BLOCKED / NOT RUN / NOT VERIFIED`, không chế tạo kết quả. Nếu một mục trong PDF mâu thuẫn với implementation hiện có, mô tả sự khác biệt và đề xuất cách giải quyết trước khi làm thay đổi hành vi mặc định. Không tự động sửa nội dung paper hoặc sửa số baseline công bố.
