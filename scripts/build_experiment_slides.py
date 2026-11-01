#!/usr/bin/env python3
"""Build a Vietnamese experiment report from verified local results; no training.

Requires python-pptx, reportlab, PyMuPDF, Pillow, matplotlib. Dependencies may
live outside the project environment (e.g. a temporary PYTHONPATH directory).
PowerPoint text and shapes remain editable; the history figure is an image.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont
from matplotlib import get_data_path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/slides/proposed_dinov2_seed1"
RESULT = ROOT / "results/cars_proposed_dinov2_seed1_user_reported_evaluation.json"
FIGURE = ROOT / "reports/figures/proposed_dinov2_seed1/history.png"
HISTORY = ROOT / "reports/figures/proposed_dinov2_seed1/history.csv"
BACKUP = ROOT / "logs/final-backup-20261101/gcd-runs-v2"
TRAIN = BACKUP / "20261031T055221Z-03368e3d"
PAPER = "https://arxiv.org/html/2504.06120v1#S4.T1"
W, H = 960, 540
INK, MUTED = "142B45", "586B7E"
BLUE, TEAL, ORANGE = "215CBD", "087F8C", "B96917"
BG, LIGHT, GREEN, AMBER = "F6F8FB", "EAF0FA", "E7F4F1", "FFF2DB"
FONTS = Path(get_data_path()) / "fonts/ttf"
ARIAL = Path("/System/Library/Fonts/Supplemental")
HAS_ARIAL = (ARIAL / "Arial.ttf").is_file() and (ARIAL / "Arial Bold.ttf").is_file()
PPT_FONT = "Arial" if HAS_ARIAL else "DejaVu Sans"
pdfmetrics.registerFont(TTFont("Deck", str(ARIAL / "Arial.ttf" if HAS_ARIAL else FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DeckBold", str(ARIAL / "Arial Bold.ttf" if HAS_ARIAL else FONTS / "DejaVuSans-Bold.ttf")))


def vn(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def wrap(text, width, size, bold=False):
    font = "DeckBold" if bold else "Deck"
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            trial = (current + " " + word).strip()
            if pdfmetrics.stringWidth(trial, font, size) > width * .96:
                if current:
                    lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    return lines


class Slide:
    def __init__(self, title, notes):
        self.title, self.notes, self.elements = title, notes, []

    def rect(self, x, y, w, h, fill, radius=8):
        self.elements.append(dict(kind="rect", x=x, y=y, w=w, h=h,
                                  fill=fill, radius=radius))

    def text(self, x, y, w, h, text, size=20, color=INK, bold=False,
             align="left", link=None):
        lines = wrap(text, w, size, bold)
        assert len(lines) * size * 1.20 <= h + 1, (self.title, text, lines, h)
        self.elements.append(dict(kind="text", x=x, y=y, w=w, h=h,
                                  text="\n".join(lines), size=size, color=color,
                                  bold=bold, align=align, link=link))

    def image(self, path, x, y, w, h):
        self.elements.append(dict(kind="image", path=path, x=x, y=y, w=w, h=h))

    def heading(self, kicker="BÁO CÁO THỰC NGHIỆM"):
        self.text(42, 25, 860, 16, kicker, 10, TEAL, True)
        self.text(42, 51, 876, 50, self.title, 29, INK, True)


def table(s, x, y, widths, rows, row_height=46, highlight=None, sizes=None):
    for r, row in enumerate(rows):
        fill = INK if r == 0 else (GREEN if r == highlight else "FFFFFF")
        s.rect(x, y + r * row_height, sum(widths), row_height - 2, fill, 3)
        xx = x
        for c, text in enumerate(row):
            font = sizes[r] if sizes else (14 if r == 0 else 17)
            s.text(xx + 12, y + r * row_height + 11, widths[c] - 24,
                   row_height - 12, text, font,
                   "FFFFFF" if r == 0 else INK,
                   bold=r == 0 or r == highlight,
                   align="left" if c == 0 else "center")
            xx += widths[c]


def build_content():
    final = json.loads(RESULT.read_text())
    raw = json.loads((BACKUP / "20261031T172513Z-881c7f46/evaluation.json").read_text())
    assert final["protocol"] == "balanced_ss_kmeans_train_unlabeled"
    for key in ("all_acc", "old_acc", "new_acc"):
        assert final[key] == raw[key]
    assert final["completed_epochs"] == 200 and final["seed"] == 1
    config = json.loads((TRAIN / "config.json").read_text())
    readiness = json.loads((TRAIN / "readiness.json").read_text())
    assert (config["epochs"], config["batch_size"], config["grad_from_block"]) == (200, 128, 10)
    assert readiness["data_counts"] == dict(labeled=2000, unlabeled=6144, disjoint_test=8041)
    scores = [100 * final[key] for key in ("all_acc", "old_acc", "new_acc")]
    paper_scores = [83.8, 93.3, 79.2]
    history = list(csv.DictReader(HISTORY.open()))
    groups = []
    for lo, hi in ((70, 89), (170, 189)):
        rows = [r for r in history if r["kind"] == "monitor" and lo <= int(r["completed_epochs"]) <= hi]
        assert len(rows) == 20
        groups.append([mean(float(r[f"{key}_percent"]) for r in rows) for key in ("All", "Old", "New")])

    slides = []
    s = Slide("Reliability weighting cho Hyp-SelEx", """Em khảo sát việc bổ sung trọng số độ tin cậy vào Hyp-SelEx cho bài toán GCD. Đây là kết quả một lần chạy với seed 1 trên Stanford Cars, không phải kết luận phương pháp đã tốt hơn baseline. Đã chạy đủ 200 epoch và đánh giá riêng checkpoint cuối bằng semi-supervised k-means. Ba chỉ số cuối là All 81,23%, Old 86,66%, New 78,61%. All là chỉ số chính. Tên hướng nghiên cứu còn là tên làm việc; tính mới chưa được xác lập.""")
    s.text(42, 36, 876, 18, "BÁO CÁO KẾT QUẢ THỰC NGHIỆM  •  01/11/2026", 11, TEAL, True)
    s.text(42, 90, 876, 108, "Reliability weighting\ncho Hyp-SelEx", 40, INK, True)
    s.text(42, 215, 860, 54, "Khảo sát độ tin cậy của cặp học trong\nGeneralized Category Discovery (GCD).", 21, MUTED)
    s.text(42, 295, 860, 24, "Stanford Cars SSB  •  DINOv2  •  200 epoch  •  1 seed", 16, INK, True)
    for i, (name, value, color) in enumerate(zip(("All — chỉ số chính", "Old — lớp đã biết", "New — lớp mới"), scores, (BLUE, INK, TEAL))):
        x = 42 + i * 298
        s.rect(x, 338, 280, 121, "FFFFFF")
        s.text(x + 18, 354, 250, 24, name, 15, MUTED)
        s.text(x + 18, 391, 250, 56, f"{vn(value)}%", 39, color, True)
    s.text(42, 479, 876, 18, "Kết quả sơ bộ; chưa có baseline đối chứng cùng điều kiện.", 13, MUTED)
    slides.append(s)

    s = Slide("Bài toán và phần mở rộng", """Trong GCD, tập có nhãn chỉ bao phủ các lớp đã biết, còn tập chưa nhãn chứa cả lớp đã biết và lớp mới. Mục tiêu là phân nhóm đúng toàn bộ ảnh chưa nhãn. DINOv2, biểu diễn hyperbolic và hierarchy là phần kế thừa từ Hyp-SelEx/HypCD. Phần em bổ sung là cách tính trọng số cho một số cặp pseudo-positive trong loss ở từng mức hierarchy. Giả thuyết là quan hệ lân cận ổn định đáng được tin cậy hơn, nhưng cần thực nghiệm để kiểm chứng. Không tạo classifier confidence và không dùng nhãn thật của ảnh chưa nhãn để tính score.""")
    s.heading()
    s.text(42, 113, 876, 65, "GCD: dùng ảnh có nhãn để hỗ trợ phân nhóm ảnh chưa nhãn, gồm cả lớp đã biết và lớp mới.", 21)
    boxes = [(42, "DINOv2", "Trích xuất đặc trưng", LIGHT),
             (340, "Hyp-SelEx", "Hyperbolic + hierarchy", LIGHT),
             (638, "Loss có trọng số", "Pseudo-positive theo level", GREEN)]
    for x, title, body, color in boxes:
        s.rect(x, 208, 280, 113, color)
        s.text(x + 16, 226, 250, 32, title, 21, bold=True)
        s.text(x + 16, 274, 250, 33, body, 14, MUTED)
    for x in (323, 621):
        s.text(x, 245, 18, 29, "→", 21, TEAL, True, "center")
    s.rect(42, 351, 876, 114, "FFFFFF")
    s.text(60, 369, 838, 28, "Phần bổ sung: reliability của các cặp học", 21, TEAL, True)
    s.text(60, 411, 838, 46, "Ước lượng từ đặc trưng và hai augmentations; giữ nhánh supervised loss trên nhãn thật và hình học gốc.", 17)
    slides.append(s)

    s = Slide("Reliability hoạt động như thế nào?", """Mỗi ảnh có hai view được biến đổi ngẫu nhiên. Ở mỗi mức hierarchy, module xây lân cận theo khoảng cách hyperbolic. Cue thứ nhất là reciprocal top-k: hai ảnh có chọn nhau hay không. Cue thứ hai đo tính nhất quán của lân cận giữa hai view. Khi bật cả hai, score là tích của hai cue, nằm trong [0,1] và được detach. Trọng số chỉ áp dụng cho pseudo-positive khác sample ID và có ít nhất một đầu chưa nhãn. Khi hoạt động đầy đủ, weight bằng 0,1 + 0,9 × score, trừ các cặp được fallback bảo vệ. Trong ramp: w = 1 − alpha + alpha × (0,1 + 0,9 × score). Loss chuẩn hóa tổng trọng số positive theo từng anchor. Chỉ số epoch bắt đầu từ 0: 0–9 tắt, 10 bắt đầu, 29 đầy đủ. Không có tham số học thêm trong reliability.""")
    s.heading()
    for x, title, body in ((42, "01  Lân cận tương hỗ", "Hai ảnh chọn nhau trong top-5 láng giềng theo khoảng cách hyperbolic."),
                           (490, "02  Ổn định giữa hai view", "Lân cận còn nhất quán khi cùng ảnh được biến đổi ngẫu nhiên hai lần.")):
        s.rect(x, 117, 428, 130, "FFFFFF")
        s.text(x + 16, 134, 396, 31, title, 20, TEAL, True)
        s.text(x + 16, 179, 396, 63, body, 16)
    s.rect(42, 271, 876, 65, GREEN)
    s.text(60, 290, 840, 35, "Score [0,1]   →   trọng số cặp   →   loss pseudo-positive", 22, INK, True, "center")
    s.text(42, 360, 876, 52, "Chỉ áp dụng cặp pseudo-positive có ít nhất một ảnh chưa nhãn.\n10 epoch đầu: tắt → tăng dần 20 epoch → đầy đủ từ index 29.", 17)
    s.text(42, 437, 876, 55, "Floor 0,1; fallback giữ weight = 1 khi thiếu hỗ trợ.\nScore không sử dụng nhãn thật của ảnh chưa nhãn.", 15, MUTED)
    slides.append(s)

    s = Slide("Thiết lập và cách tính kết quả", """Split SSB đã được kiểm tra: 98 lớp known, 98 lớp novel. Có 2.000 ảnh labeled và 6.144 ảnh unlabeled trong train; 50% ảnh train thuộc known được cấp nhãn. DINOv2 ViT-B/14 có bốn register tokens, fine-tune hai block cuối và projection head. Chỉ số chính dùng balanced semi-supervised k-means trên train-unlabeled với labeled anchors, Hungarian-v2, biết trước tổng số 196 lớp. Đây là đánh giá transductive của GCD, không phải validation hay khả năng tổng quát trên ảnh mới. Nhãn thật của ảnh unlabeled chỉ dùng tính accuracy sau dự đoán. All được tính theo số mẫu, không phải trung bình cộng Old và New. Có một disjoint test riêng 8.041 ảnh nhưng báo cáo bằng plain k-means, cần giữ tách biệt.""")
    s.heading()
    fields = [("Dữ liệu", "Stanford Cars • SSB"), ("Số lớp", "98 Old / 98 New"),
              ("Ảnh train", "2.000 có nhãn + 6.144 chưa nhãn"),
              ("Backbone", "DINOv2 ViT-B/14 • reg4"),
              ("Training", "200 epoch • batch 128 • seed 1"), ("GPU", "V100 32GB • FP32")]
    for i, (label, value) in enumerate(fields):
        y = 117 + i * 48
        s.rect(42, y, 420, 44, "FFFFFF", 3)
        s.text(54, y + 12, 95, 24, label, 14, MUTED, True)
        s.text(151, y + 12, 302, 25, value, 14)
    s.rect(486, 117, 432, 286, LIGHT)
    s.text(504, 136, 396, 32, "Chỉ số chính: All (%)", 22, BLUE, True)
    s.text(504, 184, 396, 89, "Balanced SS-k-means trên 6.144 ảnh train-unlabeled, có labeled anchors.", 19)
    s.text(504, 285, 396, 88, "Old: accuracy của lớp đã biết.\nNew: accuracy của lớp mới.\nAll: tính theo số ảnh toàn tập.", 17)
    s.rect(486, 411, 432, 34, AMBER, 3)
    s.text(501, 420, 402, 22, "Đây không phải validation accuracy.", 14, ORANGE, True)
    s.text(42, 461, 876, 43, "SGD • lr 0,1 • weight decay 5e−5 • fine-tune 2 block cuối • c = 0,1 • smoothing = 0,5.\nHungarian-v2; biết trước tổng số 196 lớp; nhãn thật unlabeled chỉ dùng chấm điểm.", 12, MUTED)
    slides.append(s)

    s = Slide(f"Kết quả cuối: All {vn(scores[0])}%", """Đây là kết quả đánh giá riêng checkpoint cuối sau 200 epoch, không chọn epoch tốt nhất sau khi nhìn đường cong. Dòng paper chỉ là mốc tham khảo. Chênh lệch ở All lớn nhất cần quan tâm, còn New gần số paper không chứng minh phương pháp tốt hơn. Chưa thể quy mức giảm cho reliability vì khác điều kiện huấn luyện và chọn checkpoint. Đặc biệt, log cuối có Train All 51,77% và Test All 56,24% do gọi plain k-means. Sau đó đánh giá lại đúng SS-k-means thu được bộ số trong bảng. Các số này thuộc những protocol khác nhau, không có bằng chứng mô hình đột ngột sụp từ 81% xuống 51%. Cần áp dụng cùng bản sửa k-means và cùng quy trình evaluation cho baseline đối chứng.""")
    s.heading()
    s.text(42, 109, 876, 28, "Checkpoint sau 200 epoch • SS-k-means • chọn final_epoch", 17, MUTED)
    rows = [["Kết quả", "All (%)", "Old (%)", "New (%)"],
            ["Đề xuất • đo thực tế, seed 1", *(vn(v) for v in scores)],
            ["Hyp-SelEx • paper, tham khảo", *(vn(v, 1) for v in paper_scores)],
            ["Chênh lệch so với paper (điểm %)", *(vn(a-b) for a,b in zip(scores, paper_scores))]]
    table(s, 42, 153, [468, 136, 136, 136], rows, 46, 1, [14, 19, 17, 15])
    s.rect(42, 356, 876, 98, AMBER)
    s.text(58, 370, 844, 29, "Chưa phải so sánh đối chứng công bằng.", 20, ORANGE, True)
    s.text(58, 408, 844, 43, "Khác số block fine-tune (2/3), smoothing (0,5/1,0), curvature (0,1/0,05), cách chọn checkpoint và xử lý k-means.", 15)
    s.text(42, 467, 876, 20, "Số 51–56% trong log cuối là plain k-means; không so trực tiếp với bảng này.", 12, MUTED)
    s.text(42, 491, 876, 16, "Nguồn: evaluation.json cuối; Liu et al., Hyperbolic Category Discovery, Table 1 (DINOv2, Cars).", 10, MUTED, link=PAPER)
    slides.append(s)

    s = Slide("Tăng sớm, sau đó chững và giảm nhẹ", """Đường cong lấy trực tiếp từ log hai lần chạy nối tiếp, loại plain-k-means ở record 199. Trục ngang là số epoch đã hoàn tất lúc đo: metrics SS ở index e được đo trước khi train e. Ngôi sao là đánh giá SS riêng sau 200 epoch. Đường đậm là trung bình trượt 10 lần đánh giá; đường mờ là số đo từng lần. Từ nhóm 70–89 sang 170–189, Old giảm rõ trong khi New chỉ cải thiện nhẹ, khiến All giảm. Đây là mô tả một lần chạy, không chứng minh overfitting hay reliability là nguyên nhân. Lịch trộn loss gốc vẫn thay đổi trong quá trình train. Dao động giữa epoch không phải độ bất định giữa các seed, và không được chọn đỉnh đường cong để thay kết quả cuối một cách hậu nghiệm.""")
    s.heading()
    s.image(FIGURE, 42, 104, 876, 319)
    s.text(42, 430, 876, 20, "Trung bình nhóm epoch 70–89 → 170–189 (cùng protocol SS-k-means):", 13, MUTED)
    for i, (name, before, after, color) in enumerate(zip(("All", "Old", "New"), groups[0], groups[1], (BLUE, INK, TEAL))):
        x = 42 + 298 * i
        s.rect(x, 460, 280, 41, "FFFFFF", 4)
        s.text(x + 12, 470, 258, 25, f"{name}: {vn(before)} → {vn(after)}%", 17, color, True)
    slides.append(s)

    s = Slide("Reliability có tác động, chưa rõ lợi ích", """Các thống kê này lấy từ diagnostics ở epoch index 189–198, 20 snapshots mỗi level. Mean weight được gộp theo số eligible ordered pairs; fallback là trung bình theo snapshot. Level 0 và level 6 có giảm trọng số quan sát được, còn level 1–3 thường gần weight 1 và fallback rất cao. Fallback là cơ chế bảo vệ khi anchor không có eligible positive với score dương; nó giữ weight 1 cho các cặp liên quan. Đây không phải tỷ lệ cặp sai hay xác suất nhãn đúng. Mỗi epoch chỉ log batch 0 và 50 trên 63 batch. Level 6 có hệ số loss 1/128 và loss chuẩn hóa theo anchor: raw weight không thể chuyển trực tiếp thành ảnh hưởng gradient. Muốn biết score chọn đúng cặp nhiễu cần audit offline trên checkpoint cố định; muốn biết accuracy tăng nhờ module cần baseline/ablation cùng điều kiện.""")
    s.heading()
    s.text(42, 112, 520, 28, "Diagnostics cuối run • epoch index 189–198", 17, MUTED)
    rows = [["Level", "Weight TB", "Giảm weight", "Fallback"],
            ["0", "0,913", "37,47%", "73,44%"],
            ["1–3", "0,987–0,994", "1,60–2,69%", "94,69–98,16%"],
            ["6", "0,793", "23,51%", "61,64%"]]
    table(s, 42, 156, [66, 132, 145, 167], rows, 44, None, [12, 16, 14, 16])
    s.text(42, 348, 508, 45, "Weight TB = trọng số trung bình của cặp đủ điều kiện.\nFallback = tỷ lệ anchor kích hoạt cơ chế bảo vệ.", 12, MUTED)
    s.rect(580, 113, 338, 119, GREEN)
    s.text(596, 129, 306, 28, "Đã thấy", 20, TEAL, True)
    s.text(596, 170, 306, 55, "Trọng số thay đổi tại một số level.", 17)
    s.rect(580, 249, 338, 146, "FFFFFF")
    s.text(596, 266, 306, 28, "Chưa chứng minh", 20, ORANGE, True)
    s.text(596, 309, 306, 75, "Cặp bị giảm là cặp nhiễu; accuracy tăng nhờ module.", 17)
    s.rect(42, 421, 876, 70, LIGHT)
    s.text(58, 435, 844, 48, "Chỉ lấy mẫu batch 0 và 50; chưa đại diện chắc chắn toàn bộ dữ liệu. Fallback cao khiến nhiều cặp vẫn giữ weight = 1.", 16)
    slides.append(s)

    s = Slide("Kết luận và bước tiếp theo", """Em đã triển khai một module có thể bật tắt, hoàn thành thí nghiệm 200 epoch, lưu checkpoint và đánh giá lại đúng protocol. Kết quả hiện tại chưa chứng minh cải tiến thuật toán: chỉ một seed, chưa có matched baseline, chưa ablation; tính mới cần đối chiếu sâu hơn với các hướng reliability, neighborhood và hierarchy đã có. Bản sửa empty cluster là xử lý kỹ thuật, không phải đóng góp học thuật; cần dùng nhất quán trong đối chứng. Bước ít tốn tài nguyên trước mắt là rà soát phương pháp và audit offline chất lượng cặp trên checkpoint cố định, không dùng nhãn evaluation để tune. Sau đó xin thống nhất với thầy thiết kế baseline và ablation trong ngân sách. Kết quả âm hoặc không cải thiện vẫn cần được báo cáo trung thực.""")
    s.heading()
    items = [("01", "Đã hoàn thành thí nghiệm có thể kiểm tra", "Đủ 200 epoch; có checkpoint, log và final evaluation đã xác minh."),
             ("02", "Chưa đủ bằng chứng về cải tiến", "Một seed; chưa baseline cùng điều kiện, chưa ablation; tính mới chưa xác lập."),
             ("03", "Ưu tiên kiểm chứng trước khi chạy thêm", "Audit offline chất lượng cặp; lập baseline và ablation theo ngân sách.")]
    for i, (n, title, body) in enumerate(items):
        y = 118 + i * 88
        s.rect(42, y, 43, 38, GREEN, 5)
        s.text(48, y + 8, 32, 25, n, 17, TEAL, True, "center")
        s.text(103, y, 806, 32, title, 21, bold=True)
        s.text(103, y + 38, 806, 45, body, 15, MUTED)
    s.rect(42, 398, 876, 81, LIGHT)
    s.text(59, 413, 840, 58, "Xin ý kiến thầy: ưu tiên tái lập baseline hay điều chỉnh reliability sau khi audit chất lượng cặp?", 20, BLUE, True)
    s.text(42, 490, 876, 17, "Tài liệu: HypCD (CVPR 2025), code chính thức; báo cáo và dữ liệu kết quả lưu cùng dự án.", 10, MUTED, link="https://github.com/Visual-AI/HypCD")
    slides.append(s)
    return slides, final, paper_scores


def export(slides):
    OUT.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W / 72), Inches(H / 72)
    prs.core_properties.title = "Reliability weighting cho Hyp-SelEx — Kết quả thực nghiệm"
    prs.core_properties.subject = "Stanford Cars SSB, DINOv2, seed 1, final 200-epoch evaluation"
    prs.core_properties.author = "gcd-final"
    pdf = canvas.Canvas(str(OUT / "bao_cao_thuc_nghiem.pdf"), pagesize=(W, H))
    pdf.setTitle(prs.core_properties.title)
    bounds = []
    for index, content in enumerate(slides, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string(BG)
        pdf.setFillColorRGB(*(int(BG[i:i+2], 16)/255 for i in (0,2,4)))
        pdf.rect(0, 0, W, H, fill=1, stroke=0)
        footer = Slide("Footer", "")
        footer.text(42, 520, 800, 14, "GCD  •  Stanford Cars  •  Kết quả một seed  •  01/11/2026", 9, MUTED)
        footer.text(867, 518, 51, 16, f"{index:02d} / {len(slides):02d}", 10, MUTED, align="right")
        for e in content.elements + footer.elements:
            x,y,w,h = (e[k] for k in ("x", "y", "w", "h"))
            assert x >= 0 and y >= 0 and x+w <= W+.1 and y+h <= H+.1, e
            bounds.append({"slide": index, "kind": e["kind"], "bounds": [x,y,w,h]})
            if e["kind"] == "rect":
                shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                    Pt(x), Pt(y), Pt(w), Pt(h))
                shape.adjustments[0] = min(e["radius"] / min(w,h), .15)
                shape.fill.solid()
                shape.fill.fore_color.rgb = RGBColor.from_string(e["fill"])
                shape.line.fill.background()
                pdf.setFillColorRGB(*(int(e["fill"][i:i+2],16)/255 for i in (0,2,4)))
                pdf.roundRect(x, H-y-h, w, h, e["radius"], fill=1, stroke=0)
            elif e["kind"] == "image":
                slide.shapes.add_picture(str(e["path"]), Pt(x), Pt(y), Pt(w), Pt(h))
                pdf.drawImage(str(e["path"]), x, H-y-h, w, h)
            else:
                tf = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h)).text_frame
                tf.clear()
                tf.word_wrap = False
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                tf.vertical_anchor = MSO_ANCHOR.TOP
                for line_index, line in enumerate(e["text"].split("\n")):
                    p = tf.paragraphs[0] if line_index == 0 else tf.add_paragraph()
                    p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[e["align"]]
                    p.space_before = p.space_after = Pt(0)
                    p.line_spacing = Pt(e["size"] * 1.20)
                    run = p.add_run()
                    run.text = line
                    run.font.name = PPT_FONT
                    run.font.size = Pt(e["size"])
                    run.font.bold = e["bold"]
                    run.font.color.rgb = RGBColor.from_string(e["color"])
                    if e["link"]:
                        run.hyperlink.address = e["link"]
                    pdf.setFont("DeckBold" if e["bold"] else "Deck", e["size"])
                    pdf.setFillColorRGB(*(int(e["color"][i:i+2],16)/255 for i in (0,2,4)))
                    yy = H-y-e["size"]*.96-line_index*e["size"]*1.20
                    xx = x if e["align"] == "left" else (x+w/2 if e["align"] == "center" else x+w)
                    {"left": pdf.drawString, "center": pdf.drawCentredString, "right": pdf.drawRightString}[e["align"]](xx, yy, line)
                if e["link"]:
                    pdf.linkURL(e["link"], (x,H-y-h,x+w,H-y), relative=0)
        slide.notes_slide.notes_text_frame.text = content.notes
        pdf.showPage()
    pdf.save()
    prs.save(OUT / "bao_cao_thuc_nghiem.pptx")
    # Read the generated files again and render actual PDF pages for inspection.
    loaded = Presentation(OUT / "bao_cao_thuc_nghiem.pptx")
    assert len(loaded.slides) == len(slides)
    doc = pymupdf.open(OUT / "bao_cao_thuc_nghiem.pdf")
    assert len(doc) == len(slides)
    previews = OUT / "preview"
    previews.mkdir(exist_ok=True)
    all_text = "\n".join(page.get_text() for page in doc)
    for label in ("81,23%", "86,66%", "78,61%", "83,8", "93,3", "79,2"):
        assert label in all_text, label
    thumbs = []
    for i, page in enumerate(doc, 1):
        page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(previews / f"slide_{i:02d}.png")
        with Image.open(previews / f"slide_{i:02d}.png") as im:
            thumbs.append(im.resize((640,360)))
    sheet = Image.new("RGB", (1328, 4*398+20), "#DDE4EE")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONTS / "DejaVuSans.ttf"), 16)
    for i, im in enumerate(thumbs):
        x, y = 16 + (i%2)*656, 14 + (i//2)*398
        draw.text((x,y), f"{i+1:02d}  {slides[i].title}", fill="#142B45", font=font)
        sheet.paste(im, (x,y+25))
    sheet.save(OUT / "tong_quan_slide.png")
    (OUT / "slide_text.txt").write_text(all_text, encoding="utf-8")
    (OUT / "layout_check.json").write_text(json.dumps({"slides": len(slides), "element_bounds_valid": True, "elements": len(bounds), "pdf_text_checked": True}, indent=2)+"\n")


def write_notes(slides, final, paper_scores):
    lines = ["# Lời dẫn báo cáo thực nghiệm", "", f"Bộ 8 slide, khoảng 5–7 phút. Có thể sửa chữ, bảng và hình khối trong PowerPoint. Font: {PPT_FONT}; bản PDF đã nhúng font. Lời dẫn cũng có trong speaker notes của PowerPoint.", ""]
    for i, s in enumerate(slides, 1):
        lines += [f"## Slide {i} — {s.title}", "", s.notes, ""]
    lines += ["## Nguồn và phạm vi", "",
              "- Kết quả đo: `results/cars_proposed_dinov2_seed1_user_reported_evaluation.json`, đối chiếu raw `evaluation.json` cuối trong backup.",
              "- Cấu hình và số mẫu: raw `config.json`, `readiness.json` của run huấn luyện đã resume.",
              "- Đường cong: `reports/figures/proposed_dinov2_seed1/history.csv` và `history.png`; gồm 199 số đo SS trước train và một đánh giá cuối sau 200 epoch. Không trộn record plain-k-means index 199.",
              "- Diagnostics: `logs/final-backup-20261101/history-audit/report.md`, epoch index 189–198; batch 0 và 50. Các mức 1–3 trình bày khoảng min–max giữa level, không phải độ lệch chuẩn.",
              f"- Liu, He, Han. *Hyperbolic Category Discovery*, CVPR 2025, [Table 1 và mục 6.2]({PAPER}); số tham khảo DINOv2 trên Cars: 83.8 / 93.3 / 79.2. Đã đối chiếu trang tác giả ngày 2026-11-01.",
              "- [Mã nguồn HypCD chính thức](https://github.com/Visual-AI/HypCD), upstream pin `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`.",
              "- `reports/13_final_assessment.md`, `reports/09_kmeans_failure.md`, `reports/06_implemented_method.md`.",
              "", "Không có matched baseline hoàn tất, ablation hay thống kê multi-seed. Không kết luận tính mới, SOTA hoặc readiness để công bố. Không chạy thêm huấn luyện để tạo slide.", ""]
    (OUT / "loi_dan.md").write_text("\n".join(lines), encoding="utf-8")
    sources = [RESULT, FIGURE, HISTORY, TRAIN / "config.json", TRAIN / "readiness.json", ROOT / "logs/final-backup-20261101/history-audit/report.md"]
    manifest = {"observed_result": final, "paper_reference": {"url": PAPER, "scores_percent": paper_scores, "verified_date": "2026-11-01", "comparison": "reference only, unmatched settings"},
                "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUT / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    content, result, reference = build_content()
    export(content)
    write_notes(content, result, reference)
    print(f"Created {len(content)} slides in {OUT.relative_to(ROOT)}")
