# Hướng dẫn train đã hợp nhất vào README

Từ 12/11/2026, **[README.md](../README.md) là hướng dẫn train chính duy nhất**
cho cả Stanford Cars, CUB và FGVC-Aircraft: SelEx baseline và CRiSH local
Keep/Abstain/Move, SS mỗi epoch, best All/Old/New, tmux, theo dõi và resume.

Không cần đọc thêm báo cáo để chạy. Nội dung hướng dẫn cũ đã được chuyển và
cập nhật trong README; bản lịch sử vẫn truy được qua Git. File này giữ làm
đường dẫn chuyển tiếp cho các liên kết cũ.

Thay đổi này chỉ hợp nhất tài liệu; không đổi runtime/config hoặc tác động
job đang chạy. Raw JSON 200 epoch vẫn mặc định policy: lệnh cho thuật toán
local trong README luôn có `--variant local`.
