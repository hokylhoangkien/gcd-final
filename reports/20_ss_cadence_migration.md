# Chuyển SS từ mỗi 10 epoch sang mỗi epoch, giữ tiến độ

2026-11-15. User requested SS every epoch; earlier assistant-provided commands
incorrectly selected --eval-every 10. Remote source run:
`20261115T110206Z-a01d46fa`, last user-observed checkpoint zero-based epoch2
(three completed epochs), 63 batches/epoch, train-loop137.44s at epoch2,
peak allocated15,826,872,832 bytes. This is a user-supplied observation,
not an independent remote process inspection. Local runtime remains unchanged.

## Công cụ đã chuẩn bị

`scripts/change_ss_cadence.py` tạo một thư mục chuyển tiếp mới:

- Từ chối nếu worker/evaluator Linux còn tham chiếu source run. Không gửi
  signal, không dừng job và không cập nhật source/config đang chạy.
- Kiểm code fingerprint/config gốc khớp checkpoint. Không bypass resume guard.
- Sao lưu model/head/optimizer/scheduler/Python/NumPy/Torch/CUDA RNG state;
  chỉ đổi `config.ss_eval_every_epochs` trên bản sao.
- Giữ báo cáo plain-test và SS best-so-far. Điều chỉnh config-bound SelEx
  best bundle trên bản sao, không sửa best weights hay điểm số.
- Ghi source checksum, config cũ và trường đổi trong migration.json và
  checkpoint chuyển tiếp. Không ghi đè resume.pt/config.json/best của run cũ.
- Từ chối nếu artifact selection đi trước checkpoint hoặc thiếu lịch sử;
  không tự reset best hoặc âm thầm bỏ kiểm tra tính toàn vẹn.

Cần dừng job an toàn **sau checkpoint boundary** trước khi chạy công cụ.
Không dừng giữa plain evaluation, vì best bundle có thể mới hơn resume.pt.
Chưa có sự đồng ý dừng job ở thời điểm chuẩn bị tài liệu này; không có
remote job nào được agent dừng hoặc khởi động lại.

Sau khi dừng và xác nhận source job/evaluator đã thoát, trên máy GPU lấy
commit có helper (runtime fingerprint không đổi), rồi:

```bash
.venv/bin/python scripts/change_ss_cadence.py \
  --source-run runs/20261115T110206Z-a01d46fa \
  --output runs/ss-every1-migration --every 1

# Chạy trong tmux; launcher bỏ smoke lặp lại khi đã có --resume,
# vẫn chạy data/CUDA preflight và giữ strict config/source guard.
bash scripts/train_selex.sh \
  --config runs/ss-every1-migration/config.json \
  --variant local --eval-every 1 \
  --resume runs/ss-every1-migration/resume.pt
```

Output phải là thư mục chưa tồn tại. Nếu source đã tiến thêm nhiều epoch,
helper lấy checkpoint thực tế lúc đã dừng; không khóa vào epoch2 trong log cũ.
Không tự sửa interval trong config gốc hoặc edit trực tiếp checkpoint gốc.

Runtime resume sẽ SS checkpoint gần nhất nếu đến lịch và chưa có report,
rồi tiếp tục epoch kế. Các epoch cũ thiếu weights sẽ vẫn thiếu SS, không
được điền từ checkpoint khác. Ghi rõ thay đổi cadence khi báo cáo best SS;
fixed-final SS vẫn là metric chính, không gán best từ tần suất khác thành
đối chứng khớp hoàn toàn. Không thay epoch tổng, LR schedule, seed hay loss.

## Kiểm chứng

Bounded CPU command, 60s subprocess limit:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_change_ss_cadence.py -q --tb=short
```

**3 passed in 1.66s**. Tests warm SGD momentum before saving, verify original
files remain byte-identical, migrated optimizer/scheduler/model match, best
history restores and missing latest SS is replayed before next SGD. They also
check active-worker refusal and different-source/config refusal. No real GPU
checkpoint migrated here; no stop/restart or loss of remote training progress.
