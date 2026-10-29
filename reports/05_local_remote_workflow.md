> Cập nhật bàn giao: trainer trực tiếp đã thay adapter; bootstrap hỗ trợ Python3.9–3.11. Dùng RUNPOD.md và reports/07_release_readiness.md cho lệnh hiện hành. Phần dưới giữ lịch sử chuẩn bị ban đầu.

# Phát triển cục bộ, huấn luyện từ xa

## Trạng thái và phạm vi
Mac chỉ dùng kiểm thử nhỏ; NVIDIA Linux là đích train. Không có GPU cloud được thuê hoặc training được chạy trong phiên này. CUDA và full baseline remote **chưa xác minh**. B0 runtime hiện chọn DINO v1 và các tham số Cars từ train script pin; vẫn còn khác biệt paper/code trong report01. Chưa kết luận faithful paper reproduction.

`gcd_runtime/` bổ sung runtime, không sửa upstream. `method=baseline` dùng loss gốc; `method=proposed` hiện chủ động báo chưa triển khai vì novelty gate chưa thông qua. Chưa có reliability module dùng trong train. DINOv2 chưa hỗ trợ; adapter bỏ import DINOv2 trong bộ chạy v1 để không ép cài xformers khi không dùng. Thay private sklearn.utils._joblib bằng public joblib qua import shim. Shim kmeans bỏ cờ tqdm_flag mà PyPI0.3 không hỗ trợ; chỉ tác động giao diện hiển thị, chưa chứng minh parity với dependency GitHub tác giả dùng. Các thay đổi đều công khai, không thay loss/geometry baseline.

## Cấu trúc và thiết lập
- `configs/cars_baseline_dino_v1.json`: cấu hình đóng băng, seed 1, 200 epoch, batch128, max runtime 86400s. Không tự sweep.
- `GCD_DATA_ROOT`: mặc định `datasets/` tính từ root dự án; Cars ở `$GCD_DATA_ROOT/cars/`.
- `GCD_WEIGHTS_ROOT`: mặc định `weights/`; file DINO `dino_vitb16_pretrain.pth`.
- `GCD_RUNS_ROOT`: mặc định `runs/`, không nằm trong Git.
- `device=auto`: CUDA nếu thực sự available, nếu không CPU cho probe nhỏ. MPS là opt-in probe, chưa xác minh. Full baseline launcher chặn máy không phải Linux và chặn CPU/MPS; upstream CUDA-only được giữ nguyên.

Mac hiện dùng `.venv` Python3.9.6 riêng đã cài torch2.4.1; không sửa Python hệ thống. Profile bootstrap cho checkout mới dùng Python3.10. Không tái chạy bootstrap lên venv3.9 hiện có: script sẽ báo mismatch và bảo toàn environment.

```bash
.venv/bin/python scripts/run_smoke.py
.venv/bin/python -m gcd_runtime.probe --device cpu --baseline
.venv/bin/python -m gcd_runtime.launch train --dry-run
```

Probe thứ hai cần đủ requirements-common và torchvision. Khi chạy thủ công nên bọc bằng subprocess timeout60 như smoke log. Không có probe tải weights. MPS chỉ dùng với `--device mps` nếu operator tương thích; không thay benchmark sang MPS.

## Git và chuyển máy
Đã `git init`, khai báo upstream thành submodule pin SHA trong report01; chỉ `.gitmodules` và gitlink đã được stage tự động bởi submodule add. Chưa commit/push, chưa có remote của dự án. Không cần cung cấp remote để tiếp tục local. Trước chuyển cloud, review rồi tự commit/push bằng các lệnh sau; URL là placeholder phải thay:

```bash
git status --short
git add AGENTS.md README.md .gitignore .gitmodules configs environments gcd_runtime scripts tests reports upstream/HypCD
git commit -m "Prepare portable GCD runtime and bounded local tests"
git remote add origin <YOUR_PROJECT_GIT_URL>
git push -u origin HEAD
```

Trên Linux (chọn đúng project commit đã review):

```bash
git clone --recurse-submodules <YOUR_PROJECT_GIT_URL> gcd-final
cd gcd-final
git checkout <REVIEWED_PROJECT_COMMIT>
python3 scripts/bootstrap_upstream.py
GCD_PYTHON=python3.10 bash scripts/bootstrap.sh cuda
```

Bootstrap không sửa/reset checkout upstream có sẵn. Python3.10 và NVIDIA driver phải có trước. Cặp torch2.4.1/torchvision0.19.1 cu121 dựa [hướng dẫn chính thức PyTorch](https://pytorch.org/get-started/previous-versions/#v241); chọn để gần upstream, không phải khẳng định tương thích mọi GPU/driver đời mới. Nếu CUDA probe lỗi kiến trúc/driver, dừng và chuẩn bị profile khác có kiểm thử, không tự nâng package trong một run đang so sánh. requirements pin dependencies trực tiếp; chưa là lock CUDA đã đo, mỗi run lưu `pip freeze`.

## Chuẩn bị dữ liệu và weights trên remote
Tự chuẩn bị dữ liệu theo quyền sử dụng; script không tải ảnh/weights. Cấu trúc phải đúng loader gốc:

```text
datasets/cars/cars_train/*.jpg
datasets/cars/cars_test/*.jpg
datasets/cars/devkit/cars_train_annos.mat
datasets/cars/devkit/cars_test_annos_withlabels.mat
weights/dino_vitb16_pretrain.pth
```

Nếu dùng mounted volume, export các biến GCD_* trước mọi lệnh. Không commit `.env`, dataset, weights, runs hoặc checkpoint. Preflight ghi checksum weights/annotations; chưa đối chiếu với checksum công bố chính thức. Chỉ nạp checkpoint/weights tin cậy.

## Trình tự trên NVIDIA

```bash
.venv/bin/python -m gcd_runtime.launch preflight --config configs/cars_baseline_dino_v1.json
bash scripts/train_linux.sh --config configs/cars_baseline_dino_v1.json
```

Preflight có timeout180s: kiểm upstream sạch đúng pin, split checksum, file/MAT/ảnh tồn tại, coverage train IDs, membership disjoint, đúng fraction và class mask; import baseline, Hungarian fixture, CUDA tensor+geometry forward/backward, ViT với pretrained state_dict thật + projection/head một bước synthetic (4 images). Chưa decode toàn bộ ảnh hoặc kiểm duplicate content; ảnh hỏng có thể chỉ lộ ra khi loader đọc. Log cho biết GPU/VRAM thật, peak allocated của **batch synthetic**, không suy ra batch128 chắc chắn đủ VRAM.

Train luôn chạy lại gate, không tái sử dụng chứng nhận CUDA cũ. Sau gate mới in PID, command, đường log và timeout worker. Batch128 là cấu hình gốc, chưa đo peak; nếu OOM phải lưu lỗi rồi tạo config/run mới, công khai thay đổi batch. Không âm thầm giảm batch trong run so sánh.

Mỗi run tạo thư mục UUID+UTC riêng với config.json, metadata.json (Git SHA/status, fingerprint code, command), environment.txt, readiness.json, run.log, metrics.jsonl và resume.pt khi epoch hoàn tất. Preflight failure cũng có metadata/log. Dùng `tail -f runs/<RUN_ID>/run.log`; progress gốc có tqdm/epoch loss, runtime thêm epoch_complete. Runtime bound áp dụng worker; gate tối đa180s riêng, terminate grace15s. Watchdog chỉ dừng child do chính launcher tạo, không tìm hoặc dừng process khác. Nếu vượt bound/OOM, epoch dở bị bỏ, checkpoint epoch hoàn tất gần nhất được giữ.

## Resume và evaluation

```bash
bash scripts/train_linux.sh --config configs/cars_baseline_dino_v1.json --checkpoint runs/<PREVIOUS_RUN_ID>/resume.pt
bash scripts/evaluate_linux.sh --config configs/cars_baseline_dino_v1.json --checkpoint runs/<TRAIN_RUN_ID>/backbone_final.pt
```

Resume tạo run ID mới, lưu parent checkpoint path; yêu cầu config giống hệt (bao gồm seed, epochs, runtime). Checkpoint gồm model/head/optimizer/scheduler, epoch và Python/NumPy/Torch/CUDA RNG; resume từ chối runtime code fingerprint khác. Chỉ resume tại biên epoch; prototype interval phải là1 để tái tạo hierarchy ở epoch tiếp theo. Chưa chứng minh bitwise equivalence trên CUDA/DataLoader thực tế; đã có fixture CPU so sánh update tiếp theo. Không gọi warmup weights là full resume. Khi worker bị ngắt trước epoch0, chưa có resume.pt.

`backbone_final.pt` chỉ sinh sau epoch cuối thành công. Eval cần backbone state_dict, không nhận resume.pt. Eval output `evaluation.json` là balanced semi-supervised k-means trên train unlabeled với labeled anchors, Hungarian v2; nhãn unlabeled chỉ dùng sau predictions. Train metrics phải đọc trường protocol: upstream chuyển từ balanced SS-kmeans sang plain kmeans mỗi200 epoch; không ghép các số này thành một curve cùng protocol. Disjoint test plain kmeans được ghi riêng, không dùng chọn checkpoint.

Runtime vô hiệu hóa test-selected best checkpoint, dùng final epoch đã đăng ký. Vì vậy đây là baseline có chính sách selection kiểm soát, không phải giữ nguyên mọi hành vi báo cáo của upstream. Code loss/parity AST đã kiểm; chưa có đo parity số học whole training. Tài liệu so sánh phải ghi rõ điều này.

## Còn thiếu trước tuyên bố sẵn sàng thực nghiệm
Hoàn tất method review/novelty và module reliability; CUDA thực, Linux environment validation, Cars data check thực, matched B0/proposed và ablation chưa chạy. Chưa có accuracy/efficiency benchmark. Không thuê/sweep tự động và không cam kết thời gian training.
