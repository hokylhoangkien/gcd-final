# Bàn giao chạy trên RunPod

Bản này hoàn tất source ứng viên và kiểm thử local; **chưa xác nhận CUDA, chưa có kết quả Cars, chưa xác nhận tính mới**. Trainer trực tiếp: `gcd_runtime/trainer.py`. Không sửa source upstream trong bộ nhớ khi chạy. Hướng nghiên cứu và giới hạn trong reports/06.

## 1. Mang bản source lên máy thuê

Có hai cách:

- Git (luồng chính): repository private `hokylhoangkien/gcd-final`. Trên RunPod đăng nhập GitHub bằng tài khoản có quyền đọc, rồi clone theo lệnh dưới. Không đưa token vào URL hoặc source.
- Không cần Git remote: dùng file `dist/gcd-final-runpod-<RELEASE>.tar.gz` và file SHA256 đi kèm. Upload hai file vào volume, kiểm checksum, giải nén. Gói không có dataset, weights, logs, checkpoints, .venv, .git hoặc prototype legacy; upstream là snapshot có hash manifest đúng commit pin.

Luồng Git trên RunPod (cần `git` và GitHub CLI `gh`):

```bash
cd /workspace
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
git clone --recurse-submodules https://github.com/hokylhoangkien/gcd-final.git
cd gcd-final
git rev-parse HEAD
python3 scripts/bootstrap_upstream.py
```

Để cập nhật một checkout hiện có sau khi run trước đã dừng và các thay đổi local được bảo toàn:

```bash
git status --short
git pull --ff-only
git submodule update --init --recursive
```

Không update source giữa một run hoặc ép resume checkpoint với code fingerprint khác. Ghi lại commit bằng `git rev-parse HEAD` cho mỗi thực nghiệm.

Ví dụ sau upload, thay RELEASE bằng tên thật:

```bash
cd /workspace
sha256sum -c gcd-final-runpod-<RELEASE>.tar.gz.sha256
tar -xzf gcd-final-runpod-<RELEASE>.tar.gz
cd gcd-final-runpod-<RELEASE>
python3 scripts/bootstrap_upstream.py
```

Đặt data, weights và runs trên volume đã gắn. Theo [RunPod storage docs](https://docs.runpod.io/pods/storage/types), volume disk mặc định `/workspace` tồn tại qua stop/restart nhưng mất khi xóa Pod; network volume tồn tại độc lập. Kiểm tra loại volume trong Pod của bạn và sao lưu kết quả trước khi xóa Pod. Script dự án không tạo Pod hoặc tự chi tiền.

## 2. Môi trường

Máy phải là Linux x86_64 với NVIDIA driver và Python3.9/3.10/3.11. Profile pin torch2.4.1 + torchvision0.19.1 CUDA12.1 để gần upstream; không giả định hỗ trợ mọi kiến trúc GPU đời mới. Nếu template chỉ có Python3.12+, chuẩn bị Python phù hợp trước, không ép cài dependency incompatible. Không có cam kết VRAM cho batch128 khi chưa đo trên GPU của bạn.

```bash
python3 --version
nvidia-smi
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
```

Bootstrap chỉ cài trong `.venv`, chạy pip check và bounded CPU tests, không tải dataset/weights hoặc bắt đầu training. Pythonminor và toàn bộ pip freeze sẽ được ghi trong run; baseline/proposed phải dùng cùng môi trường. Lỗi bootstrap phải giải quyết trước bước train.

## 3. Chuẩn bị dữ liệu và pretrained weights

Export đường dẫn đến các thư mục của bạn; ví dụ dưới là đường dẫn volume trên RunPod, không phải đường dẫn hardcode trong code:

```bash
export GCD_DATA_ROOT=/workspace/gcd-data
export GCD_WEIGHTS_ROOT=/workspace/gcd-weights
export GCD_RUNS_ROOT=/workspace/gcd-runs
```

Loader cần đúng layout:

```text
$GCD_DATA_ROOT/cars/cars_train/*.jpg
$GCD_DATA_ROOT/cars/cars_test/*.jpg
$GCD_DATA_ROOT/cars/devkit/cars_train_annos.mat
$GCD_DATA_ROOT/cars/devkit/cars_test_annos_withlabels.mat
$GCD_WEIGHTS_ROOT/dino_vitb16_pretrain.pth
```

Dùng checkpoint DINO ViT-B/16 chính thức, không thay bằng DINOv2 hoặc ViT-S. Dataset/split/weights chưa có trong release. Chỉ dùng weights/checkpoints tin cậy; run lưu checksum để tái lập, chưa là chứng nhận nguồn tải. Không đổi split SSB đi kèm.

## 4. Kiểm tra GPU, rồi mới train

```bash
.venv/bin/python -m gcd_runtime.launch preflight --config configs/cars_proposed_dino_v1.json
```

Gate có timeout180s; ghi GPU/VRAM thật, CUDA tensor+hyperbolic backward, metric fixture, dữ liệu/split, pretrained ViT image-pipeline probe và reliability active-branch backward. Không coi `nvidia-smi` đơn thuần là CUDA đã hoạt động. Chỉ sau khi gate trả `preflight_passed` mới chạy:

```bash
bash scripts/train_proposed_linux.sh
```

Lệnh train chạy lại gate, rồi training foreground, 200 epoch, max86400s mặc định. Mỗi plain-kmeans call có timeout120s để tránh vòng hội tụ vô hạn. Nếu cần thay deadline, tạo config mới có `kmeans_timeout_seconds` và ghi lại. Không âm thầm thay optimizer/batch khi OOM. Mọi job có run ID riêng; không sweep tự động, không tự dừng Pod.

Để có baseline so sánh cùng cấu hình:

```bash
bash scripts/train_linux.sh --config configs/cars_baseline_dino_v1.json
```

Chạy lần lượt, không chạy hai job cạnh tranh trên cùng GPU. Config B1/B2 đã có; `python3 scripts/ablation_plan.py` chỉ tạo configs/commands cho seeds1,2,3, không thực thi sweep. Chỉ chạy nhiều seed khi ngân sách cho phép.

## 5. Theo dõi, resume và đánh giá

Launcher in PID và đường log. Trong terminal khác:

```bash
tail -f /workspace/gcd-runs/<RUN_ID>/run.log
```

Nếu ngắt/kết thúc runtime budget, dùng checkpoint epoch hoàn tất gần nhất và giữ nguyên config/source:

```bash
bash scripts/train_proposed_linux.sh --checkpoint /workspace/gcd-runs/<RUN_ID>/resume.pt
```

Resume tạo run ID mới, kiểm config và source fingerprint; checkpoint gồm model, head, optimizer, scheduler và RNG. Không phải mid-batch resume. Epoch dở phải chạy lại; ngắt trước epoch đầu thì chưa có checkpoint. Không sửa source rồi ép dùng checkpoint cũ.

Sau epoch cuối, evaluate checkpoint **backbone_final.pt** (không phải resume.pt):

```bash
bash scripts/evaluate_linux.sh --config configs/cars_proposed_dino_v1.json --checkpoint /workspace/gcd-runs/<RUN_ID>/backbone_final.pt
```

`evaluation.json` ghi balanced SS-kmeans/Hungarian-v2 trên train-unlabeled pool có labeled anchors, unit fraction. Muốn % nhân100. `metrics.jsonl` gắn protocol vì train monitor của upstream dùng plain-kmeans tại report interval200, khác SS-kmeans ở các epoch còn lại. Không trộn hai protocol khi lập bảng chính. Chọn final epoch, không chọn best bằng test labels.

Artifacts cần giữ: metadata.json, config.json, environment.txt, readiness.json, run.log, metrics.jsonl, reliability.jsonl (khi active), resume.pt, backbone_final.pt và evaluation.json ở eval run. Backup cả train/eval run trước khi xóa Pod.

## 6. Khi có lỗi

- Thiếu data/weights: gate báo file cần có; bổ sung đúng layout, không tạo dữ liệu giả.
- CUDA unavailable/unsupported architecture: dừng, kiểm driver và profile PyTorch; không fallback CPU cho training chính thức.
- OOM: giữ log và metadata; thử config batch nhỏ hơn ở run mới, phải dùng cùng batch khi so sánh B0/proposed.
- Kmeans timeout/NaN: dừng có log; không xem partial metrics là benchmark hoàn tất. Kiểm dependency và convergence trước khi tăng deadline.
- Novelty: reciprocal hyperbolic neighbor đã có trong HypGCD. Phần đang kiểm chứng là reliability weighting tại từng mức hierarchy của Hyp-SelEx; chưa có bằng chứng ưu thế.

Bản source sẵn để chuyển máy và chạy các gate trên. Không có lần NVIDIA/RunPod thực tế nào được thực hiện trong phiên Mac; preflight trên GPU của bạn là bước xác minh bắt buộc.
