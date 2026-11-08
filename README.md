# CRiSH-GCD v2.1

Source nghiên cứu GCD với **một pipeline huấn luyện**: CRiSH mặc định, B0 dùng cùng pipeline và tắt sửa phân hoạch để đối chứng. Backbone mặc định: DINOv2 ViT-B/14 reg4. Hỗ trợ Stanford Cars SSB (mặc định) và CUB SSB (`--dataset cub`); Cars vẫn là benchmark chính.

**Bản chốt cấu hình ngày 08/11/2026:** entrypoint mặc định dùng `--profile final`: USE `legacy`, evidence shortlist bằng hàng xóm trên toàn bộ frozen features, probe mỗi fold 16 labeled / 32 unlabeled và lấy labeled cân bằng theo lớp. Loss, lịch train và điều kiện chấp nhận repair giữ nguyên. `final` là tên bản cấu hình đã chốt để thử nghiệm; chưa có bằng chứng tăng accuracy. Profile `reference` và `upgraded` vẫn giữ để đối chứng.

Code đã có kiểm thử CPU tổng hợp. Log GPU được cung cấp ngày 05/11/2026 cho thấy bản trước chạy hết epoch 0–2 trên A40 rồi lỗi prototype khi dựng hierarchy. Runtime hiện có xử lý cụm rỗng dùng chung cho CRiSH/B0 và thêm kiểm tra HSSK vào preflight. **Bản sửa chưa được xác nhận lại trên CUDA/dữ liệu thật; chưa có kết quả benchmark CRiSH hoàn chỉnh.** Hiệu quả và tính mới của phương pháp vẫn cần thực nghiệm.

## 1. Cài trên máy train

Linux + NVIDIA CUDA, Python 3.9–3.11. Dung lượng VRAM và thời gian thực tế phải đo bằng smoke test; cấu hình đầy đủ dùng batch 128, projector 8192 chiều, 200 epoch.

Giải nén gói source rồi mở terminal trong thư mục vừa giải nén. Gói đã chứa upstream HypCD đúng phiên bản, không cần clone lại.

```bash
GCD_PYTHON=python3 bash scripts/bootstrap.sh cuda
```

Lệnh tạo `.venv` nếu chưa có, cài PyTorch 2.4.1/CUDA 12.1 và dependencies, xác minh upstream rồi chạy test CPU có timeout. Cần Internet để cài dependencies. Nếu driver không chạy được bộ CUDA này, xử lý môi trường trước khi train.

## 2. Đặt dataset và pretrained weights

**Gói source không kèm ảnh dataset hoặc pretrained weights.** Tải chúng trực tiếp trên máy Linux NVIDIA sau bước cài môi trường ở mục 1; không cần tải bộ dữ liệu lớn về Mac. Lệnh train không tải data/weights tự động. Các lệnh dưới chạy từ thư mục gốc project, dùng đường dẫn mặc định. Nếu đã có đủ file ở ổ khác, bỏ qua bước tải và đặt biến môi trường ở cuối mục này.

### 2.1. Tải Stanford Cars

Nguồn ảnh và annotations dạng gốc: [mirror cộng đồng trên Hugging Face](https://huggingface.co/datasets/iharabukhouski/stanford_cars/tree/main), khoảng 2 GB theo danh sách file. Nguồn Stanford cũ hiện không còn tải được theo [Torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html). Mirror chưa được kiểm chứng toàn bộ trên workspace này; preflight sẽ kiểm tra ảnh, annotation và membership trước khi train.

Cài thêm công cụ tải rồi tải nguyên tên file/thư mục, giữ nguyên train/test gốc. Cú pháp dùng [Hugging Face CLI 0.26.2](https://huggingface.co/docs/huggingface_hub/v0.26.2/guides/cli):

```bash
.venv/bin/python -m pip install 'huggingface_hub[cli]==0.26.2'
timeout 3600 .venv/bin/huggingface-cli download iharabukhouski/stanford_cars \
  --repo-type dataset --local-dir datasets/cars \
  --include 'cars_train/**' 'cars_test/**' 'devkit/**' 'cars_test_annos_withlabels.mat'
```

`timeout` là lệnh trên Linux; giới hạn tải một giờ, không phải ước lượng thời gian hoàn tất. Nếu tải bị ngắt, chạy lại lệnh download để tiếp tục các file còn thiếu. Nếu nhóm đã có bản Cars dùng trong thí nghiệm trước, có thể dùng lại bản đó và chạy preflight.

Mirror đặt annotation test ở thư mục gốc, còn loader của project đọc trong `devkit/`. Chép file vào đúng vị trí, giữ file đã có:

```bash
mkdir -p datasets/cars/devkit
cp -n datasets/cars/cars_test_annos_withlabels.mat datasets/cars/devkit/
```

Không đổi tên ảnh, tự chia lại train/test hoặc tự sắp ảnh theo tên lớp. Split Cars SSB đã đi kèm upstream: 98 lớp known, 98 lớp novel; 50% ảnh của các lớp known được gán nhãn theo loader gốc. Nhãn của tập unlabeled chỉ được dùng trong evaluator riêng sau khi khóa checkpoint.

### 2.2. Tải pretrained DINOv2

Dùng **DINOv2 ViT-B/14 với 4 register tokens**, file `dinov2_vitb14_reg4_pretrain.pth`. [Weights tải trực tiếp từ Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) được liệt kê trong [repo DINOv2 chính thức](https://github.com/facebookresearch/dinov2). Kiến trúc model đã nằm trong upstream, chỉ cần bổ sung weights.

Nếu chưa có weights, chạy:

```bash
mkdir -p weights
curl --fail --location --continue-at - --connect-timeout 20 --max-time 1800 \
  --output weights/dinov2_vitb14_reg4_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
```

Lệnh giới hạn 30 phút và có thể chạy lại để tiếp tục tải dở. Preflight nạp state dict vào đúng kiến trúc, ghi SHA256 weights và thử forward/backward CUDA; tải thành công chưa phải xác nhận CUDA.

### 2.3. Kiểm tra vị trí file

```text
datasets/cars/
├── cars_train/                         # ảnh train gốc
├── cars_test/                          # ảnh test gốc
└── devkit/
    ├── cars_train_annos.mat
    └── cars_test_annos_withlabels.mat
weights/
└── dinov2_vitb14_reg4_pretrain.pth
```

Nếu data/weights đã ở ổ khác, đặt biến môi trường trỏ tới **thư mục cha** tương ứng. Ví dụ `GCD_DATA_ROOT` chứa thư mục `cars/`, còn `GCD_WEIGHTS_ROOT` chứa trực tiếp file `.pth`:

```bash
export GCD_DATA_ROOT=/path/to/datasets
export GCD_WEIGHTS_ROOT=/path/to/weights
export GCD_RUNS_ROOT=/path/to/runs
```

Sau khi đặt đủ file và biến môi trường, kiểm tra trên máy CUDA:

```bash
bash scripts/train_crish.sh check
```

Chỉ tiếp tục mục 3 khi preflight thành công. Nếu báo thiếu file, sửa vị trí theo thông báo; không tự tạo annotation hoặc split thay thế.

## 3. Chạy train

Kiểm tra lệnh và cấu hình, không dùng GPU:

```bash
bash scripts/train_crish.sh --dry-run
```

Train CRiSH:

```bash
bash scripts/train_crish.sh
```

Lệnh này chạy **preflight → smoke 1 epoch/tối đa 2 batch, có selector và hai final evaluator → một run mới 200 epoch**. Mỗi stage chỉ tiếp tục khi stage trước thành công. Preflight kiểm tra split/ảnh/weights/import, forward/backward CUDA thật, backbone và loss/scorer. Smoke kiểm tra toàn bộ đường chọn best và báo best/last trên dữ liệu thật; accuracy của smoke không phải kết quả benchmark. Smoke không tiếp tục thành checkpoint của run chính.

Preflight giới hạn 180 giây; worker smoke gồm train/selector tối đa 1.800 giây; worker full train gồm selector tối đa 86.400 giây. Mỗi evaluator có giới hạn 1.800 giây; sau worker train/smoke có tối đa **hai** final evaluator, mỗi worker tối đa 1.800 giây. Đây là giới hạn dừng, không phải dự báo thời gian hoàn thành. Chương trình chạy foreground, in PID và đường dẫn log, lưu resume trước mỗi selector. Nếu đã có tiến trình GPU, lệnh dừng để người vận hành kiểm tra; không tự dừng tiến trình khác. Preflight kiểm tra HSSK với embedding trùng nhau và plain k-means trên CUDA; log `KMEANS ... empty_cluster_repairs=...` ghi số lần xử lý cụm rỗng.

Môi trường CUDA đã có từ bản trước cần bổ sung backend Git đúng phiên bản (không tải lại weights):

```bash
.venv/bin/python -m pip install -r environments/requirements-selector-cuda.txt
.venv/bin/python -m pip install --force-reinstall --no-deps -r environments/requirements-selector-cuda.txt
```

File này ghim `kmeans_pytorch` ở commit `f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805` của repo chính thức, kèm Numba. Lệnh thứ hai buộc thay bản PyPI cùng version bằng bản Git; `--no-deps` giữ nguyên Torch và các dependencies đã được giải quyết ở lệnh đầu. PyPI 0.3 thiếu `tqdm_flag`, có khác biệt về GPU/empty-cluster; CUDA preflight chặn backend cũ. CPU fixture vẫn có thể dùng PyPI 0.3 trên dữ liệu nhỏ, không xác nhận CUDA. Không chạy bootstrap/update dependency khi GPU run cũ còn dùng môi trường đó.

Các lệnh bổ sung:

```bash
# Chỉ kiểm tra độ sẵn sàng trên máy CUDA
bash scripts/train_crish.sh check

# Chỉ smoke dữ liệu thật, chưa chạy đủ 200 epoch
bash scripts/train_crish.sh smoke

# Đối chứng B0: cùng objective, backbone, split và lịch tối ưu
bash scripts/train_crish.sh --baseline

# Thêm SS-k-means trên train-unlabeled mỗi epoch (tốn thêm thời gian)
bash scripts/train_crish.sh --eval-every 1

# Mặc định: test selector mỗi epoch; SS chỉ cho best/last khi train xong
bash scripts/train_crish.sh --eval-every 0

# Tiếp tục từ cuối epoch đã lưu; dùng đúng source/config của run đó
bash scripts/train_crish.sh --resume runs/RUN_ID/resume.pt

# Đánh giá lại backbone đã khóa ở epoch cuối
bash scripts/train_crish.sh evaluate --checkpoint runs/RUN_ID/backbone_final.pt

# Đánh giá lại best với đúng config snapshot (kể cả --eval-every đã override)
.venv/bin/python -m gcd_runtime.launch evaluate --config runs/RUN_ID/config.json \
  --checkpoint runs/RUN_ID/backbone_best_old_test.pt
```

Thêm `--baseline` khi resume/evaluate checkpoint B0 bằng entrypoint chung. Checkpoint resume có optimizer/RNG nên chỉ nạp file tin cậy. Resume đòi đúng config (gồm `--eval-every` nếu đã dùng) và fingerprint source; checkpoint từ source cũ cần chạy bằng bản source cũ đã lưu. Với chế độ chọn best, giữ **cả thư mục run**: `resume.pt`, `backbone_best_old_test.pt`, `test_evaluations/` và các histories. Chỉ chép `resume.pt` sẽ không đủ để khôi phục best đã chọn. Cập nhật code này không thay đổi run đang chạy trên máy khác và không bổ sung được lịch sử accuracy cho các epoch chưa từng đánh giá.

Sau khi cập nhật bản sửa lỗi prototype ngày 05/11/2026, giữ nguyên thư mục run/checkpoint cũ để đối chiếu. Checkpoint của bản trước bị chặn resume do fingerprint source đã đổi; bản sửa này không bỏ qua kiểm tra đó. Dùng source mới trên máy Linux NVIDIA, chạy `bash scripts/train_crish.sh check`, tiếp theo `bash scripts/train_crish.sh smoke`. Nếu cả hai thành công, `bash scripts/train_crish.sh` tạo một run mới, có preflight và smoke trước training đầy đủ. Chạy lại B0 bằng cùng bản sửa khi so sánh.

## 4. Cấu hình và kết quả

`configs/crish_final.json` là cấu hình mặc định; `configs/baseline_final.json` chỉ khác trường `method`. Hai file `_final_smoke.json` dùng riêng cho kiểm tra ngắn. `configs/crish.json` / `baseline.json` giữ cấu hình reference cũ, chọn bằng `--profile reference`; `_upgraded.json` là ablation đổi USE sang `row`. CUB có các file tương ứng với tiền tố `cub_`. Các siêu tham số CRiSH hiện là lựa chọn kỹ thuật chưa được xác nhận bằng benchmark. M2 stability mặc định tắt; lookahead chỉ cập nhật bản sao projector trong một bước SGD.

**Cấu hình Cars/DINOv2 đã chốt theo script upstream:** `unsupervised_smoothing=0.5`, **`hyper_max_weight=0.5`**, `c=0.1`, `cr=2.0`, `grad_from_block=10`, batch 128, 200 epoch, SGD lr 0.1, weight decay 5e-5, supervised weight 0.35, projector 8192, seed 1, official SSB split/labeled fraction 0.5. B0 và CRiSH dùng cùng các giá trị này; smoke chỉ rút epoch/step. Mốc đối chiếu là lệnh `scars v2 0.5 0.5 0.1 2.0` trong [script Hyp-SelEx được ghim](upstream/HypCD/scripts/train_HypSelEx.cmd), không lấy default 1.0 của parser thay cho lệnh Cars-v2.

`scripts/check_protocol.py` đọc trực tiếp script/parser upstream và so với **CLI thực tế của worker**, không chạy Slurm. Kiểm tra này nằm trong CPU smoke; preflight CUDA cũng ghi đối chiếu vào `readiness.json`. Chạy riêng bằng `.venv/bin/python scripts/check_protocol.py`. Kết quả khớp tham số không có nghĩa đã khôi phục chính xác mọi số trong paper: các sửa lỗi runtime, RNG isolation và checkpoint mapping chưa công bố vẫn được ghi riêng. Run cũ dùng weight 1.0 phải giữ nguyên cấu hình/nhãn thí nghiệm của nó; chuyển sang 0.5 là run mới, không đổi số hay metadata cũ.

Mỗi lần chạy tạo `runs/RUN_ID/` với `config.json`, `environment.txt`, `metadata.json`, `readiness.json`, `run.log`, checkpoint và diagnostics. Theo dõi đường dẫn log được in ra:

```bash
tail -f runs/RUN_ID/run.log
```

Mặc định B0 và CRiSH dùng `checkpoint_selection=best_old_test_v2`, theo quy tắc chọn checkpoint trong **code Hyp-SelEx phát hành**, với hai bước:

1. **Mỗi epoch:** khóa snapshot sau SGD, chờ worker riêng trích normalized backbone features trên **disjoint test**, chạy **plain Euclidean k-means**, Hungarian-v2 toàn cục và in All/Old/New. Khi **Old-v2 tăng**, lưu best; hòa điểm giữ epoch sớm hơn. `test_metrics.jsonl` và `test_evaluations/epoch_0001.json` lưu kết quả từng epoch. `selection.json` ghi epoch/điểm/quy tắc chọn và hash best. Không early-stop, không đưa score vào loss hay CRiSH action screening.
2. **Sau epoch cuối:** worker riêng chạy **balanced SS-k-means trên train-unlabeled**, với labeled anchors, cho cả **best và last**; báo metric Hungarian-v2 toàn cục. `evaluation_best_old_test.json` là best, `evaluation_last.json` là last; `evaluation.json` là bản sao kết quả best trong chế độ này. Best được chọn theo Old-test, không phải All cao nhất trong lịch sử.

`--eval-every N` chỉ thêm bước SS monitoring trên train-unlabeled mỗi N epoch; mặc định **0**, không ảnh hưởng selector disjoint test mỗi epoch. `1` có đủ SS metrics mỗi epoch nhưng tốn thêm một lượt trích feature/clustering. Các kết quả SS monitoring nằm trong `epoch_metrics.jsonl` và `evaluations/epoch_NNNN.json`. Best/last SS cũng xuất hiện trong lịch sử SS; khi không bật monitoring, các đỉnh SS chỉ xét hai checkpoint đã đánh giá, không phải mọi epoch.

`run.log` in `EVAL epoch=… protocol=… All=…% Old=…% New=…%`, cùng dòng `SELECTION` báo best hiện tại. `results_summary.json` và lệnh bên dưới tổng kết riêng **epoch max-All, max-Old, max-New**, kèm nguyên bộ All/Old/New tại từng epoch và tách theo protocol. File JSON lưu accuracy dạng **fraction**, kèm sample counts, thời gian, seed/RNG seed, epoch, timestamp và hash checkpoint; nhân 100 để lấy %. `metrics.jsonl` vẫn chỉ ghi loss/thời gian của worker train, không nhận oracle score.

Để xem lại bảng kết quả đã lưu:

```bash
.venv/bin/python scripts/show_metrics.py runs/RUN_ID
```

Chỉ giữ snapshot gần nhất `backbone_monitor.pt`, best `backbone_best_old_test.pt`, `resume.pt` và `backbone_final.pt`; không giữ weights của cả 200 epoch. Best là một bundle gồm backbone và metadata chọn model được thay thế nguyên tử; không có optimizer để resume từ best. Resume dùng last optimizer/RNG và sao chép best/history từ run nguồn. Nếu selector của epoch vừa lưu thất bại, resume đánh giá lại epoch đó trước khi chạy SGD tiếp; kể cả khi đã hết epoch, có thể khôi phục selector rồi chuyển sang final evaluations. Thiếu lịch sử hoặc best sẽ báo lỗi, không tự reset điểm best. Tổng thời gian/VRAM bổ sung cần đo trên NVIDIA.

Đây là **checkpoint selection có dùng nhãn đánh giá** theo yêu cầu đối chiếu code gốc, thay cho quy tắc chỉ lấy epoch cuối trước đây; không gọi đây là protocol không dùng oracle. Với v2, global Hungarian dùng nhãn cả known/novel trước khi tính Old. Nhãn/score vẫn ở evaluator và không đi vào SGD/CRiSH. Muốn giữ quy tắc cũ, đặt `checkpoint_selection=final_epoch` và `test_every_epochs=0` cho cả hai cấu hình; khi đó `evaluation.json` là kết quả last.

Để so sánh dùng cùng selector, lịch, backend và cấu hình cho B0/CRiSH. SelEx gốc mặc định chọn theo Old-v1 (helper trả metric đầu tiên); không gọi chế độ Old-v2 này là tái lập nguyên xi SelEx. HypCD eval script nạp `model_test.pt`; chưa xác minh checkpoint nào sinh từng số paper. B0 vẫn có các sửa runtime/siêu tham số đã ghi nhận, nên việc khớp selector không tự chứng minh tái lập đầy đủ bảng công bố. Evaluation RNG được tách khỏi training, seed được ghi lại; đây cũng là khác biệt so với script gốc dùng chung RNG.

## 5. Kiểm thử và bố cục

Trên Mac/CPU, chỉ chạy:

```bash
.venv/bin/python scripts/run_smoke.py
```

Nếu máy chưa có môi trường local: `GCD_PYTHON=python3 bash scripts/bootstrap.sh local`. Không chạy train dài trên Mac. Biên nhận test nằm ở `results/local_validation.json`; không phải số liệu benchmark.

```text
crish/          thuật toán, objective adapter, repair/probe/lookahead và vòng train chung
gcd_runtime/   data boundary, CLI, preflight, checkpoint và evaluation
configs/        CRiSH, B0 và smoke
scripts/        bootstrap, integrity check, smoke, train và đóng gói
tests/         unit, protocol và synthetic integration tests
audit/         oracle đối chiếu loss với source baseline
upstream/      HypCD giữ nguyên, pin commit
reports/       audit, protocol, phương pháp, giới hạn và đối chiếu bản thảo
paper/         nguồn LaTeX và PDF bản thảo nghiên cứu hiện tại
results/       biên nhận kiểm tra; không có accuracy CRiSH giả lập
```

[Phương pháp và kế hoạch nghiên cứu](reports/04_method_and_research_plan.md) · [Protocol](reports/01_baseline_protocol.md) · [Đối chiếu source/bản thảo](reports/06_manuscript_source_alignment.md) · [Nguồn LaTeX](paper/manuscript.tex) · [Bản thảo PDF](paper/manuscript.pdf) · [Nguồn gốc mã](NOTICE) · [Giấy phép HypCD](LICENSES/HypCD-MIT.txt).

`paper/` giữ bản thảo đã đồng bộ với thuật toán/cấu hình hiện tại; các giới hạn
và kế hoạch chưa thực nghiệm được ghi riêng. Chưa phải paper hoàn chỉnh. `.venv/`
được giữ để chạy local. Slide, preview và các release cũ đã được dọn; chỉ
giữ tài liệu phương pháp ở `reports/`. Checkpoint/log dữ liệu thật của hướng
cũ nằm trong `logs/final-backup-20261101/`, không phải kết quả CRiSH. Bản
source khớp fingerprint của run lịch sử được giữ cùng backup để tái lập.

`AGENTS.md` và upstream giữ nguyên. Đóng gói khi cần bằng
`.venv/bin/python scripts/build_release.py`; `dist/` được tạo lại với manifest
và SHA256. Gói gồm code, ghi chú nghiên cứu và bản thảo LaTeX/PDF hiện tại,
không kèm `.venv`, data/weights, log/checkpoint, slide hoặc backup lịch sử.

Kiểm tra code–paper và độ khớp LaTeX/PDF:

```bash
.venv/bin/python scripts/check_manuscript.py
```

Bảng cấu hình trong paper được sinh từ config thật. Nếu thuật toán/config
đổi, phải rà lại công thức và pseudocode trước khi cập nhật bảng/digest bằng
`scripts/check_manuscript.py --write`, rồi dựng PDF bằng
`scripts/build_paper.py --compiler PATH_TO_TEX_COMPILER`. Compiler hỗ trợ
Tectonic, pdfLaTeX hoặc XeLaTeX đã có sẵn; script không cài vào `.venv` và
để log/intermediate trong thư mục tạm hoặc `logs/`. Biên nhận
`paper/manuscript.build.json` kiểm tra PDF đi cùng đúng nguồn LaTeX và source
đã review; smoke local cũng chạy kiểm tra này. Hash không thay thế việc đọc
và đối chiếu ngữ nghĩa thuật toán.

Trước khi train lại để báo cáo paper, đọc [audit đánh giá HypCD/SelEx](reports/07_evaluation_and_retraining_audit.md).
SelEx đã xác nhận lỗi ordering ở tầng thô; runtime này sửa chung cho B0/CRiSH,
nên cần baseline chạy lại cùng source. Không ghép peak All/Old/New từ các epoch
khác nhau, và không dùng plain-test score thay final SS train-unlabeled.

Chuẩn bị snapshot ghép cặp cho nhiều seed, **chỉ sinh plan, không chạy GPU**:

```bash
.venv/bin/python scripts/prepare_retraining.py --dataset scars --profile final --seeds 1 2 3
```

Plan nằm ở `runs/plans/`, có `manifest.json` và `commands.txt`; trên máy Linux
NVIDIA tạo lại từ đúng source rồi chạy từng stage B0 preflight → smoke → full,
sau đó CRiSH cùng seed. Không chạy các job đồng thời. Mặc định giữ
`best_old_test_v2`, đồng thời báo riêng fixed-last. Nếu chọn `--selection final_epoch`,
phải khóa trước khi xem kết quả và dùng cho cả hai arm. Script không xác nhận
data/weights/CUDA hoặc VRAM cho máy target. Sửa provenance trong audit này
làm đổi source fingerprint: bắt đầu run mới; checkpoint cũ cần source cũ để resume.

## Chuyển sang CUB-200-2011

Dùng cùng loss, CRiSH, checkpoint selector và evaluator; CLI chọn loader, preflight và cấu hình riêng. Chuẩn bị dữ liệu gốc trên máy Linux NVIDIA:

```text
datasets/cub/CUB_200_2011/
  images/
  images.txt
  image_class_labels.txt
  train_test_split.txt
  attributes/image_attribute_labels.txt
```

File attributes bắt buộc vì loader upstream đọc nó. Runtime không tự tải hoặc giải nén CUB. `GCD_DATA_ROOT` trỏ tới thư mục cha chứa `cub/` và/hoặc `cars/`, không phải thư mục images. Dùng cùng pretrained DINOv2 reg4 ở mục 2.

```bash
# Xem cấu hình trên Mac, không chạy train
bash scripts/train_crish.sh --dataset cub --dry-run

# Trên Linux NVIDIA, kiểm tra dữ liệu/weights và CUDA
bash scripts/train_crish.sh check --dataset cub

# Smoke dữ liệu thật, có giới hạn
bash scripts/train_crish.sh smoke --dataset cub

# Smoke rồi chạy một thí nghiệm 200 epoch
bash scripts/train_crish.sh --dataset cub

# Đối chứng B0 trên cùng dataset
bash scripts/train_crish.sh --dataset cub --baseline
```

CUB dùng official SSB 100 known/100 novel, labeled fraction 0.5; không chia lại lớp theo ID liên tiếp. Theo lệnh `cub v2 1.0 1.0 0.1 1.5` trong script upstream, CUB dùng smoothing 1.0, hyper weight 1.0, `c=0.1`, `cr=1.5`. Cars giữ smoothing/hyper weight 0.5 và `cr=2.0`. Các giá trị này nằm trong `configs/cub_*.json` và `configs/*.json`; `scripts/check_protocol.py` đối chiếu mọi profile full/smoke với source được ghim. Chưa tuning CRiSH riêng cho CUB.

Không chuyển checkpoint Cars sang CUB bằng `--resume`; resume phải giữ nguyên dataset, source và config của run. Cơ chế lưu lịch sử từng epoch, chọn best theo Old-test-v2 và báo SS-kmeans best/last giữ nguyên giữa hai dataset. CPU tests với ảnh giả lập xác nhận wiring/split và cách ly nhãn; chưa có kết quả benchmark CUB thật hoặc xác nhận khớp bảng paper.

## Các profile và bản chốt để train (08/11/2026)

`--profile final` là mặc định cho cả Cars/CUB và CRiSH/B0, chốt riêng thay đổi
evidence/probe và giữ loss reference. `--profile reference` chọn cấu hình cũ;
`--profile upgraded` thêm thay đổi USE normalization. Đây là các lựa chọn có thể
đối chứng, chưa có bằng chứng tăng accuracy hoặc đạt novelty gate. CUB dùng
cùng lựa chọn CRiSH để kiểm tra khả năng tổng quát khi có compute.

| Thành phần | Reference | Final (mặc định) | Upgraded (ablation) |
|---|---|---|---|
| `use_target_normalization` | `legacy` | `legacy` | `row` |
| `crish.evidence_sampling` | `random` | `global_neighbors` | `global_neighbors` |
| `crish.probe_sampling` | `random` | `class_balanced` | `class_balanced` |
| Probe mỗi fold | 4 labeled / 8 unlabeled | 16 labeled / 32 unlabeled | 16 labeled / 32 unlabeled |

`global_neighbors` chỉ shortlist từ frozen features hợp lệ; kiểm tra hai view
vẫn diễn ra trong cache 128 mẫu. Nó không dùng hidden labels hay global
augmented-view kNN. Probe lớn hơn và tìm hàng xóm có thể tốn thời gian hơn;
giữ ngân sách refresh 120 giây và đo fallback/acceptance trong smoke thật.
Đối chứng reference/final giúp tách thay đổi evidence/probe khỏi objective;
để gán tác dụng cho từng thành phần vẫn cần ablation riêng.

USE ở chế độ `row` có fallback về view còn lại của cùng ảnh khi không có
smoothing support; không cần nhãn ẩn. Các loss class upstream và true-label
supervised terms được giữ nguyên. Đây là thay đổi objective, nên B0 upgraded
cũng dùng chuẩn hóa mới; không gọi nó là baseline paper nguyên vẹn.

Shortlist dùng normalized backbone features hiện có, khoảng cách Euclidean,
chia block tham chiếu 1.024 hàng và loại cả hai probe khỏi tập tham chiếu.
Cache vẫn tối đa 128 ảnh; kNN hai augmentation chỉ đo trong cache này, **không
phải global cross-view kNN**. Hai probe độc lập về sample ID; unlabeled chỉ có
ảnh/feature và membership, không có known/novel identity. Warmup 10 epoch,
refresh mỗi 5 epoch, tối đa một repair và các acceptance guard giữ nguyên.
Chi phí thực tế trên CUDA cần được đo. `crish.jsonl` ghi số reference/cache,
scope, kích thước/độ phủ lớp labeled của probe, candidate, quyết định và thời gian.

Trên Linux NVIDIA có dữ liệu/weights và GPU rảnh:

```bash
# Kiểm tra rồi bounded smoke riêng trước khi quyết định chạy đủ lịch
bash scripts/train_crish.sh check --profile upgraded
bash scripts/train_crish.sh smoke --profile upgraded

# Run mới: preflight -> bounded smoke -> training hữu hạn -> SS best/last
bash scripts/train_crish.sh --profile upgraded

# Đối chứng cùng chuẩn hóa USE và cùng protocol
bash scripts/train_crish.sh --profile upgraded --baseline
```

Thêm `--dataset cub` cho CUB. Mac chỉ xem kế hoạch bằng
`bash scripts/train_crish.sh --profile upgraded --dry-run` và chạy
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_smoke.py`.
Không chạy các full run đồng thời hoặc tự động sweep nhiều seed.

Các file `configs/*_upgraded*.json` giữ config riêng cho từng dataset/method và
smoke. Từng thành phần có thể tắt riêng bằng các giá trị trong bảng; khi đối
chứng probe cần khóa cả cách sampling và kích thước. Dùng config riêng qua
`python -m gcd_runtime.launch ... --config PATH` sau check/smoke tương ứng;
không chỉnh config đang dùng bởi một run. Checkpoint source cũ vẫn cần source
cũ để resume; `--profile upgraded` không chuyển đổi checkpoint cũ.

Protocol checker ghi riêng CLI hyperparameter match và
`reference_use_arithmetic_preserved`; upgraded có trường thứ hai là `false`.
Kết quả accuracy/All/Old/New và lợi ích của từng thay đổi: **chưa chạy**.
