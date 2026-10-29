# Protocol Hyp-SelEx — kiểm tra ban đầu

## Nguồn và pin
Upstream https://github.com/Visual-AI/HypCD, commit `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`, MIT, bản clone tại `upstream/HypCD`, chưa sửa upstream. Workspace cha chưa có Git SHA.
Paper đọc qua https://arxiv.org/html/2504.06120v1 (method và supplementary §6); CVF trả HTTP 403. Chưa đối chiếu bản PDF camera-ready.

## Split và dữ liệu
`data/get_datasets.py:get_class_splits`: dataset CLI là `scars`, SSB enabled. File `data/ssb_splits/scars_osr_splits.pkl` có SHA256 `28d471f0fb9e2c773e1255b8b447e542974f6617952203056a0635c638513d39`. Đã kiểm tra bằng công cụ offline: 98 known, 98 novel, novel theo thứ tự Hard (7), Medium (15), Easy (76), disjoint và phủ 0..195.
`data/stanford_cars.py`: ảnh `cars_train/`, `cars_test/`; metadata `devkit/cars_train_annos.mat`, `devkit/cars_test_annos_withlabels.mat`. MAT labels 1-based chuyển về 0-based trước remapping.
CLI `prop_train_labels=0.5`; lấy floor(0.5 * tổng ảnh known), không phải bảo đảm đúng 50% từng lớp. `subsample_instances` reset NumPy seed về 0; chia membership cố định dù đổi training seed. Unlabeled là phần bù của labeled trong train pool. `split_train_val=False`. Test pool riêng. Chưa có ảnh/MAT nên chưa xác minh số mẫu thực tế; con số làm tròn trong paper không phải counts đã đo.

## Luồng code
`train/train_HypSelEx.py`:
- Hai augmentation qua `ContrastiveLearningViewGenerator`; imagenet resize 256, random crop 224, flip, ColorJitter mặc định, normalize ImageNet; eval center crop.
- ViT-B backbone 768 chiều; DINO v1 ViT-B/16 hoặc DINOv2 theo `models/vision_transformer2.py`; cần xác minh tương thích weights reg4 B/14 thực tế trước chạy.
- DINOHead 3 tầng, output 8192 theo script; `hyptorch.nn.ToPoincare` có clipping, curvature; dùng manifold hiện có.
- Mỗi epoch extract normalized backbone features, balanced semi-supervised k-means; fit nhận unlabeled features, labeled features, labeled targets. Hierarchy giảm đôi số known/novel; pseudo-label dùng cho SupCon nhiều mức.
- USE: confusion_factor từ pair distance và cluster radii, vào soft-target CrossEntropy (`LabelSmoothingLoss`, `info_nce_logits`). Không phải classifier confidence. SSE: true labels chỉ qua mask_lab; thêm hierarchy pseudo-label trên toàn batch.
- Trộn hai nhánh Euclidean distance (gọi angle trong code) và hyperbolic distance theo epoch. Chưa chạy numerical test soft targets; không tự chuẩn hóa lại baseline.
- Supervised/hierarchy terms ở dòng 348–359 là điểm chèn dự kiến cần phân rã theo pair, không được nhân trọng số vào supervised loss toàn cục.

## Hyperparameters từ train script Cars
Batch 128, epochs 200, grad_from_block 10, SGD lr 0.1 momentum 0.9, weight decay 5e-5, cosine lr min=0.001*lr; sup_con_weight .35, temperature 1, seed mặc định 1, workers 4, 2 views, contrast_unlabel_only False, cluster_momentum 1, prototype interval 1.
DINO v1: smoothing .5, hyper_max_weight 1, c .1, cr 2. DINOv2: smoothing .5, hyper_max_weight .5, c .1, cr 2.

Lệnh tham chiếu đã chuyển khỏi Slurm, **chưa chạy**; chỉ dùng khi config.py, dependencies, data, weights, CUDA, log, timeout/resume đã chuẩn bị:

```bash
cd upstream/HypCD
python -m train.train_HypSelEx --dataset_name scars --batch_size 128 --grad_from_block 10 --epochs 200 --base_model vit_dino --num_workers 4 --use_ssb_splits True --sup_con_weight 0.35 --weight_decay 5e-5 --contrast_unlabel_only False --transform imagenet --lr 0.1 --eval_funcs v2 --c 0.1 --cr 2.0 --dino v1 --mlp_out_dim 8192 --unsupervised_smoothing 0.5 --hyper_max_weight 1.0
```

## Evaluation và rò rỉ
`project_utils/cluster_and_log_utils.py:split_cluster_acc_v2`: một Hungarian mapping toàn pool rồi tính đúng/tổng cho All, Old, New. All = n_old/n * Old + n_new/n * New; không phải trung bình hai số. Accuracy code ở [0,1], bảng % nhân 100.
`test()` eval_only: semi-supervised k-means trên merged train pool, fit chỉ nhận labeled targets; unlabeled targets dùng sau predictions để báo cáo. `test_kmeans()` khác: plain k-means trên normalized backbone features, bao gồm disjoint test khi gọi mỗi epoch. Không đánh đồng hai protocol.
Code lưu best khi `old_acc_test` tăng, rồi báo train metrics epoch đó. Đây là lựa chọn checkpoint dựa test labels; không phải nhãn trực tiếp trong loss, nhưng gây selection bias. Cần báo nguyên bản tách riêng; đánh giá phương pháp mới phải preregister final epoch hoặc validation known riêng, không tune bằng evaluation ground truth.

## Discrepancies và blockers
Paper supplementary ghi Cars fine-tune 3 blocks (code script 10 => blocks 10,11, tức 2); smoothing Cars 1.0 trong paper nhưng train script .5. Eval script hardcode smoothing 1, hmw 1, cr 1.5, khác Cars train. Chưa tự quyết bản nào là đúng để công bố fair reproduction.
Hardcoded CUDA device và `.cuda()`; đổi flag không đủ để chạy CPU/MPS. Máy chưa có torch/numpy/scipy/pytest; chưa có dataset/weights. requirements pin torch 2.4.1, numpy 1.24.4, scipy 1.10.1; README Python3.8 chưa bảo đảm tương thích mọi dependency hiện tại. Chưa có full optimizer/scheduler/RNG resume; warmup load weights không phải exact resume. Chưa chạy imports/forward/backward/metric parity.

## Runtime portable bổ sung (không sửa upstream)
`gcd_runtime/adapter.py` kiểm checksum training source rồi áp dụng thay đổi trong bộ nhớ: bỏ import DINOv2 không dùng, shim import joblib, hỗ trợ epoch checkpoint, final_epoch selection và metrics có nhãn protocol. Các class/hàm loss quan trọng được so sánh AST giữ nguyên qua test. Chưa đo parity whole training trên CUDA.
Phát hiện dependency mismatch thực tế: `kmeans-pytorch==0.3` trên PyPI không nhận `tqdm_flag` mà upstream truyền. `gcd_runtime/compat.py` chỉ bỏ cờ hiển thị nếu signature không hỗ trợ; fixture clustering 6 điểm đã chạy. Chưa đối chiếu numerical parity với commit GitHub kmeans_pytorch mà tác giả sử dụng; cần ghi đây là reproducibility caveat. PyPI kmeans có vòng lặp hội tụ không có iteration limit, do đó outer worker deadline vẫn bắt buộc; trường hợp NaN/empty cluster cần điều tra log, không coi job treo là kết quả.
