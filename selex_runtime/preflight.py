"""Real-image, actual-head NVIDIA preflight; never install or download inputs."""
import math
import torch
from gcd_runtime.config import paths, ROOT
from gcd_runtime.readiness import inspect_inputs, audit_dataset
from .reference import configure_environment, definitions
from .trainer import make_args


def preflight(cfg):
    if cfg['device'] != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('SelEx full runs require Linux NVIDIA CUDA')
    configure_environment(ROOT/'runs/probe-placeholder', cfg)
    result = inspect_inputs(cfg)
    result['data_counts'] = audit_dataset(cfg)
    from data.get_datasets import get_class_splits, get_datasets
    from data.augmentations import get_transform
    from models import vision_transformer, vision_transformer2
    from methods.clustering.faster_mix_k_means_pytorch import K_Means
    args = get_class_splits(make_args(cfg))
    architecture = vision_transformer2 if cfg['dino'] == 'v2' else vision_transformer
    model = architecture.vit_base()
    filename = 'dinov2_vitb14_reg4_pretrain.pth' if cfg['dino'] == 'v2' else 'dino_vitb16_pretrain.pth'
    model.load_state_dict(torch.load(paths()[1]/filename, map_location='cpu', weights_only=True))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ('blocks.' in name and int(name.split('.')[1]) >= cfg['grad_from_block'])
    model.cuda()
    head = architecture.DINOHead(in_dim=768, out_dim=cfg['mlp_out_dim'], nlayers=3).cuda()
    transform, evaluation = get_transform('imagenet', 224, args)
    merged, _, _, _ = get_datasets(cfg['dataset_name'], None, evaluation, args)
    images = [merged[0][0], merged[len(merged.labelled_dataset)][0]]
    views = torch.stack([transform(image) for _ in range(2) for image in images]).cuda()
    raw = head(model(views))
    normalized = torch.nn.functional.normalize(raw, dim=-1)
    paired = torch.stack(normalized.chunk(2), dim=1)
    loss = definitions('cuda')['SupConLoss']()(paired, labels=torch.arange(2, device='cuda'))
    loss.backward()
    gradients = [p.grad for p in list(model.parameters())+list(head.parameters()) if p.grad is not None]
    if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('SelEx pretrained image/head CUDA backward failed')
    torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=cfg['lr']).step()
    # Exercise the exact pinned fitter, not the historical Hyp-SelEx adapter.
    points = torch.tensor([[1., 0.], [0., 1.]], device='cuda')
    fit = K_Means(k=2, n_init=1, max_iterations=2,
                  cluster_size=None if cfg.get('unbalanced', False) else 2, random_state=0)
    fit.fit_mix(points.clone(), points.clone(), torch.arange(2, device='cuda'))
    if not torch.isfinite(fit.cluster_centers_).all():
        raise FloatingPointError('Pinned SelEx fitter readiness failed')
    from gcd_runtime.disjoint_evaluation import disjoint_probe
    result.update(cuda_verified=True, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        head_dimension=cfg['mlp_out_dim'], real_image_backward=True,
        disjoint_selection_probe=disjoint_probe('cuda'),
        synthetic_probe_not_benchmark=True)
    return result
