"""Small real tensor checks; never downloads weights or opens a dataset."""
import json
from .config import ROOT, choose_device


def tensor_probe(requested='auto', baseline=False):
    import sys
    import torch
    device = choose_device(torch, requested)
    torch.manual_seed(7)
    layer = torch.nn.Linear(8, 4).to(device)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    loss = layer(torch.randn(4, 8, device=device)).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in layer.parameters()):
        raise RuntimeError('Non-finite tensor probe')
    optimizer.step()
    sys.path.insert(0, str(ROOT / 'upstream/HypCD'))
    from hyptorch.nn import ToPoincare
    from hyptorch.pmath import dist_matrix
    features = torch.randn(4, 8, device=device, requires_grad=True)
    points = ToPoincare(c=0.1, clip_r=2.0)(features)
    distances = dist_matrix(points, points, c=0.1)
    geometric_loss = distances.square().mean()
    geometric_loss.backward()
    if not torch.isfinite(distances).all() or not torch.isfinite(features.grad).all():
        raise RuntimeError('Non-finite hyperbolic probe')
    if not (points.norm(dim=-1) < 0.1 ** -0.5).all():
        raise RuntimeError('Invalid Poincare radius')
    if baseline:
        from .worker import configure_environment
        configure_environment(ROOT / 'runs' / 'probe-placeholder')
        from . import trainer
        scope = vars(trainer)
        from types import SimpleNamespace
        scope['args'] = SimpleNamespace(c=0.1, n_views=2, temperature=1.0)
        scope['device'] = device
        z = points.detach().reshape(2, 2, 8).requires_grad_()
        loss = scope['SupConLoss'](hyper_c=0.1)(z, labels=torch.tensor([0, 1], device=device))
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(z.grad).all():
            raise RuntimeError('Upstream SupCon probe failed')
        contrast = points.detach().requires_grad_()
        logits, targets, similarity = scope['info_nce_logits'](
            contrast, torch.ones(4, 4, device=device), scope['args'], hyper_c=0.1)
        use_loss = scope['LabelSmoothingLoss']()(logits, targets, similarity, smoothing=0.5)
        use_loss.backward()
        if not torch.isfinite(use_loss) or not torch.isfinite(contrast.grad).all():
            raise RuntimeError('Upstream USE probe failed')
        from project_utils.cluster_and_log_utils import split_cluster_acc_v2
        import numpy as np
        all_acc, old, new = split_cluster_acc_v2(np.array([0, 0, 1, 1]),
                                                np.array([1, 1, 0, 0]),
                                                np.array([True, True, False, False]))
        if (all_acc, old, new) != (1.0, 1.0, 1.0):
            raise RuntimeError('Hungarian metric fixture failed')
        all_acc, old, new = split_cluster_acc_v2(np.array([0, 0, 0, 1]),
                                                np.array([1, 1, 0, 0]),
                                                np.array([True, True, True, False]))
        if not np.isclose(all_acc, 0.75) or not np.isclose(all_acc, 0.75 * old + 0.25 * new):
            raise RuntimeError('Sample-weighted All/Old/New fixture failed')
        # The previous method smoke mocked HSSK, so it missed empty-cluster
        # NaNs. Exercise the production fitter on this actual device as well.
        from .kmeans import SemiSupKMeans
        from crish.rebuild import prototype_radii
        clustering = SemiSupKMeans(k=4, max_iterations=2, n_init=1,
                                   random_state=7, pairwise_batch_size=16, cluster_size=2)
        anchors = torch.tensor([0, 1], device=device)
        clustering.fit_mix(torch.ones(6, 3, device=device), torch.ones(2, 3, device=device), anchors)
        if (not torch.equal(clustering.labels_[:2], anchors)
                or not (torch.bincount(clustering.labels_, minlength=4) > 0).all()
                or not torch.isfinite(prototype_radii(clustering.cluster_centers_)).all()):
            raise RuntimeError('Empty-cluster HSSK probe failed')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return dict(device=str(device), torch=torch.__version__, cuda_build=torch.version.cuda,
                cuda_verified=device.type == 'cuda', baseline_probe=baseline,
                kmeans_empty_cluster_probe=baseline,
                gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device.type == 'cuda' else None,
                synthetic_loss=float(geometric_loss.detach().cpu()))


def backbone_probe(cfg):
    """Remote-only synthetic ViT forward/backward using the actual pretrained file."""
    import torch
    from .config import paths
    from .worker import configure_environment
    configure_environment(ROOT / 'runs/probe-placeholder')
    from . import trainer
    from types import SimpleNamespace
    scope = vars(trainer)
    scope['args'] = SimpleNamespace(c=cfg['c'])
    from models.vision_transformer import vit_base, DINOHead
    from hyptorch.nn import ToPoincare
    _, weights, _ = paths()
    if cfg['dino'] == 'v2':
        from models.vision_transformer2 import vit_base as vit_base_v2
        model = vit_base_v2()
        filename = 'dinov2_vitb14_reg4_pretrain.pth'
    else:
        model = vit_base()
        filename = 'dino_vitb16_pretrain.pth'
    model.load_state_dict(torch.load(weights / filename, map_location='cpu', weights_only=True))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = 'blocks.' in name and int(name.split('.')[1]) >= cfg['grad_from_block']
    model = model.cuda()
    head = torch.nn.Sequential(DINOHead(in_dim=768, out_dim=cfg['mlp_out_dim'], nlayers=3),
                               ToPoincare(c=cfg['c'], clip_r=cfg['cr'], riemannian=False)).cuda()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=cfg['lr'])
    # Exercise the selected dataset's decoding and two-view train transforms.
    from data.get_datasets import get_class_splits, get_datasets
    from data.augmentations import get_transform
    name = cfg.get('dataset_name', 'scars')
    image_args = SimpleNamespace(dataset_name=name, use_ssb_splits=True, prop_train_labels=.5, interpolation=3, crop_pct=.875)
    image_args = get_class_splits(image_args)
    transform, test_transform = get_transform('imagenet', image_size=224, args=image_args)
    view_transform = scope['ContrastiveLearningViewGenerator'](transform, n_views=2)
    merged, _, _, _ = get_datasets(name, view_transform, test_transform, image_args)
    views = [merged[0][0], merged[len(merged.labelled_dataset)][0]]
    images = torch.cat([torch.stack([sample[view] for sample in views]) for view in range(2)]).cuda()
    features = head(model(images))
    paired = torch.stack(features.chunk(2), dim=1)
    loss = scope['SupConLoss'](hyper_c=cfg['c'])(paired, labels=torch.tensor([0, 1], device='cuda'))
    loss.backward()
    grads = [p.grad for p in list(model.parameters()) + list(head.parameters()) if p.grad is not None]
    if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Pretrained ViT synthetic backward failed')
    optimizer.step()
    torch.cuda.synchronize()
    return dict(status='real_image_pipeline_probe_passed', batch_images=4, loss=float(loss.detach()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    print(json.dumps(tensor_probe(args.device, args.baseline), indent=2))
def method_probe(cfg, requested='cuda'):
    from crish.smoke import crish_method_probe
    return crish_method_probe(cfg, requested)
