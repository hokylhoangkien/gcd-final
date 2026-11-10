"""SelEx setup/evaluation; CRiSH is the sole matched-arm intervention."""
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.training_data import sanitize_training
from gcd_runtime.cluster_alignment import restore_input_order
from crish.training import train_strict
from .objective import selex_loss
from .hierarchy import extract_state
from .rng import upstream_policy, seed_training, seed_ss_setup, rng_digest


def make_args(cfg):
    args = SimpleNamespace(**cfg)
    args.use_ssb_splits, args.prop_train_labels = True, .5
    args.interpolation, args.crop_pct, args.image_size = 3, .875, 224
    args.n_views, args.contrast_unlabel_only, args.cluster_momentum = 2, False, 1.
    return args


def run(cfg, run_dir, mode, checkpoint='', epoch=None):
    from data.get_datasets import get_class_splits, get_datasets
    from data.augmentations import get_transform
    from models import vision_transformer, vision_transformer2
    from project_utils.general_utils import seed_torch
    # Upstream imports these before main seeds. Do not let worker-only import
    # initialization consume the restored per-epoch evaluation stream.
    from project_utils.cluster_and_log_utils import split_cluster_acc_v2
    from kmeans_pytorch import kmeans
    from methods.clustering.faster_mix_k_means_pytorch import K_Means
    from gcd_runtime.disjoint_evaluation import plain_kmeans
    from gcd_runtime.config import paths
    if not torch.cuda.is_available():
        raise RuntimeError('SelEx real-data worker requires NVIDIA CUDA')
    args = get_class_splits(make_args(cfg))
    if upstream_policy(cfg):
        seed_training(cfg['seed'])
    else:
        seed_torch(cfg['seed'])
    args.num_labeled_classes, args.num_unlabeled_classes = len(args.train_classes), len(args.unlabeled_classes)
    architecture = vision_transformer2 if cfg['dino'] == 'v2' else vision_transformer
    model = architecture.vit_base()
    filename = 'dinov2_vitb14_reg4_pretrain.pth' if cfg['dino'] == 'v2' else 'dino_vitb16_pretrain.pth'
    model.load_state_dict(torch.load(paths()[1]/filename, map_location='cpu', weights_only=True))
    model.cuda()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ('blocks.' in name and int(name.split('.')[1]) >= cfg['grad_from_block'])
    transform, evaluation_transform = get_transform('imagenet', image_size=224, args=args)
    class TwoViews:
        def __call__(self, image):
            return [transform(image), transform(image)]
    if upstream_policy(cfg) and mode not in ('train', 'select'):
        seed_ss_setup()
    merged, test, _, _ = get_datasets(cfg['dataset_name'], TwoViews(), evaluation_transform, args)
    runtime = SelExRuntime(run_dir, cfg, checkpoint if mode == 'train' and checkpoint else None,
        evaluation_epoch=epoch, evaluated_checkpoint=checkpoint if mode != 'train' else None,
        evaluation_kind='test' if mode == 'select' else 'ss')
    if mode != 'train':
        from gcd_runtime.selection import load_evaluation_checkpoint
        model.load_state_dict(load_evaluation_checkpoint(checkpoint, cfg, runtime))
        model.eval()
        context = nullcontext() if upstream_policy(cfg) else runtime.evaluation_context(
            epoch if epoch is not None else cfg['epochs']-1,
            'disjoint_test' if mode == 'select' else 'final_ss', model)
        with context:
            if mode == 'select':
                from .rng import begin_plain_worker, finish_plain_worker
                request = begin_plain_worker(runtime) if upstream_policy(cfg) else None
                result = evaluate_plain(model, test, args, runtime)
                if request is not None:
                    finish_plain_worker(runtime, request)
                return result
            merged.labelled_dataset.transform = merged.unlabelled_dataset.transform = evaluation_transform
            return evaluate_ss(model, merged, args, runtime)
    # No annotation-bearing evaluation dataset remains reachable by train/repair.
    merged = sanitize_training(merged, cfg['dataset_name'])
    test = None
    count = len(merged.labelled_dataset)
    weights = torch.DoubleTensor([1. if i < count else count/len(merged) for i in range(len(merged))])
    sampler = WeightedRandomSampler(weights, num_samples=len(merged))
    train = DataLoader(merged, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], sampler=sampler, drop_last=True)
    extraction = DataLoader(merged, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False)
    head = architecture.DINOHead(in_dim=768, out_dim=cfg['mlp_out_dim'], nlayers=3).cuda()
    def extract(model, loader, args):
        return extract_state(model, loader, args, runtime)
    return train_strict(head, model, train, extraction, args, runtime,
                        objective=selex_loss, extractor=extract)


class SelExRuntime(EpochRuntime):
    @property
    def monitor_before_save(self):
        return upstream_policy(self.config)

    allow_completed_resume = True

    def save(self, epoch, model, head, optimizer, scheduler):
        self.checkpoint_extra = dict(plain_rng_pending=getattr(self, 'plain_rng_pending', False))
        return super().save(epoch, model, head, optimizer, scheduler)

    def restore(self, model, head, optimizer, scheduler):
        start = super().restore(model, head, optimizer, scheduler)
        if self.resume and self.restored_plain_rng_pending:
            from .monitoring import monitor_plain
            monitor_plain(self, self.restored_epoch, model)
            self.save(self.restored_epoch, model, head, optimizer, scheduler)
        elif self.resume and start == self.config['epochs']:
            torch.save(model.state_dict(), self.run_dir/'backbone_final.tmp')
            import os
            os.replace(self.run_dir/'backbone_final.tmp', self.run_dir/'backbone_final.pt')
        return start

    def monitor_epoch(self, epoch, model):
        from gcd_runtime.monitoring import monitor_epoch
        from .monitoring import monitor_plain
        if upstream_policy(self.config):
            self.plain_rng_pending = True
        monitor_plain(self, epoch, model)
        monitor_epoch(self, epoch, model)

    def evaluation(self, metrics):
        if self.evaluation_kind != 'test':
            return super().evaluation(metrics)
        # Plain monitoring is not model selection: fixed-final is primary.
        from gcd_runtime.metrics import atomic_json, write_epoch
        from datetime import datetime, timezone
        from gcd_runtime.readiness import sha256
        record = dict(status='observed', unit='fraction', protocol='plain_kmeans_disjoint_test',
            epoch=self.evaluation_epoch, epoch_number=self.evaluation_epoch+1,
            seed=self.config['seed'], purpose='monitor_only_not_selection',
            timestamp_utc=datetime.now(timezone.utc).isoformat(), matching='hungarian_v2_global',
            checkpoint_sha256=sha256(Path(self.evaluated_checkpoint)), **metrics)
        atomic_json(self.run_dir/'test_evaluations'/f'epoch_{self.evaluation_epoch+1:04d}.json', record)
        write_epoch(self.run_dir/'test_metrics.jsonl', record)
        print(f'EVAL epoch={self.evaluation_epoch+1} protocol={record["protocol"]} '
              f'All={100*metrics["all_acc"]:.2f}% Old={100*metrics["old_acc"]:.2f}% '
              f'New={100*metrics["new_acc"]:.2f}% purpose=monitor_only_not_selection', flush=True)


@torch.no_grad()
def evaluate_plain(model, dataset, args, runtime):
    from gcd_runtime.disjoint_evaluation import plain_kmeans
    from project_utils.cluster_and_log_utils import split_cluster_acc_v2
    features, targets = [], []
    native = upstream_policy(runtime.config)
    before = rng_digest()
    for images, labels, _ in DataLoader(dataset, batch_size=args.batch_size if native else args.batch_size//2,
                                        num_workers=args.num_workers, shuffle=False):
        features.append(F.normalize(model(images.cuda()), dim=-1).cpu())
        targets.append(labels)
    truth = torch.cat(targets).numpy()
    predictions, backend = plain_kmeans(torch.cat(features), args.num_labeled_classes+args.num_unlabeled_classes,
                                        'cuda', iteration_limit=1000 if native else None)
    known = np.isin(truth, np.arange(args.num_labeled_classes))
    all_acc, old_acc, new_acc = split_cluster_acc_v2(truth, predictions, known)
    runtime.evaluation(dict(all_acc=float(all_acc), old_acc=float(old_acc), new_acc=float(new_acc),
        n_samples=len(truth), n_old=int(known.sum()), n_new=int((~known).sum()),
        rng_before_evaluation_sha256=before, rng_after_evaluation_sha256=rng_digest(),
        evaluation_rng=runtime.config['evaluation_rng'], **backend))


@torch.no_grad()
def evaluate_ss(model, dataset, args, runtime):
    from methods.clustering.faster_mix_k_means_pytorch import K_Means
    from project_utils.cluster_and_log_utils import split_cluster_acc_v2
    features, targets, memberships = [], [], []
    for images, labels, _, mask in DataLoader(dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers, shuffle=False):
        encoded = F.normalize(model(images.cuda()), dim=-1)
        # Upstream caches normalized features, then normalizes that cache again.
        features.append((F.normalize(encoded, dim=-1) if upstream_policy(runtime.config) else encoded).cpu())
        targets.append(labels); memberships.append(mask[:, 0].bool())
    features, truth, mask = torch.cat(features), torch.cat(targets), torch.cat(memberships)
    k = args.num_labeled_classes+args.num_unlabeled_classes
    # SelEx standalone evaluator seeds clustering at 0 and uses ten restarts.
    if not upstream_policy(runtime.config):
        np.random.seed(0); torch.manual_seed(0)
    before_fit = rng_digest()
    fit = K_Means(k=k, tolerance=1e-4, max_iterations=args.ss_max_iter, init='k-means++',
        n_init=args.ss_n_init, random_state=None, n_jobs=None, pairwise_batch_size=1024,
        mode=None, protos=None, cluster_size=math.ceil(len(truth)/k))
    fit.fit_mix(features[~mask].cuda(), features[mask].cuda(), truth[mask].cuda())
    predictions = restore_input_order(fit.labels_, mask).cpu().numpy()[~mask.numpy()]
    known = np.isin(truth[~mask].numpy(), np.arange(args.num_labeled_classes))
    all_acc, old_acc, new_acc = split_cluster_acc_v2(truth[~mask].numpy(), predictions, known)
    runtime.evaluation(dict(all_acc=float(all_acc), old_acc=float(old_acc), new_acc=float(new_acc),
        n_unlabeled=int((~mask).sum()), n_old=int(known.sum()), n_new=int((~known).sum()),
        ss_n_init=args.ss_n_init, ss_max_iter=args.ss_max_iter, clustering_seed=0,
        evaluation_seed=0, evaluation_rng=runtime.config['evaluation_rng'],
        rng_before_fit_sha256=before_fit,
        seed_stage='before_dataset_setup' if upstream_policy(runtime.config) else 'before_fit',
        baseline_family='selex', upstream_sha=args.upstream_sha))
