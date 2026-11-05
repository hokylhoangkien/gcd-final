# Reference losses/HSSK derived from Visual-AI/HypCD (MIT), commit
# ffc895b7ff38be8502bb7e3bd851f1d2a55dff73. See NOTICE and LICENSES/HypCD-MIT.txt.
"""Single CRiSH/B0 setup, source-faithful losses, HSSK and final evaluation."""
import argparse
import math
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import hyptorch.nn as hypnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from project_utils.general_utils import init_experiment, str2bool
from project_utils.cluster_and_log_utils import log_accs_from_preds
from gcd_runtime.kmeans import SemiSupKMeans
from models import vision_transformer as vits1
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from hyptorch.pmath import dist_matrix
from config import exp_root, dino_pretrain_path, dinov2_pretrain_path
from tqdm import tqdm
warnings.filterwarnings("ignore")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity, smoothing=0.5):
        target_smooth = F.one_hot(target, input.size(1)).float() * (1 - smoothing) + smoothing * similarity  # F.one_hot(similarity,input.size(1)).float()#s1/input.size(0)#coef# / self.num_classes
        return torch.nn.CrossEntropyLoss()(input, target_smooth)


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, hyper_c=0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.hyper_c = hyper_c

    def forward(self, features, labels=None, mask=None, is_code=False):  # , smoothing=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if is_code:
            dist = torch.cdist(anchor_feature, contrast_feature)
            dist = -dist / (dist.sum(dim=1) + 1e-10)
        else:
            if self.hyper_c == 0:
                dist = -torch.cdist(anchor_feature, contrast_feature)
            else:
                dist = -dist_matrix(anchor_feature, contrast_feature, c=args.c)

        anchor_dot_contrast = torch.div(dist, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, confusion_factor, args, is_code=False, hyper_c=0):
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    if is_code:
        dist = torch.cdist(features, features, p=2)
        similarity_matrix = -dist / (dist.sum(dim=1) + 1e-10)
    else:
        if hyper_c == 0:
            similarity_matrix = -torch.cdist(features, features)
        else:
            similarity_matrix = -dist_matrix(features, features, c=args.c)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    pos_confs = confusion_factor[labels.bool()].view(confusion_factor.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg_confs = confusion_factor[~labels.bool()].view(confusion_factor.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    log_confs = torch.cat([pos_confs, neg_confs], dim=1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels, log_confs


def train(projection_head, model, train_loader, merge_train_loader, args):
    """Both methods use the same loop; only CrishEngine's repair switch differs."""
    from crish.training import train_strict
    return train_strict(projection_head, model, train_loader, merge_train_loader, args, _runtime)



def extract_labeled_protos(model, train_loader, args):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    ids = np.array([])

    for batch_idx, (images, label, uq_idx, mask_lab_) in enumerate(tqdm(train_loader)):
        images = images[0].to(device)
        label, mask_lab_ = label.to(device), mask_lab_.to(device).bool()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        ids = np.append(ids, uq_idx.cpu().numpy())
        mask = np.append(mask, mask_lab_.cpu().bool().numpy())

    mask = mask.astype(bool)
    # -----------------------
    # K-MEANS
    # -----------------------
    # print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = targets[mask]  # Get labelled targets
    n_samples = len(targets)

    if args.unbalanced:
        cluster_size = None
    else:
        cluster_size = math.ceil(n_samples / (args.num_labeled_classes + args.num_unlabeled_classes))
    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None, cluster_size=cluster_size)

    l_feats, u_feats, l_targets = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats, l_targets))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    all_preds = _runtime.cluster_labels(kmeanssem.labels_, mask)
    prototype_higher = []
    prototypes = kmeanssem.cluster_centers_
    prototype_higher.append(prototypes.clone())
    n_labeled = args.num_labeled_classes
    n_novel = args.num_unlabeled_classes
    label_proto = prototypes.cpu().numpy()[:args.num_labeled_classes, :]
    preds_higher = []

    preds_higher.append(all_preds.clone())
    print('Hierarchy clustering')
    mask_known = (all_preds < args.num_labeled_classes).cpu().numpy()
    l_feats = all_feats[mask_known]  # Get labelled set
    u_feats = all_feats[~mask_known]
    l_feats, u_feats = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats))

    while n_labeled > 1:
        n_labeled = max(int(n_labeled / 2), 1)
        n_novel = max(int(n_novel / 2), 1)
        # kmeans on labeled proto
        kmeans_l = KMeans(n_clusters=n_labeled, random_state=0).fit(label_proto)
        preds_labels = torch.from_numpy(kmeans_l.labels_).to(device)
        level_l_targets = preds_labels[all_preds[mask_known]]  # all labeled data at the new cluster level
        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil(n_samples / (n_labeled + n_novel))
        # sskmeans on features with new level labels
        kmeans_higher = SemiSupKMeans(k=n_labeled + n_novel, tolerance=1e-4,
                                      max_iterations=10, init='k-means++',
                                      n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                                      mode=None, protos=None, cluster_size=cluster_size)
        kmeans_higher.fit_mix(u_feats, l_feats, level_l_targets)
        preds_level = _runtime.cluster_labels(kmeans_higher.labels_, mask_known)
        prototypes_level = kmeans_higher.cluster_centers_
        prototype_higher.append(prototypes_level.clone())
        preds_higher.append(preds_level.to(device).clone())

    _runtime.capture_hierarchy(ids, all_feats, mask, l_targets.detach().cpu(), preds_higher, prototype_higher)
    return _runtime.hierarchy_state


def evaluate(model, test_loader, args):
    # test the model accuracy using BSSKMeans
    model.eval()

    with torch.no_grad():
        all_feats = []
        targets = np.array([])
        mask = np.array([])
        ids = np.array([])
        mask_cls = np.array([])
        metrics = dict()

        for batch_idx, (images, label, uq_idx, mask_lab_) in enumerate(tqdm(test_loader)):
            if isinstance(images, list):
                images = images[0]
            images = images.to(device)
            label, mask_lab_ = label.to(device), mask_lab_.to(device).bool()

            # Pass features through base model and then additional learnable transform (linear layer)
            feats = model(images)
            all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            ids = np.append(ids, uq_idx.cpu().numpy())
            mask = np.append(mask, mask_lab_.cpu().bool().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

        mask = mask.astype(bool)
        mask_cls = mask_cls.astype(bool)
        # -----------------------
        # K-MEANS
        # -----------------------
        # print('Fitting K-Means...')
        all_feats = np.concatenate(all_feats)
        l_feats = all_feats[mask]  # Get labelled set
        u_feats = all_feats[~mask]  # Get unlabelled set
        l_targets = targets[mask]  # Get labelled targets
        u_targets = targets[~mask]  # Get unlabelled targets
        n_samples = len(targets)

        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil(n_samples / (args.num_labeled_classes + args.num_unlabeled_classes))
        kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                                  max_iterations=10, init='k-means++',
                                  n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                                  mode=None, protos=None, cluster_size=cluster_size)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats, l_targets, u_targets))

        kmeanssem.fit_mix(u_feats, l_feats, l_targets)
        all_preds = _runtime.cluster_labels(kmeanssem.labels_, mask)
        mask_cls = mask_cls[~mask]
        preds = all_preds.cpu().numpy()[~mask]
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls, eval_funcs=args.eval_funcs, save_name='SS-K-Means Train ACC Unlabelled Test', print_output=True)
        metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
        _runtime.evaluation(metrics)
        return metrics


def run(argv, runtime):
    """Run one configured process; dataset/weights are provided externally."""
    global args, device, _runtime
    _runtime = runtime

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--kmeans_timeout_seconds', default=120, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--grad_from_block', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--mlp_out_dim', default=256, type=int)

    parser.add_argument('--strategy', type=str, default='zero_one')
    parser.add_argument('--cluster_momentum', type=float, default=1)

    parser.add_argument('--unsupervised_smoothing', type=float, default=1)
    parser.add_argument('--distance', type=str, default='euclidean', help='options: euclidean, cosine')

    parser.add_argument('--train_report_interval', default=200, type=int)
    parser.add_argument('--prototype_extraction_interval', default=1, type=int)

    parser.add_argument('--gpu_clustering', type=str2bool, default=True)
    parser.add_argument('--unbalanced', type=str2bool, default=False)

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--report', type=str2bool, default=True)

    # hyperbolic
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_model_path', default=None, type=str)
    parser.add_argument('--hyper_start_epoch', default=0, type=int)
    parser.add_argument('--hyper_end_epoch', default=200, type=int)
    parser.add_argument('--c', default=0.05, type=float)
    parser.add_argument('--cr', type=float, default=0)
    parser.add_argument('--riemannian', type=bool, default=False)
    parser.add_argument('--dino', default='v1', type=str)
    parser.add_argument('--hyper_max_weight', default=1.0, type=float)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args(argv)
    from .config import choose_device
    device = choose_device(torch, runtime.config.get('device', 'auto'))
    if device.type != 'cuda':
        raise RuntimeError('Full trainer requires NVIDIA CUDA; local tests use synthetic train fixtures')
    args = get_class_splits(args)
    set_random_seed(args.seed)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'HypSelEx_{args.dataset_name}'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':
        args.interpolation = 3
        args.crop_pct = 0.875
        if args.dino == 'v1':
            model = vits1.__dict__['vit_base']()
            torch.cuda.empty_cache()
            state_dict = torch.load(dino_pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)
        elif args.dino == 'v2':
            from models import vision_transformer2 as vits2
            model = vits2.vit_base()
            torch.cuda.empty_cache()
            model.load_state_dict(torch.load(dinov2_pretrain_path, map_location='cpu', weights_only=True))
        else:
            raise ValueError('Unsupported DINO variant')

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir + 'model_best.pt', map_location='cpu'), strict=False)
        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        max_block = 0
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num > max_block:
                    max_block = block_num

                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:
        raise NotImplementedError
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)
    if args.eval_only:
        if args.eval_model_path is None:
            raise ValueError('Evaluation requires a locked backbone checkpoint')
        train_dataset.labelled_dataset.transform = test_transform
        train_dataset.unlabelled_dataset.transform = test_transform
        evaluation_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                                       batch_size=args.batch_size, shuffle=False)
        model.load_state_dict(torch.load(args.eval_model_path, map_location='cpu', weights_only=True))
        with torch.no_grad(), runtime.evaluation_context(args.epochs-1, 'final_ss', model):
            return evaluate(model, evaluation_loader, args)

    from .training_data import sanitize_cars_training
    train_dataset = sanitize_cars_training(train_dataset)
    # Release all annotation-bearing evaluation objects before training.
    test_dataset = unlabelled_train_examples_test = datasets = None

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / (unlabelled_len + label_len) for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    merge_train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True)

    projection_head = vits1.__dict__['DINOHead'](in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    if args.cr != 0:
        hyperbolic_projector = hypnn.ToPoincare(c=args.c, ball_dim=args.mlp_out_dim, riemannian=args.riemannian, clip_r=args.cr).to(device)
    else:
        hyperbolic_projector = hypnn.ToPoincare(c=args.c, ball_dim=args.mlp_out_dim, riemannian=args.riemannian).to(device)
    if args.warmup_model_dir is not None:
        print(f'Loading projection head weights from {args.warmup_model_dir}')
        projection_head.load_state_dict(torch.load(args.warmup_model_dir + 'model_proj_head_best.pt', map_location='cpu'), strict=False)

    projection_head = nn.Sequential(projection_head, hyperbolic_projector)
    projection_head.to(device)

    return train(projection_head, model, train_loader, merge_train_loader, args)
