"""Synthetic full-loss/repair probe using the actual DINOHead implementation."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from crish.lookahead import preserve_rng
from crish.rebuild import prototype_radii
from crish.state import SupervisionState
from gcd_runtime.config import ROOT, choose_device
from .config import CrishSettings
from .engine import CrishEngine
from .loss import hypselex_loss
from gcd_runtime.worker import configure_environment


class FeatureFixture(Dataset):
    """Only synthetic features, legal labels and IDs. Never opens real data."""
    def __init__(self, features, labeled, targets):
        self.features = features.cpu()
        self.labeled = labeled.cpu()
        self.targets = torch.full((len(features),), -1, dtype=torch.long)
        self.targets[self.labeled] = targets.cpu()
        self.sample_ids = tuple(17+11*i for i in range(len(features)))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        views = [x+torch.randn_like(x)*.005 for _ in range(2)]
        return views, self.targets[index], self.sample_ids[index], torch.tensor([self.labeled[index]])


def crish_method_probe(cfg, requested='cuda', *, backbone_dim=768, hidden_dim=2048):
    """Scaled dimensions are test-only; remote defaults match the real projector."""
    configure_environment(ROOT/'runs/probe-placeholder')
    from gcd_runtime import trainer
    from models.vision_transformer import DINOHead
    from hyptorch.nn import ToPoincare
    from gcd_runtime.checkpoint import EpochRuntime
    device = choose_device(torch, requested)
    args = SimpleNamespace(c=cfg['c'], n_views=2, temperature=1., contrast_unlabel_only=False,
                           strategy='zero_one', unsupervised_smoothing=cfg['unsupervised_smoothing'],
                           sup_con_weight=cfg['sup_con_weight'], hyper_start_epoch=0, hyper_end_epoch=200,
                           hyper_max_weight=cfg['hyper_max_weight'])
    prior_args, prior_device = getattr(trainer, 'args', None), getattr(trainer, 'device', None)
    try:
        with preserve_rng():
            torch.manual_seed(604)
            trainer.args, trainer.device = args, device
            raw = torch.randn(32, backbone_dim)
            labels = torch.arange(32) % 4
            labeled = torch.zeros(32, dtype=torch.bool)
            labeled[[0, 1, 4, 5]] = True
            dataset = FeatureFixture(raw, labeled, labels[labeled])
            normalized = F.normalize(raw, dim=-1)
            levels = (labels, labels // 2)
            centers = tuple(torch.stack([normalized[y == g].mean(0) for g in range(int(y.max())+1)]) for y in levels)
            state = SupervisionState(torch.tensor(dataset.sample_ids), normalized, labeled, labels[labeled],
                                     levels, centers, tuple(prototype_radii(p) for p in centers),
                                     tuple(torch.zeros(32, dtype=torch.bool) for _ in levels))
            head = torch.nn.Sequential(DINOHead(in_dim=backbone_dim, out_dim=cfg['mlp_out_dim'],
                                                nlayers=3, hidden_dim=hidden_dim,
                                                bottleneck_dim=min(256, hidden_dim)),
                                       ToPoincare(c=cfg['c'], clip_r=cfg['cr'] or None, riemannian=False)).to(device)
            optimizer = torch.optim.SGD(head.parameters(), lr=.001, momentum=.9, weight_decay=cfg['weight_decay'])
            test_cfg = dict(cfg, crish=asdict(CrishSettings(warmup_epochs=0, refresh_every=1,
                            evidence_samples=16, evidence_trials=1, k=2, boundaries=1, max_targets=1,
                            max_accepted=1, min_support=1, train_batch=8, probe_known=1, probe_unlabeled=2,
                            max_protected_shift=1., max_refresh_seconds=60, save_partitions=False)))
            with tempfile.TemporaryDirectory(prefix='crish-runtime-probe-') as temp:
                runtime = EpochRuntime(temp, test_cfg)
                repaired = CrishEngine(runtime).prepare(state, torch.nn.Identity().to(device), head, optimizer,
                                                       dataset, args, 0)
                ids = state.ids[:8]
                views = torch.cat([raw[:8], raw[:8]+.01]).to(device)
                loss = hypselex_loss(head, views, ids, labeled[:8], labels[:8][labeled[:8]], repaired, args, 0)['loss']
                loss.backward()
                grads = [p.grad for p in head.parameters() if p.grad is not None]
                if not grads or not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
                    raise RuntimeError('CRiSH full-objective synthetic backward failed')
                path = Path(temp)/'crish.jsonl'
                record = json.loads(path.read_text().splitlines()[-1]) if path.exists() else None
                if cfg['method'] == 'crish' and (record is None or not record.get('gains')):
                    raise RuntimeError(f'CRiSH repair scorer was not exercised: {record}')
            return dict(status='synthetic_strict_training_probe_passed', method=cfg['method'], device=str(device),
                        backbone_dim=backbone_dim, projector_dim=cfg['mlp_out_dim'], loss=float(loss.detach()),
                        scoring_attempts=len(record['gains']) if record else 0,
                        accepted=record['accepted'] if record else 0, trained_epochs=0)
    finally:
        trainer.args, trainer.device = prior_args, prior_device
