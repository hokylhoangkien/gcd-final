"""Atomic epoch checkpoints, RNG restoration and protocol-labelled metrics."""
import json
import os
import random
from pathlib import Path
from .config import code_fingerprint


class EpochRuntime:
    def __init__(self, run_dir, config, resume=None):
        self.run_dir = Path(run_dir)
        self.config = config
        self.resume = resume
        self.epoch = 0
        self.batch_idx = 0
        self._weights_cache = {}

    def restore(self, model, head, optimizer, scheduler):
        if not self.resume:
            return 0
        import numpy as np
        import torch
        # Trusted, self-produced checkpoints only; includes NumPy RNG objects.
        state = torch.load(self.resume, map_location='cpu', weights_only=False)
        if state['code_fingerprint'] != code_fingerprint():
            raise ValueError('Runtime source changed since checkpoint; resume rejected')
        if state['config'] != self.config:
            raise ValueError('Resume config differs; epochs and seed must also match')
        model.load_state_dict(state['model'])
        head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        return state['epoch'] + 1

    def save(self, epoch, model, head, optimizer, scheduler, all_acc, old, new,
             test_all, test_old, test_new):
        import numpy as np
        import torch
        state = dict(epoch=epoch, config=self.config, code_fingerprint=code_fingerprint(), model=model.state_dict(),
                     head=head.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        temporary = self.run_dir / 'resume.tmp'
        torch.save(state, temporary)
        os.replace(temporary, self.run_dir / 'resume.pt')
        if epoch + 1 == self.config['epochs']:
            final_tmp = self.run_dir / 'backbone_final.tmp'
            torch.save(model.state_dict(), final_tmp)
            os.replace(final_tmp, self.run_dir / 'backbone_final.pt')
        # Upstream training reports SS-kmeans except its final reporting epoch,
        # where it switches to plain kmeans. Preserve that distinction explicitly.
        protocol = 'plain_kmeans_train_unlabeled' if (epoch + 1) % 200 == 0 else 'balanced_ss_kmeans_train_unlabeled'
        record = dict(epoch=epoch, status='synthetic_fixture' if self.config.get('synthetic_fixture') else 'observed', unit='fraction', protocol=protocol,
                      All=float(all_acc), Old=float(old), New=float(new),
                      disjoint_test=dict(protocol='plain_kmeans', All=float(test_all),
                                         Old=float(test_old), New=float(test_new)))
        with (self.run_dir / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        print(f'RUNTIME epoch_complete={epoch} resume={self.run_dir / "resume.pt"}', flush=True)

    def evaluation(self, metrics):
        record = dict(status='observed', unit='fraction',
                      protocol='balanced_ss_kmeans_train_unlabeled', **metrics)
        (self.run_dir / 'evaluation.json').write_text(json.dumps(record, allow_nan=False, indent=2))

    def begin_batch(self, epoch, batch_idx):
        self.epoch, self.batch_idx = epoch, batch_idx
        self._weights_cache.clear()

    def supervised_losses(self, distance_criterion, angle_criterion, features, labels):
        if self.config.get('method') == 'proposed' and len(features) == 0:
            # A sampler may draw an all-unlabeled batch; no supervised term exists.
            zero = features.sum() * 0
            return zero, zero
        return distance_criterion(features, labels=labels), angle_criterion(features, labels=labels)

    def normalize_confusion(self, confusion):
        if self.config.get('method') != 'proposed':
            return confusion / confusion.sum(dim=1)
        import torch
        if not torch.isfinite(confusion).all():
            raise ValueError('Non-finite confusion input before row normalization')
        # Preserve upstream's broadcast convention (column-wise denominator).
        # Only zero-mass columns receive a finite uniform fallback.
        mass = confusion.sum(dim=1)
        uniform = torch.ones_like(confusion) - torch.eye(len(confusion), device=confusion.device)
        uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1)
        safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
        return torch.where(mass > 0, confusion / safe_mass, uniform)

    @staticmethod
    def check_loss(loss, epoch, batch_idx):
        import torch
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at epoch={epoch}, batch={batch_idx}')
        return loss

    def hierarchy_losses(self, distance_criterion, angle_criterion, features,
                         pseudo_labels, labeled, sample_ids, epoch, level):
        return (self.hierarchy_loss(distance_criterion, features, pseudo_labels, labeled, sample_ids, epoch, level),
                self.hierarchy_loss(angle_criterion, features, pseudo_labels, labeled, sample_ids, epoch, level))

    def hierarchy_loss(self, criterion, features, pseudo_labels, labeled, sample_ids, epoch, level):
        if self.config.get('method') != 'proposed':
            return criterion(features, labels=pseudo_labels)
        import torch
        from .method_config import ReliabilitySettings
        from .reliability import neighborhood_scores, pair_weights, weighted_supcon
        settings = ReliabilitySettings(**self.config['reliability'])
        if (not settings.reciprocal and not settings.cross_view) or settings.strength(epoch) == 0 or settings.floor == 1:
            return criterion(features, labels=pseudo_labels)
        key = (epoch, level, features.data_ptr(), tuple(features.shape))
        if key not in self._weights_cache:
            sample_ids = torch.as_tensor(sample_ids, device=features.device)
            scores, details = neighborhood_scores(features, sample_ids, self.config['c'], settings)
            weights, eligible, fallback = pair_weights(scores, pseudo_labels, labeled, sample_ids, settings, epoch)
            self._weights_cache[key] = weights
            if self.batch_idx % settings.diagnostics_every == 0:
                self._diagnostics(weights, scores, eligible, fallback, labeled, details, epoch, level, settings)
        return weighted_supcon(features, pseudo_labels, self._weights_cache[key], criterion, self.config['c'])

    def _diagnostics(self, weights, scores, eligible, fallback, labeled, details, epoch, level, settings):
        import torch
        selected = weights[eligible].detach().float()
        selected_scores = scores[eligible].detach().float()
        def mean_or_none(values):
            return float(values.mean().cpu()) if values.numel() else None
        record = dict(epoch=epoch, batch=self.batch_idx, level=level,
                      strength=settings.strength(epoch), eligible_pairs=int(eligible.sum().cpu()),
                      weight_mean=mean_or_none(selected), raw_score_mean=mean_or_none(selected_scores),
                      downweighted_fraction=mean_or_none((selected < 1).float()),
                      zero_weight_fraction=mean_or_none((selected == 0).float()),
                      effective_weight_mass=float(selected.sum().cpu()),
                      fallback_anchor_fraction=float(fallback.float().mean().cpu()),
                      agreement_mean=float(details['agreement'].mean().cpu()),
                      unique_samples=int(details['unique_samples']),
                      labeled_anchor_weight_mean=mean_or_none(weights[eligible & labeled[:, None]]),
                      unlabeled_anchor_weight_mean=mean_or_none(weights[eligible & ~labeled[:, None]]),
                      weight_histogram=torch.histc(selected, bins=10, min=0, max=1).long().cpu().tolist())
        with (self.run_dir / 'reliability.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
