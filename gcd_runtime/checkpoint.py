"""Atomic checkpoints and diagnostics shared by CRiSH and the B0 control."""
import json
import os
import random
from pathlib import Path
from .config import code_fingerprint


class EpochRuntime:
    def __init__(self, run_dir, config, resume=None):
        self.run_dir, self.config, self.resume = Path(run_dir), config, resume
        self.epoch = self.batch_idx = 0

    def capture_hierarchy(self, ids, features, labeled, legal_targets, labels, prototypes):
        import torch
        from crish.state import SupervisionState
        from crish.rebuild import prototype_radii
        radii = []
        for level, centers in enumerate(prototypes):
            try:
                radii.append(prototype_radii(centers))
            except ValueError as exc:
                raise ValueError(f'Hierarchy level={level}: {exc}') from exc
        self.hierarchy_state = SupervisionState(torch.as_tensor(ids), torch.as_tensor(features),
            torch.as_tensor(labeled, dtype=torch.bool), torch.as_tensor(legal_targets, dtype=torch.long),
            tuple(x.long() for x in labels), tuple(prototypes),
            tuple(radii),
            tuple(torch.zeros(len(ids), dtype=torch.bool) for _ in labels))

    def restore(self, model, head, optimizer, scheduler):
        if not self.resume:
            return 0
        import numpy as np
        import torch
        state = torch.load(self.resume, map_location='cpu', weights_only=False)
        if state['code_fingerprint'] != code_fingerprint():
            raise ValueError('Source changed since checkpoint; use the original source release to resume')
        if state['config'] != self.config:
            raise ValueError('Resume requires the exact original config, epochs and seed')
        model.load_state_dict(state['model'])
        head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        return state['epoch']+1

    def save(self, epoch, model, head, optimizer, scheduler):
        # No oracle metric argument exists: checkpoint choice cannot depend on it.
        import numpy as np
        import torch
        state = dict(epoch=epoch, config=self.config, code_fingerprint=code_fingerprint(),
            model=model.state_dict(), head=head.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        temporary = self.run_dir/'resume.tmp'
        torch.save(state, temporary)
        os.replace(temporary, self.run_dir/'resume.pt')
        if epoch+1 == self.config['epochs']:
            temporary = self.run_dir/'backbone_final.tmp'
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, self.run_dir/'backbone_final.pt')
        record = dict(epoch=epoch, status='synthetic_fixture' if self.config.get('synthetic_fixture') else 'observed',
            protocol='training_without_oracle_labels', All=None, Old=None, New=None,
            **getattr(self, 'last_training_summary', {}))
        with (self.run_dir/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')
        print(f'RUNTIME epoch_complete={epoch} resume={self.run_dir / "resume.pt"}', flush=True)

    def evaluation(self, metrics):
        record = dict(status='synthetic_fixture' if self.config.get('synthetic_fixture') else 'observed',
            unit='fraction', protocol='balanced_ss_kmeans_train_unlabeled', **metrics)
        (self.run_dir/'evaluation.json').write_text(json.dumps(record, allow_nan=False, indent=2))

    def evaluation_context(self, epoch, kind, model):
        from .evaluation_policy import evaluation_context
        return evaluation_context(self.config, epoch, kind, model)

    def begin_batch(self, epoch, batch_idx):
        self.epoch, self.batch_idx = epoch, batch_idx

    def cluster_labels(self, grouped_labels, anchor_mask):
        from .cluster_alignment import restore_input_order
        return restore_input_order(grouped_labels, anchor_mask)

    def log_loss_geometry(self, angle_loss, distance_loss, features, labeled, distance_weight):
        every = self.config.get('training_diagnostics_every', 0)
        if not every or self.batch_idx % every:
            return
        from .training_diagnostics import loss_geometry
        record = loss_geometry(angle_loss, distance_loss, features, labeled, distance_weight)
        record.update(epoch=self.epoch, batch=self.batch_idx,
            status='synthetic_fixture' if self.config.get('synthetic_fixture') else 'observed')
        with (self.run_dir/'loss_geometry.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')
