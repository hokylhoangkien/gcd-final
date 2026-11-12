"""Atomic checkpoints and diagnostics shared by CRiSH and the B0 control."""
import json
import os
import random
from pathlib import Path
from .config import code_fingerprint


class EpochRuntime:
    def __init__(self, run_dir, config, resume=None, *, evaluation_epoch=None, evaluated_checkpoint=None,
                 evaluation_kind='ss'):
        self.run_dir, self.config, self.resume = Path(run_dir), config, resume
        self.epoch = self.batch_idx = 0
        self.evaluation_epoch = evaluation_epoch
        self.evaluated_checkpoint = evaluated_checkpoint
        self.evaluation_kind = evaluation_kind
        self.evaluation_role = 'monitor_only' if evaluation_epoch is not None else 'final'

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
        from .ss_best import restore_best
        restore_best(Path(self.resume).parent, self.run_dir, self.config, state['epoch'])
        model.load_state_dict(state['model'])
        head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        from .selection import enabled, restore_artifacts
        if enabled(self.config):
            restore_artifacts(Path(self.resume).parent, self.run_dir, self.config, state['epoch'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.restored_epoch = state['epoch']
        self.restored_plain_rng_pending = state.get('plain_rng_pending', False)
        if enabled(self.config):
            # A failure may have happened after saving SGD but before selecting
            # this epoch. Recover that evaluation before advancing any SGD.
            report = self.run_dir/'test_evaluations'/f'epoch_{state["epoch"]+1:04d}.json'
            if not report.is_file():
                from .monitoring import monitor_epoch
                monitor_epoch(self, state['epoch'], model, selection_only=True)
            if state['epoch']+1 == self.config['epochs']:
                torch.save(model.state_dict(), self.run_dir/'backbone_final.tmp')
                os.replace(self.run_dir/'backbone_final.tmp', self.run_dir/'backbone_final.pt')
        return state['epoch']+1

    def save(self, epoch, model, head, optimizer, scheduler):
        # No oracle metric argument enters the training/resume checkpoint.
        import numpy as np
        import torch
        state = dict(epoch=epoch, config=self.config, code_fingerprint=code_fingerprint(),
            model=model.state_dict(), head=head.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        state.update(getattr(self, 'checkpoint_extra', {}))
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
        metrics = dict(metrics)
        from datetime import datetime, timezone
        import hashlib
        import math
        from .evaluation_policy import evaluation_seed
        from .metrics import atomic_json, summarize_run, write_epoch
        from .selection import TEST_PROTOCOL, enabled, select_checkpoint
        from selex_runtime.selection import enabled as selex_best_enabled
        secondary_best = selex_best_enabled(self.config)
        for key in ('all_acc', 'old_acc', 'new_acc'):
            if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
                raise ValueError(f'Invalid evaluation metric: {key}')
        epoch = self.config['epochs']-1 if self.evaluation_epoch is None else self.evaluation_epoch
        is_test = self.evaluation_kind == 'test'
        role = 'selection_test' if is_test else self.evaluation_role
        record = dict(status='synthetic_fixture' if self.config.get('synthetic_fixture') else 'observed',
            unit='fraction', protocol=TEST_PROTOCOL if is_test else 'balanced_ss_kmeans_train_unlabeled',
            matching='hungarian_v2_global',
            epoch=epoch, epoch_number=epoch+1, seed=self.config['seed'],
            evaluation_seed=metrics.pop('evaluation_seed', evaluation_seed(self.config, epoch, 'disjoint_test' if is_test else 'final_ss')),
            purpose=role, checkpoint_selection=self.config.get('checkpoint_selection', 'final_epoch'),
            # A fixed last/monitor snapshot is not chosen by test scores even
            # when this run also maintains a separate test-selected best.
            selected_using_evaluation_labels=(enabled(self.config) or secondary_best) and role == 'best_old_test',
            checkpoint_selection_uses_evaluation_labels=enabled(self.config) or (secondary_best and role == 'best_old_test'),
            matching_uses_evaluation_labels=True,
            used_for_checkpoint_selection=is_test,
            timestamp_utc=datetime.now(timezone.utc).isoformat(), **metrics)
        if secondary_best:
            record.update(primary='fixed_final_ss_v2',
                secondary_best_checkpoint_rule='max_old_v1_disjoint_plain_test_strict_improvement',
                selection_matching=getattr(self, 'selection_matching', None))
        for key in ('all_acc', 'old_acc', 'new_acc'):
            record[key] = float(record[key])
        if self.evaluated_checkpoint:
            checkpoint = Path(self.evaluated_checkpoint)
            digest = hashlib.sha256()
            with checkpoint.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(block)
            record.update(checkpoint=str(checkpoint), checkpoint_sha256=digest.hexdigest())
        if is_test:
            select_checkpoint(self.run_dir, self.config, self.evaluated_checkpoint, record)
            destination = self.run_dir/'test_evaluations'/f'epoch_{epoch+1:04d}.json'
            history = self.run_dir/'test_metrics.jsonl'
        elif role == 'final':
            destination = self.run_dir/'evaluation_last.json'
            if not enabled(self.config):
                atomic_json(self.run_dir/'evaluation.json', record)
            history = self.run_dir/'epoch_metrics.jsonl'
        elif role == 'best_old_test':
            destination = self.run_dir/'evaluation_best_old_test.json'
            if enabled(self.config):
                atomic_json(self.run_dir/'evaluation.json', record)
            # Separate best/last histories: they can share the same epoch.
            history = self.run_dir/('best_old_ss_metrics.jsonl' if secondary_best else 'epoch_metrics.jsonl')
        else:
            destination = self.run_dir/'evaluations'/f'epoch_{epoch+1:04d}.json'
            history = self.run_dir/'epoch_metrics.jsonl'
        atomic_json(destination, record)
        write_epoch(history, record)
        from .ss_best import update_best
        update_best(self.run_dir, self.config, record)
        summarize_run(self.run_dir, write=True)
        print(f'EVAL epoch={epoch+1}/{self.config["epochs"]} protocol={record["protocol"]} '
              f'All={100*metrics["all_acc"]:.2f}% Old={100*metrics["old_acc"]:.2f}% '
              f'New={100*metrics["new_acc"]:.2f}% purpose={role} report={destination}', flush=True)

    def monitor_epoch(self, epoch, model):
        from .monitoring import monitor_epoch
        monitor_epoch(self, epoch, model)

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
