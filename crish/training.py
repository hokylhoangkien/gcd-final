"""One training path for strict matched B0 and executable CRiSH v2.1.

No benchmark labels or evaluation metrics are consumed here. Periodic and final
evaluation run in separate workers on locked backbone snapshots.
"""
import json
import time

import numpy as np
import torch

from crish.partition_audit import audit_stack
from .config import CrishSettings
from .engine import CrishEngine
from .loss import hypselex_loss
from .diagnostics import supervision_dose


def train_strict(head, model, train_loader, merge_loader, args, runtime, *,
                 objective=hypselex_loss, extractor=None):
    if extractor is None:
        from gcd_runtime import trainer
        extractor = trainer.extract_labeled_protos
    args.use_target_normalization = runtime.config.get('use_target_normalization', 'legacy')
    args.method = runtime.config['method']
    args.pair_reliability = runtime.config.get('pair_reliability')
    optimizer = torch.optim.SGD(list(head.parameters())+list(model.parameters()),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*1e-3)
    start_epoch = runtime.restore(model, head, optimizer, scheduler)
    if start_epoch >= args.epochs:
        if (runtime.config.get('checkpoint_selection') == 'best_old_test_v2'
                or getattr(runtime, 'allow_completed_resume', False)):
            print('TRAIN already complete; restored final checkpoint and recovered selection; no SGD steps', flush=True)
            return
        raise ValueError('Checkpoint reached final epoch; evaluate instead of resuming')
    engine = CrishEngine(runtime, objective=objective)
    from .population import PopulationEngine
    population = PopulationEngine(runtime, objective=objective)
    settings = CrishSettings(**runtime.config['crish'])
    device = next(head.parameters()).device
    print(f'TRAIN plan method={runtime.config["method"]} device={device} epochs={args.epochs} '
          f'batch={runtime.config["batch_size"]} max_steps_per_epoch={runtime.config.get("max_steps_per_epoch", 0)} '
          f'trainable_backbone={sum(p.numel() for p in model.parameters() if p.requires_grad)} '
          f'trainable_projector={sum(p.numel() for p in head.parameters() if p.requires_grad)}', flush=True)
    for epoch in range(start_epoch, args.epochs):
        runtime.epoch = epoch
        start = time.monotonic()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        runtime.hierarchy_state = None
        with torch.no_grad():
            extractor(model, merge_loader, args)
        state = runtime.hierarchy_state
        if state is None:
            raise RuntimeError('HSSK extraction failed to capture a sanitized supervision snapshot')
        audit = audit_stack(state.labels)
        audit.update(epoch=epoch, kind='observed_partition_geometry_not_semantic_accuracy')
        with (runtime.run_dir/'partition_audit.jsonl').open('a') as handle:
            handle.write(json.dumps(audit, allow_nan=False)+'\n')
        if settings.save_partitions and (epoch == start_epoch or settings.due(epoch)):
            np.savez_compressed(runtime.run_dir/f'partitions_epoch_{epoch:04d}.npz',
                                ids=state.ids.numpy(), labeled=state.labeled.numpy(),
                                **{f'level_{i}': labels.numpy() for i, labels in enumerate(state.labels)})
        before_intervention = state
        state = engine.prepare(state, model, head, optimizer, train_loader.dataset, args, epoch)
        state = population.prepare(state, model, head, optimizer, train_loader.dataset, args, epoch)
        if runtime.config.get('require_intervention_smoke') and runtime.config['method'] == 'crish':
            if population.settings.due(epoch):
                record = getattr(runtime, 'last_population_record', {})
                exercised = any('discovery' in c for c in record.get('candidates', []))
            elif settings.due(epoch):
                record = getattr(runtime, 'last_local_record', {})
                exercised = bool(record.get('gains'))
            else:
                record, exercised = {}, True
            if record.get('status') == 'fallback_keep' or not exercised:
                raise RuntimeError('Bounded intervention smoke did not execute a counterfactual: '+
                                   record.get('reason', 'no scored candidate; inspect support/cache/budget'))
        dose = supervision_dose(before_intervention, state)
        with (runtime.run_dir/'intervention_diagnostics.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(epoch=epoch, **dose), allow_nan=False)+'\n')
        if settings.save_partitions and dose['changed_ids']:
            np.savez_compressed(runtime.run_dir/f'supervision_epoch_{epoch:04d}.npz',
                ids=state.ids.numpy(), labeled=state.labeled.numpy(),
                **{f'level_{i}': y.numpy() for i, y in enumerate(state.labels)},
                **{f'abstained_{i}': y.numpy() for i, y in enumerate(state.abstained)})
        runtime.hierarchy_state = state
        model.train()
        head.train()
        total, steps = 0., 0
        pair_summary = dict(eligible_pairs=0, changed_pairs=0, weight_sum=0., weight_histogram=[0]*5)
        pair_batches = 0
        for batch_index, (views, targets, ids, membership) in enumerate(train_loader):
            limit = runtime.config.get('max_steps_per_epoch', 0)
            if limit and steps >= limit:
                break
            runtime.begin_batch(epoch, batch_index)
            membership = membership[:, 0].bool()
            # Hidden labels are neither inspected nor forwarded. The materialized
            # production dataset stores -1 for every unlabeled row.
            legal = targets[membership].long()
            raw = model(torch.cat(views).to(device))
            result = objective(head, raw, ids, membership, legal, state, args, epoch)
            pair_record = result.get('pair_reliability')
            if pair_record is not None:
                pair_batches += 1
                for key in ('eligible_pairs', 'changed_pairs', 'weight_sum'):
                    pair_summary[key] += pair_record[key]
                pair_summary['weight_histogram'] = [a+b for a, b in zip(
                    pair_summary['weight_histogram'], pair_record['weight_histogram'])]
                if batch_index % 20 == 0:
                    with (runtime.run_dir/'pair_reliability.jsonl').open('a') as handle:
                        handle.write(json.dumps(dict(epoch=epoch, batch=batch_index, **pair_record), allow_nan=False)+'\n')
                    print(f'PAIR epoch={epoch} step={batch_index} status={pair_record["status"]} '
                          f'changed={pair_record["changed_pairs"]}/{pair_record["eligible_pairs"]} '
                          f'mean_weight={pair_record["mean_weight"]}', flush=True)
            runtime.log_loss_geometry(result['angle'], result['distance'], result['features'],
                                      membership.to(device), result['distance_weight'])
            optimizer.zero_grad(set_to_none=True)
            result['loss'].backward()
            parameters = [p for p in list(head.parameters())+list(model.parameters()) if p.grad is not None]
            if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f'Nonfinite/absent training gradients at epoch={epoch} step={batch_index}')
            optimizer.step()
            total += float(result['loss'].detach())
            steps += 1
            if batch_index % 20 == 0:
                print(f'TRAIN method={runtime.config["method"]} epoch={epoch} step={batch_index} loss={total/steps:.6f}', flush=True)
        if not steps:
            raise ValueError('No complete training batches; check data/batch size')
        scheduler.step()
        runtime.last_training_summary = dict(loss=total/steps, steps=steps, seconds=time.monotonic()-start,
                                             repairs=state.version, intervention=dose,
                                             method=runtime.config['method'],
                                             peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        if pair_batches:
            pair_summary.update(batches=pair_batches, mean_weight=(pair_summary['weight_sum']/pair_summary['eligible_pairs']
                                if pair_summary['eligible_pairs'] else None))
            runtime.last_training_summary['pair_reliability'] = pair_summary
        if getattr(runtime, 'monitor_before_save', False):
            # Upstream SelEx consumes plain-evaluation RNG before saving. On a
            # bounded evaluator failure save SGD + the pending RNG phase so
            # resume can replay evaluation without repeating an SGD epoch.
            try:
                runtime.monitor_epoch(epoch, model)
            except BaseException:
                runtime.save(epoch, model, head, optimizer, scheduler)
                raise
            runtime.save(epoch, model, head, optimizer, scheduler)
        else:
            # Historical isolated monitoring keeps the original save order.
            runtime.save(epoch, model, head, optimizer, scheduler)
            runtime.monitor_epoch(epoch, model)
    print(f'TRAIN complete method={runtime.config["method"]}; final={runtime.run_dir / "backbone_final.pt"}; '
          'oracle evaluation is a separate worker', flush=True)
