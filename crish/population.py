"""Global base-supervision abstention; bounded, training-only counterfactuals."""
from dataclasses import replace
import json
import math
import random
import time

import numpy as np
import torch

from .config import CrishSettings
from .diagnostics import base_target_mass, supervision_dose
from .global_policy import apply_threshold, epoch_arrays, knn_evidence
from .engine import (_check_deadline, _draw_context, cached_views, fixed_modes, projector_optimizer)
from .ids import rows_for_ids
from .lookahead import clone_snapshot, exact_sgd_comparison, preserve_rng
from .population_settings import PopulationSettings
from .probe import LegalProbe, ProbeWeights
from .rebuild import target_drift
from .sampling import known_fold_structure


@torch.no_grad()
def assignment_reliability(state, *, device='cpu', block_size=256, deadline=None):
    """Occupied-prototype squared-distance margin over ALL training rows."""
    support = torch.bincount(state.labels[0], minlength=len(state.prototypes[0]))
    occupied = support > 0
    if int(occupied.sum()) < 2:
        raise ValueError('Population ranking needs >=2 occupied base clusters')
    centers = state.prototypes[0].to(device).float()
    scores = []
    for start in range(0, len(state.ids), block_size):
        if deadline is not None: _check_deadline(deadline)
        features = state.features[start:start+block_size].to(device).float()
        distance = torch.cdist(features, centers).square()
        own = state.labels[0][start:start+block_size].to(device)
        assigned = distance.gather(1, own[:, None]).squeeze(1)
        distance[:, ~occupied.to(device)] = torch.inf
        distance.scatter_(1, own[:, None], torch.inf)
        alternative = distance.min(1).values
        score = .5+.5*(alternative-assigned)/(alternative+assigned+1e-8)
        if not torch.isfinite(score).all():
            raise ValueError('Nonfinite global geometric assignment reliability')
        scores.append(score.clamp(0, 1).cpu())
    return torch.cat(scores)


def abstain_fraction(state, reliability, fraction, settings, *, generator=None):
    """Global low-margin selection with quotas; never invent missing support."""
    if (reliability.shape != state.ids.shape or not torch.isfinite(reliability).all()
            or ((reliability < 0) | (reliability > 1)).any()
            or not math.isfinite(fraction) or not 0 <= fraction < 1):
        raise ValueError('Invalid population scores/fraction')
    rows = (~state.labeled).nonzero().flatten()
    if state.abstained[0].any():
        raise ValueError('Population policy requires a fresh unmodified base snapshot')
    count = int(len(rows)*fraction)
    if not count:
        return state, dict(requested_fraction=fraction, actual_fraction=0., selected=0,
                           retained_mass=base_target_mass(state), saturated=False)
    # Tie order is stable by sample ID, not source-loader position.
    rows = rows[state.ids[rows].argsort(stable=True)]
    if settings.mode == 'random':
        rows = rows[torch.randperm(len(rows), generator=generator)]
    else:
        rows = rows[reliability[rows].argsort(stable=True)]
    labels = state.labels[0]
    support = torch.bincount(labels[~state.labeled], minlength=len(state.prototypes[0]))
    capacity = torch.minimum((support.double()*settings.max_cluster_fraction).floor().long(),
                             (support-settings.min_cluster_kept).clamp_min(0))
    used = torch.zeros_like(capacity)
    selected = []
    for row in rows.tolist():
        group = int(labels[row])
        if used[group] < capacity[group]:
            selected.append(row); used[group] += 1
            if len(selected) == count: break
    if not selected:
        return state, dict(requested_fraction=fraction, actual_fraction=0., selected=0,
                           retained_mass=base_target_mass(state), saturated=True)
    abstained = list(state.abstained)
    abstained[0] = abstained[0].clone()
    abstained[0][selected] = True
    proposed = replace(state, abstained=tuple(abstained), version=state.version+1)
    return proposed, dict(requested_fraction=fraction, actual_fraction=len(selected)/len(rows),
        selected=len(selected), retained_mass=base_target_mass(proposed),
        saturated=len(selected) < count, per_cluster_selected=used.tolist())


class PopulationEngine:
    def __init__(self, runtime, *, objective):
        self.runtime, self.objective = runtime, objective
        self.settings = PopulationSettings(**runtime.config.get('population_policy', {}))
        self.probe_settings = CrishSettings(**runtime.config['crish'])
        # CRiSH-G persists the chosen threshold, never a list of sample IDs.
        self.policy = dict(tau=0., selected_epoch=None, source=None)
        restored = getattr(runtime, 'restored_population_policy', None)
        if self.settings.is_global and restored is not None:
            self.policy = dict(tau=float(restored['tau']), selected_epoch=restored.get('selected_epoch'),
                               source=restored.get('source'))

    def prepare(self, state, model, head, optimizer, dataset, args, epoch):
        cfg, settings = self.runtime.config, self.settings
        if settings.is_global:
            return self._prepare_global(state, model, head, optimizer, dataset, args, epoch)
        if cfg['method'] != 'crish' or not settings.due(epoch): return state
        started = time.monotonic(); deadline = started+settings.max_refresh_seconds
        seed = (cfg['seed']*1000003+epoch*103+71) % (2**32)
        record = dict(epoch=epoch, seed=seed, mode=settings.mode, status='keep', accepted=0,
                      chosen_fraction=0., candidates=[], ranking='global_occupied_prototype_margin',
                      lookahead_scope='projector', scoring_sampling='uniform_nonprobe_without_replacement')
        try:
            with preserve_rng(), fixed_modes(model):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                generator = torch.Generator().manual_seed(seed)
                device = next(head.parameters()).device
                reliability = assignment_reliability(state, device=device, block_size=settings.rank_block,
                                                     deadline=deadline)
                scores = reliability[~state.labeled]
                record.update(ranked_rows=len(state.ids), ranked_unlabeled=len(scores),
                              reliability_min=float(scores.min()), reliability_max=float(scores.max()))
                if settings.mode != 'random' and float(scores.max()-scores.min()) <= 1e-8:
                    record['reason'] = 'No varying geometric evidence'; return state
                folds, _ = _draw_context(state, self.probe_settings, generator, deadline, fold_count=2)
                reserved = torch.cat([torch.cat([k, u]) for k, u in folds])
                available = torch.ones(len(state.ids), dtype=torch.bool); available[reserved] = False
                available = available.nonzero().flatten()
                if len(available) < 4:
                    raise ValueError('Insufficient disjoint policy-gradient rows')
                rows = available[torch.randperm(len(available), generator=generator)[:settings.train_batch]]
                ids, membership = state.ids[rows], state.labeled[rows]
                views = cached_views(model, dataset, ids, device, deadline)
                flat = torch.cat([views[:, v] for v in range(2)]).detach()
                frozen, virtual_optimizer = clone_snapshot(head, projector_optimizer(head, optimizer))
                frozen.eval()
                parameters = tuple(p for p in frozen.parameters() if p.requires_grad)
                registry = dict(zip(state.ids[state.labeled].tolist(), state.known_targets.tolist()))
                probes = []
                for known, other in folds:
                    probe_ids = torch.cat([state.ids[known], state.ids[other]])
                    x = cached_views(model, dataset, probe_ids, device, deadline)
                    with torch.no_grad(): teacher = frozen(x[len(known):].flatten(0, 1)).reshape(len(other), 2, -1)
                    probes.append(LegalProbe(known_ids=state.ids[known], known_views=x[:len(known)],
                        known_targets=state.labels[0][known], unlabeled_ids=state.ids[other],
                        unlabeled_views=x[len(known):], teacher_views=teacher, proposal_ids=ids,
                        legal_labels=registry, weights=ProbeWeights(self.probe_settings.probe_known_weight,
                            self.probe_settings.probe_view_weight, self.probe_settings.probe_variance_weight,
                            self.probe_settings.probe_std_floor)))
                probes[0].assert_disjoint(probes[1])
                record.update(scoring_rows=len(ids), probe_folds=[dict(known_fold_structure(state.labels[0][k]),
                    unlabeled_rows=len(u)) for k, u in folds])
                targets = state.labels[0][rows][membership]
                result = self.objective(frozen, flat, ids, membership, targets, state, args, epoch)
                g0 = torch.autograd.grad(result['loss'], parameters, allow_unused=True)
                pair_weights = result.get('pseudo_pair_weights')
                best = None
                for q in settings.candidates(epoch):
                    if q == 0: continue  # exact Keep comparison has gain zero
                    _check_deadline(deadline)
                    proposed, coverage = abstain_fraction(state, reliability, q, settings,
                                                          generator=torch.Generator().manual_seed(seed+1))
                    entry = dict(coverage, status='rejected'); record['candidates'].append(entry)
                    if proposed is state or coverage['retained_mass'] < settings.min_retained_mass:
                        entry['reason'] = 'No admissible coverage/retained target mass'; continue
                    drift = target_drift(state, proposed, ids, flat, smoothing=args.unsupervised_smoothing,
                        strategy=args.strategy, normalization=getattr(args, 'use_target_normalization', 'legacy'),
                        pair_weights=pair_weights)
                    entry['target_drift'] = drift
                    if drift['sse_levels'][0]['remaining_mass_mean'] < settings.min_retained_mass:
                        entry['reason'] = 'Scoring-batch retained weighted target mass too low'; continue
                    if drift['sse_levels'][0]['coefficient_half_l1_mean'] <= 0:
                        entry['reason'] = 'Policy has no changed pseudo-SSE targets in the scoring batch'; continue
                    if max(x['protected_coefficient_shift_max'] for x in drift['sse_levels']) > self.probe_settings.max_protected_shift:
                        raise ValueError('Population policy changed protected coefficients')
                    loss = self.objective(frozen, flat, ids, membership, targets, proposed, args, epoch)['loss']
                    ga = torch.autograd.grad(loss, parameters, allow_unused=True)
                    comparison = exact_sgd_comparison(frozen, virtual_optimizer, g0, ga, probes[0])
                    entry['discovery'] = comparison
                    gain, known = comparison['gain'], comparison['known_increase']
                    if (not math.isfinite(gain) or not math.isfinite(known)
                            or gain <= self.probe_settings.acceptance_gain or known > self.probe_settings.known_tolerance):
                        entry['reason'] = 'Discovery gain/known guard failed'; continue
                    entry['status'] = 'eligible'
                    if best is None or gain > best[0]: best = (gain, proposed, ga, entry)
                if best is None:
                    record['reason'] = 'No eligible population candidate'; return state
                _check_deadline(deadline)
                _, proposed, ga, entry = best
                comparison = exact_sgd_comparison(frozen, virtual_optimizer, g0, ga, probes[1])
                entry['confirmation'] = comparison
                gain, known = comparison['gain'], comparison['known_increase']
                if (not math.isfinite(gain) or not math.isfinite(known)
                        or gain <= self.probe_settings.acceptance_gain or known > self.probe_settings.known_tolerance):
                    entry.update(status='rejected', reason='Winner confirmation failed; no runner-up search')
                    record['reason'] = entry['reason']; return state
                _check_deadline(deadline)
                entry['status'] = 'accepted'
                record.update(status='completed', accepted=1, chosen_fraction=entry['requested_fraction'],
                              actual_fraction=entry['actual_fraction'], dose=supervision_dose(state, proposed))
                return proposed
        except (ValueError, RuntimeError) as error:
            record.update(status='fallback_keep', reason=str(error)); return state
        finally:
            record['seconds'] = time.monotonic()-started
            self.runtime.last_population_record = record
            with (self.runtime.run_dir/'crish_p.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
            print(f'CRISH_P epoch={epoch} status={record["status"]} q={record["chosen_fraction"]} '
                  f'accepted={record["accepted"]} seconds={record["seconds"]:.2f}', flush=True)

    # ----------------------------------------------------------------- CRiSH-G
    def _evidence(self, state, device, deadline=None):
        return knn_evidence(state, k=self.settings.knn_k, nonreciprocal_weight=self.settings.nonreciprocal_weight,
                            device=device, query_block=self.settings.rank_block, deadline=deadline)

    def _apply(self, state, evidence, tau, epoch):
        return apply_threshold(state, evidence, tau, self.settings, min_support=self.probe_settings.min_support,
                               random_seed=(self.runtime.config['seed'], epoch))

    def _remember(self, tau, epoch):
        self.policy = dict(tau=float(tau), selected_epoch=epoch, source='due')
        self.runtime.checkpoint_extra = dict(getattr(self.runtime, 'checkpoint_extra', {}),
                                             population_policy_state=dict(self.policy))

    def _prepare_global(self, state, model, head, optimizer, dataset, args, epoch):
        """Due epochs choose tau*; with persist, every later epoch re-applies pi_tau*.

        Non-due epochs only recompute deterministic kNN evidence on the freshly
        extracted features/partition: no probe, no RNG draw, no model forward.
        """
        cfg, settings = self.runtime.config, self.settings
        if cfg['method'] != 'crish' or not settings.active(epoch):
            return state
        started = time.monotonic()
        evidence = coverage = None
        outcome, status, reason = state, 'completed', None
        if settings.due(epoch):
            tau, outcome, coverage, evidence = self._select_threshold(state, model, head, optimizer, dataset, args, epoch)
            self._remember(tau, epoch)
            source = 'due'
            status = self.runtime.last_population_record['status']
            reason = self.runtime.last_population_record.get('reason')
        else:
            tau, source = self.policy['tau'], 'persisted'
            try:
                device = next(head.parameters()).device if head is not None else 'cpu'
                evidence = self._evidence(state, device)
                outcome, coverage = self._apply(state, evidence, tau, epoch)
            except (ValueError, RuntimeError) as error:
                outcome, status, reason, coverage = state, 'fallback_keep', str(error), None
        self._log_epoch(epoch, tau, source, state, outcome, evidence, coverage, status, reason,
                        time.monotonic()-started)
        return outcome

    def _log_epoch(self, epoch, tau, source, before, after, evidence, coverage, status, reason, seconds):
        applied = after is not before
        levels = []
        for entry in (coverage or {}).get('levels', []):
            entry = {key: entry[key] for key in (
                'level', 'scoped', 'candidates', 'abstained_unlabeled', 'fraction', 'cluster_capped',
                'truncated_by_mass_floor', 'retained_mass', 'reliability_histogram',
                'reliability_mean_unlabeled', 'random_shortfall') if key in entry}
            if not applied:
                entry.update(abstained_unlabeled=0, fraction=0., retained_mass=1.)
            levels.append(entry)
        moved = int((coverage or {}).get('moved', 0)) if applied else 0
        record = dict(epoch=epoch, tau=float(tau), source=source, status=status, reason=reason,
                      applied=applied, moved=moved, levels=levels, seconds=seconds,
                      knn_seconds=None if evidence is None else evidence.seconds,
                      mutual_fraction=None if evidence is None else evidence.mutual_fraction,
                      knn_k=self.settings.knn_k, ranking='global_knn_agreement',
                      sample_choice=self.settings.sample_choice, levels_scope=self.settings.levels,
                      actions=self.settings.actions, persist=self.settings.persist)
        self.runtime.last_policy_epoch_record = record
        with (self.runtime.run_dir/'crish_policy_epochs.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')
        if self.settings.save_arrays and evidence is not None:
            np.savez_compressed(self.runtime.run_dir/f'crish_g_epoch_{epoch:04d}.npz',
                                **epoch_arrays(before, after, evidence))
        print(f'CRISH_G epoch={epoch} tau={float(tau)} source={source} '
              f'abstained={[x["abstained_unlabeled"] for x in levels]} moved={moved} '
              f'seconds={seconds:.2f}', flush=True)

    def _scoring_rows(self, state, available, seed, generator):
        settings = self.settings
        count = min(len(available), settings.scoring_batches*settings.train_batch)
        if settings.scoring_sampling == 'uniform':
            return available[torch.randperm(len(available), generator=generator)[:count]]
        # Same marginal weights as the training WeightedRandomSampler, drawn
        # without replacement from a refresh-seeded private generator.
        n_labeled = int(state.labeled.sum())
        weights = torch.where(state.labeled[available], torch.ones((), dtype=torch.float64),
                              torch.full((), n_labeled/len(state.ids), dtype=torch.float64))
        private = torch.Generator().manual_seed((seed+2) % (2**32))
        return available[torch.multinomial(weights, count, replacement=False, generator=private)]

    def _select_threshold(self, state, model, head, optimizer, dataset, args, epoch):
        """Counterfactual dose choice over tau; Keep (tau=0) is always the reference."""
        cfg, settings = self.runtime.config, self.settings
        started = time.monotonic(); deadline = started+settings.max_refresh_seconds
        seed = (cfg['seed']*1000003+epoch*103+71) % (2**32)
        record = dict(epoch=epoch, seed=seed, mode=settings.mode, gate=settings.gate, status='keep', accepted=0,
                      chosen_threshold=0., chosen_fraction=0., candidates=[], ranking='global_knn_agreement',
                      levels_scope=settings.levels, actions=settings.actions, sample_choice=settings.sample_choice,
                      persist=settings.persist, lookahead_scope='projector', probe_float64=settings.probe_float64,
                      scoring_sampling=('train_sampler_weights_without_replacement'
                                        if settings.scoring_sampling == 'train_sampler'
                                        else 'uniform_nonprobe_without_replacement'))
        result = dict(tau=0., chosen=state, coverage=None, evidence=None)
        try:
            with preserve_rng(), fixed_modes(model):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                generator = torch.Generator().manual_seed(seed)
                device = next(head.parameters()).device
                evidence = result['evidence'] = self._evidence(state, device, deadline)
                unlabeled = ~state.labeled
                record.update(ranked_rows=len(state.ids), ranked_unlabeled=int(unlabeled.sum()),
                              knn_seconds=evidence.seconds, mutual_fraction=evidence.mutual_fraction,
                              reliability=[dict(level=level, min=float(r[unlabeled].min()),
                                                max=float(r[unlabeled].max()), mean=float(r[unlabeled].mean()))
                                           for level, r in enumerate(evidence.reliability)])
                keep_coverage = result['coverage'] = self._apply(state, evidence, 0., epoch)[1]
                if settings.gate == 'none':
                    tau = settings.fixed_threshold
                    chosen, coverage = self._apply(state, evidence, tau, epoch)
                    record['candidates'].append(dict(threshold=float(tau), status='applied_without_probe',
                                                     coverage=coverage))
                    record.update(status='completed', accepted=int(chosen is not state), chosen_threshold=float(tau),
                                  dose=supervision_dose(state, chosen))
                    result.update(tau=float(tau), chosen=chosen, coverage=coverage)
                    return result['tau'], result['chosen'], result['coverage'], evidence
                folds, _ = _draw_context(state, self.probe_settings, generator, deadline, fold_count=2)
                reserved = torch.cat([torch.cat([k, u]) for k, u in folds])
                available = torch.ones(len(state.ids), dtype=torch.bool); available[reserved] = False
                available = available.nonzero().flatten()
                if len(available) < 4:
                    raise ValueError('Insufficient disjoint policy-gradient rows')
                rows = self._scoring_rows(state, available, seed, generator)
                size = settings.train_batch
                if len(rows) >= size:
                    rows = rows[:len(rows)//size*size]
                else:
                    size = len(rows)
                dtype = torch.float64 if settings.probe_float64 else None
                views = cached_views(model, dataset, state.ids[rows], device, deadline)
                chunks = []
                for start in range(0, len(rows), size):
                    part, v = rows[start:start+size], views[start:start+size]
                    flat = torch.cat([v[:, 0], v[:, 1]]).detach()
                    membership = state.labeled[part]
                    chunks.append(dict(ids=state.ids[part], membership=membership, drift_views=flat,
                                       flat=flat if dtype is None else flat.to(dtype),
                                       targets=state.labels[0][part][membership]))
                frozen, virtual_optimizer = clone_snapshot(head, projector_optimizer(head, optimizer))
                frozen.eval()
                if dtype is not None:
                    # Cast the clone in place: optimizer references stay valid.
                    frozen.to(dtype)
                    for slot in virtual_optimizer.state.values():
                        for key, value in list(slot.items()):
                            if torch.is_tensor(value) and value.is_floating_point():
                                slot[key] = value.to(dtype)
                parameters = tuple(p for p in frozen.parameters() if p.requires_grad)
                registry = dict(zip(state.ids[state.labeled].tolist(), state.known_targets.tolist()))
                probes = []
                for known, other in folds:
                    probe_ids = torch.cat([state.ids[known], state.ids[other]])
                    x = cached_views(model, dataset, probe_ids, device, deadline)
                    if dtype is not None:
                        x = x.to(dtype)
                    with torch.no_grad(): teacher = frozen(x[len(known):].flatten(0, 1)).reshape(len(other), 2, -1)
                    probes.append(LegalProbe(known_ids=state.ids[known], known_views=x[:len(known)],
                        known_targets=state.labels[0][known], unlabeled_ids=state.ids[other],
                        unlabeled_views=x[len(known):], teacher_views=teacher, proposal_ids=state.ids[rows],
                        legal_labels=registry, weights=ProbeWeights(self.probe_settings.probe_known_weight,
                            self.probe_settings.probe_view_weight, self.probe_settings.probe_variance_weight,
                            self.probe_settings.probe_std_floor)))
                probes[0].assert_disjoint(probes[1])
                record.update(scoring_rows=len(rows), scoring_chunks=len(chunks), scoring_chunk_rows=size,
                              scoring_labeled_rows=int(state.labeled[rows].sum()),
                              probe_folds=[dict(known_fold_structure(state.labels[0][k]), unlabeled_rows=len(u))
                                           for k, u in folds])

                def mean_gradients(current):
                    """Average of per-chunk gradients keeps SelEx's per-batch normalization."""
                    total, weights = None, []
                    for chunk in chunks:
                        output = self.objective(frozen, chunk['flat'], chunk['ids'], chunk['membership'],
                                                chunk['targets'], current, args, epoch)
                        grads = torch.autograd.grad(output['loss'], parameters, allow_unused=True)
                        weights.append(output.get('pseudo_pair_weights'))
                        total = grads if total is None else tuple(
                            b if a is None else a if b is None else a+b for a, b in zip(total, grads))
                    return tuple(None if g is None else g/len(chunks) for g in total), weights

                g0, pair_weights = mean_gradients(state)
                best = None
                for threshold in settings.threshold_candidates():
                    if threshold == 0:
                        # Exact Keep comparison: identical gradients, gain zero.
                        record['candidates'].append(dict(threshold=0., status='keep_reference', gain=0.,
                                                         known_increase=0., coverage=keep_coverage))
                        continue
                    _check_deadline(deadline)
                    proposed, coverage = self._apply(state, evidence, threshold, epoch)
                    entry = dict(threshold=float(threshold), status='rejected', coverage=coverage)
                    record['candidates'].append(entry)
                    if proposed is state:
                        entry['reason'] = 'No admissible abstention/move at this threshold'; continue
                    drifts = [target_drift(state, proposed, chunk['ids'], chunk['drift_views'],
                                           smoothing=args.unsupervised_smoothing, strategy=args.strategy,
                                           normalization=getattr(args, 'use_target_normalization', 'legacy'),
                                           pair_weights=weights)
                              for chunk, weights in zip(chunks, pair_weights)]
                    levels = [dict(level=level,
                                   coefficient_half_l1_mean=sum(d['sse_levels'][level]['coefficient_half_l1_mean']
                                                                for d in drifts)/len(drifts),
                                   remaining_mass_mean=sum(d['sse_levels'][level]['remaining_mass_mean']
                                                           for d in drifts)/len(drifts),
                                   protected_coefficient_shift_max=max(
                                       d['sse_levels'][level]['protected_coefficient_shift_max'] for d in drifts))
                              for level in range(len(state.labels))]
                    entry['target_drift'] = dict(sse_levels=levels, chunks=len(drifts))
                    if max(x['protected_coefficient_shift_max'] for x in levels) > self.probe_settings.max_protected_shift:
                        raise ValueError('Population policy changed protected coefficients')
                    if all(x['coefficient_half_l1_mean'] <= 0 for x in levels):
                        entry['reason'] = 'Policy has no changed pseudo-SSE targets in the scoring rows'; continue
                    ga, _ = mean_gradients(proposed)
                    comparison = exact_sgd_comparison(frozen, virtual_optimizer, g0, ga, probes[0])
                    entry['discovery'] = comparison
                    gain, known = comparison['gain'], comparison['known_increase']
                    if (not math.isfinite(gain) or not math.isfinite(known)
                            or gain <= self.probe_settings.acceptance_gain or known > self.probe_settings.known_tolerance):
                        entry['reason'] = 'Discovery gain/known guard failed'; continue
                    entry['status'] = 'eligible'
                    if best is None or gain > best[0]: best = (gain, threshold, proposed, coverage, ga, entry)
                if best is None:
                    record['reason'] = 'No eligible population threshold'
                    return 0., state, keep_coverage, evidence
                _check_deadline(deadline)
                _, threshold, proposed, coverage, ga, entry = best
                comparison = exact_sgd_comparison(frozen, virtual_optimizer, g0, ga, probes[1])
                entry['confirmation'] = comparison
                gain, known = comparison['gain'], comparison['known_increase']
                if (not math.isfinite(gain) or not math.isfinite(known)
                        or gain <= self.probe_settings.acceptance_gain or known > self.probe_settings.known_tolerance):
                    entry.update(status='rejected', reason='Winner confirmation failed; no runner-up search')
                    record['reason'] = entry['reason']
                    return 0., state, keep_coverage, evidence
                _check_deadline(deadline)
                entry['status'] = 'accepted'
                record.update(status='completed', accepted=1, chosen_threshold=float(threshold),
                              dose=supervision_dose(state, proposed))
                result.update(tau=float(threshold), chosen=proposed, coverage=coverage)
                return result['tau'], proposed, coverage, evidence
        except (ValueError, RuntimeError) as error:
            record.update(status='fallback_keep', reason=str(error), accepted=0, chosen_threshold=0.)
            result.update(tau=0., chosen=state)
            return 0., state, result['coverage'], result['evidence']
        finally:
            record['seconds'] = time.monotonic()-started
            applied = result['chosen'] is not state and record['chosen_threshold'] > 0
            record['abstained_unlabeled'] = ([x['abstained_unlabeled'] for x in result['coverage']['levels']]
                                             if applied and result['coverage'] else [])
            record['moved'] = int(result['coverage'].get('moved', 0)) if applied and result['coverage'] else 0
            self.runtime.last_population_record = record
            with (self.runtime.run_dir/'crish_p.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
            print(f'CRISH_P epoch={epoch} status={record["status"]} tau={record["chosen_threshold"]} '
                  f'accepted={record["accepted"]} seconds={record["seconds"]:.2f}', flush=True)
