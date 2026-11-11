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

    def prepare(self, state, model, head, optimizer, dataset, args, epoch):
        cfg, settings = self.runtime.config, self.settings
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
