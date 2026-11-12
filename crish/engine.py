"""Executable bounded refresh policy on frozen backbone features.

Lookahead is projector-only with the real current projector SGD state. Evidence
uses normalized backbone Euclidean neighbors; this is logged explicitly. M2 is
optional and defaults off until incremental value has been measured.
"""
from contextlib import contextmanager
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from crish.candidates import propose_actions
from crish.ids import rows_for_ids
from crish.lookahead import clone_snapshot, exact_sgd_comparison, gradient_gain, preserve_rng
from crish.policy import DecisionEvidence, sequential_repair
from crish.probe import LegalProbe, ProbeWeights
from crish.rebuild import target_drift
from crish.reliability import blockwise_euclidean_knn, sparse_evidence
from crish.sampling import balanced_known_folds, global_neighbor_pool, known_fold_structure
from crish.scoring_batch import scoring_ids
from crish.stability import partition_stability
from .config import CrishSettings
from .loss import hypselex_loss


@contextmanager
def fixed_modes(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def projector_optimizer(head, optimizer):
    """Copy the SGD state for the projector subset; never reset momentum."""
    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError('CRiSH lookahead supports SGD only')
    owned = set(head.parameters())
    groups = []
    for group in optimizer.param_groups:
        parameters = [p for p in group['params'] if p in owned]
        if parameters:
            groups.append(dict(group, params=parameters))
    if {p for g in groups for p in g['params']} != owned:
        raise ValueError('Projector parameters missing from the live optimizer')
    subset = torch.optim.SGD(groups, lr=optimizer.defaults['lr'])
    for parameter in owned:
        if parameter in optimizer.state:
            subset.state[parameter] = copy.deepcopy(optimizer.state[parameter])
    return subset


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise RuntimeError('CRiSH refresh budget exhausted; remaining actions Keep')


@torch.no_grad()
def cached_views(model, dataset, ids, device, deadline, batch_size=16):
    if not hasattr(dataset, 'sample_ids'):
        raise ValueError('Strict training dataset must expose stable sample_ids')
    indices = rows_for_ids(dataset.sample_ids, ids).tolist()
    first, second = [], []
    with fixed_modes(model):
        for start in range(0, len(indices), batch_size):
            _check_deadline(deadline)
            samples = [dataset[index] for index in indices[start:start+batch_size]]
            expected = ids[start:start+batch_size].tolist()
            if [int(sample[2]) for sample in samples] != expected:
                raise ValueError('Dataset ID lookup changed during repair refresh')
            if any(not isinstance(sample[0], (tuple, list)) or len(sample[0]) != 2 for sample in samples):
                raise ValueError('Repair cache requires two independent augmented views')
            x = torch.cat([torch.stack([sample[0][v] for sample in samples]) for v in range(2)]).to(device)
            a, b = model(x).detach().chunk(2)
            first.append(a)
            second.append(b)
    return torch.stack([torch.cat(first), torch.cat(second)], dim=1)


def _draw_context(state, settings, generator, deadline=None, *, fold_count=None):
    fold_count = fold_count or 1+min(settings.max_accepted, settings.boundaries)
    known = state.labeled.nonzero().flatten()
    other = (~state.labeled).nonzero().flatten()
    if len(known) < fold_count*settings.probe_known or len(other) < fold_count*settings.probe_unlabeled:
        raise ValueError('Insufficient legal probe support; refresh Keep')
    if settings.probe_sampling == 'class_balanced':
        known_folds = balanced_known_folds(known, state.known_targets, settings.probe_known, generator,
                                          per_class=settings.probe_known_per_class, folds=fold_count)
    else:
        known = known[torch.randperm(len(known), generator=generator)]
        known_folds = [known[fold*settings.probe_known:(fold+1)*settings.probe_known] for fold in range(fold_count)]
    other = other[torch.randperm(len(other), generator=generator)]
    folds = []
    for fold in range(fold_count):
        k = known_folds[fold]
        u = other[fold*settings.probe_unlabeled:(fold+1)*settings.probe_unlabeled]
        folds.append((k, u))
    reserved = set(torch.cat(known_folds).tolist()+other[:fold_count*settings.probe_unlabeled].tolist())
    available = torch.tensor([i for i in range(len(state.ids)) if i not in reserved], dtype=torch.long)
    if len(available) <= settings.k:
        raise ValueError('Insufficient disjoint evidence support; refresh Keep')
    if settings.evidence_sampling == 'global_neighbors':
        pool = global_neighbor_pool(state, available, settings.evidence_samples, settings.k,
                                    generator, reference_block=settings.neighbor_reference_block,
                                    deadline=deadline)
    else:
        pool = available[torch.randperm(len(available), generator=generator)[:settings.evidence_samples]]
    return folds, pool


def _optional_stability(state, pool, settings, seed):
    if not settings.stability_trials:
        return {}
    from sklearn.cluster import KMeans
    reference = state.features[pool].numpy()
    result = {}
    for level, labels in enumerate(state.labels):
        clusters = len(state.prototypes[level])
        if clusters >= len(pool):
            result[level] = dict(status='insufficient_subset_coverage', stability={},
                                 observed_global_fraction=len(pool)/len(state.ids))
            continue
        draws = []
        for trial in range(settings.stability_trials):
            random_state = seed+trial+level*1009
            noise = np.random.RandomState(random_state).normal(0, settings.stability_noise, reference.shape)
            predicted = KMeans(n_clusters=clusters,
                               n_init=1, max_iter=10, random_state=random_state).fit_predict(reference+noise)
            draws.append(torch.from_numpy(predicted))
        result[level] = partition_stability(labels[pool], draws)
        result[level]['observed_global_fraction'] = len(pool)/len(state.ids)
    return result


def _shortlist(state, pool, evidence, settings, stability):
    boundaries = []
    pool_ids = set(state.ids[pool].tolist())
    adjacency = {sample: [] for sample in pool_ids}
    for (a, b), record in evidence.items():
        if record.joint > 0:
            adjacency[a].append((b, record.joint))
            adjacency[b].append((a, record.joint))
    for row in pool.tolist():
        if state.labeled[row]:
            continue
        sample = int(state.ids[row])
        for level, labels in enumerate(state.labels):
            source = int(labels[row])
            candidates = propose_actions(state, sample, level, evidence,
                                         max_targets=settings.max_targets, min_support=settings.min_support,
                                         min_edges=settings.min_edges, min_reliability=settings.min_reliability)
            # Rank measured cross-cluster disagreement; unmeasured edges are not
            # counted as negative evidence. Abstain remains a scored alternative.
            measured = adjacency[sample]
            if not measured:
                continue
            neighbor_rows = rows_for_ids(state.ids, [j for j, _ in measured])
            cross = sum(r for (j, r), neighbor in zip(measured, neighbor_rows) if int(labels[neighbor]) != source)
            within = sum(r for (j, r), neighbor in zip(measured, neighbor_rows) if int(labels[neighbor]) == source)
            score = (cross-within)/len(measured)
            if level in stability:
                score += 1-stability[level]['stability'].get(source, 1.)
            boundaries.append((score, sample, level, candidates))
    boundaries.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen, used_samples = [], set()
    for _, sample, level, candidates in boundaries:
        if sample in used_samples:
            continue
        chosen.extend(a for a in candidates if a.kind != 'keep')
        used_samples.add(sample)
        if len(used_samples) >= settings.boundaries:
            break
    return chosen, adjacency


class CrishEngine:
    def __init__(self, runtime, *, objective=hypselex_loss):
        self.runtime = runtime
        self.objective = objective
        self.settings = CrishSettings(**runtime.config['crish'])

    def _append(self, record):
        with (self.runtime.run_dir/'crish.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')

    def prepare(self, state, model, head, optimizer, dataset, args, epoch):
        settings = self.settings
        if self.runtime.config['method'] != 'crish' or not settings.due(epoch):
            return state
        start = time.monotonic()
        deadline = start+settings.max_refresh_seconds
        seed = (self.runtime.config['seed']*1000003+epoch*103+7) % (2**32)
        record = dict(epoch=epoch, lookahead_scope='projector', evidence_space='normalized_backbone_euclidean',
                      seed=seed, accepted=0, decisions=[], gains=[], status='keep',
                      evidence_sampling=settings.evidence_sampling, probe_sampling=settings.probe_sampling,
                      use_target_normalization=getattr(args, 'use_target_normalization', 'legacy'))
        try:
            with preserve_rng(), fixed_modes(model):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                folds, pool = _draw_context(state, settings, torch.Generator().manual_seed(seed), deadline)
                record['evidence_cache_rows'] = len(pool)
                record['evidence_reference_rows'] = (len(state.ids)-sum(len(k)+len(u) for k, u in folds)
                                                     if settings.evidence_sampling == 'global_neighbors' else len(pool))
                record['view_neighbor_scope'] = 'local_cached_two_views'
                record['probe_folds'] = [dict(known_fold_structure(state.labels[0][k]),
                                              unlabeled_rows=len(u)) for k, u in folds]
                record['confirmation_fold_policy'] = 'fresh_fold_per_accepted_action'
                device = next(head.parameters()).device
                trial_neighbors = []
                first_cache = None
                for trial in range(settings.evidence_trials):
                    views = cached_views(model, dataset, state.ids[pool], device, deadline)
                    if first_cache is None:
                        first_cache = views
                    normalized = F.normalize(views, dim=-1)
                    trial_neighbors.append(tuple(blockwise_euclidean_knn(normalized[:, v], min(settings.k, len(pool)-1)) for v in range(2)))
                evidence = sparse_evidence(state.ids[pool], trial_neighbors)
                record['evidence_edges'] = len(evidence)
                _check_deadline(deadline)
                stability = _optional_stability(state, pool, settings, seed)
                record['stability'] = stability
                actions, adjacency = _shortlist(state, pool, evidence, settings, stability)
                record['candidate_count'] = len(actions)
                if not actions:
                    record['reason'] = 'No measured supported boundaries'
                    return state
                reserved = torch.cat([torch.cat([k, u]) for k, u in folds])
                batch_ids, record['scoring_batch'] = scoring_ids(
                    state, actions, adjacency, pool, reserved, max(settings.train_batch, settings.boundaries))
                cached = {int(sample): first_cache[i] for i, sample in enumerate(state.ids[pool])}
                missing = torch.tensor([int(sample) for sample in batch_ids if int(sample) not in cached], dtype=torch.long)
                if len(missing):
                    extra = cached_views(model, dataset, missing, device, deadline)
                    cached.update({int(sample): extra[i] for i, sample in enumerate(missing)})
                train_views = torch.stack([cached[int(sample)] for sample in batch_ids])
                flat = torch.cat([train_views[:, v] for v in range(2)]).detach()
                metadata_rows = rows_for_ids(state.ids, batch_ids)
                membership = state.labeled[metadata_rows]
                targets = state.labels[0][metadata_rows][membership]
                # Clone once for the frozen refresh snapshot; all scoring stays
                # off the live head/model, with the actual current SGD state.
                frozen, virtual_optimizer = clone_snapshot(head, projector_optimizer(head, optimizer))
                frozen.eval()
                parameters = tuple(p for p in frozen.parameters() if p.requires_grad)
                legal_registry = {int(i): int(y) for i, y in zip(state.ids[state.labeled], state.known_targets)}
                probes = []
                for known, other in folds:
                    ids = torch.cat([state.ids[known], state.ids[other]])
                    views = cached_views(model, dataset, ids, device, deadline)
                    known_views, other_views = views[:len(known)], views[len(known):]
                    with torch.no_grad():
                        teacher = frozen(other_views.flatten(0, 1)).reshape(len(other), 2, -1)
                    probes.append(LegalProbe(known_ids=state.ids[known], known_views=known_views,
                                            known_targets=state.labels[0][known], unlabeled_ids=state.ids[other],
                                            unlabeled_views=other_views, teacher_views=teacher,
                                            proposal_ids=state.ids[pool], legal_labels=legal_registry,
                                            weights=ProbeWeights(settings.probe_known_weight, settings.probe_view_weight,
                                                                 settings.probe_variance_weight, settings.probe_std_floor)))
                for i, probe in enumerate(probes):
                    for other in probes[i+1:]:
                        probe.assert_disjoint(other)
                def gradients(loss):
                    return torch.autograd.grad(loss, parameters, allow_unused=True)
                query = gradients(probes[0].loss(frozen))
                cache = {}
                candidates = {}
                def scorer(current, proposed, action):
                    _check_deadline(deadline)
                    if action.kind == 'move':
                        refreshed = propose_actions(current, action.sample_id, action.level, evidence,
                            max_targets=settings.max_targets, min_support=settings.min_support,
                            min_edges=settings.min_edges, min_reliability=settings.min_reliability)
                        if not any(a.kind == 'move' and a.target == action.target for a in refreshed):
                            raise ValueError('Move lost neighborhood support after a prior repair')
                    if current.version not in cache:
                        result = self.objective(frozen, flat, batch_ids, membership, targets, current, args, epoch)
                        cache[current.version] = (gradients(result['loss']), result.get('pseudo_pair_weights'))
                    keep_gradient, pair_weights = cache[current.version]
                    drift = target_drift(current, proposed, batch_ids, flat,
                                         smoothing=args.unsupervised_smoothing, strategy=args.strategy,
                                         normalization=getattr(args, 'use_target_normalization', 'legacy'),
                                         pair_weights=pair_weights)
                    if max(x['protected_coefficient_shift_max'] for x in drift['sse_levels']) > settings.max_protected_shift:
                        record.setdefault('scoring_rejections', []).append(dict(
                            sample_id=action.sample_id, level=action.level, kind=action.kind, target_drift=drift))
                        raise ValueError('Move exceeds predeclared protected-coefficient shift cap')
                    mass = sum(x['coefficient_half_l1_mean']/2**(x['level']+1) for x in drift['sse_levels'])
                    # SSE and USE remain separate diagnostics; SSE mass is the
                    # declared normalization reference, not a fabricated sum.
                    if mass <= 0:
                        record.setdefault('scoring_rejections', []).append(dict(
                            sample_id=action.sample_id, level=action.level, kind=action.kind, target_drift=drift))
                        raise ValueError('No pseudo-SSE change in the scoring batch')
                    action_gradient = gradients(self.objective(frozen, flat, batch_ids, membership, targets,
                                                               proposed, args, epoch)['loss'])
                    zero = lambda grad, p: torch.zeros_like(p) if grad is None else grad
                    proxy = gradient_gain([zero(q, p) for q, p in zip(query, parameters)],
                                          [zero(g, p) for g, p in zip(action_gradient, parameters)],
                                          [zero(g, p) for g, p in zip(keep_gradient, parameters)], changed_mass=mass)
                    discovery = exact_sgd_comparison(frozen, virtual_optimizer, keep_gradient, action_gradient, probes[0])
                    entry = dict(version=current.version, sample_id=action.sample_id, level=action.level,
                                 kind=action.kind, source=action.source, target=action.target,
                                 proxy=proxy, exact_discovery=discovery,
                                 confirmation=None, target_drift=drift)
                    record['gains'].append(entry)
                    # Action includes source, target and snapshot version: two
                    # Move destinations must never share confirmation gradients.
                    candidates[action] = (keep_gradient, action_gradient, entry)
                    return DecisionEvidence((discovery['gain'],), 0., discovery['known_increase'], mass)
                confirmation_index = 1
                def confirm(current, proposed, action):
                    nonlocal confirmation_index
                    _check_deadline(deadline)
                    g0, ga, entry = candidates[action]
                    comparison = exact_sgd_comparison(frozen, virtual_optimizer, g0, ga, probes[confirmation_index])
                    entry['confirmation_fold'] = confirmation_index
                    confirmation_index += 1
                    entry['confirmation'] = comparison
                    return comparison
                repaired, trace = sequential_repair(state, actions, scorer, max_accepted=settings.max_accepted,
                                                    threshold=settings.acceptance_gain, agreement=1.,
                                                    known_tolerance=settings.known_tolerance, min_support=settings.min_support,
                                                    confirmer=confirm)
                record.update(decisions=trace, accepted=sum(x['status'] == 'accepted' for x in trace), status='completed')
                return repaired
        except (ValueError, RuntimeError) as exc:
            # Proposal failure never damages the baseline snapshot. Training
            # still validates the original loss; it never masks baseline NaN.
            record.update(status='fallback_keep', reason=str(exc))
            return state
        finally:
            record['seconds'] = time.monotonic()-start
            self.runtime.last_local_record = record
            self._append(record)
