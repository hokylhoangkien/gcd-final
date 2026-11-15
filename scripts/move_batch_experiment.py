"""Opt-in one-epoch Move exposure experiment. CLI runs a synthetic fixture only."""
import argparse
from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from crish.candidates import Action
from crish.ids import rows_for_ids
from crish.lookahead import preserve_rng
from crish.rebuild import pseudo_masks, rebuild, prototype_radii
from crish.state import SupervisionState
from selex_runtime.objective import selex_loss
from selex_runtime.trainer import make_args
from gcd_runtime.config import load_config, code_fingerprint


def changed_pairs(before, after, ids, level):
    """Unique unordered original-image pairs, independent of views/multiplicity."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    old, new = pseudo_masks(before, ids, level)[1], pseudo_masks(after, ids, level)[1]
    def pairs(mask):
        return {tuple(sorted((int(ids[i]), int(ids[j])))) for i, j in mask.triu(1).nonzero().tolist()
                if ids[i] != ids[j]}
    return pairs(old & ~new), pairs(new & ~old)


def validate_plan(original, proposed, state):
    if len(original) != len(proposed) or any(len(a) != len(b) for a, b in zip(original, proposed)):
        raise ValueError('Batch count/size changed')
    if Counter(x for b in original for x in b) != Counter(x for b in proposed for x in b):
        raise ValueError('Per-image occurrence counts changed')
    for a, b in zip(original, proposed):
        if state.labeled[rows_for_ids(state.ids, a)].sum() != state.labeled[rows_for_ids(state.ids, b)].sum():
            raise ValueError('Per-batch labeled quota changed')
        ca, cb = Counter(a), Counter(b)
        if any(n > max(1, ca[x]) for x, n in cb.items()):
            raise ValueError('Introduced an intra-batch duplicate')


def active_plan(plan, before, after, action, *, enabled=False, search_budget=10000):
    """Swap existing same-membership slots only; rollback incomplete placement."""
    original = [list(b) for b in plan]
    validate_plan(original, original, before)
    if action.kind != 'move':
        raise ValueError('A single locked Move is required')
    if not enabled:
        return original, dict(status='disabled', swaps=[])
    attempts = 0
    # Bounded mask blocks, never a dataset-sized NxN allocation.
    lost, added = set(), set()
    for block in before.ids.split(128):
        l, a = changed_pairs(before, after, [action.sample_id]+block.tolist(), action.level)
        lost |= l; added |= a
    peers = [{b if a == action.sample_id else a for a, b in group if action.sample_id in (a, b)}
             for group in (lost, added)]
    membership = dict(zip(before.ids.tolist(), before.labeled.tolist()))
    for batch_index, batch in enumerate(original):
        if action.sample_id not in batch:
            continue
        def place(current, relation, swaps):
            nonlocal attempts
            if relation == 2:
                return current, swaps
            if any(x in peers[relation] for x in current[batch_index]):
                return place(current, relation+1, swaps)
            for src, source in enumerate(current):
                if src == batch_index:
                    continue
                for sj, sample in enumerate(source):
                    if sample not in peers[relation]:
                        continue
                    for tj, replaced in enumerate(current[batch_index]):
                        if replaced == action.sample_id or membership[replaced] != membership[sample]:
                            continue
                        # Preserve a previously placed relation witness.
                        trial = copy.deepcopy(current)
                        trial[src][sj], trial[batch_index][tj] = replaced, sample
                        if relation and not any(x in peers[0] for x in trial[batch_index]):
                            continue
                        attempts += 1
                        if attempts > search_budget:
                            return None
                        try:
                            validate_plan(original, trial, before)
                        except ValueError:
                            continue
                        result = place(trial, relation+1, swaps+[[src, sj, batch_index, tj]])
                        if result is not None:
                            return result
            return None
        found = place(original, 0, [])
        if found is not None:
            proposed, swaps = found
            return proposed, dict(status='placed', anchor_batch=batch_index, swaps=swaps, attempts=attempts)
        if attempts > search_budget:
            break
    return original, dict(status='no_feasible_placement_within_budget', swaps=[], attempts=attempts)


def seeded_views(dataset, sample_id, epoch, seed):
    """One deterministic two-view draw per image/epoch; independent of batch order."""
    index = int(rows_for_ids(torch.as_tensor(dataset.sample_ids), [sample_id])[0])
    value = int.from_bytes(hashlib.sha256(f'{seed}/{epoch}/{sample_id}'.encode()).digest()[:4], 'big')
    with preserve_rng():
        random.seed(value); np.random.seed(value); torch.manual_seed(value)
        views, _, observed_id, _ = dataset[index]
        if int(observed_id) != sample_id or len(views) != 2:
            raise ValueError('Stable IDs and two views required')
        return [x.clone() for x in views]


def run_ab(before, after, action, plan, model, head, optimizer, scheduler, dataset, args,
           epoch, seed, output, *, provenance, max_seconds=60):
    """Public adapter for sanitized real inputs; does not select or re-run repair."""
    if max_seconds <= 0:
        raise ValueError('A positive wall-time bound is required')
    before.validate(); after.validate()
    expected, _ = rebuild(before, action)
    for key in ('ids', 'features', 'labeled', 'known_targets'):
        if not torch.equal(getattr(expected, key), getattr(after, key)):
            raise ValueError('Move snapshot mismatch: '+key)
    for key in ('labels', 'prototypes', 'radii', 'abstained'):
        if len(getattr(expected, key)) != len(getattr(after, key)) or any(not torch.equal(a, b) for a, b in zip(getattr(expected, key), getattr(after, key))):
            raise ValueError('After state must contain exactly the locked Move: '+key)
    if provenance.get('kind') != 'synthetic_fixture' and not provenance.get('accepted_move_evidence'):
        raise ValueError('Real inputs require accepted Move provenance')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic()+max_seconds
    plans = dict(A=[list(b) for b in plan])
    plans['B'], placement = active_plan(plan, before, after, action, enabled=True)
    initial = copy.deepcopy(dict(model=model.state_dict(), head=head.state_dict(),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()))
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    torch.save(dict(before=before, after=after, action=action, initial=initial, rng=rng), output/'locked_input.pt')
    manifest = dict(status='running', epoch_log=epoch, epoch_number=epoch+1, seed=seed,
        move=asdict(action), provenance=provenance, placement=placement,
        code_fingerprint=code_fingerprint(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plans=plans, timestamp_utc=datetime.now(timezone.utc).isoformat(), max_seconds=max_seconds,
        command=sys.argv, python=sys.version, torch=torch.__version__, All=None, New=None)
    version = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10)
    manifest['git_sha'] = version.stdout.strip() if version.returncode == 0 else 'source_archive'
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = []
    device = next(head.parameters()).device
    try:
        for arm, batches in plans.items():
            model.load_state_dict(initial['model']); head.load_state_dict(initial['head'])
            optimizer.load_state_dict(copy.deepcopy(initial['optimizer']))
            scheduler.load_state_dict(copy.deepcopy(initial['scheduler']))
            random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
            if rng['cuda']:
                torch.cuda.set_rng_state_all(rng['cuda'])
            model.train(); head.train()
            started = time.monotonic(); lost, added, changed_batches = set(), set(), 0
            for step, ids in enumerate(batches):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Bounded A/B epoch deadline')
                l, a = changed_pairs(before, after, ids, action.level)
                lost |= l; added |= a; changed_batches += bool(l or a)
                views = [seeded_views(dataset, i, epoch, seed) for i in ids]
                images = torch.cat([torch.stack([v[k] for v in views]) for k in range(2)]).to(device)
                ids = torch.tensor(ids)
                rows = rows_for_ids(after.ids, ids); labeled = after.labeled[rows]
                optimizer.zero_grad(set_to_none=True)
                loss = selex_loss(head, model(images), ids, labeled, after.labels[0][rows][labeled], after, args, epoch)['loss']
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in list(model.parameters())+list(head.parameters())):
                    raise FloatingPointError('Nonfinite A/B gradients')
                optimizer.step()
                print(f'AB arm={arm} epoch={epoch} step={step+1}/{len(batches)} loss={float(loss):.6f}', flush=True)
            scheduler.step()
            if device.type == 'cuda': torch.cuda.synchronize()
            row = dict(arm=arm, move_id=action.sample_id, level=action.level, steps=len(batches),
                changed_batches=changed_batches, lost_pairs=len(lost), added_pairs=len(added),
                seconds=time.monotonic()-started, All=None, New=None)
            results.append(row)
            torch.save(dict(model=model.state_dict(), head=head.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict()), output/f'{arm}_after_epoch.pt')
        manifest.update(status='completed', results=results)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc), results=results)
        raise
    finally:
        (output/'result.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def fixture(output, *, max_seconds=60):
    # Explicit synthetic Move, not an acceptance attributed to the CRiSH probe.
    torch.manual_seed(27)
    from crish.smoke import FeatureFixture
    ids = torch.tensor([40, 7, 91, 15, 22, 66, 80, 12])
    x = torch.randn(8, 3)
    labels = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3])
    labeled = torch.tensor([True, True]+[False]*6)
    centers = torch.stack([x[labels == g].mean(0) for g in range(4)])
    before = SupervisionState(ids, x, labeled, labels[labeled], (labels,), (centers,),
        (prototype_radii(centers),), (torch.zeros(8, dtype=torch.bool),))
    action = Action('move', 91, 0, 0, 2)
    after, _ = rebuild(before, action)
    dataset = FeatureFixture(x, labeled, labels[labeled]); dataset.sample_ids = tuple(ids.tolist())
    model, head = torch.nn.Linear(3, 3), torch.nn.Linear(3, 16)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3)); model.bias.zero_()
    optimizer = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=.001, momentum=.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    cfg = load_config(ROOT/'configs/selex_crish.json'); args = make_args(cfg)
    args.hierarchy_levels = 1
    return run_ab(before, after, action, [[91, 7, 15, 12], [40, 22, 66, 80]],
        model, head, optimizer, scheduler, dataset, args, 0, 27, output,
        provenance=dict(kind='synthetic_fixture', accepted_move_evidence=False,
                        real_checkpoint=None, real_data_comparison='not_run'), max_seconds=max_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', action='store_true', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=int, default=60)
    opt = parser.parse_args()
    output = opt.output or ROOT/'logs'/('move-batch-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    print(json.dumps(fixture(output, max_seconds=opt.max_seconds)['results'], indent=2))
