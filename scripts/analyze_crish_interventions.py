"""Read-only temporal SS audit; observed changes are not causal action effects.

Stdlib only: safe to copy this file to /tmp on a live training host without
updating the runtime/source fingerprint. Optional output lives where specified.
"""
import argparse
import csv
import json
import math
from pathlib import Path

PROTOCOL = 'balanced_ss_kmeans_train_unlabeled'
SS_PROTOCOLS = (PROTOCOL, 'unbalanced_ss_kmeans_train_unlabeled')
METRICS = ('all_acc', 'old_acc', 'new_acc')


def rows(path):
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path}: incomplete JSON; retry after the log write completes') from exc


def ss_history(run_dir):
    records = rows(run_dir/'epoch_metrics.jsonl')
    records.extend(json.loads(p.read_text()) for p in sorted((run_dir/'evaluations').glob('epoch_*.json')))
    final = run_dir/'evaluation_last.json'
    if final.is_file():
        records.append(json.loads(final.read_text()))
    result = {}
    for row in records:
        if (row.get('protocol') not in SS_PROTOCOLS or row.get('purpose') not in ('monitor_only', 'final')
                or row.get('status') != 'observed'):
            continue
        if row.get('unit') != 'fraction':
            raise ValueError('SS audit requires fraction units')
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in METRICS):
            raise ValueError('Invalid SS metric')
        epoch = row['epoch']
        if epoch in result and any(result[epoch][k] != row[k] for k in METRICS):
            raise ValueError(f'Conflicting SS reports at zero-based epoch {epoch}')
        result[epoch] = row
    if len({r['protocol'] for r in result.values()}) > 1:
        raise ValueError('Cannot mix balanced and unbalanced SS histories')
    return result


def identity(action):
    return tuple(action.get(k) for k in ('version', 'sample_id', 'level', 'kind', 'source', 'target'))


def audit(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir/'crish.jsonl').is_file():
        raise ValueError('Missing local crish.jsonl; verify the run ID/variant')
    history = ss_history(run_dir)
    refreshes = rows(run_dir/'crish.jsonl')
    dose = {r['epoch']: r for r in rows(run_dir/'intervention_diagnostics.jsonl')}
    events, summaries = [], []
    for r in refreshes:
        epoch = r['epoch']
        gains = {identity(g): g for g in r.get('gains', [])}
        accepted = [d for d in r.get('decisions', []) if d.get('status') == 'accepted']
        applied = r.get('status') == 'completed'
        if not applied:
            accepted = []
        if len(accepted) != r.get('accepted', 0):
            raise ValueError(f'Accepted-action mismatch at log epoch {epoch}')
        if any(d['kind'] == 'move' for d in accepted):
            group = 'move_epoch'
        elif accepted:
            group = 'abstain_only_epoch'
        else:
            group = 'fallback_epoch' if r.get('status') == 'fallback_keep' else 'keep_epoch'
        summaries.append(dict(epoch_log=epoch, epoch_number=epoch+1, group=group,
                              status=r.get('status'), accepted=len(accepted),
                              candidates=r.get('candidate_count'), scored=len(r.get('gains', [])),
                              seconds=r.get('seconds'), reason=r.get('reason')))
        for d in accepted:
            g = gains.get(identity(d))
            if g is None:
                raise ValueError(f'No matching scored action at log epoch {epoch}: {identity(d)}')
            discovery, confirmation = g.get('exact_discovery'), g.get('confirmation')
            entry = dict(epoch_log=epoch, epoch_number=epoch+1, action=d['kind'],
                         sample_id=d['sample_id'], level=d['level'], source=d.get('source'),
                         target=d.get('target'), version=d.get('version'),
                         discovery_gain=discovery.get('gain') if discovery else None,
                         confirmation_gain=confirmation.get('gain') if confirmation else None,
                         discovery_known_increase=discovery.get('known_increase') if discovery else None,
                         confirmation_known_increase=confirmation.get('known_increase') if confirmation else None,
                         action_changed_directed_pairs=d.get('changed_global_pseudo_pairs'),
                         epoch_changed_ids=dose.get(epoch, {}).get('changed_ids'),
                         accepted_actions_same_epoch=len(accepted))
            # Action at log epoch e affects SGD of e, reported as EVAL epoch=e+1.
            # Before=e-1, during=e, next=e+1; never substitute a nearby/best report.
            for label, index in (('before', epoch-1), ('during', epoch), ('next', epoch+1)):
                record = history.get(index)
                entry[label+'_epoch_number'] = index+1
                for metric in METRICS:
                    entry[f'{label}_{metric}_percent'] = 100*record[metric] if record else None
            for metric in METRICS:
                before = entry[f'before_{metric}_percent']
                for label in ('during', 'next'):
                    after = entry[f'{label}_{metric}_percent']
                    entry[f'{label}_minus_before_{metric}_pp'] = (
                        after-before if before is not None and after is not None else None)
            entry['attribution'] = 'temporal_association_not_causal_effect'
            events.append(entry)
    protocol = next(iter(history.values()))['protocol'] if history else None
    result = dict(run_id=run_dir.name, protocol=protocol, events=events, refreshes=summaries,
        missing_values='null means not evaluated/available; never imputed',
        interpretation='Before/after changes include ordinary SGD, hierarchy and clustering changes. '
                       'Keep/Abstain epochs are observational controls, not no-Move counterfactuals. '
                       'Multiple accepted actions in one epoch share its SS change; do not sum it per action.',
        no_move_counterfactual='not_observed_requires_matched_baseline_or_locked_snapshot_replay')
    result['moves_with_temporal_all_drop'] = [e['epoch_log'] for e in events if e['action'] == 'move'
        and e['during_minus_before_all_acc_pp'] is not None and e['during_minus_before_all_acc_pp'] < 0]
    result['moves_with_temporal_new_drop'] = [e['epoch_log'] for e in events if e['action'] == 'move'
        and e['during_minus_before_new_acc_pp'] is not None and e['during_minus_before_new_acc_pp'] < 0]
    return result


def number(value, signed=False):
    if value is None:
        return 'NA'
    return f'{value:+.3f}' if signed else f'{value:.3f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, help='Optional CSV plus .json report; source run remains unchanged')
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error('Run directory does not exist')
    report = audit(args.run_dir)
    print('Epoch(log/display) Action Sample  Discovery Confirmation  All before/during/next  dAll(pp) dNew(pp)')
    for e in report['events']:
        print(f'{e["epoch_log"]:3}/{e["epoch_number"]:<3} {e["action"]:7} {e["sample_id"]:6} '
              f'{e["discovery_gain"] if e["discovery_gain"] is not None else "NA"!s:>12} '
              f'{e["confirmation_gain"] if e["confirmation_gain"] is not None else "NA"!s:>12}  '
              f'{number(e["before_all_acc_percent"])}/{number(e["during_all_acc_percent"])}/'
              f'{number(e["next_all_acc_percent"])} '
              f'{number(e["during_minus_before_all_acc_pp"], True):>8} '
              f'{number(e["during_minus_before_new_acc_pp"], True):>8}')
    print('Move epochs followed by lower All (association only):', report['moves_with_temporal_all_drop'])
    print('Move epochs followed by lower New (association only):', report['moves_with_temporal_new_drop'])
    print('No-Move alternative: NOT OBSERVED; requires matched baseline or replay.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as handle:
            fields = list(report['events'][0]) if report['events'] else ['epoch_log', 'action']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report['events'])
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('Saved:', args.output, 'and', args.output.with_suffix('.json'))


if __name__ == '__main__':
    main()
