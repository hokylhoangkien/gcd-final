"""Offline-only semantic action audit. Never imported by a training entry point.

Snapshots JSONL: one record per (epoch, level), with arrays sample_ids,
true_labels, clusters and labeled (boolean). Require full pre-refresh population.
Target class is its unique majority class excluding the edited image. This is
an offline cluster-purity diagnostic, not Hungarian accuracy or a causal effect.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_registry import Source


def majority(labels):
    counts = Counter({k:v for k,v in Counter(labels).items() if v > 0})
    if not counts:
        return None
    ranked = counts.most_common()
    return ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else None


def fraction(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                fraction=numerator/denominator if denominator else None)


def leave_one_out(groups, cluster, label):
    counts = groups.get(cluster, Counter()).copy()
    counts[label] -= 1
    counts = Counter({k:v for k,v in counts.items() if v > 0})
    return majority(counts)


def analyze(run, snapshots):
    snapshots = Path(snapshots)
    table = {}
    for line in snapshots.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row['epoch'], row['level']
        if key in table:
            raise ValueError('Duplicate snapshot')
        ids, truth, clusters, labeled = (row[k] for k in ('sample_ids','true_labels','clusters','labeled'))
        if not (len(ids) == len(truth) == len(clusters) == len(labeled)) or len(set(ids)) != len(ids):
            raise ValueError('Invalid snapshot alignment')
        if any(type(v) is not bool for v in labeled) or any(y is None or y == -1 for y in truth):
            raise ValueError('Require actual offline labels and membership')
        table[key] = (row.get('run_id'), {i:dict(truth=y, cluster=c, labeled=l) for i,y,c,l in zip(ids,truth,clusters,labeled)})
    source = Source(run)
    try:
        final = source.obj('evaluation_last.json')
        counts = Counter()
        missing = []
        events = []
        for refresh in source.rows('crish.jsonl'):
            epoch = refresh['epoch']
            accepted = [d for d in refresh.get('decisions', []) if d.get('status') == 'accepted']
            if accepted and refresh.get('status') != 'completed':
                raise ValueError('Cannot analyze rolled-back actions as applied')
            for level in {d['level'] for d in accepted}:
                snapshot = table.get((epoch, level))
                if snapshot is None:
                    missing.append([epoch, level])
                    continue
                run_id, population = snapshot
                if run_id != source.obj('metadata.json').get('run_id'):
                    raise ValueError('Snapshot run ID mismatch')
                if sum(not v['labeled'] for v in population.values()) != final.get('n_unlabeled'):
                    raise ValueError('Snapshot does not cover full unlabeled population')
                groups = {}
                for v in population.values():
                    groups.setdefault(v['cluster'], Counter())[v['truth']] += 1
                # Baseline error uses leave-one-out cluster majorities.
                for sample, v in population.items():
                    if v['labeled']:
                        continue
                    predicted = leave_one_out(groups, v['cluster'], v['truth'])
                    if predicted is not None:
                        counts['baseline_n'] += 1
                        counts['baseline_wrong'] += predicted != v['truth']
                for d in (d for d in accepted if d['level'] == level):
                    sample = d['sample_id']
                    if sample not in population or population[sample]['labeled']:
                        raise ValueError('Edited ID missing or legally labeled')
                    v = population[sample]
                    if v['cluster'] != d['source']:
                        raise ValueError('Snapshot/source mismatch')
                    target = d['target'] if d['kind'] == 'move' else d['source']
                    predicted = (leave_one_out(groups, target, v['truth']) if target == v['cluster']
                                 else majority(groups.get(target, Counter())))
                    if predicted is not None:
                        counts[d['kind']+'_n'] += 1
                        counts[d['kind']+'_correct'] += predicted == v['truth']
                    events.append(dict(epoch=epoch, level=level, sample_id=sample, kind=d['kind'],
                        majority_class=predicted, true_label=v['truth'],
                        correct=None if predicted is None else predicted == v['truth']))
                    if d['kind'] == 'move':
                        groups[v['cluster']][v['truth']] -= 1
                        groups.setdefault(target, Counter())[v['truth']] += 1
                        v['cluster'] = target
            for gain in refresh.get('gains', []):
                a, b = gain.get('exact_discovery'), gain.get('confirmation')
                if a and b and all(isinstance(x.get('gain'), (float,int)) for x in (a,b)):
                    counts['sign_n'] += 1
                    counts['sign_same'] += (a['gain'] > 0) == (b['gain'] > 0)
        return dict(status='PARTIAL' if missing else 'offline_analysis_only', missing_snapshots=missing,
            snapshots_sha256=hashlib.sha256(snapshots.read_bytes()).hexdigest(),
            move_target_majority_accuracy=fraction(counts['move_correct'], counts['move_n']),
            abstain_source_error=fraction(counts['abstain_n']-counts['abstain_correct'], counts['abstain_n']),
            baseline_source_error=fraction(counts['baseline_wrong'], counts['baseline_n']),
            discovery_confirmation_sign_agreement=fraction(counts['sign_same'], counts['sign_n']),
            events=events, definition='Leave-one-out unique cluster majority; ties excluded. Baseline denominator is evaluable unlabeled image-level-refresh rows. Sign agreement only among confirmed candidates; selection biased.')
    finally:
        source.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run, args.snapshots)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
