"""Offline-only CRiSH-G audit of crish_g_epoch_*.npz against evaluation labels.

Uses evaluation ground truth: never import from training, dose selection or
checkpoint selection. Pair statistics are counted per cluster from class
histograms (no N-by-N matrix) on the pre-intervention partition; a pair is
"removed" when either endpoint is abstained at that level. Majorities are
unique leave-one-out cluster majorities (ties excluded), as scripts/paper_posthoc.py.
Every rate reports numerator/denominator; a zero denominator gives null.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np

DEFINITION = ('Pairs: unordered same-cluster pseudo-positive pairs (i<j) of the pre-intervention level-l '
              'partition; wrong = different true labels; removed = either endpoint abstained. Level-0 '
              'pseudo-label error: true label differs from the unique leave-one-out true-class majority of '
              'its cluster (ties excluded), unlabeled rows only. AUROC scores (1-r^0) for that error. Move: '
              'true label vs unique majority of the target cluster, and of the source cluster leave-one-out.')


def fraction(numerator, denominator):
    numerator, denominator = int(numerator), int(denominator)
    return dict(numerator=numerator, denominator=denominator,
                fraction=numerator/denominator if denominator else None)


def read_labels(path):
    truth = {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            sample, label = int(row['sample_id']), int(row['true_label'])
            if sample in truth or label < 0:
                raise ValueError('Duplicate sample ID or missing offline label')
            truth[sample] = label
    return truth


def pairs(n):
    return n*(n-1)//2


def pair_counts(clusters, truth, abstained):
    """Total/wrong pseudo-positive pairs, split into removed and retained, per cluster."""
    out = Counter()
    for group in np.unique(clusters):
        members = clusters == group
        all_classes = Counter(truth[members].tolist())
        kept_classes = Counter(truth[members & ~abstained].tolist())
        n, m = int(members.sum()), int((members & ~abstained).sum())
        same_all = sum(pairs(c) for c in all_classes.values())
        same_kept = sum(pairs(c) for c in kept_classes.values())
        out['retained'] += pairs(m)
        out['retained_wrong'] += pairs(m)-same_kept
        out['removed'] += pairs(n)-pairs(m)
        out['removed_wrong'] += (pairs(n)-pairs(m))-(same_all-same_kept)
    return out


def unique_majority(counts):
    counts = [(c, k) for k, c in counts.items() if c > 0]
    if not counts:
        return None
    counts.sort(reverse=True)
    return counts[0][1] if len(counts) == 1 or counts[0][0] > counts[1][0] else None


def auroc(scores, positive):
    """Mann-Whitney AUROC with average ranks for ties; None without both classes."""
    scores, positive = np.asarray(scores, dtype=np.float64), np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return dict(value=None, positives=n_pos, negatives=n_neg)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start
        while stop+1 < len(scores) and sorted_scores[stop+1] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop+1]] = (start+stop)/2+1
        start = stop+1
    value = (ranks[positive].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg)
    return dict(value=float(value), positives=n_pos, negatives=n_neg)


def audit_epoch(arrays, truth_map):
    ids = arrays['ids'].astype(np.int64)
    missing = [int(i) for i in ids if int(i) not in truth_map]
    if missing:
        raise ValueError(f'Offline labels do not cover snapshot IDs: {missing[:5]}')
    truth = np.array([truth_map[int(i)] for i in ids], dtype=np.int64)
    labeled = arrays['labeled'].astype(bool)
    unlabeled = ~labeled
    levels = sorted(int(k.split('_')[1]) for k in arrays.files if re.fullmatch(r'label_\d+', k))
    record = dict(labeled_anchor_agreement=fraction((truth[labeled] == arrays['label_0'][labeled]).sum(),
                                                    labeled.sum()), levels=[])
    for level in levels:
        clusters, abstained = arrays[f'label_{level}'], arrays[f'abstained_{level}'].astype(bool)
        if abstained[labeled].any():
            raise ValueError('Snapshot abstains a labeled row')
        counts = pair_counts(clusters, truth, abstained)
        record['levels'].append(dict(level=level,
            abstained_unlabeled=fraction((abstained & unlabeled).sum(), unlabeled.sum()),
            removed_pair_wrong_rate=fraction(counts['removed_wrong'], counts['removed']),
            retained_pair_wrong_rate=fraction(counts['retained_wrong'], counts['retained'])))
    clusters, r0 = arrays['label_0'], arrays['reliability_0'].astype(np.float64)
    abstained = arrays['abstained_0'].astype(bool)
    groups = {}
    for group, label in zip(clusters.tolist(), truth.tolist()):
        groups.setdefault(group, Counter())[label] += 1
    evaluable, wrong = np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=bool)
    for row in np.flatnonzero(unlabeled):
        counts = groups[int(clusters[row])].copy(); counts[int(truth[row])] -= 1
        predicted = unique_majority(counts)
        if predicted is not None:
            evaluable[row] = True
            wrong[row] = predicted != truth[row]
    record['level0'] = dict(
        pseudo_label_error=fraction(wrong[evaluable].sum(), evaluable.sum()),
        abstained_error=fraction(wrong[evaluable & abstained].sum(), (evaluable & abstained).sum()),
        kept_error=fraction(wrong[evaluable & ~abstained].sum(), (evaluable & ~abstained).sum()),
        auroc_one_minus_r0=auroc(1-r0[evaluable], wrong[evaluable]))
    base, inside = record['level0']['pseudo_label_error']['fraction'], record['level0']['abstained_error']['fraction']
    record['level0']['abstained_error_ratio_to_base'] = inside/base if base and inside is not None else None
    moved = arrays['moved'].astype(bool)
    target_ok = source_ok = target_n = source_n = 0
    for row in np.flatnonzero(moved):
        if labeled[row]:
            raise ValueError('Snapshot moves a labeled row')
        predicted = unique_majority(groups.get(int(arrays['move_target'][row]), Counter()))
        if predicted is not None:
            target_n += 1; target_ok += predicted == truth[row]
        counts = groups[int(clusters[row])].copy(); counts[int(truth[row])] -= 1
        predicted = unique_majority(counts)
        if predicted is not None:
            source_n += 1; source_ok += predicted == truth[row]
    record['move'] = dict(moved=int(moved.sum()),
                          target_majority_correct=fraction(target_ok, target_n),
                          source_majority_correct=fraction(source_ok, source_n))
    return record


def audit(run, labels, epochs=None):
    run = Path(run)
    truth = read_labels(labels)
    paths = sorted(run.glob('crish_g_epoch_*.npz'))
    if epochs:
        paths = [p for p in paths if int(p.stem.split('_')[-1]) in set(epochs)]
    if not paths:
        raise ValueError('No crish_g_epoch_*.npz arrays (run with population_policy.save_arrays=true)')
    report = dict(status='offline_analysis_only', uses_evaluation_labels=True, run=str(run),
                  labels_sha256=hashlib.sha256(Path(labels).read_bytes()).hexdigest(),
                  definition=DEFINITION, epochs=[])
    from scripts.global_policy_statistics import extended_epoch, persistence
    snapshots={}
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            epoch=int(path.stem.split('_')[-1])
            snapshots[epoch]=dict(arrays)
            report['epochs'].append(dict(epoch=epoch, file=path.name,
                                         **audit_epoch(arrays, truth), extended=extended_epoch(arrays,truth)))
    report['persistence']=persistence(snapshots,truth)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--labels', type=Path, required=True, help='Offline CSV: sample_id,true_label')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, nargs='+', help='0-based log epochs; default all arrays')
    parser.add_argument('--compare-run',type=Path,help='Paired image-bootstrap enrichment ratio vs this run')
    parser.add_argument('--bootstrap',type=int,default=2000)
    args = parser.parse_args()
    result = audit(args.run, args.labels, args.epochs)
    if args.compare_run is not None:
        from scripts.global_policy_statistics import enrichment_ratio
        truth=read_labels(args.labels)
        ratios=[]
        for row in result['epochs']:
            path=args.run/row['file'];other=args.compare_run/row['file']
            if not other.is_file():continue
            with np.load(path,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
                order={int(i):j for j,i in enumerate(b['ids'])}
                aligned={k:v[np.array([order[int(i)] for i in a['ids']])] for k,v in dict(b).items()}
                ratios.append(dict(epoch=row['epoch'],**enrichment_ratio(a,aligned,truth,repetitions=args.bootstrap)))
        result['enrichment_ratios']=ratios
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
