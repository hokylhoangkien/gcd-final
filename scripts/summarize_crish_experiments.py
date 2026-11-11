"""Export observed fixed-last metrics and realized dose; never choose a winner."""
import argparse
import csv
import json
import hashlib
import math
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcd_runtime.metrics import read_rows


def summarize(root):
    result = []
    for path in sorted(Path(root).glob('*/config.json')):
        directory, cfg = path.parent, json.loads(path.read_text())
        metrics_path = directory/'evaluation_last.json'
        if not metrics_path.is_file(): continue
        metrics = json.loads(metrics_path.read_text())
        # Exclude synthetic, plain-test, best-selected and incomplete reports.
        if (metrics.get('status') != 'observed' or metrics.get('unit') != 'fraction'
                or metrics.get('protocol') != 'balanced_ss_kmeans_train_unlabeled'
                or metrics.get('epoch') != cfg['epochs']-1): continue
        if any(not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1 for k in ('all_acc','old_acc','new_acc')):
            raise ValueError(f'Invalid observed metrics: {metrics_path}')
        inputs = {}
        for name in ('metadata', 'readiness'):
            p = directory/(name+'.json')
            inputs[name] = json.loads(p.read_text()) if p.is_file() else {}
        signature = dict(config={k:v for k,v in cfg.items() if k != 'seed'},
            code=inputs['metadata'].get('code_fingerprint'),
            environment_sha256=(hashlib.sha256((directory/'environment.txt').read_bytes()).hexdigest()
                                if (directory/'environment.txt').is_file() else None),
            gpu=inputs['readiness'].get('gpu'), torch=inputs['readiness'].get('torch'),
            inputs={k: inputs['readiness'].get(k) for k in
                    ('upstream_sha','split','weight_sha256','annotation_sha256')})
        provenance = bool(signature['code'] and signature['inputs']['weight_sha256'] and signature['environment_sha256'])
        if not provenance: signature['unverified_run'] = directory.name
        group_key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        local, population = read_rows(directory/'crish.jsonl'), read_rows(directory/'crish_p.jsonl')
        dose, train = read_rows(directory/'intervention_diagnostics.jsonl'), read_rows(directory/'metrics.jsonl')
        result.append(dict(run_id=directory.name, seed=cfg['seed'], dataset=cfg['dataset_name'], method=cfg['method'],
            experiment_name=cfg.get('experiment_name','explicit_run_config'), group_key=group_key,
            aggregation_provenance=provenance,
            config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            all_percent=100*metrics['all_acc'], old_percent=100*metrics['old_acc'], new_percent=100*metrics['new_acc'],
            local_accepted=sum(x.get('accepted', 0) for x in local),
            population_accepted=sum(x.get('accepted', 0) for x in population),
            changed_id_epochs=sum(x.get('changed_ids', 0) for x in dose),
            local_seconds=sum(x.get('seconds', 0) for x in local),
            population_seconds=sum(x.get('seconds', 0) for x in population),
            training_seconds=sum(x.get('seconds', 0) for x in train),
            peak_cuda_bytes=max((x.get('peak_cuda_allocated_bytes') or 0 for x in train), default=0),
            fallback_refreshes=sum(x.get('status') == 'fallback_keep' for x in local+population),
            checkpoint_sha256=metrics.get('checkpoint_sha256'), protocol=metrics['protocol']))
    return result


def aggregate(rows):
    """Never pool different configs/source/weights or count reruns as seeds."""
    groups = {}
    for row in rows: groups.setdefault(row['group_key'],[]).append(row)
    result = []
    for key, values in groups.items():
        valid = all(v['aggregation_provenance'] for v in values) and len({v['seed'] for v in values}) == len(values)
        record = dict(group_key=key, experiment_name=values[0]['experiment_name'],
            status='matched_observed_seeds' if valid else 'not_aggregated_missing_provenance_or_duplicate_seed',
            n_seeds=len({v['seed'] for v in values}), per_seed=values)
        if valid:
            for metric in ('all_percent','old_percent','new_percent'):
                data = [v[metric] for v in values]
                record[metric] = dict(mean=statistics.mean(data),
                    std=statistics.stdev(data) if len(data) > 1 else None)
        result.append(record)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    args.output.with_suffix('.summary.json').write_text(json.dumps(aggregate(rows),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='exported_observed_only', observed_runs=len(rows), output=str(args.output))))
