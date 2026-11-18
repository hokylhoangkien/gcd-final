"""Read-only final-run registry, refresh counts and measured epoch overhead.

Accept explicit run directories or a text export ZIP. Never load model tensors,
run evaluations, choose best checkpoints, or pass analysis labels to training.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_crish_experiments import aggregate

METRICS = ('all_acc', 'old_acc', 'new_acc')


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(path) if self.path.suffix == '.zip' else None
        self.tar = tarfile.open(path) if str(path).endswith('.tar.gz') else None
        self.prefix = ''
        if self.archive or self.tar:
            names = self.archive.namelist() if self.archive else self.tar.getnames()
            configs = [n for n in names if n.endswith('/config.json') or n == 'config.json']
            if len(configs) != 1:
                self.close()
                raise ValueError('ZIP must contain exactly one run config')
            self.prefix = configs[0][:-len('config.json')]

    def read(self, name):
        if self.tar:
            try:
                handle = self.tar.extractfile(self.prefix+name)
            except KeyError:
                return None
            if handle is None:
                return None
            with handle:
                return handle.read()
        if self.archive:
            try:
                return self.archive.read(self.prefix+name)
            except KeyError:
                return None
        p = self.path/name
        return p.read_bytes() if p.is_file() else None

    def obj(self, name):
        payload = self.read(name)
        return json.loads(payload) if payload else {}

    def rows(self, name):
        if self.tar:
            try:
                handle = self.tar.extractfile(self.prefix+name)
            except KeyError:
                return
            if handle is None:
                return
        elif self.archive:
            try:
                handle = self.archive.open(self.prefix+name)
            except KeyError:
                return
        else:
            path = self.path/name
            if not path.is_file():
                return
            handle = path.open('rb')
        with handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def close(self):
        if self.archive:
            self.archive.close()
        if self.tar:
            self.tar.close()

    def exists(self, name):
        if self.archive:
            return self.prefix+name in self.archive.namelist()
        if self.tar:
            return self.prefix+name in self.tar.getnames()
        return (self.path/name).is_file()


def mean(values):
    return statistics.mean(values) if values else None


def inspect_run(path):
    source = Source(path)
    try:
        cfg, meta = source.obj('config.json'), source.obj('metadata.json')
        readiness, final = source.obj('readiness.json'), source.obj('evaluation_last.json')
        if not cfg:
            raise ValueError(f'Missing config: {path}')
        status = 'NOT_VERIFIED_FINAL'
        valid = (final.get('status') == 'observed' and final.get('purpose') == 'final'
            and final.get('checkpoint_selection') == 'final_epoch'
            and final.get('epoch') == cfg['epochs']-1 and final.get('seed') == cfg['seed']
            and final.get('unit') == 'fraction' and final.get('matching') == 'hungarian_v2_global'
            and final.get('protocol') == 'balanced_ss_kmeans_train_unlabeled'
            and not final.get('selected_using_evaluation_labels', False)
            and not final.get('checkpoint_selection_uses_evaluation_labels', False)
            and not final.get('used_for_checkpoint_selection', False)
            and not cfg.get('synthetic_fixture', False))
        if valid:
            if any(type(final.get(k)) not in (int, float) or not math.isfinite(final[k]) or not 0 <= final[k] <= 1 for k in METRICS):
                raise ValueError('Invalid final metrics')
            n, old, new = (final[k] for k in ('n_unlabeled', 'n_old', 'n_new'))
            if old+new != n or min(n, old, new) <= 0 or not math.isclose(
                    final['all_acc'], (old*final['old_acc']+new*final['new_acc'])/n, abs_tol=1e-10):
                raise ValueError('Final sample-weighted accuracy mismatch')
            status = 'observed_final_single_seed'
        refreshes, actions, accepted_ids = [], [], set()
        for r in source.rows('crish.jsonl'):
            accepted = [d for d in r.get('decisions', []) if d.get('status') == 'accepted']
            if (len(accepted) != r.get('accepted', 0) or
                (accepted and r.get('status') != 'completed') or
                r.get('applied_actions', len(accepted)) != len(accepted)):
                raise ValueError('Accepted/applied mismatch')
            counts = Counter(d['kind'] for d in accepted)
            paired_gains = [(g['exact_discovery']['gain'], g['confirmation']['gain'])
                for g in r.get('gains', []) if g.get('exact_discovery') and g.get('confirmation')]
            if any(not math.isfinite(v) for pair in paired_gains for v in pair):
                raise ValueError('Nonfinite paired probe gains')
            touched = {d['sample_id'] for d in accepted}
            accepted_ids.update(touched)
            decisions = r.get('decisions', [])
            refreshes.append(dict(epoch=r['epoch'], epoch_number=r['epoch']+1,
                proposed=r.get('candidate_count'), accepted=len(accepted),
                not_applied=(r['candidate_count']-len(accepted)) if 'candidate_count' in r else None,
                keep_shortlisted=(r['candidate_samples']-len(touched)) if 'candidate_samples' in r else None,
                move=counts['move'], abstain=counts['abstain'], seconds=r.get('seconds'),
                sign_agreement_numerator=sum((a > 0) == (b > 0) for a,b in paired_gains),
                sign_agreement_denominator=len(paired_gains),
                stop_reason=r.get('stop_reason'),
                touched_unlabeled_fraction=len(touched)/final['n_unlabeled'] if valid else None,
                rejection_trace=json.dumps(Counter(d.get('reason', d['status']) for d in decisions
                                                   if d['status'] != 'accepted'), sort_keys=True)))
            def identity(d):
                return tuple(d.get(k) for k in ('version','sample_id','level','kind','source','target'))
            gains = {identity(g): g for g in r.get('gains', [])}
            for d in accepted:
                g = gains.get(identity(d), {})
                discovery, confirmation = g.get('exact_discovery'), g.get('confirmation')
                actions.append(dict(epoch=r['epoch'], **{k:d.get(k) for k in
                    ('sample_id','level','kind','source','target','version')},
                    discovery_gain=discovery.get('gain') if discovery else None,
                    confirmation_gain=confirmation.get('gain') if confirmation else None))
        epochs = [r['epoch'] for r in refreshes]
        if len(epochs) != len(set(epochs)):
            raise ValueError('Duplicate refresh epochs; resolve resume provenance before pooling')
        train = list(source.rows('metrics.jsonl'))
        if len({r['epoch'] for r in train}) != len(train):
            raise ValueError('Duplicate training epochs')
        refresh_set = set(epochs)
        settings = cfg.get('crish', {})
        scheduled = {e for e in range(cfg['epochs']) if cfg['method'] == 'crish'
            and settings.get('enabled', False) and e >= settings['warmup_epochs']
            and (e-settings['warmup_epochs']) % settings['refresh_every'] == 0}
        missing_refreshes = sorted(scheduled-refresh_set)
        with_refresh = [r['seconds'] for r in train if r['epoch'] in refresh_set]
        without_refresh = [r['seconds'] for r in train if r['epoch'] not in refresh_set | scheduled]
        env = source.read('environment.txt')
        signature = dict(config={k:v for k,v in cfg.items() if k != 'seed'},
            code=meta.get('code_fingerprint'), environment=hashlib.sha256(env).hexdigest() if env else None,
            inputs={k:readiness.get(k) for k in ('upstream_sha','split','weight_sha256','annotation_sha256')})
        provenance = bool(signature['code'] and env and signature['inputs']['weight_sha256']
                          and signature['inputs']['split'] and signature['inputs']['annotation_sha256'])
        if not provenance:
            signature['unverified_run'] = str(path)
        row = dict(run_id=meta.get('run_id', source.prefix.rstrip('/') or source.path.name),
            source=str(path), dataset=cfg['dataset_name'], method=cfg['method'], seed=cfg['seed'],
            variant=cfg.get('crish_ablation', 'full'), experiment_name=cfg.get('crish_ablation', 'full'),
            status=status, git_sha=meta.get('git_sha'), code_fingerprint=meta.get('code_fingerprint'),
            config_sha256=hashlib.sha256(source.read('config.json')).hexdigest(),
            config_json=json.dumps(cfg, sort_keys=True),
            timestamp_utc=meta.get('timestamp_utc'), metadata_status=meta.get('status'),
            epoch=final.get('epoch'), protocol=final.get('protocol'),
            checkpoint_sha256=final.get('checkpoint_sha256'),
            all_percent=100*final['all_acc'] if valid else None,
            old_percent=100*final['old_acc'] if valid else None,
            new_percent=100*final['new_acc'] if valid else None,
            action_log_status='observed' if source.exists('crish.jsonl') else 'MISSING',
            action_coverage='partial' if missing_refreshes else 'complete_for_current_schedule',
            missing_scheduled_refresh_epochs=json.dumps(missing_refreshes),
            local_accepted=len(actions) if source.exists('crish.jsonl') else None,
            move=sum(a['kind']=='move' for a in actions) if source.exists('crish.jsonl') else None,
            abstain=sum(a['kind']=='abstain' for a in actions) if source.exists('crish.jsonl') else None,
            distinct_touched_unlabeled=len(accepted_ids) if source.exists('crish.jsonl') else None,
            touched_unlabeled_fraction=len(accepted_ids)/final['n_unlabeled'] if valid and source.exists('crish.jsonl') else None,
            recorded_training_seconds=sum(r['seconds'] for r in train) if train else None,
            recorded_refresh_seconds=sum(r['seconds'] for r in refreshes) if source.exists('crish.jsonl') else None,
            timed_epochs_with_refresh=len(with_refresh) if source.exists('crish.jsonl') else None,
            timed_epochs_without_refresh=len(without_refresh) if source.exists('crish.jsonl') else None,
            mean_epoch_seconds_with_refresh=mean(with_refresh),
            mean_epoch_seconds_without_refresh=mean(without_refresh) if source.exists('crish.jsonl') else None,
            sign_agreement_numerator=sum(r['sign_agreement_numerator'] for r in refreshes),
            sign_agreement_denominator=sum(r['sign_agreement_denominator'] for r in refreshes),
            aggregation_provenance=provenance,
            group_key=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest())
        return row, refreshes, actions
    finally:
        source.close()


def csv_write(path, rows):
    with Path(path).open('x', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def export(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    registry, refreshes, actions = [], [], []
    seen = set()
    for path in paths:
        row, rr, aa = inspect_run(path)
        if row['run_id'] in seen:
            raise ValueError('Duplicate run ID (including duplicate imports)')
        seen.add(row['run_id'])
        registry.append(row)
        refreshes.extend(dict(run_id=row['run_id'], dataset=row['dataset'], **r) for r in rr)
        actions.extend(dict(run_id=row['run_id'], dataset=row['dataset'], **a) for a in aa)
    csv_write(output/'runs.csv', registry)
    csv_write(output/'refreshes.csv', refreshes)
    csv_write(output/'actions.csv', actions)
    groups = aggregate([r for r in registry if r['status'] == 'observed_final_single_seed'])
    for group in groups:
        group['paper_ready'] = group['status'] == 'matched_observed_seeds' and group['n_seeds'] >= 3
    report = dict(runs=registry, groups=groups,
        timing_note='Measured partial log coverage; epoch time includes extraction/refresh/SGD, excludes evaluation. Observational, not causal overhead.',
        keep_note='Unique shortlisted IDs without applied edit; not the entire unlabeled pool.',
        k5_status='NOT_VERIFIED: offline labels and pre-refresh assignments required')
    (output/'registry.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = export(args.runs, args.output)
    print(json.dumps(dict(exported_runs=len(report['runs']), output=str(args.output))))
