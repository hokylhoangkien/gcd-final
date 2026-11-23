"""Prepare CRiSH-G (global reliability policy) profiles; never launch jobs or download inputs."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.config import code_fingerprint, load_config
from selex_runtime.config import validate_config

PROFILES = ('g_fixed', 'g_cf', 'g_cf_move', 'g_random', 'g_cf_nopersist')
OPTIONAL = ('g_cf_local',)
G2_PROFILES = ('matched_selex','g2_fixed','g2_random','g2_cf','g2_cf_move','g2_cf_nopersist','g2_cf_noconfirm','g2_oracle_dose')


def common(template, known_guard=True):
    if template.get('baseline_family') != 'selex':
        raise ValueError('CRiSH-G profiles require the pinned SelEx path')
    cfg = copy.deepcopy(template)
    cfg['method'] = 'crish'
    cfg['crish']['enabled'] = False
    if 'pair_reliability' in cfg:
        cfg['pair_reliability']['enabled'] = False
    cfg.pop('crish_ablation', None)
    policy = dict(cfg.get('population_policy', {}))
    policy.update(enabled=True, ranking='knn_agreement', selection='threshold', levels='active',
                  persist=True, warmup_epochs=10, refresh_every=5, scoring_batches=4,
                  scoring_sampling='train_sampler', probe_float64=True, save_arrays=True,
                  thresholds=[0., .3, .5, .7], fixed_threshold=.5, sample_choice='evidence',
                  actions='abstain', gate='counterfactual', mode='counterfactual', compose_local=False,
                  known_guard=known_guard)
    cfg['population_policy'] = policy
    return cfg


def g2_variant(template,name,known_guard=True):
    cfg=common(template,known_guard)
    cfg['experiment_profile']=name
    cfg['crish']['probe_view_reduction']='per_row_sum'
    cfg['crish'].update(probe_known=16,probe_unlabeled=32,probe_sampling='class_balanced',probe_known_per_class=2)
    cfg['report_best_old_test']=False
    cfg['population_policy'].update(scorer='first_order_fp64',selector='normalized_v1',dose_mode='rank_quantile',
        dose_quantiles=[0.,.1,.25,.4],fixed_dose=.25,abstain_ceiling=.5,min_boundary_size=20,
        hierarchical_abstain=True,tie_break='hash',t_min=1.,t_min_confirm=0.,guard_se=2.,sigma_floor_ratio=.25,
        scoring_batches=8,confirmation_batches=8,confirm_fresh_chunks=True,probe_float64=False,
        emergency_deadline_seconds=900.,diagnostics=dict(use_saturation=True,exposure=True),confirm=True,
        reliability_ema=dict(enabled=False,beta=.7,reset_jaccard=.5),evidence_views=1)
    p=cfg['population_policy']
    p.pop('oracle_labels_csv',None)
    if name=='matched_selex':cfg['method']='baseline';p['enabled']=False
    if name in ('g2_fixed','g2_random','g2_oracle_dose'):p.update(mode='fixed',gate='none')
    if name=='g2_random':p['random_control']='count_matched'
    if name=='g2_cf_move':p['actions']='abstain_move'
    if name=='g2_cf_nopersist':p['persist']=False
    if name=='g2_cf_noconfirm':p['confirm']=False
    if name=='g2_oracle_dose':
        p['diagnostic_oracle']=True
        p['oracle_labels_csv']=template.get('population_policy',{}).get('oracle_labels_csv')
        if not p['oracle_labels_csv']:raise ValueError('g2_oracle_dose requires --oracle-labels CSV')
    return cfg


def global_variants(template, profiles=PROFILES, known_guard=True):
    base = common(template, known_guard)
    out = {}
    def make(**changes):
        cfg = copy.deepcopy(base)
        cfg['population_policy'].update(changes)
        return cfg
    builders = dict(
        g_fixed=lambda: make(mode='fixed', gate='none', fixed_threshold=.5, sample_choice='evidence'),
        g_cf=lambda: make(),
        g_cf_move=lambda: make(actions='abstain_move'),
        g_random=lambda: make(mode='fixed', gate='none', fixed_threshold=.5, sample_choice='random'),
        g_cf_nopersist=lambda: make(persist=False),
        g_cf_local=lambda: local(make(compose_local=True)))
    for name in profiles:
        if name in G2_PROFILES:
            out[name]=validate_config(g2_variant(template,name,known_guard))
            continue
        if name not in builders:
            raise ValueError(f'Unknown CRiSH-G profile: {name}')
        out[name] = validate_config(builders[name]())
    return out


def local(cfg):
    """Same local adaptive budgets as scripts/prepare_local_adaptive.py."""
    cfg['crish'].update(enabled=True, max_accepted=8, confirmation_budget=8, boundaries=16,
        train_batch=128, evidence_samples=128, evidence_trials=2, k=5,
        probe_known=16, probe_unlabeled=32, probe_sampling='class_balanced',
        probe_known_per_class=2, max_refresh_seconds=120)
    return cfg


SS_KEYS = ('epochs', 'ss_eval_every_epochs', 'ss_n_init', 'ss_max_iter', 'ss_eval_timeout_seconds',
           'evaluation_rng', 'final_ss_evaluation', 'checkpoint_selection', 'report_best_old_test',
           'final_eval_timeout_seconds', 'unbalanced')
POLICY_KEYS = ('method', 'population_policy', 'crish', 'pair_reliability', 'crish_ablation')


def control_config(treatment, control_base):
    """Matched SelEx control: baseline config, same seed, epochs and SS settings as the profiles."""
    control = load_config(control_base)
    if control.get('method') != 'baseline' or control['dataset_name'] != treatment['dataset_name']:
        raise ValueError('Control base must be the SelEx baseline config of the same dataset')
    control['seed'] = treatment['seed']
    for key in SS_KEYS:
        if key in treatment:
            control[key] = treatment[key]
        else:
            control.pop(key, None)
    return validate_config(control)


def config_differences(first, second):
    return sorted(k for k in set(first) | set(second) if first.get(k) != second.get(k))


def prepare(base, output, seeds=None, profiles=PROFILES, known_guard=True, control=False, oracle_labels=None, phase=None):
    template = load_config(base)
    if oracle_labels is not None:
        template.setdefault('population_policy',{})['oracle_labels_csv']=str(oracle_labels)
    seeds = [template['seed']] if not seeds else list(seeds)
    if len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for seed in seeds:
        for name, cfg in global_variants(dict(template, seed=seed), profiles, known_guard).items():
            if phase is not None:
                if name not in G2_PROFILES:raise ValueError('--phase is for g2 profiles only')
                cfg['ss_eval_every_epochs']=5 if phase=='phase3' else 1
                cfg['ss_eval_last_epochs']=10 if phase=='phase3' else 0
            path = output/f'{name}_seed{seed}.json'
            payload = json.dumps(cfg, indent=2)+'\n'
            path.write_text(payload)
            variant = 'global_local' if cfg['population_policy'].get('compose_local') else 'global'
            command = ['bash', 'scripts/train_selex.sh', '--config', str(path), '--variant', variant,
                       '--eval-every', str(cfg['ss_eval_every_epochs'] if phase else 1)]
            if name=='matched_selex':
                variant='baseline'
                command=['bash','scripts/train_selex.sh','--config',str(path),'--baseline','--eval-every',str(cfg['ss_eval_every_epochs'] if phase else 1)]
            entries.append(dict(name=name, seed=seed, config=path.name, variant=variant,
                sha256=hashlib.sha256(payload.encode()).hexdigest(), command=command))
        if control:
            reference = global_variants(dict(template, seed=seed), ('g_cf',), known_guard)['g_cf']
            cfg = control_config(reference, ROOT/f'configs/selex_{template["dataset_name"]}_baseline_200.json')
            differences = config_differences(cfg, reference)
            if not set(differences) <= set(POLICY_KEYS):
                raise ValueError(f'Control differs from g_cf outside method/policy: {differences}')
            path = output/f'control_seed{seed}.json'
            payload = json.dumps(cfg, indent=2)+'\n'
            path.write_text(payload)
            entries.append(dict(name='control', seed=seed, config=path.name, variant='baseline',
                sha256=hashlib.sha256(payload.encode()).hexdigest(), differs_from_g_cf=differences,
                command=['bash', 'scripts/train_selex.sh', '--config', str(path), '--baseline', '--eval-every', '1']))
    version = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
        text=True, timeout=10).stdout.strip() or 'source_archive'
    manifest = dict(status='prepared_not_executed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=version, code_fingerprint=code_fingerprint(), base=str(base), seeds=seeds,
        profiles=list(profiles), known_guard=known_guard, entries=entries, dataset=template['dataset_name'],
        phase=phase, preregistration_commit='f70279b' if any(n in G2_PROFILES for n in profiles) else None,
        epochs=template['epochs'], cuda_verified=False, benchmark_metrics=None)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'commands.txt').write_text('\n\n'.join(
        f'# {entry["name"]} seed{entry["seed"]}; run one job at a time\n'+shlex.join(entry['command'])
        for entry in entries)+'\n')
    return dict(status=manifest['status'], directory=str(output), jobs=len(entries), profiles=list(profiles))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'configs/selex_scars_crish_200.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--profiles', nargs='+', choices=PROFILES+OPTIONAL+G2_PROFILES, default=list(PROFILES))
    parser.add_argument('--with-local', action='store_true', help='Also write optional g_cf_local')
    parser.add_argument('--phase',choices=['g1','phase3'],help='g2: SS every epoch (G1), or every 5 + final 10 (phase3)')
    parser.add_argument('--oracle-labels',type=Path,help='Diagnostic g2_oracle_dose only: offline CSV')
    parser.add_argument('--control', action='store_true',
                        help='Also write the matched SelEx baseline control (same seed/epochs/SS settings)')
    parser.add_argument('--known-guard', choices=['on', 'off'], default='on',
                        help='off: counterfactual gates on total probe gain only (known_increase still logged)')
    args = parser.parse_args()
    profiles = tuple(args.profiles)+(OPTIONAL if args.with_local and 'g_cf_local' not in args.profiles else ())
    result = prepare(args.base, args.output, args.seeds, profiles, args.known_guard == 'on', args.control, args.oracle_labels, args.phase)
    if args.control:
        manifest = json.loads((Path(args.output)/'manifest.json').read_text())
        for entry in manifest['entries']:
            if entry['name'] == 'control':
                print(f'control seed{entry["seed"]} differs from g_cf only in: {", ".join(entry["differs_from_g_cf"])}')
    print(json.dumps(result, indent=2))
