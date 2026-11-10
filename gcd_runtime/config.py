"""One supported training protocol: CRiSH or its matched baseline control."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = 'ffc895b7ff38be8502bb7e3bd851f1d2a55dff73'
TRAIN_SHA = 'af872b72e0d0068f0492a032299a59decbbc049f6d45a4934fdb57bb5453a1d5'
PLAIN_KMEANS_SHA = 'f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805'


def load_config(path, *, eval_every_epochs=None):
    from crish.config import CrishSettings
    cfg = json.loads(Path(path).read_text())
    if cfg.get('baseline_family') == 'selex':
        from selex_runtime.config import validate_config
        return validate_config(cfg, eval_every_epochs=eval_every_epochs)
    if cfg.get('method') not in ('crish', 'baseline'):
        raise ValueError('method must be crish or baseline')
    if 'reliability' in cfg:
        raise ValueError('Retired reliability settings are not part of this source release')
    defaults = dict(dataset_name='scars', training_protocol='crish_v2_1', kmeans_timeout_seconds=120,
        align_cluster_rows=True, training_diagnostics_every=0, test_every_epochs=0,
        plain_train_eval_every=0, final_ss_evaluation=True, evaluation_rng='isolated',
        max_steps_per_epoch=0, final_eval_timeout_seconds=1800, momentum=.9,
        ss_eval_every_epochs=0, ss_eval_timeout_seconds=1800,
        temperature=1., distance='euclidean', strategy='zero_one', cluster_momentum=1.,
        n_views=2, hyper_start_epoch=0, hyper_end_epoch=200,
        use_target_normalization='legacy')
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    from .datasets import dataset_spec
    dataset_spec(cfg['dataset_name'])
    if cfg['use_target_normalization'] not in ('legacy', 'row'):
        raise ValueError('use_target_normalization must be legacy or row')
    if eval_every_epochs is not None:
        cfg['ss_eval_every_epochs'] = eval_every_epochs
    if cfg['training_protocol'] != 'crish_v2_1':
        raise ValueError('Only the CRiSH/B0 training protocol is supported')
    if cfg['align_cluster_rows'] is not True or cfg['evaluation_rng'] != 'isolated':
        raise ValueError('Both arms require row alignment and isolated evaluation RNG')
    if cfg.get('checkpoint_selection') not in ('final_epoch', 'best_old_test_v2'):
        raise ValueError('checkpoint_selection must be final_epoch or best_old_test_v2')
    expected_test_interval = 1 if cfg['checkpoint_selection'] == 'best_old_test_v2' else 0
    if type(cfg['test_every_epochs']) is not int or cfg['test_every_epochs'] != expected_test_interval:
        raise ValueError('best_old_test_v2 requires test_every_epochs=1; final_epoch requires 0')
    if type(cfg['plain_train_eval_every']) is not int or cfg['plain_train_eval_every'] != 0:
        raise ValueError('Plain train k-means is not the reported balanced SS protocol')
    if type(cfg['final_ss_evaluation']) is not bool or cfg.get('early_stopping', False) is not False:
        raise ValueError('Invalid final-evaluation or model-selection setting')
    if expected_test_interval and not cfg['final_ss_evaluation']:
        raise ValueError('best_old_test_v2 requires final SS evaluations of both best and last')
    for key in ('epochs', 'batch_size', 'max_runtime_seconds', 'kmeans_timeout_seconds', 'final_eval_timeout_seconds', 'ss_eval_timeout_seconds'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'num_workers', 'grad_from_block', 'training_diagnostics_every', 'max_steps_per_epoch', 'ss_eval_every_epochs'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if cfg['device'] not in ('auto', 'cpu', 'mps', 'cuda') or cfg['dino'] not in ('v1', 'v2'):
        raise ValueError('Unsupported device or backbone')
    if cfg['grad_from_block'] > 11 or type(cfg['mlp_out_dim']) is not int or cfg['mlp_out_dim'] < 2:
        raise ValueError('Invalid trainable block or projection dimension')
    if cfg['prototype_extraction_interval'] != 1:
        raise ValueError('Use epoch-wise HSSK')
    for key in ('lr', 'weight_decay', 'c', 'cr', 'sup_con_weight', 'unsupervised_smoothing',
                'hyper_max_weight', 'momentum', 'temperature', 'cluster_momentum'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if cfg['lr'] <= 0 or cfg['c'] <= 0 or cfg['temperature'] <= 0 or cfg['momentum'] >= 1:
        raise ValueError('Invalid learning rate, curvature, temperature or momentum')
    if any(cfg[key] > 1 for key in ('sup_con_weight', 'unsupervised_smoothing', 'hyper_max_weight')):
        raise ValueError('Loss weights/smoothing must be bounded by one')
    if cfg['distance'] != 'euclidean' or cfg['cluster_momentum'] != 1 or type(cfg['n_views']) is not int or cfg['n_views'] != 2:
        raise ValueError('Verified rebuild requires Euclidean prototypes, momentum1, two views')
    if cfg['strategy'] not in ('zero_one', 'pair_dist', 'pair_cluster'):
        raise ValueError('Unsupported USE strategy')
    if any(type(cfg[k]) is not int for k in ('hyper_start_epoch', 'hyper_end_epoch')) or cfg['hyper_end_epoch'] <= cfg['hyper_start_epoch']:
        raise ValueError('Invalid hyperbolic ramp interval')
    cfg['crish'] = asdict(CrishSettings(**cfg.get('crish', {})))
    return cfg


def paths():
    data = Path(os.environ.get('GCD_DATA_ROOT', ROOT/'datasets')).expanduser().resolve()
    weights = Path(os.environ.get('GCD_WEIGHTS_ROOT', ROOT/'weights')).expanduser().resolve()
    runs = Path(os.environ.get('GCD_RUNS_ROOT', ROOT/'runs')).expanduser().resolve()
    return data, weights, runs


def choose_device(torch, requested='auto'):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    return torch.device(requested)


def code_fingerprint():
    import hashlib
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('gcd_runtime', 'crish', 'selex_runtime')
            for p in sorted((ROOT/directory).glob('*.py'))}
