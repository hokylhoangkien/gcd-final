"""Explicit SelEx protocol, separate from historical Hyp-SelEx profiles."""
from dataclasses import asdict
import math

from crish.config import CrishSettings
from .reference import COMMIT


def validate_config(cfg, *, eval_every_epochs=None):
    cfg = dict(cfg)
    if (cfg.get('baseline_family') != 'selex' or cfg.get('upstream_sha') != COMMIT
            or cfg.get('training_protocol') != 'selex_updated_crish_v1'):
        raise ValueError('Expected pinned updated SelEx protocol')
    if cfg.get('method') not in ('baseline', 'crish'):
        raise ValueError('SelEx method must be baseline or crish')
    if cfg.get('dataset_name') != 'scars' or cfg.get('dino') not in ('v1', 'v2'):
        raise ValueError('Initial SelEx adapter supports Stanford Cars only')
    # Standardized, predeclared reporting: final SS is primary, as the updated
    # README table reports last checkpoints. We do not relabel Old-v2 as SelEx's
    # native Old-v1 selector. Plain-test monitoring remains available every epoch.
    if cfg.get('checkpoint_selection') != 'final_epoch' or not cfg.get('final_ss_evaluation'):
        raise ValueError('SelEx path requires fixed-final primary SS evaluation')
    if cfg.get('use_target_normalization') != 'selex_legacy':
        raise ValueError('Preserve updated SelEx USE arithmetic in both arms')
    if any(key in cfg for key in ('c', 'cr', 'hyper_max_weight', 'hyper_start_epoch', 'hyper_end_epoch')):
        raise ValueError('Hyperbolic settings do not belong to SelEx')
    if eval_every_epochs is not None:
        cfg['ss_eval_every_epochs'] = eval_every_epochs
    for key in ('epochs', 'batch_size', 'mlp_out_dim', 'level_interval', 'max_runtime_seconds',
                'final_eval_timeout_seconds', 'ss_eval_timeout_seconds', 'ss_n_init', 'ss_max_iter'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Invalid SelEx integer: {key}')
    for key in ('seed', 'num_workers', 'ss_eval_every_epochs', 'max_steps_per_epoch'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'Invalid SelEx integer: {key}')
    if cfg['device'] not in ('cuda', 'cpu') or cfg['grad_from_block'] not in range(12):
        raise ValueError('Invalid SelEx device/block')
    for key in ('lr', 'weight_decay', 'temperature', 'momentum', 'sup_con_weight', 'unsupervised_smoothing'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'Invalid SelEx setting: {key}')
    if (cfg['lr'] <= 0 or cfg['temperature'] <= 0 or cfg['momentum'] >= 1
            or cfg['sup_con_weight'] > 1 or cfg['unsupervised_smoothing'] > 1):
        raise ValueError('Invalid SelEx loss/optimizer settings')
    if cfg.get('strategy') != 'zero_one' or cfg.get('evaluation_rng') not in ('isolated', 'upstream'):
        raise ValueError('Expected zero_one and an explicit SelEx RNG policy')
    if cfg['evaluation_rng'] == 'upstream' and (cfg['epochs'] >= 200 or cfg.get('train_report_interval', 200) != 200):
        raise ValueError('Upstream RNG adapter supports the updated sub-200-epoch schedule; extra train-plain evaluation is unsupported')
    cfg['crish'] = asdict(CrishSettings(**cfg['crish']))
    return cfg
