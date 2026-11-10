"""Internal worker, launched only after the remote readiness gate."""
import os
from pathlib import Path
import runpy
import sys
import types
from .checkpoint import EpochRuntime
from .config import ROOT, paths, load_config


def configure_environment(run_dir, cfg=None):
    if (cfg or {}).get('baseline_family') == 'selex':
        from selex_runtime.reference import configure_environment as configure_selex
        return configure_selex(run_dir, cfg)
    cache = ROOT / '.cache' / 'matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    upstream = ROOT / 'upstream/HypCD'
    data, weights, _ = paths()
    sys.path.insert(0, str(upstream))
    module = types.ModuleType('config')
    for name, value in runpy.run_path(str(upstream / 'config.py')).items():
        if not name.startswith('_'):
            if isinstance(value, str):
                value = value.replace('{DATA_PATH}', str(data)).replace('{PRETRAINED_PATH}', str(weights))
            setattr(module, name, value)
    module.osr_split_dir = str(upstream / 'data/ssb_splits')
    module.exp_root = str(Path(run_dir) / 'upstream_outputs')
    sys.modules['config'] = module
    # sklearn removed a private import used upstream; joblib is the public provider.
    import joblib
    sys.modules['sklearn.utils._joblib'] = joblib
    from .datasets import configure_modules
    configure_modules(module)
    return upstream


def build_flags(cfg, mode='train', checkpoint=''):
    """Effective CLI is inspectable without importing Torch or starting a worker."""
    flags = ['--dataset_name', cfg.get('dataset_name', 'scars'), '--eval_funcs', 'v2', '--use_ssb_splits', 'True',
             '--contrast_unlabel_only', 'False', '--transform', 'imagenet',
             '--base_model', 'vit_dino', '--prop_train_labels', '0.5']
    keys = ('seed', 'epochs', 'batch_size', 'num_workers', 'grad_from_block', 'lr',
            'weight_decay', 'sup_con_weight', 'unsupervised_smoothing', 'hyper_max_weight',
            'c', 'cr', 'mlp_out_dim', 'prototype_extraction_interval', 'dino', 'kmeans_timeout_seconds')
    for key in keys:
        flags.extend(['--' + key, str(cfg[key])])
    for key in ('momentum', 'temperature', 'distance', 'strategy', 'cluster_momentum',
                'n_views', 'hyper_start_epoch', 'hyper_end_epoch'):
        flags.extend(['--'+key, str(cfg[key])])
    if mode in ('evaluate', 'monitor', 'select'):
        flags.extend(['--eval_only', '--eval_model_path', checkpoint])
    return flags


def main():
    config_path, run_dir, mode, checkpoint, *extra = sys.argv[1:]
    cfg = load_config(config_path)
    if mode not in ('train', 'evaluate', 'monitor', 'select'):
        raise ValueError('Unsupported worker mode')
    epoch_mode = mode in ('monitor', 'select')
    epoch = int(extra[0]) if epoch_mode and len(extra) == 1 else None
    if (epoch_mode and (epoch is None or not 0 <= epoch < cfg['epochs'])) or (not epoch_mode and extra):
        raise ValueError('Monitor requires exactly one valid zero-based epoch')
    configure_environment(run_dir, cfg)
    if cfg.get('baseline_family') == 'selex':
        from selex_runtime.trainer import run
        return run(cfg, run_dir, mode, checkpoint, epoch)
    flags = build_flags(cfg, mode, checkpoint)
    sys.argv = ['train.train_HypSelEx'] + flags
    from . import trainer
    runtime = EpochRuntime(run_dir, cfg, checkpoint if mode == 'train' and checkpoint else None,
                           evaluation_epoch=epoch, evaluated_checkpoint=checkpoint if mode != 'train' else None,
                           evaluation_kind='test' if mode == 'select' else 'ss')
    trainer.run(flags, runtime)


if __name__ == '__main__':
    main()
