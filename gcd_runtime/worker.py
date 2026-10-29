"""Internal worker, launched only after the remote readiness gate."""
import json
import os
from pathlib import Path
import runpy
import sys
import types
from .checkpoint import EpochRuntime
from .config import ROOT, paths


def configure_environment(run_dir):
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
    return upstream


def main():
    config_path, run_dir, mode, checkpoint = sys.argv[1:]
    cfg = json.loads(Path(config_path).read_text())
    upstream = configure_environment(run_dir)
    flags = ['--dataset_name', 'scars', '--eval_funcs', 'v2', '--use_ssb_splits', 'True',
             '--contrast_unlabel_only', 'False', '--transform', 'imagenet']
    keys = ('seed', 'epochs', 'batch_size', 'num_workers', 'grad_from_block', 'lr',
            'weight_decay', 'sup_con_weight', 'unsupervised_smoothing', 'hyper_max_weight',
            'c', 'cr', 'mlp_out_dim', 'prototype_extraction_interval', 'dino', 'kmeans_timeout_seconds')
    for key in keys:
        flags.extend(['--' + key, str(cfg[key])])
    if mode == 'evaluate':
        flags.extend(['--eval_only', '--eval_model_path', checkpoint])
    sys.argv = ['train.train_HypSelEx'] + flags
    from . import trainer
    runtime = EpochRuntime(run_dir, cfg, checkpoint if mode == 'train' and checkpoint else None)
    trainer.run(flags, runtime)


if __name__ == '__main__':
    main()
