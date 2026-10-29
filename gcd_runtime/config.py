"""Dependency-free configuration validation shared by local and remote tools."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = 'ffc895b7ff38be8502bb7e3bd851f1d2a55dff73'
TRAIN_SHA = 'af872b72e0d0068f0492a032299a59decbbc049f6d45a4934fdb57bb5453a1d5'


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault('kmeans_timeout_seconds', 120)
    if cfg['method'] not in ('baseline', 'proposed'):
        raise ValueError('method must be baseline or proposed')
    from .method_config import ReliabilitySettings
    from dataclasses import asdict
    settings = ReliabilitySettings(**cfg.get('reliability', {}))
    cfg['reliability'] = asdict(settings)
    if cfg['device'] not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('Invalid device')
    for key in ('epochs', 'batch_size', 'max_runtime_seconds', 'kmeans_timeout_seconds'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if cfg['dino'] != 'v1':
        raise ValueError('Initial runtime supports audited DINO v1 only')
    if cfg['prototype_extraction_interval'] != 1:
        raise ValueError('Epoch resume currently requires prototype extraction every epoch')
    if cfg['checkpoint_selection'] != 'final_epoch':
        raise ValueError('Only preregistered final_epoch selection is supported')
    for key in ('num_workers', 'grad_from_block'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if cfg['grad_from_block'] > 11:
        raise ValueError('grad_from_block must select a ViT block 0..11')
    if type(cfg['mlp_out_dim']) is not int or cfg['mlp_out_dim'] < 2:
        raise ValueError('Invalid projection dimension')
    import math
    for key in ('lr', 'weight_decay', 'c', 'cr', 'sup_con_weight', 'unsupervised_smoothing', 'hyper_max_weight'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if cfg['lr'] == 0 or cfg['c'] == 0 or cfg['sup_con_weight'] > 1 or cfg['unsupervised_smoothing'] > 1:
        raise ValueError('Invalid learning rate, curvature or loss weight')
    return cfg


def paths():
    data = Path(os.environ.get('GCD_DATA_ROOT', ROOT / 'datasets')).expanduser().resolve()
    weights = Path(os.environ.get('GCD_WEIGHTS_ROOT', ROOT / 'weights')).expanduser().resolve()
    runs = Path(os.environ.get('GCD_RUNS_ROOT', ROOT / 'runs')).expanduser().resolve()
    return data, weights, runs


def choose_device(torch, requested='auto'):
    if requested == 'auto':
        # Prefer CPU for local tests; MPS is explicit opt-in until verified.
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    return torch.device(requested)


def code_fingerprint():
    import hashlib
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'gcd_runtime').glob('*.py'))}
