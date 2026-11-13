"""Pinned, unmodified SelEx sources; import definitions without running main."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import types

from gcd_runtime.config import ROOT, paths

COMMIT = 'd8107b5458f3337b2d5342f79345a976037a1265'
SOURCE_ROOT = ROOT/'upstream/SelEx'
TRAIN = SOURCE_ROOT/'methods/contrastive_training/contrastive_training.py'


def verify_reference():
    snapshot = json.loads((ROOT/'upstream/SelEx.SNAPSHOT.json').read_text())
    if snapshot['commit'] != COMMIT:
        raise ValueError('SelEx snapshot commit mismatch')
    for name, digest in snapshot['files'].items():
        path = (SOURCE_ROOT/name).resolve()
        if (not path.is_relative_to(SOURCE_ROOT.resolve()) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError(f'SelEx snapshot mismatch: {name}')
    return dict(sha=COMMIT, kind='verified_selex_snapshot')


def definitions(device='cpu'):
    """Original arithmetic is the executable oracle, not a translated loss."""
    import torch
    import numpy as np
    import torch.nn.functional as F
    verify_reference()
    tree = ast.parse(TRAIN.read_text())
    names = {'SupConLoss', 'LabelSmoothingLoss', 'info_nce_logits'}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    scope = dict(torch=torch, np=np, F=F, device=torch.device(device))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), 'exec'), scope)
    return scope


def configure_environment(run_dir, cfg):
    verify_reference()
    cache = ROOT/'.cache/matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    data, weights, _ = paths()
    # Each production worker is a fresh process. Refuse accidental mixed imports.
    for name in ('data', 'models', 'project_utils', 'methods'):
        module = sys.modules.get(name)
        filename = getattr(module, '__file__', None)
        if filename and not Path(filename).resolve().is_relative_to(SOURCE_ROOT.resolve()):
            raise RuntimeError('Use a fresh worker: mixed SelEx/HypCD module imports')
    sys.path.insert(0, str(SOURCE_ROOT))
    module = types.ModuleType('config')
    for name, value in runpy.run_path(str(SOURCE_ROOT/'config.py')).items():
        if not name.startswith('_'):
            setattr(module, name, value)
    module.car_root = str(data/'cars')
    module.cub_root = str(data/'cub')
    module.cifar_100_root = str(data/'cifar100')
    module.aircraft_root = str(data/'aircraft/fgvc-aircraft-2013b')
    module.herbarium_dataroot = str(data/'herbarium_19')
    module.osr_split_dir = str(SOURCE_ROOT/'data/ssb_splits')
    module.dino_pretrain_path = str(weights/'dino_vitb16_pretrain.pth')
    module.dino_pretrain_path2 = str(weights/'dinov2_vitb14_reg4_pretrain.pth')
    module.dino_v1 = cfg['dino'] == 'v1'
    module.exp_root = str(Path(run_dir)/'upstream_outputs')
    sys.modules['config'] = module
    from gcd_runtime.datasets import configure_modules
    configure_modules(module)
    return SOURCE_ROOT
