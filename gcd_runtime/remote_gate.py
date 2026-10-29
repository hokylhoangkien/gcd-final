"""Internal bounded remote gate: real CUDA + upstream imports + actual data audit."""
import json
from pathlib import Path
import sys
from .config import load_config
from .probe import tensor_probe, backbone_probe, method_probe
from .readiness import inspect_inputs, audit_dataset

if __name__ == '__main__':
    cfg = load_config(sys.argv[1])
    if cfg['device'] not in ('auto', 'cuda'):
        raise RuntimeError('Remote baseline requires CUDA; CPU/MPS are smoke-only')
    result = inspect_inputs()
    result['probe'] = tensor_probe('cuda', baseline=True)
    result['data_counts'] = audit_dataset()
    result['backbone_probe'] = backbone_probe(cfg)
    result['method_probe'] = method_probe(cfg, 'cuda')
    Path(sys.argv[2], 'readiness.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
