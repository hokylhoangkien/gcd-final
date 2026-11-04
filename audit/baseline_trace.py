"""Execute only pinned loss definitions/USE block as independent CPU test oracles.

This is not a runtime source adapter. No training/main function is executed.
HypCD source is MIT licensed; see LICENSES/HypCD-MIT.txt.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'upstream/HypCD/train/train_HypSelEx.py'
SHA256 = 'af872b72e0d0068f0492a032299a59decbbc049f6d45a4934fdb57bb5453a1d5'


def baseline_oracle(curvature=.1):
    import sys
    upstream = str(ROOT / 'upstream/HypCD')
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from hyptorch.pmath import dist_matrix
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise RuntimeError('Pinned Hyp-SelEx training source changed; audit required')
    tree = ast.parse(source)
    scope = dict(torch=torch, np=np, F=F, device=torch.device('cpu'), dist_matrix=dist_matrix,
                 args=SimpleNamespace(c=curvature, n_views=2, temperature=1.0))
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                   and node.name in ('SupConLoss', 'LabelSmoothingLoss', 'info_nce_logits')]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), scope)
    # Exact original confusion-factor block + two normalization statements.
    # Its line span is protected by the full training-source hash above.
    lines = source.decode().splitlines()
    import textwrap
    block = textwrap.dedent('\n'.join(lines[275:322]))
    normalize = textwrap.dedent('\n'.join(lines[331:333]))
    confusion_code = compile(block + '\n' + normalize, str(SOURCE) + ':276-333', 'exec')

    def confusion(backbone_views, labels, prototypes, radii, *, strategy='zero_one'):
        count = len(labels[0])
        local = dict(scope, all_features=backbone_views, preds_ind_list=labels,
                     cluster_protos_list=prototypes, cluster_radius_list=[r.clone() for r in radii],
                     Distance='euclidean', cluster_momentum=1.0, strategy=strategy,
                     uq_index=np.arange(count), uq_idxs=np.arange(count),
                     args=SimpleNamespace(num_labeled_classes=len(prototypes[0]) // 2,
                                          num_unlabeled_classes=len(prototypes[0]) // 2))
        exec(confusion_code, local)
        return local['confusion_factor']

    scope['confusion_oracle'] = confusion
    return scope
