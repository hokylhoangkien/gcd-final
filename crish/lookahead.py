"""First-order score and one exact cloned SGD step with supplied train gradients.

No unrolling of 5/10-step training is claimed. Probe accepts LegalProbe only.
"""
from contextlib import contextmanager
import copy
import random
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .probe import LegalProbe, trusted_weight_norm_hooks


def clone_snapshot(model, optimizer=None):
    """Copy original DINOHead WeightNorm without touching live derived weights.

    WeightNorm exposes a non-leaf `weight` tensor after forward. Deepcopy's memo
    supplies detached copies for those derived attributes; cloned hooks recreate
    them from cloned weight_g/weight_v on forward. Optimizer parameters remain
    linked to their corresponding cloned registered parameters.
    """
    memo = {id(getattr(module, hook.name)): getattr(module, hook.name).detach().clone()
            for module, hook in trusted_weight_norm_hooks(model)}
    return copy.deepcopy(model if optimizer is None else (model, optimizer), memo)


def gradient_gain(q: Sequence[Tensor], action: Sequence[Tensor], keep: Sequence[Tensor], *,
                  changed_mass: float) -> dict:
    if not q or len(q) != len(action) or len(q) != len(keep):
        raise ValueError('Gradient parameter sets differ')
    if not np.isfinite(changed_mass) or changed_mass < 0:
        raise ValueError('Invalid changed-target mass')
    raw = normq = normd = 0.0
    for query, ga, g0 in zip(q, action, keep):
        if query.shape != ga.shape or query.shape != g0.shape or any(not torch.isfinite(g).all() for g in (query, ga, g0)):
            raise ValueError('Invalid gradient shape/values')
        delta = ga.detach().double() - g0.detach().double()
        query = query.detach().double()
        raw += float((query * delta).sum())
        normq += float(query.square().sum())
        normd += float(delta.square().sum())
    scale = (normq * normd) ** .5
    if not all(np.isfinite(x) for x in (raw, normq, normd, scale)):
        raise ValueError('Gradient score overflow; reject proposal')
    per_mass = raw / changed_mass if changed_mass > 0 else None
    if per_mass is not None and not np.isfinite(per_mass):
        raise ValueError('Mass-normalized score overflow; reject proposal')
    return dict(raw=raw, per_mass=per_mass,
                cosine=raw / scale if scale > 0 else None, changed_mass=changed_mass)


@contextmanager
def preserve_rng():
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def exact_sgd_comparison(model: nn.Module, optimizer: torch.optim.Optimizer,
                         keep_gradients: Sequence[Optional[Tensor]],
                         action_gradients: Sequence[Optional[Tensor]], probe: LegalProbe) -> dict:
    """Joint deepcopy preserves optimizer->cloned-parameter references and state.

    Gradients correspond to model.parameters() filtered by requires_grad, from
    the same frozen snapshot. None means unused, preserving SGD skip semantics.
    """
    if not isinstance(optimizer, torch.optim.SGD) or type(probe) is not LegalProbe:
        raise TypeError('Only verified SGD and LegalProbe are supported')
    params = [p for p in model.parameters() if p.requires_grad]
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    if not params or any(id(p) not in owned for p in params) or not owned <= {id(p) for p in model.parameters()}:
        raise ValueError('Optimizer must own model trainable parameters only')
    scores = []
    for gradients in (keep_gradients, action_gradients):
        if len(gradients) != len(params):
            raise ValueError('Gradient parameter count differs')
        for p, grad in zip(params, gradients):
            if grad is not None and (grad.shape != p.shape or not torch.isfinite(grad).all()):
                raise ValueError('Invalid virtual-step gradient')
        # Each alternative starts with identical RNG and optimizer/model state.
        with preserve_rng():
            virtual, opt = clone_snapshot(model, optimizer)
            opt.zero_grad(set_to_none=True)
            for p, grad in zip((p for p in virtual.parameters() if p.requires_grad), gradients):
                p.grad = None if grad is None else grad.detach().to(p).clone()
            opt.step()
            virtual.eval()
            with torch.no_grad():
                components = probe.components(virtual)
                loss = sum(getattr(probe.weights, key) * value for key, value in components.items())
            if not torch.isfinite(loss) or any(not torch.isfinite(p).all() for p in virtual.parameters()):
                raise ValueError('Nonfinite cloned optimizer step')
            scores.append(dict(loss=float(loss), **{key: float(value) for key, value in components.items()}))
    return dict(gain=scores[0]['loss'] - scores[1]['loss'],
                known_increase=scores[1]['known'] - scores[0]['known'],
                keep=scores[0], action=scores[1])
