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


def assert_plain_sgd(optimizer):
    """First-order displacement formula assumes dampening=0, no Nesterov."""
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError('first_order_fp64 requires SGD')
    if any(g.get('nesterov', False) or g.get('dampening', 0) != 0 for g in optimizer.param_groups):
        raise ValueError('first_order_fp64 requires nesterov=False and dampening=0')


def first_order_component_gains(probe_grads, delta_grads, lr):
    """Positive is improvement; cast operands BEFORE multiplying and accumulating.

    A scalar lr is usual; per-parameter rates support multiple SGD groups.
    None is an unused parameter and contributes zero.
    """
    import math
    rates = [lr]*len(delta_grads) if isinstance(lr, (int, float)) else list(lr)
    if len(rates) != len(delta_grads) or any(not math.isfinite(x) or x < 0 for x in rates):
        raise ValueError('Invalid first-order learning rates')
    gains = {}
    for name, gradients in probe_grads.items():
        if len(gradients) != len(delta_grads):
            raise ValueError('Gradient parameter sets differ')
        terms = []
        for query, delta, rate in zip(gradients, delta_grads, rates):
            if query is None or delta is None:
                continue
            if query.shape != delta.shape or not torch.isfinite(query).all() or not torch.isfinite(delta).all():
                raise ValueError('Invalid component/delta gradients')
            terms.append(rate*float((query.detach().double()*delta.detach().double()).sum()))
        gain = math.fsum(terms)
        if not math.isfinite(gain):
            raise ValueError('Nonfinite first-order gain')
        gains[name] = gain
    return gains


def component_gradients_after_keep(model, optimizer, keep_gradients, probes):
    """One shared Keep clone, with real momentum/decay; no live mutations."""
    assert_plain_sgd(optimizer)
    with preserve_rng():
        virtual, opt = clone_snapshot(model, optimizer)
        parameters = tuple(p for p in virtual.parameters() if p.requires_grad)
        if len(parameters) != len(keep_gradients):
            raise ValueError('Keep gradient count differs')
        opt.zero_grad(set_to_none=True)
        for p, grad in zip(parameters, keep_gradients):
            p.grad = None if grad is None else grad.detach().to(p).clone()
        opt.step()
        virtual.eval()
        out = []
        for probe in probes:
            if type(probe) is not LegalProbe:
                raise TypeError('Component gradients require LegalProbe')
            components = probe.components(virtual)
            out.append({name: tuple(None if g is None else g.detach() for g in
                torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True))
                for name, value in components.items()})
        return out
