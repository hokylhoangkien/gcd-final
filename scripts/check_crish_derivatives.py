"""CPU identity checks against the checked-out method; no model/optimizer step on real data."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crish.repair import Action, VirtualClassifier, corrected_distill
from model import DINOHead, DistillLoss


def main():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    n, k, d, row = 128, 12, 24, 87
    head = DINOHead(d, k)
    h = torch.randn(2 * n, d, requires_grad=True)
    _, z = head(h)
    v = head.last_layer.weight_v
    teacher = z.detach().clone().reshape(2, n, k)
    teacher[:, row, 2] += 3  # legally supported fixed test action
    teacher = teacher.flatten(0, 1)
    q = (teacher / .04).softmax(-1).reshape(2, n, k).flip(0)
    p = (z.detach() / .1).softmax(-1).reshape_as(q)
    criterion = DistillLoss(30, 200)
    factual = corrected_distill(z, teacher, criterion, 100, [], .2)
    inputs = (z, h, v)
    keep = torch.autograd.grad(.65 * factual, inputs, retain_graph=True)
    torch.testing.assert_close(keep[0], (.65 * (p - q) / (2 * n * .1)).reshape_as(z))
    for kind in ("move", "abstain"):
        edits = [Action(row, kind, 2, 3)]
        for mass in (.1, .2):
            cf = corrected_distill(z, teacher, criterion, 100, edits, mass)
            actual = torch.autograd.grad(.65 * (cf - factual), inputs, retain_graph=True)
            dz = torch.zeros_like(q)
            if kind == "move":
                dz[:, row, 2] = mass
                dz[:, row, 3] = -mass
            else:
                dz[:, row] = q[:, row] - p[:, row]
            dz = dz.flatten(0, 1) * (.65 / (2 * n * .1))
            u = F.normalize(h.detach())
            w = F.normalize(v.detach())
            g = head.last_layer.weight_g.detach()
            du = dz @ (w * g)
            dw = dz.T @ u * g
            dh = (du - u * (du * u).sum(-1, keepdim=True)) / h.detach().norm(dim=-1, keepdim=True)
            dv = (dw - w * (dw * w).sum(-1, keepdim=True)) / v.detach().norm(dim=-1, keepdim=True)
            for measured, expected in zip(actual, (dz, dh, dv)):
                torch.testing.assert_close(measured, expected, atol=2e-8, rtol=1e-4)
            optimizer = torch.optim.SGD([v], lr=.03, momentum=.9, weight_decay=5e-5)
            optimizer.state[v]["momentum_buffer"] = torch.randn_like(v)
            virtual = VirtualClassifier(head.last_layer, optimizer)
            legal = torch.arange(n) % k
            legal[64:] = -1
            opts = SimpleNamespace(crish_mass=mass, sup_weight=.35, memax_weight=1.)
            x = u.reshape(2, n, -1)
            def step(plan):
                return virtual.step(x, teacher.reshape_as(q), legal, criterion, 100, plan, opts)
            torch.testing.assert_close(step(edits) - step([]), -.03 * dv, atol=6e-8, rtol=1e-3)
            print(kind, mass, f"||delta logits||={dz.norm().item():.8f}")
    print("PASS: KEEP logits, differential MOVE/ABSTAIN logits, feature and weight Jacobians, virtual SGD")


if __name__ == "__main__":
    main()
