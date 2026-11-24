"""Bounded scientific integration checks, using the official head and losses."""

import argparse
import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from crish.data import ImageOnly
from crish.repair import Action, VirtualClassifier, corrected_distill, select
from crish.runtime import Repairs, arguments
from model import DINOHead, DistillLoss, SupConLoss, get_params_groups, info_nce_logits

torch.set_num_threads(1)


def fixture(seed=1, device="cpu"):
    torch.manual_seed(seed)
    head = DINOHead(8, 4, hidden_dim=12, bottleneck_dim=6).to(device)
    backbone = nn.Linear(8, 8).to(device)
    model = nn.Sequential(backbone, head)
    opt = torch.optim.SGD(
        get_params_groups(model), lr=0.018, momentum=0.9, weight_decay=0.001
    )
    images = torch.randn(48, 8)
    images = torch.cat([images, images + 0.13 * torch.randn_like(images)]).to(device)
    # Populate actual momentum buffers before the virtual comparison.
    model(images)[1].square().mean().backward()
    opt.step()
    opt.zero_grad()
    x = backbone(images)
    proj, out = head(x)
    legal = torch.arange(48, device=device) % 3
    legal[24:] = -1
    ids = [101 + 7 * i for i in range(48)]
    crit = DistillLoss(0, 1)
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parser.parse_args([])
    args.sup_weight, args.memax_weight, args.epochs, args.n_views = 0.35, 1.0, 1, 2
    args.crish_warmup, args.run_seed, args.crish_mode = 0, seed, "E3"
    args.crish_cap = 2
    return model, opt, x, proj, out, legal, ids, crit, args


class Checks(unittest.TestCase):
    def test_distillation_and_gradients(self):
        _, _, _, _, out, _, _, crit, _ = fixture()
        loss = crit(out, out.detach(), 0)
        keep = corrected_distill(out, out, crit, 0, [], 0.2)
        self.assertTrue(torch.equal(loss, keep))
        (grad,) = torch.autograd.grad(loss, out, retain_graph=True)
        (got,) = torch.autograd.grad(keep, out, retain_graph=True)
        self.assertTrue(torch.equal(grad, got))
        i = 24
        a = Action(i, "abstain", 0)
        masked = corrected_distill(out, out, crit, 0, [a], 0.2)
        (abgrad,) = torch.autograd.grad(masked, out, retain_graph=True)
        self.assertEqual(abgrad[[i, i + 48]].abs().sum().item(), 0)
        torch.testing.assert_close(abgrad[:i], grad[:i])
        teacher = torch.zeros_like(out)
        teacher[:, 0] = 1000
        move = corrected_distill(out, teacher, crit, 0, [Action(i, "move", 0, 1)], 0.2)
        original = crit(out, teacher, 0)
        (shift,) = torch.autograd.grad(move - original, out)
        expected = torch.zeros_like(out)
        expected[[i, i + 48], 0] = 0.2 / (96 * 0.1)
        expected[[i, i + 48], 1] = -0.2 / (96 * 0.1)
        torch.testing.assert_close(shift, expected)

    def test_virtual_sgd_matches_real_classifier_step(self):
        model, opt, x, _, out, legal, _, crit, args = fixture()
        layer = model[1].last_layer
        virtual = VirtualClassifier(layer, opt)
        features = F.normalize(x.detach(), dim=-1).reshape(2, 48, -1)
        teachers = out.detach().reshape(2, 48, -1)
        before = copy.deepcopy(model.state_dict())
        buffer = opt.state[layer.weight_v]["momentum_buffer"].clone()
        for actions in ([], [Action(24, "abstain", 0)]):
            actual = virtual.step(features, teachers, legal, crit, 0, actions, args)
            torch.testing.assert_close(
                actual,
                virtual.step(features, teachers, legal, crit, 0, actions, args),
                rtol=0,
                atol=0,
            )
            # Independent real torch.optim.SGD with the same state and stock head.
            replica = DINOHead(8, 4, hidden_dim=12, bottleneck_dim=6)
            replica.load_state_dict(model[1].state_dict())
            param = replica.last_layer.weight_v
            other = torch.optim.SGD([param], lr=0.018, momentum=0.9, weight_decay=0.001)
            other.state[param]["momentum_buffer"] = buffer.clone()
            logits = replica(x.detach())[1]
            dist = corrected_distill(logits, out.detach(), crit, 0, actions, 0.2)
            p = (logits / 0.1).softmax(-1).mean(0)
            mask = legal >= 0
            ce = F.cross_entropy(
                logits.reshape(2, 48, -1)[:, mask].flatten(0, 1) / 0.1,
                legal[mask].repeat(2),
            )
            loss = 0.65 * (dist + (p * p.log()).sum() + np.log(4)) + 0.35 * ce
            loss.backward()
            other.step()
            torch.testing.assert_close(actual, param)
        for k, v in before.items():
            self.assertTrue(torch.equal(v, model.state_dict()[k]))
        self.assertTrue(
            torch.equal(buffer, opt.state[layer.weight_v]["momentum_buffer"])
        )

    def test_threshold_keep_and_identical_discovery_inputs(self):
        from unittest.mock import patch

        model, opt, x, _, out, legal, ids, crit, args = fixture()
        args.crish_gain = 1e5
        args.crish_cap = 2
        chosen, rec = select(
            x, out, ids, legal, model[1].last_layer, opt, crit, 0, args
        )
        self.assertEqual(chosen, [])
        self.assertEqual(rec["stop"], "discovery")
        args.crish_gain = 0
        # Observe actual probe inputs without changing its scores.
        ref = VirtualClassifier.probe
        calls = []

        def veto(self, v, inputs, labels, anchor):
            calls.append(inputs.detach().clone())
            return ref(self, v, inputs, labels, anchor)

        with patch.object(VirtualClassifier, "probe", veto):
            select(x, out, ids, legal, model[1].last_layer, opt, crit, 0, args)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0], calls[1]))

    def test_image_only(self):
        class Trap:
            uq_idxs = np.arange(4)
            data = ("a", "b", "c", "d")
            loader = staticmethod(lambda path: path)
            transform = staticmethod(lambda path: path.upper())

            def __getitem__(self, i):
                raise AssertionError("annotation-bearing getter")

            @property
            def target(self):
                raise AssertionError("hidden targets")

        self.assertEqual(ImageOnly(Trap(), "scars")[2], ("C", -1, 2))

    def test_active_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ["E3", "E1", "E2", "E4"]:
                model, opt, x, proj, out, legal, ids, crit, args = fixture(
                    SMOKE_SEED, DEVICE
                )
                directory = Path(tmp) / mode
                directory.mkdir()
                args.model_path = directory / "model.pt"
                args.crish_mode = mode
                if mode in ("E1", "E2"):
                    args.crish_dose = Path(tmp) / "E3" / "crish.jsonl"
                process = Repairs(args)
                self.assertTrue(process.due(0, 0))
                self.assertFalse(process.due(0, 1))
                ids[-1] = ids[-2]  # replacement sampler collision
                snapshot = copy.deepcopy(model.state_dict())
                momentum = opt.state[model[1].last_layer.weight_v][
                    "momentum_buffer"
                ].clone()
                rng, py = torch.get_rng_state(), random.getstate()
                gpu_rng = torch.cuda.get_rng_state() if DEVICE == "cuda" else None
                repaired = process.loss(
                    x, out, ids, legal, model[1].last_layer, opt, crit, 0
                )
                self.assertTrue(
                    torch.equal(rng, torch.get_rng_state()) and py == random.getstate()
                )
                if gpu_rng is not None:
                    self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))
                self.assertTrue(
                    all(
                        torch.equal(v, model.state_dict()[k])
                        for k, v in snapshot.items()
                    )
                )
                self.assertTrue(
                    torch.equal(
                        momentum,
                        opt.state[model[1].last_layer.weight_v]["momentum_buffer"],
                    )
                )
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                r = process.pending
                partitions = [r["score_ids"]] + r["probe_ids"]
                unique = [key for part in partitions for key in part]
                self.assertEqual(len(unique), len(set(unique)))
                self.assertNotIn(ids[-1], unique)
                self.assertTrue(all(a["id"] in ids[24:] for a in r["accepted"]))
                self.assertGreater(len(r["accepted"]), 0, r)
                if mode in ("E3", "E4"):
                    for d in r["decisions"]:
                        if d["accepted"]:
                            self.assertGreater(
                                min(
                                    d["discovery"],
                                    d["confirmation"],
                                    d["group_discovery"],
                                    d["group_confirmation"],
                                ),
                                0,
                            )
                # All four official losses, one real joint optimizer update.
                known = legal >= 0
                cls = F.cross_entropy(
                    out.reshape(2, 48, -1)[:, known].flatten(0, 1) / 0.1,
                    legal[known].repeat(2),
                )
                p = (out / 0.1).softmax(-1).mean(0)
                logits, labels = info_nce_logits(proj, device=DEVICE)
                rep = F.cross_entropy(logits, labels)
                sup = SupConLoss()(
                    F.normalize(torch.stack(proj.chunk(2), dim=1)[known], dim=-1),
                    labels=legal[known],
                )
                total = 0.65 * (
                    repaired + (p * p.log()).sum() + np.log(4) + rep
                ) + 0.35 * (cls + sup)
                total.backward()
                self.assertTrue(
                    all(
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )
                opt.step()
                process.finish_step(True)
                print(
                    mode,
                    r["counts"],
                    r["stop"],
                    round(r["elapsed_seconds"] * 1000, 2),
                    "ms",
                )


SMOKE_SEED = 2
DEVICE = "cpu"
if __name__ == "__main__":
    import sys

    if "--cuda" in sys.argv:
        DEVICE = "cuda"
        sys.argv.remove("--cuda")
    unittest.main()
