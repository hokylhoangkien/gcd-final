"""Focused invariants for the D1 selector and exact checkpoint boundary."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

from crish.data import ImageOnly
from crish.global_repair import frozen
from crish.repair import corrected_distill
from model import DINOHead, DistillLoss
from relational.export import export, verify
from relational.state import append, restore, save, write_json
from relational.train import train_epoch

from . import cache
from .policy import Plan, Settings, folds, scores
from .run import drive
from .train import optimizer_for, scheduler_for, step


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 16)])

    def forward(self, x):
        return self.blocks[-1](x)


def net():
    return nn.Sequential(
        Backbone(), DINOHead(16, 196, hidden_dim=24, bottleneck_dim=12)
    )


def fixture():
    bits = torch.ones(8, 1, dtype=torch.bool)
    bits[4:] = False
    return (
        [torch.randn(8, 8), torch.randn(8, 8)],
        torch.tensor([0, 1, 2, 3, -1, -1, -1, -1]),
        torch.arange(8),
        bits,
    )


class D1Checks(unittest.TestCase):
    def test_folds_and_audited_slot_support(self):
        ids = np.arange(8)
        part = folds(len(ids), 10, 1)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([a for a, _ in part])), ids
        )
        for q, b in part:
            self.assertFalse(set(q) & set(b))
        x = F.normalize(torch.randn(8, 5), dim=1)
        slots = np.arange(8) % 3
        r, d = scores(x, slots, ids, part)
        for q, b in part:
            for i in q:
                support = {
                    k: float((x[b[slots[b] == k]] @ x[i]).max()) for k in set(slots[b])
                }
                own = support.pop(slots[i], -1)
                dst = max(support, key=lambda k: (support[k], -k))
                self.assertEqual(dst, d[i])
                self.assertAlmostEqual(r[i], support[dst] - own, places=5)
        with self.assertRaisesRegex(ValueError, "overlap"):
            scores(x, slots, ids, [(ids, ids)])

    def test_live_dose_and_leakage(self):
        cfg = Settings(batch_cap=1, mass=0.8)
        plan = Plan([0], [(1, 0, 1, 0.2), (2, 0, 1, 0.1), (3, 0, 1, 0.05)], 2, cfg, {})
        logits = torch.tensor([[0.6, 0, 0], [0.6, 0, 0], [0, 0.6, 0], [0.03, 0, 0]] * 2)
        crit = DistillLoss(30, 200)
        actions, rows = plan.select(logits, [0, -1, -1, -1], list(range(4)), 11, crit)
        self.assertEqual([(a.row, a.source, a.target) for a in actions], [(1, 0, 1)])
        self.assertEqual(plan.applied, [])
        plan.commit(1, rows)
        plan.commit(
            2, plan.select(logits, [0, -1, -1, -1], list(range(4)), 11, crit)[1]
        )
        self.assertEqual(
            plan.select(logits, [0, -1, -1, -1], list(range(4)), 11, crit)[0], []
        )
        self.assertEqual(plan.counts["stale_source"], 3)
        self.assertEqual(plan.counts["low_teacher_mass"], 3)
        self.assertEqual(plan.record(12)["mass"], 1.6)
        self.assertEqual(plan.record(12)["unique_applied"], 1)
        with self.assertRaisesRegex(ValueError, "mask disagree"):
            plan.select(logits, [0, 2, -1, -1], list(range(4)), 11, crit)
        actual = corrected_distill(logits, logits, crit, 11, actions, cfg.mass)
        q = (logits / crit.teacher_temp_schedule[11]).softmax(-1).reshape(2, 4, 3)
        q[:, 1, 0] -= 0.8
        q[:, 1, 1] += 0.8
        self.assertTrue((q >= 0).all())
        torch.testing.assert_close(q.sum(-1), torch.ones(2, 4))
        t = (logits / 0.1).reshape(2, 4, 3).log_softmax(-1)
        torch.testing.assert_close(actual, -(q.flip(0) * t).sum(-1).mean())

    def test_cache_split_and_order_rejected(self):
        blind = ImageOnly.__new__(ImageOnly)
        blind.ids = np.arange(6144, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "split fingerprint"):
            cache.check_split(blind, "different")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wrong.npz"
            np.savez(path, ids=blind.ids[::-1], features=[], provenance="{}")
            with self.assertRaisesRegex(ValueError, "ID ordering"):
                cache.load(path, blind, cache.SPLIT, "cpu")
        with self.assertRaises(TypeError):
            cache.check_split(object(), cache.SPLIT)

    def test_zero_move_path_matches_sim_gcd(self):
        a, b = net(), net()
        b.load_state_dict(a.state_dict())
        batch = fixture()
        cfg = DistillLoss(30, 200)
        train_epoch(a, [batch], optimizer_for(a), cfg, 0, None, torch.device("cpu"))
        step(b, optimizer_for(b), batch, cfg, 0, None, 1)
        for x, y in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_exact_resume_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "D1"
            folder.mkdir()
            a = net()
            opt = optimizer_for(a)
            scheduler = scheduler_for(opt)
            criterion = DistillLoss(30, 200)
            step(a, opt, fixture(), criterion, 0, None, 1)
            scheduler.step()
            keys = ("metrics.jsonl", "relationships.jsonl")
            for k in keys:
                append(folder / k, {"epoch": 1})
            identity = {"variant": "D1"}
            save(folder, 1, identity, a, opt, scheduler)
            keys_next = list(WeightedRandomSampler([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5], 8))
            state_next = (random.random(), float(np.random.random()))
            expected = fixture()
            step(a, opt, expected, criterion, 1, None, 1)
            scheduler.step()
            b = net()
            opt_b = optimizer_for(b)
            sb = scheduler_for(opt_b)
            append(folder / keys[0], {"incomplete": True})
            self.assertEqual(restore(folder, identity, b, opt_b, sb), 1)
            self.assertEqual(
                keys_next,
                list(WeightedRandomSampler([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5], 8)),
            )
            self.assertEqual(state_next, (random.random(), float(np.random.random())))
            resumed = fixture()
            for i, j in zip(expected[0], resumed[0], strict=True):
                torch.testing.assert_close(i, j, rtol=0, atol=0)
            step(b, opt_b, resumed, criterion, 1, None, 1)
            sb.step()
            self.assertEqual(sb.state_dict(), scheduler.state_dict())
            for x, y in zip(a.parameters(), b.parameters(), strict=True):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                if x.requires_grad:
                    torch.testing.assert_close(
                        opt.state[x]["momentum_buffer"],
                        opt_b.state[y]["momentum_buffer"],
                        rtol=0,
                        atol=0,
                    )
            self.assertEqual(len((folder / keys[0]).read_text().splitlines()), 1)
            for k in (
                "config.json",
                "environment.json",
                "training.log",
                "guards.jsonl",
            ):
                (folder / k).touch()
            verify(export(folder, recovery=True))

    def test_refresh_rng_and_modes_are_isolated(self):
        model = net().train()
        model[1].eval()
        expected = [m.training for m in model.modules()]
        torch.manual_seed(19)
        torch_state = torch.get_rng_state()
        random.seed(11)
        np.random.seed(13)
        with frozen(model, 22, torch.device("cpu")):
            self.assertFalse(any(m.training for m in model.modules()))
            torch.rand(3)
            random.random()
            np.random.random()
        self.assertEqual(expected, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(random.random(), random.Random(11).random())
        self.assertEqual(
            float(np.random.random()), float(np.random.RandomState(13).random_sample())
        )

    def test_milestones_pause_and_require_explicit_continue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "D1"
            folder.mkdir()
            (folder / "epoch-100.pt").touch()
            decisions = []
            calls = []

            def publish(f, row, *, geometry):
                self.assertTrue(geometry)
                write_json(f / "status.json", row)
                decisions.append((row["epoch"], row["decision"]))

            def worker(module, args, log):
                self.assertEqual(module, "dual.train")
                calls.append(args[1])
                (folder / f"epoch-{args[1]}.pt").touch()

            with (
                patch("dual.run.review", return_value={"mock": True}),
                patch("dual.run.publish", side_effect=publish),
                patch("dual.run.call", side_effect=worker),
            ):
                drive(root, Path("test.npz"), False, False, None)
                self.assertEqual(calls, [])
                drive(root, Path("test.npz"), False, False, ("CONTINUE", 100))
            self.assertEqual(calls, [150])
            self.assertEqual(
                decisions, [(100, "PAUSED"), (100, "CONTINUE"), (150, "PAUSED")]
            )


if __name__ == "__main__":
    unittest.main()
