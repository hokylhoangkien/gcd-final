"""CPU contract tests and optional CUDA parity against the pinned upstream train function."""

import argparse
import copy
import json
import logging
import random
import subprocess
import tempfile
import unittest
from decimal import Decimal as D
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from crish.data import ImageOnly
from crish.experiment import load_checkpoint
from model import DINOHead, DistillLoss, get_params_groups
from relational.export import ARTIFACTS, export, verify
from relational.graph import Graph, Settings, evidence, keep
from relational.guard import evaluate, grade
from relational.state import JOURNALS, append, checkpoints, restore, save, write_json
from relational.train import sampler_for, train_epoch


class Contracts(unittest.TestCase):
    def graph(self, policy):
        keys = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-0.1, 0.9]], requires_grad=True
        )
        return Graph(
            torch.tensor([71, 5, 82, 9], dtype=torch.int32),
            keys,
            keys,
            keys,
            policy,
            Settings(k=1),
        )

    def test_keep_and_evidence(self):
        g = self.graph(keep)
        self.assertTrue((g.edge_ids == torch.tensor([[5, 71], [9, 82]])).all())
        h = torch.randn(4, 2, requires_grad=True)
        a, b, _ = g.loss(h, torch.tensor([71, 82]))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(
            torch.equal(
                torch.autograd.grad(a, h, retain_graph=True)[0],
                torch.autograd.grad(b, h)[0],
            )
        )
        self.assertFalse(g.keys.requires_grad)
        self.assertEqual(
            evidence(None, torch.tensor([0, 1, 2]), Settings()).tolist(),
            [-0.25, 0.0, 0.25],
        )

    def test_gradient_direction_zero_and_denominator(self):
        h = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
        ids = torch.tensor([71])
        key = F.normalize(torch.tensor([0.9, 0.1]), dim=0)
        for delta in (0.25, -0.25, -0.5):
            graph = self.graph(
                lambda edges, votes, cfg, delta=delta: torch.full_like(
                    votes, delta, dtype=torch.float
                )
            )
            edited, base, pending = graph.loss(h, ids)
            (diff,) = torch.autograd.grad(edited - base, h, retain_graph=True)
            self.assertEqual(float(torch.sign(-diff[0] @ key)), float(np.sign(delta)))
            if delta == -0.5:
                self.assertEqual(edited.item(), 0.0)
                self.assertEqual(torch.autograd.grad(edited, h)[0].norm().item(), 0.0)
            graph.applied(pending)
            self.assertEqual(graph.exposure_record(1)["intervention_mass"], abs(delta))
            wider = torch.stack([h[0], h[0], h[1], h[1]])
            longer = graph.loss(wider, torch.tensor([71, 1001]))[0]
            torch.testing.assert_close(longer, edited / 2)

    def test_blind_and_sampler(self):
        class Poison:
            uq_idxs = np.array([12, 29])
            data = (1, 2)
            transform = staticmethod(lambda x: x + 1)
            loader = staticmethod(lambda x: x)

            @property
            def target(self):
                raise AssertionError("Hidden label touched")

            def __getitem__(self, index):
                raise AssertionError("Annotated sample read")

        data = ImageOnly(Poison(), "scars")
        self.assertEqual(data[1], (3, -1, 29))

        class Sizes:
            labelled_dataset = range(3)
            unlabelled_dataset = range(7)

            def __len__(self):
                return 10

        sampler = sampler_for(Sizes())
        reference = torch.utils.data.WeightedRandomSampler(
            torch.DoubleTensor([1] * 3 + [3 / 7] * 7), 10
        )
        torch.manual_seed(99)
        expected = list(reference)
        torch.manual_seed(99)
        self.assertEqual(list(sampler), expected)
        self.assertTrue(sampler.replacement)

    def test_linear_gradient_basis_matches_direct_grid(self):
        from relational.sensitivity_metrics import COEFFICIENTS, MAGNITUDES

        # Fixed graph with one weakened and one strengthened edge, a replacement
        # draw and a labeled row. Compare recombination to direct autograd.
        def policy(edges, votes, cfg):
            return torch.tensor([-cfg.magnitude, cfg.magnitude])

        ids = torch.tensor([71, 82, 71, 999])
        graph = self.graph(policy)
        weight = torch.nn.Parameter(torch.tensor([[0.2, 0.3], [0.1, -0.4]]))
        features = torch.arange(16).reshape(8, 2).float() @ weight
        edited, plain, _ = graph.loss(features, ids)

        def grad(value):
            return torch.autograd.grad(value, weight, retain_graph=True)[0]

        keep = grad(plain)
        unit_edit = grad((edited - plain) / graph.settings.magnitude)
        for delta in MAGNITUDES:
            direct = self.graph(lambda e, v, cfg, d=delta: torch.tensor([-d, d]))
            total = direct.loss(features, ids)[0]
            for coeff in COEFFICIENTS:
                torch.testing.assert_close(
                    grad(coeff * total), coeff * (keep + delta * unit_edit)
                )

    def test_repeated_draws_and_rejected_inputs(self):
        graph = self.graph(evidence)
        _, _, pending = graph.loss(torch.randn(8, 2), torch.tensor([5, 71, 71, 999]))
        graph.applied(pending)
        out = graph.exposure_record(4)
        self.assertEqual(out["applied_edges"], [[5, 71, 3]])
        self.assertEqual(out["unique_edited_anchors"], 2)
        self.assertEqual(graph.settings.coefficient, 1.0)
        self.assertEqual(out["effective_mass"], graph.settings.coefficient * 0.75 / 4)
        with self.assertRaises(ValueError):
            self.graph(lambda edges, votes, cfg: torch.full_like(votes, 2.0))

    def test_guard_boundaries(self):
        for epoch, a, n, verdict in [
            (100, "-.5", "3", "NO-GO"),
            (100, "4", "-1", "NO-GO"),
            (100, "1", ".5", "GO"),
            (100, ".99", ".5", "GRAY ZONE"),
            (150, ".29999", "2", "NO-GO"),
            (150, "1", "-.00001", "NO-GO"),
            (150, ".3", "0", "MARGINAL"),
            (150, "1", ".5", "STRONG GO"),
        ]:
            self.assertEqual(grade(epoch, D(a), D(n)), verdict)

    def test_matched_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in [("R0", 0.5), ("R1", 0.52), ("R2", 0.51)]:
                p = root / name
                p.mkdir()
                write_json(
                    p / "config.json", {"protocol": {"seed": 1}, "variant": name}
                )
                (p / "epoch-100.pt").touch()
                for epoch in range(1, 101):
                    acc = 0.99 if epoch <= 80 else value
                    append(
                        p / "metrics.jsonl", {"epoch": epoch, "all": acc, "new": acc}
                    )
            self.assertEqual(evaluate(root, "R2", 100)["decision"], "BLOCKED")
            (root / "R0" / "epoch-200.pt").touch()
            result = evaluate(root, "R2", 100)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertEqual(result["comparisons"]["R0"]["decision"], "GO")
            self.assertEqual(result["comparisons"]["R1"]["delta_all_pp"], -1.0)
            write_json(
                root / "R1" / "config.json", {"protocol": {"seed": 2}, "variant": "R1"}
            )
            self.assertEqual(evaluate(root, "R2", 100)["decision"], "BLOCKED")

    def test_resume_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "R2"
            folder.mkdir()
            identity = {"variant": "R2", "protocol": {"seed": 1}}

            def setup():
                m = torch.nn.Linear(3, 1)
                opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, 200, eta_min=0.0001
                )
                return m, opt, sched

            def update(m, opt, sched):
                sampled = list(torch.utils.data.WeightedRandomSampler([0.2, 1, 0.5], 3))
                x = torch.randn(3, 3)[sampled] + random.random() + np.random.rand()
                opt.zero_grad()
                m(x).square().mean().backward()
                opt.step()
                sched.step()
                return sampled

            m, opt, sched = setup()
            for name in JOURNALS:
                append(folder / name, {"epoch": 100})
            update(m, opt, sched)
            save(folder, 100, identity, m, opt, sched)
            sampled = update(m, opt, sched)
            expected = copy.deepcopy(m.state_dict())
            for name in JOURNALS:
                append(folder / name, {"uncommitted": True})
            m2, opt2, sched2 = setup()
            self.assertEqual(restore(folder, identity, m2, opt2, sched2), 100)
            self.assertEqual(update(m2, opt2, sched2), sampled)
            for key, x in expected.items():
                self.assertTrue(torch.equal(x, m2.state_dict()[key]))
            for a, b in zip(opt.state.values(), opt2.state.values()):
                self.assertTrue(torch.equal(a["momentum_buffer"], b["momentum_buffer"]))
            self.assertEqual(sched.state_dict(), sched2.state_dict())
            self.assertEqual(len((folder / JOURNALS[0]).read_text().splitlines()), 1)
            for e in (110, 120):
                save(folder, e, identity, m2, opt2, sched2)
            self.assertEqual(
                [p.stem for p in checkpoints(folder)], ["epoch-100", "epoch-120"]
            )
            for name in ARTIFACTS:
                (folder / name).touch()
            bundle = export(folder)
            verify(bundle)
            bundle.with_suffix(".tar.sha256").write_text("bad")
            with self.assertRaises(ValueError):
                verify(bundle)

    def test_normal_training_path_changes_backbone(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

            def forward(self, x):
                return self.blocks[0](x)

        batch = (
            [
                torch.tensor([[1.0, 0.4], [0.3, 0.8], [0.2, 0.6], [0.7, 0.3]]),
                torch.tensor([[0.8, 0.3], [0.4, 0.7], [0.2, 0.8], [0.9, 0.4]]),
            ],
            torch.tensor([0, 1, -1, -1]),
            torch.tensor([400, 401, 71, 82]),
            torch.tensor([[1], [1], [0], [0]]),
        )
        weights = []
        for policy in (keep, evidence):
            torch.manual_seed(1)
            m = torch.nn.Sequential(
                Backbone(), DINOHead(2, 3, hidden_dim=4, bottleneck_dim=3)
            )
            opt = torch.optim.SGD(get_params_groups(m), lr=0.1, momentum=0.9)
            graph = self.graph(policy)
            stats = train_epoch(
                m, [batch], opt, DistillLoss(30, 200), 10, graph, torch.device("cpu")
            )
            delta = stats["gradient_first_batch"]["edit_delta"]["norm"]
            self.assertEqual(delta == 0, policy is keep)
            weights.append(m[0].blocks[0].weight.detach().clone())
        self.assertFalse(torch.equal(*weights))

    def test_controller_stops_exports_and_reference_resume(self):
        from relational.run import drive

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fixture(name):
                folder = root / name
                folder.mkdir()
                for artifact in ARTIFACTS:
                    (folder / artifact).touch()
                write_json(
                    folder / "config.json", {"protocol": {"seed": 1}, "variant": name}
                )
                for e in range(1, 151):
                    append(
                        folder / "metrics.jsonl",
                        {"epoch": e, "all": 0.4 if name == "R1" else 0.5, "new": 0.5},
                    )
                (folder / "epoch-100.pt").touch()
                return folder

            target = fixture("R1")
            stages = []

            def fake_call(module, args, log):
                if module == "relational.train":
                    stages.append(args[-1])
                    (target / f"epoch-{args[-1]}.pt").touch()
                else:
                    write_json(target / "geometry.json", {"offline_fixture": True})

            with patch("relational.run.call", side_effect=fake_call):
                drive(root, "R1")
                self.assertEqual(
                    json.loads((target / "status.json").read_text())["decision"],
                    "BLOCKED",
                )
                ref = fixture("R0")
                for e in (150, 200):
                    (ref / f"epoch-{e}.pt").touch()
                drive(root, "R1")
                self.assertEqual(
                    json.loads((target / "status.json").read_text())["decision"],
                    "NO-GO",
                )
                self.assertEqual(stages, [])
                verify(root / "exports" / "R1-epoch-100.tar")
                drive(root, "R1", reference_through=150)
                self.assertEqual(stages, [150])
                self.assertTrue(
                    json.loads((target / "status.json").read_text())["reference_only"]
                )


def cuda_parity():
    source = subprocess.check_output(
        ["git", "show", "f843f27195701b7c8c5713cfd17499ad67c86247:train.py"]
    )
    ref = ModuleType("official")
    exec(compile(source, "pinned-official/train.py", "exec"), ref.__dict__)  # noqa: S102 - pinned local upstream blob
    torch.manual_seed(19)
    x = torch.randn(32, 8)
    ids = torch.arange(32)
    known = ids < 16
    batch = (
        [x, x + 0.1 * torch.randn_like(x)],
        (ids % 3).masked_fill(~known, -1),
        ids,
        known[:, None],
    )
    with tempfile.TemporaryDirectory() as tmp:
        args = SimpleNamespace(
            epochs=1,
            fp16=False,
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-5,
            warmup_teacher_temp_epochs=1,
            n_views=2,
            warmup_teacher_temp=0.07,
            teacher_temp=0.04,
            sup_weight=0.35,
            memax_weight=1.0,
            print_freq=1,
            logger=logging.getLogger(),
            train_classes=[0, 1],
            eval_funcs=["v2"],
            model_path=str(Path(tmp) / "upstream.pt"),
        )

        def make():
            torch.manual_seed(27)
            return torch.nn.Sequential(
                torch.nn.Linear(8, 8), DINOHead(8, 4, hidden_dim=12, bottleneck_dim=6)
            ).cuda()

        up = make()
        ref.train(up, [batch], None, [(x, ids % 4, ids)], args)
        expected = load_checkpoint(args.model_path)
        own = make()
        opt = torch.optim.SGD(
            get_params_groups(own), lr=0.1, momentum=0.9, weight_decay=5e-5
        )
        train_epoch(
            own, [batch], opt, DistillLoss(30, 200), 0, None, torch.device("cuda")
        )
        for k, v in own.state_dict().items():
            assert torch.equal(v.cpu(), expected["model"][k]), k
        for i, z in opt.state_dict()["state"].items():
            assert torch.equal(
                z["momentum_buffer"].cpu(),
                expected["optimizer"]["state"][i]["momentum_buffer"],
            )
    print(
        "PASS: R0 and upstream one-step weights and momentum buffers agree bit-for-bit"
    )


def cuda_resume():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def forward(self, x):
            return self.blocks[0](x)

    device = torch.device("cuda")

    def setup():
        model = torch.nn.Sequential(
            Backbone(), DINOHead(4, 4, hidden_dim=8, bottleneck_dim=4)
        ).to(device)
        opt = torch.optim.SGD(
            get_params_groups(model), lr=0.1, momentum=0.9, weight_decay=5e-5
        )
        return (
            model,
            opt,
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200, eta_min=0.0001),
        )

    def step(state, epoch):
        model, opt, sched = state
        ids = torch.arange(32)
        mask = ids < 16
        a = torch.randn(32, 4)
        b = a + 0.15 * torch.randn_like(a)
        with torch.no_grad():
            x = model[0](a[~mask].to(device))
            y = model[0](b[~mask].to(device))
        graph = Graph(ids[~mask], x, x, y, evidence)
        draws = torch.tensor(
            list(torch.utils.data.WeightedRandomSampler([1.0] * 32, 32))
        )
        batch = (
            [a[draws], b[draws]],
            (ids % 3).masked_fill(~mask, -1)[draws],
            ids[draws],
            mask[draws, None],
        )
        # Include CUDA RNG in the exact-continuation check.
        torch.rand(3, device=device)
        train_epoch(model, [batch], opt, DistillLoss(30, 200), epoch, graph, device)
        sched.step()
        return draws, graph.exposure_record(32), torch.rand(3, device=device)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name in JOURNALS:
            (folder / name).touch()
        original = setup()
        step(original, 10)
        save(folder, 10, "fixture", *original)
        draws, ledger, rng = step(original, 11)
        replay = setup()
        restore(folder, "fixture", *replay)
        draws2, ledger2, rng2 = step(replay, 11)
        assert (
            torch.equal(draws, draws2) and ledger == ledger2 and torch.equal(rng, rng2)
        )
        for key, value in original[0].state_dict().items():
            assert torch.equal(value, replay[0].state_dict()[key]), key
        for a, b in zip(original[1].state.values(), replay[1].state.values()):
            assert torch.equal(a["momentum_buffer"], b["momentum_buffer"])
        assert original[2].state_dict() == replay[2].state_dict()
    print(
        "PASS: CUDA R2 epoch replay, sampled IDs, exposures, parameters, momentum, scheduler and RNG"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cuda-parity", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.cuda_parity:
        cuda_parity()
        cuda_resume()
    else:
        unittest.main(argv=["validate_relational"], verbosity=2)
