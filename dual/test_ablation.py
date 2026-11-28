"""Focused matched-ablation tests on CPU with deterministic, legally masked batches."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import DistillLoss

from .ablation import (
    ARMS,
    matched_rows,
    matched_step,
    random_destinations,
    ranked,
    refresh,
    run,
)
from .policy import Plan, Settings
from .test_d1 import fixture, net
from .train import optimizer_for, step


class AblationChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_ranked_dino_preserves_positive_gate_and_reports_pool_size(self):
        ids = np.array([10, 12, 11, 13, 14, 15])
        src = np.zeros(6, dtype=int)
        r = np.array([0.1, 0.5, 0.5, 0, np.inf, 0.9])
        d = np.array([1, 2, 3, 4, 5, -1])
        proposals, n = ranked((r, d), src, ids, 2)
        self.assertEqual(n, 3)
        self.assertEqual(proposals, [(11, 0, 3, 0.5), (12, 0, 2, 0.5)])

    def test_random_destination_is_disjoint_seeded_in_bank_and_has_shortfalls(self):
        rows = [(0, 0, 1, 0.1), (1, 0, 2, 0.2)]
        ids = np.arange(5)
        slots = np.array([99, 98, 0, 1, 2])
        partitions = [(ids[:2], ids[2:]), (ids[2:], ids[:2])]
        expected = ([(0, 0, 2, 0.1), (1, 0, 1, 0.2)], 0)
        self.assertEqual(
            random_destinations(rows, slots, ids, partitions, 11, 1), expected
        )
        self.assertEqual(
            random_destinations(rows, slots, ids, partitions, 11, 1), expected
        )
        actual, short = random_destinations(
            rows, np.array([99, 98, 0, 1, 1]), ids, partitions, 11, 1
        )
        self.assertEqual((actual, short), ([(1, 0, 1, 0.2)], 1))
        with self.assertRaisesRegex(ValueError, "overlap"):
            random_destinations(rows, slots, ids, [(ids, ids)], 11, 1)

    def test_refresh_uses_clip_or_current_dino_with_identical_folds(self):
        ids = np.arange(4, 12)
        data = SimpleNamespace(
            unlabelled_dataset=SimpleNamespace(ids=ids),
            labelled_dataset=SimpleNamespace(uq_idxs=np.arange(4)),
        )
        loader = SimpleNamespace(dataset=data, batch_size=128)
        series = [torch.ones(8, 3) * i for i in (1, 2, 3)]
        slots = np.arange(8) % 4
        clip = torch.ones(8, 4)
        expected = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        models = dict(zip(ARMS, (net() for _ in ARMS), strict=True))

        class Loader:
            dataset = data
            batch_size = loader.batch_size

            def __len__(self):
                return 2

        with (
            patch("dual.ablation.snapshot", side_effect=[(slots, x) for x in series]),
            patch(
                "dual.ablation.scores", return_value=(expected, (slots + 1) % 4)
            ) as score,
        ):
            plans = refresh(
                models,
                Loader(),
                None,
                clip,
                10,
                Settings(coverage=0.5),
                torch.device("cpu"),
                0,
            )
        self.assertIs(score.call_args_list[0].args[0], clip)
        self.assertIs(score.call_args_list[1].args[0], series[-1])
        self.assertIs(score.call_args_list[0].args[3], score.call_args_list[1].args[3])
        self.assertEqual(plans["D1-DINO"].details["candidate_shortfall"], 2)

    def test_matching_never_pads_or_bypasses_a_guard(self):
        cfg = Settings(batch_cap=3)
        plans = {a: Plan([], [], 5, cfg, {}) for a in ARMS}
        eligible = dict(
            zip(
                ARMS,
                (
                    [[0, 10, 1, 2], [2, 12, 1, 2], [3, 13, 1, 2]],
                    [[3, 13, 1, 4], [0, 10, 1, 4]],
                    [[1, 11, 2, 5]],
                ),
                strict=True,
            )
        )
        rows, counts = matched_rows(plans, eligible)
        self.assertEqual([len(rows[a]) for a in ARMS], [1, 1, 1])
        self.assertEqual(rows[ARMS[0]][0][:3], rows[ARMS[1]][0][:3])
        self.assertEqual(counts["dino_shortfall_before_matching"], 2)
        plans[ARMS[-1]].applied = [None] * 5
        rows, counts = matched_rows(plans, eligible)
        self.assertFalse(any(rows.values()))
        self.assertEqual(counts["applied_per_arm"], 0)

    def test_real_updates_match_dose_and_keep_is_bitwise_when_empty(self):
        batch, criterion = fixture(), DistillLoss(30, 200)
        torch.manual_seed(19)
        original = net()
        models = {a: net() for a in ARMS}
        for model in models.values():
            model.load_state_dict(original.state_dict())
        opts = {a: optimizer_for(model) for a, model in models.items()}
        plans = {a: Plan(range(4), [], 8, Settings(), {}) for a in ARMS}
        step(original, optimizer_for(original), batch, criterion, 10, None, 1)
        matched_step(models, opts, batch, criterion, 10, plans, 1, micro=False)
        for model in models.values():
            for key, val in original.state_dict().items():
                self.assertTrue(torch.equal(model.state_dict()[key], val), key)

        src_net = net()
        models = {a: net() for a in ARMS}
        for model in models.values():
            model.load_state_dict(src_net.state_dict())
        opts = {a: optimizer_for(model) for a, model in models.items()}
        with torch.no_grad():
            prob = (
                src_net(torch.cat(batch[0]))[1] / criterion.teacher_temp_schedule[10]
            ).softmax(-1)
        source = prob.reshape(2, 8, -1).mean(0)[4].argmax().item()
        mass = float(prob[[4, 12], source].min()) / 2
        self.assertGreater(mass, 0)
        plans = {
            a: Plan(
                range(4),
                [[4, source, (source + i + 1) % 196, 1.0]],
                4,
                Settings(mass=mass),
                {},
            )
            for i, a in enumerate(ARMS)
        }
        _, counts = matched_step(
            models, opts, batch, criterion, 10, plans, 1, micro=False
        )
        self.assertEqual(counts["applied_per_arm"], 1)
        self.assertEqual(
            [p.applied[0][:4] for p in plans.values()], [[1, 4, 4, source]] * 3
        )

    def test_joint_resume_matches_uninterrupted_model_state(self):
        sample = fixture()

        class Samples:
            def __getitem__(self, i):
                return (
                    [x[i] for x in sample[0]],
                    sample[1][i],
                    sample[2][i],
                    sample[3][i],
                )

            def __len__(self):
                return len(sample[1])

        # No real data is accessed in this resume test. The shared loader still
        # advances torch RNG and shuffle order; the real trainer uses the
        # original weighted sampler returned by relational.train.build().
        data = Samples()
        data.unlabelled_dataset = SimpleNamespace(ids=np.arange(4, 8))
        data.labelled_dataset = SimpleNamespace(uq_idxs=np.arange(4))

        def build_cpu():
            torch.manual_seed(53)
            return (
                net(),
                DataLoader(data, batch_size=8, shuffle=True),
                None,
                None,
                None,
                {"split": "fixture", "relational": {}, "workers": 0},
            )

        @torch.no_grad()
        def active_fixture(
            models, loader, center, clip, epoch, settings, device, workers
        ):
            plans = {}
            temp = DistillLoss(30, 200).teacher_temp_schedule[epoch]
            for offset, (arm, model) in enumerate(models.items(), 1):
                p = (
                    (model(torch.cat(sample[0]))[1] / temp)
                    .softmax(-1)
                    .reshape(2, 8, -1)
                )
                src = int(p[:, 4].mean(0).argmax())
                settings = Settings(mass=0.0001)
                plans[arm] = Plan(
                    range(4), [(4, src, (src + offset) % 196, 1.0)], 1, settings, {}
                )
            return plans

        def evaluate(*_):
            return tuple(torch.rand(3).tolist())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "fixture"
            with (
                patch("dual.ablation.build", side_effect=build_cpu),
                patch(
                    "dual.ablation.cache.load",
                    return_value=(torch.ones(4, 2), {"fixture": True}),
                ),
                patch("dual.ablation.official_test", side_effect=evaluate),
                patch("dual.ablation.refresh", side_effect=active_fixture),
            ):
                run(root / "a", 2, clip)
                run(root / "b", 1, clip)
                with (root / "b/matched.jsonl").open("ab") as stream:
                    stream.write(b'{"epoch":2')
                run(root / "b", 2, clip)
            a, b = (
                torch.load(root / name / "recovery.pt", weights_only=False)
                for name in ("a", "b")
            )
            for arm in ARMS:
                self.assertEqual(a["schedulers"][arm], b["schedulers"][arm])
                for key, val in a["models"][arm].items():
                    self.assertTrue(torch.equal(val, b["models"][arm][key]))
                for key, item in a["optimizers"][arm]["state"].items():
                    self.assertTrue(
                        torch.equal(
                            item["momentum_buffer"],
                            b["optimizers"][arm]["state"][key]["momentum_buffer"],
                        )
                    )
            self.assertTrue(torch.equal(a["rng"][2], b["rng"][2]))
            rows = [
                json.loads(line)
                for line in (root / "b/matched.jsonl").read_bytes().splitlines()
            ]
            self.assertTrue(any(r["arms"][ARMS[0]]["applied"] for r in rows))
            for r in rows:
                self.assertEqual(len({r["arms"][a]["applied"] for a in ARMS}), 1)
            self.assertEqual(
                (root / "a/matched.jsonl").read_bytes(),
                (root / "b/matched.jsonl").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
