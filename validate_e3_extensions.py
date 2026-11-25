"""Bounded method regressions, including direct parity with the pinned E3 source."""

import argparse
import functools
import operator
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from crish.confirmation import confirm_ranked, sign_audit
from crish.global_repair import EpochRepairs, Proposal
from crish.runtime import arguments, global_defaults
from data.data_utils import MergedDataset
from model import DINOHead, DistillLoss
from validate_global_crish import Rows


def pinned_e3():
    """Load all method dependencies from the same commit without a second checkout."""
    name = "_pinned_e3"
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    for part in (
        "ids",
        "reliability",
        "repair",
        "sampling",
        "scoring_batch",
        "global_repair",
    ):
        path = f"crish/{part}.py"
        blob = subprocess.check_output(
            ["git", "show", f"a63f29c:{path}"], cwd=Path(__file__).parent
        )
        module = types.ModuleType(f"{name}.{part}")
        sys.modules[module.__name__] = module
        module.__package__ = name
        exec(compile(blob, f"a63f29c/{path}", "exec"), module.__dict__)  # noqa: S102 - trusted pinned repository blob
    return module.EpochRepairs


def fixture(root):
    torch.manual_seed(23)
    parser = argparse.ArgumentParser()
    arguments(parser)
    global_defaults(parser)
    a = parser.parse_args([])
    a.crish_mode, a.n_views, a.num_workers = "E3", 2, 0
    a.crish_cap, a.crish_confirmations = 3, 3
    a.crish_probe_known, a.crish_probe_unknown = 4, 8
    a.crish_scoring, a.crish_evidence, a.crish_boundaries = 64, 128, 24
    a.memax_weight, a.sup_weight = 1, 0.35
    a.model_path = str(root / "model.pt")
    x = torch.randn(512, 12)
    ds = MergedDataset(
        Rows(x[:128], torch.arange(128) % 4, 1000),
        Rows(x[128:], torch.full((384,), -1), 1128),
    )
    head = DINOHead(12, 8, hidden_dim=16, bottleneck_dim=8)
    model = nn.Sequential(nn.Linear(12, 12), head)
    opt = torch.optim.SGD(model.parameters(), lr=0.06, momentum=0.9)
    return a, (model, opt, ds, lambda x: x, DistillLoss(0, 20))


class Extensions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_disabled_matches_a63f29c(self):
        a, inputs = fixture(self.root)
        baseline = pinned_e3()(a)
        sub = self.root / "new"
        sub.mkdir()
        a.model_path = str(sub / "model.pt")
        current = EpochRepairs(a)
        accepted = 0
        for epoch in range(5):
            old = baseline.prepare(*inputs, epoch)
            new = current.prepare(*inputs, epoch)
            for rec in (old, new):
                del rec["runtime_seconds"], rec["stages_seconds"]
            self.assertEqual(old, new)
            accepted += len(new["accepted"])
            labels = torch.arange(512) < 128
            ids = torch.arange(1000, 1512)
            logits = torch.randn(1024, 8, requires_grad=True)
            gradients = []
            for impl in (baseline, current):
                loss = impl.loss(logits, ids, labels, inputs[-1], epoch)
                gradients.append(torch.autograd.grad(loss, logits)[0])
                impl.finish_step(True)
            self.assertTrue(torch.equal(*gradients))
            self.assertEqual(baseline.close_epoch(), current.close_epoch())
        self.assertGreater(accepted, 0)

    def test_rank_veto_fresh_fold_next_best_and_hard_limit(self):
        # 1 fails the known-class guard despite improving total gain; 2 passes.
        opts = [Proposal(i, "abstain", 0) for i in range(1, 16)]
        ranked = [(a, a, None, None, {"gain": 1 / a.sample_id}) for a in opts]
        used, rejected = [], set()
        rec = {"confirmations": 0, "confirmation_attempts": []}

        def get_fold(i):
            used.append(i)
            return (i,)

        kwargs = {
            "fallback": True,
            "limit": 12,
            "rejected": rejected,
            "rec": rec,
            "clock": SimpleNamespace(check=lambda: None, stage=lambda _: nullcontext()),
            "get_probe": get_fold,
            "step": lambda edits: edits[-1].sample_id,
            "accepted": [],
            "current": 0,
            "original": 0,
            "threshold": 0,
            "round_index": 0,
            "audit": lambda *args: {},
        }

        def probe(v, i):
            if not v:
                return (10, 4)
            return (9, 5 if i == 1 else 3)

        winner = confirm_ranked(ranked, probe=probe, **kwargs)
        self.assertEqual(winner[0], opts[1])
        self.assertEqual(used, [1, 2])
        self.assertEqual(rejected, {opts[0]})
        self.assertEqual(rec["confirmation_attempts"][0]["result"], "known_veto")
        self.assertIsNone(
            confirm_ranked(ranked[2:], probe=lambda v, i: (10 + v, 4), **kwargs)
        )
        self.assertEqual(used, list(range(1, 13)))
        self.assertEqual(rec["confirmations"], 12)
        self.assertEqual(rec["stop"], "confirmation_budget")

    def test_fallback_real_probe_and_support(self):
        a, inputs = fixture(self.root)
        a.crish_confirmation_fallback = True
        repair = EpochRepairs(a)
        saw_retry = False
        for epoch in range(12):
            r = repair.prepare(*inputs, epoch)
            folds = r["probe_ids"]
            all_ids = functools.reduce(operator.iadd, folds, [])
            self.assertEqual(len(folds), 13)
            self.assertEqual(len(set(all_ids)), 13 * 12)
            self.assertFalse(
                set(all_ids).intersection(r["score_ids"] + r["evidence_ids"])
            )
            attempts = r["confirmation_attempts"]
            self.assertEqual(
                [v["fold"] for v in attempts], list(range(1, 1 + len(attempts)))
            )
            tested = [(v["id"], v["kind"]) for v in attempts]
            self.assertEqual(len(set(tested)), len(tested))
            self.assertLessEqual(len(attempts), 12)
            for v in attempts:
                if v["result"] == "accepted":
                    self.assertGreater(min(v["marginal"], v["group"]), 0)
                    self.assertLessEqual(
                        max(v["known_delta"], v["group_known_delta"]), 0
                    )
            saw_retry |= any(v["rank"] > 1 for v in attempts)
            repair.close_epoch()
            if saw_retry:
                break
        self.assertTrue(saw_retry)
        a.crish_probe_known = 64
        r = repair.prepare(*inputs, 13)
        self.assertEqual(r["stop"], "insufficient_known_probes")
        self.assertFalse(repair.plan)
        repair.close_epoch()
        a.crish_probe_known, a.crish_probe_unknown = 4, 64
        r = repair.prepare(*inputs, 14)
        self.assertEqual(r["stop"], "insufficient_unlabeled_probes")
        self.assertFalse(repair.plan)

    def test_carry_is_inactive_until_verified_and_origin_expires(self):
        a, inputs = fixture(self.root)
        a.crish_revalidated_persistence = True
        action = Proposal(1130, "move", 999, 1)
        a.resume_carry = [{"action": asdict(action), "origin_epoch": 0}]
        repair = EpochRepairs(a)
        for epoch in range(1, 5):
            rec = repair.prepare(*inputs, epoch)
            if epoch <= 3:
                self.assertEqual(rec["carried"], a.resume_carry)
                self.assertIn(1130, rec["evidence_ids"])
                self.assertNotIn(
                    1130, functools.reduce(operator.iadd, rec["probe_ids"], [])
                )
                self.assertNotIn(1130, rec["carry_geometry_ids"])
                self.assertNotIn(1130, repair.plan)
            else:
                self.assertEqual(rec["expired"], a.resume_carry)
            # Don't add unrelated synthetic proposals to this lifecycle test.
            repair.applied_ids.update(repair.plan)
            repair.close_epoch()
        self.assertNotIn(1130, repair.carry)

    def test_reaccept_keeps_origin_and_only_actual_application_retires(self):
        a, inputs = fixture(self.root)
        a.crish_revalidated_persistence = True
        action = Proposal(1130, "abstain", 0)
        a.resume_carry = [{"action": asdict(action), "origin_epoch": 3}]
        repair = EpochRepairs(a)
        for epoch, stepped in [(4, False), (5, True)]:
            repair.prepare(*inputs, epoch)
            repair.plan = {1130: action}
            logits = torch.tensor([[1.0, 0.0]] * 2, requires_grad=True)
            repair.loss(
                logits, torch.tensor([1130]), torch.tensor([False]), inputs[-1], epoch
            )
            repair.finish_step(stepped)
            info = repair.close_epoch()
            self.assertEqual(info["applied_ids"], [1130] if stepped else [])
            self.assertEqual(info["carry_pending"], [] if stepped else a.resume_carry)

    def test_adjusted_sign_audit(self):
        labels = torch.cat(
            [torch.arange(16).repeat_interleave(2), torch.full((64,), -1)]
        )
        y = labels.clamp_min(0)
        targets = torch.nn.functional.one_hot(y, 16).float().expand(2, -1, -1)
        virtual = SimpleNamespace(logits=lambda x, v: v * targets)
        fold = (None, labels, targets)
        audit = sign_audit(virtual, 0, 1, fold, 12)
        self.assertEqual(audit["positive"], 16)
        self.assertEqual(audit["p_bonferroni"], 12 / 2**16)
        self.assertEqual(sign_audit(virtual, 0, 0, fold, 12)["p_bonferroni"], 1)

    def test_checkpoint_recovers_unapplied_origin_and_subsequent_decisions(self):
        from crish.experiment import load_checkpoint
        from validate_cars_suite import run

        saved, continuous = run(self.root, "E3", "continuous", extensions=True)
        with self.assertRaises(InterruptedError):
            run(self.root, "E3", "restart", extensions=True, interrupt=True)
        checkpoint = load_checkpoint(self.root / "restart" / "model.pt")
        self.assertTrue(checkpoint["carry"])
        new, replay = run(self.root, "E3", "restart", extensions=True, resume=True)
        self.assertEqual(checkpoint["carry"], replay[2]["carried"])
        self.assertEqual(saved["carry"], new["carry"])
        self.assertTrue(
            all(torch.equal(v, new["model"][k]) for k, v in saved["model"].items())
        )
        self.assertEqual(
            [r["accepted"] for r in continuous], [r["accepted"] for r in replay]
        )


if __name__ == "__main__":
    unittest.main()
