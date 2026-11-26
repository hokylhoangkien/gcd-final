"""Bounded controller integration tests using committed fixture trajectories."""

import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from relational.export import export, verify
from relational.graph import Graph, Settings, evidence
from relational.guard import evaluate
from relational.run import standalone
from relational.state import append, write_json


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.trained = []
        self.ref = self.root / "R2"
        self.ledger = self.graph_record()
        patched = patch("relational.run.call", side_effect=self.worker)
        patched.start()
        self.addCleanup(patched.stop)

    def graph_record(self):
        h = torch.tensor([[1.0, 0.0], [0.8, 0.2], [-0.2, 0.9], [0.0, 1.0]])
        graph = Graph(torch.arange(4), h, h, h, evidence, Settings(k=1))
        _, _, pending = graph.loss(torch.cat([h, h]), torch.arange(4))
        graph.applied(pending)
        record = graph.record()
        self.assertEqual(record["edge_count"], len(record["edges"]))
        return {"graph": record, "applied": graph.exposure_record(4)}

    def fixture(self, name, end, acc=0.5):
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        write_json(folder / "config.json", {"variant": name, "protocol": {"seed": 1}})
        write_json(folder / "environment.json", {})
        metrics = folder / "metrics.jsonl"
        start = len(metrics.read_text().splitlines()) if metrics.exists() else 0
        for e in range(start + 1, end + 1):
            val = acc + (e - 100) * 0.001
            append(
                metrics,
                {
                    "epoch": e,
                    "all": val,
                    "old": val + 0.1,
                    "new": val - 0.1,
                    "gradient_first_batch": {
                        k: {"ratio": 0.02} for k in ("edit_delta", "relational")
                    },
                },
            )
            append(folder / "relationships.jsonl", dict(epoch=e, **self.ledger))
        (folder / f"epoch-{end:03}.pt").write_bytes(b"fixture")

    def worker(self, module, args, log):
        if module == "relational.train":
            self.trained.append(args[-1])
            self.fixture("R2", args[-1])
        else:
            epoch = int(args[0].stem[6:])
            write_json(args[-1], {"epoch": epoch, "groups": {"all": {}}})

    def state(self):
        return json.loads((self.ref / "status.json").read_text())

    def test_manual_guards_and_stop(self):
        standalone(self.root)
        self.assertEqual(self.trained, [100])
        self.assertEqual(self.state()["decision"], "PAUSED")
        report = json.loads((self.ref / "review-100.json").read_text())
        self.assertEqual(report["comparative"]["decision"], "UNVERIFIED")
        self.assertAlmostEqual(report["accuracy"]["old"]["mean_pct"], 59.05)
        self.assertAlmostEqual(report["accuracy"]["new"]["slope_pp_per_epoch"], 0.1)
        self.assertEqual(
            report["gradient_first_batch"]["edit_delta"]["measured_epochs"], 20
        )
        self.assertEqual(report["exposure"]["unique_edited_anchors"], 4)
        self.assertEqual(report["exposure"]["intervention_mass"], 20)
        self.assertEqual(evaluate(self.root, "R2", 100)["decision"], "BLOCKED")
        with self.assertRaises(ValueError):
            standalone(self.root, continue_at=150)
        standalone(self.root, continue_at=100)
        self.assertEqual(self.trained, [100, 150])
        self.assertEqual(self.state()["decision"], "PAUSED")
        standalone(self.root, stop_at=150)
        standalone(self.root, auto=True)
        self.assertEqual(self.trained, [100, 150])
        self.assertEqual(self.state()["decision"], "STOPPED")
        verify(self.root / "exports" / "R2-epoch-150.tar")

    def test_auto_exports_before_next_stage(self):
        standalone(self.root, auto=True)
        self.assertEqual(self.trained, [100, 150, 200])
        self.assertEqual(self.state()["decision"], "COMPLETE")
        for epoch in self.trained:
            verify(self.root / "exports" / f"R2-epoch-{epoch}.tar")

    def test_rolling_export_before_first_review(self):
        self.fixture("R2", 10)
        for name in ("training.log", "guards.jsonl"):
            (self.ref / name).touch()
        write_json(
            self.ref / "standalone.json", {"mode": "standalone", "variant": "R2"}
        )
        first = export(self.ref, recovery=True)
        verify(first)
        self.fixture("R2", 20)
        second = export(self.ref, recovery=True)
        self.assertEqual(first, second)
        verify(second)
        with tarfile.open(second) as bundle:
            self.assertIn("epoch-020.pt", bundle.getnames())
            self.assertIn("standalone.json", bundle.getnames())
            self.assertNotIn("epoch-010.pt", bundle.getnames())

    def test_matched_references_and_no_go(self):
        for name in ("R0", "R1"):
            self.fixture(name, 100, acc=0.52)
        (self.root / "R0" / "epoch-200.pt").touch()
        standalone(self.root, auto=True)
        self.assertEqual(self.trained, [100])
        self.assertEqual(self.state()["decision"], "PAUSED")
        self.assertEqual(self.state()["comparative"], "NO-GO")
        self.assertEqual(evaluate(self.root, "R2", 100)["decision"], "NO-GO")

    def test_operator_pause_off_milestone(self):
        self.fixture("R2", 37)
        write_json(self.ref / "operator-request.json", {"action": "PAUSE"})
        standalone(self.root, auto=True)
        self.assertEqual(self.state()["epoch"], 37)
        self.assertFalse(self.trained)
        standalone(self.root, auto=True)
        self.assertFalse(self.trained)
        standalone(self.root, continue_at=37)
        self.assertEqual(self.trained, [100])

    def test_partial_reference_veto_survives_unverified_grade(self):
        self.fixture("R1", 100, acc=0.52)
        standalone(self.root, auto=True)
        self.assertEqual(self.trained, [100])
        self.assertEqual(self.state()["decision"], "PAUSED")
        report = json.loads((self.ref / "review-100.json").read_text())
        comparative = report["comparative"]
        self.assertEqual(comparative["decision"], "UNVERIFIED")
        self.assertEqual(comparative["comparisons"]["R1"]["decision"], "NO-GO")
        self.assertIn("R0", comparative["unavailable_references"])


class BoundaryReplay(unittest.TestCase):
    def test_extra_checkpoint_replays_exactly(self):
        from contextlib import ExitStack

        from crish.experiment import load_checkpoint
        from relational.train import stage

        def components():
            torch.manual_seed(73)
            return torch.nn.Linear(4, 1), None, None, None, None, {"seed": 1}

        def run_to(folder, stop):
            def update(model, loader, opt, criterion, epoch, graph, device):
                opt.zero_grad()
                model(torch.randn(4, 4)).square().mean().backward()
                opt.step()
                if epoch + 1 == stop:
                    write_json(folder / "operator-request.json", {"action": "PAUSE"})
                return {"steps": 1}

            with ExitStack() as stack:
                for path, value in (
                    ("relational.train.build", components),
                    ("relational.train.train_epoch", update),
                    ("train.test", lambda *a, **kw: (0.5, 0.6, 0.4)),
                    ("torch.cuda.get_device_name", lambda *args: "CPU fixture"),
                    ("torch.cuda.reset_peak_memory_stats", lambda: None),
                    ("torch.cuda.max_memory_allocated", lambda: 0),
                    ("subprocess.check_output", lambda *a, **kw: ""),
                ):
                    stack.enter_context(patch(path, value))
                stage(folder, "R2", 100)
            (folder / "operator-request.json").unlink()

        with tempfile.TemporaryDirectory() as tmp:
            a, b = (Path(tmp) / name for name in ("direct", "resumed"))
            for folder in (a, b):
                folder.mkdir()
                write_json(folder / "standalone.json", {"mode": "standalone"})
            run_to(a, 3)
            run_to(b, 1)
            self.assertTrue((b / "epoch-001.pt").is_file())
            run_to(b, 3)
            left, right = (
                load_checkpoint(folder / "epoch-003.pt") for folder in (a, b)
            )
            self.assertEqual(left["scheduler"], right["scheduler"])
            self.assertEqual(right["scheduler"]["T_max"], 200)
            self.assertEqual(right["scheduler"]["last_epoch"], 3)
            for key, tensor in left["model"].items():
                self.assertTrue(torch.equal(tensor, right["model"][key]))
            for key, state in left["optimizer"]["state"].items():
                self.assertTrue(
                    torch.equal(
                        state["momentum_buffer"],
                        right["optimizer"]["state"][key]["momentum_buffer"],
                    )
                )
            self.assertTrue(torch.equal(left["rng"][2], right["rng"][2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
