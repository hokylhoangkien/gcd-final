"""Small CPU integration fixture for full-run modes and epoch-boundary recovery."""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

import train
from crish.experiment import atomic_checkpoint, load_checkpoint
from crish.runtime import arguments, global_defaults
from data.data_utils import MergedDataset
from model import DINOHead, info_nce_logits
from validate_global_crish import Rows


def run(root, mode, folder, resume=False, interrupt=False, seed=23):
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()
    arguments(parser)
    global_defaults(parser)
    a = parser.parse_args([])
    a.run_seed, a.crish_mode, a.crish_global_suite = seed, mode, True
    a.resume_run, a.fp16, a.epochs, a.num_workers = resume, False, 3, 0
    a.crish_probe_known, a.crish_probe_unknown = 4, 5
    a.crish_warmup, a.crish_cap, a.crish_confirmations = 0, 3 if mode == "E4" else 1, 3
    a.crish_scoring, a.crish_evidence, a.crish_boundaries = 64, 128, 24
    a.lr, a.momentum, a.weight_decay = 0.06, 0.9, 5e-5
    a.memax_weight, a.sup_weight, a.n_views = 1, 0.35, 2
    a.teacher_temp, a.warmup_teacher_temp, a.warmup_teacher_temp_epochs = 0.04, 0.07, 0
    a.logger, a.print_freq = logging.getLogger("suite-fixture"), 100
    folder = root / folder
    folder.mkdir(exist_ok=True)
    a.model_path = str(folder / "model.pt")
    if mode in ("E1", "E2"):
        a.crish_dose = root / "reference" / "crish-global.jsonl"
    x = torch.randn(512, 12)
    ds = MergedDataset(
        Rows(x[:128], torch.arange(128) % 4, 1000),
        Rows(x[128:], torch.full((384,), -1), 1128),
    )
    head = DINOHead(12, 8, hidden_dim=16, bottleneck_dim=8)
    model = nn.Sequential(nn.Linear(12, 12), head)
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    evaluation = SimpleNamespace(dataset=SimpleNamespace(transform=lambda v: v))

    def checkpoint(state, path):
        atomic_checkpoint(state, path)
        if interrupt:
            raise InterruptedError("simulated interruption after first committed epoch")

    with (
        patch.object(torch.Tensor, "cuda", lambda x, **kw: x),
        patch.object(
            train,
            "info_nce_logits",
            lambda features: info_nce_logits(features, device="cpu"),
        ),
        patch.object(train, "test", return_value=(0.0, 0.0, 0.0)),
        patch("crish.experiment.atomic_checkpoint", side_effect=checkpoint),
    ):
        train.train(model, loader, None, evaluation, a)
    records = list(
        map(json.loads, (folder / "crish-global.jsonl").read_text().splitlines())
    )
    return load_checkpoint(a.model_path), records


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference, records = run(root, "E3", "reference")
        assert len(records) == 6 and reference["epoch"] == 3
        assert any(r["accepted"] for r in records[::2])
        for mode in ("E1", "E2", "E4"):
            state, logs = run(root, mode, mode)
            assert state["epoch"] == 3 and len(logs) == 6
            assert [r["probe_ids"] for r in logs[::2]] == [
                r["probe_ids"] for r in records[::2]
            ]
            if mode in ("E1", "E2"):
                for r, exp, ref in zip(logs[::2], logs[1::2], records[1::2]):
                    assert not r["decisions"] and r["confirmations"] == 0
                    assert exp["reference_applied"] == ref["applied"]
                    assert all(
                        v <= ref["applied"].get(k, 0) for k, v in exp["applied"].items()
                    )
                    assert exp["exposure_matched"] == (exp["applied"] == ref["applied"])
                assert any(r["accepted"] for r in logs[::2])
                assert any(r["applied"] for r in logs[1::2])
        try:
            run(root, "E3", "recovery", interrupt=True)
        except InterruptedError:
            pass
        with (root / "recovery" / "crish-global.jsonl").open("ab") as f:
            f.write(b'{"partial"')
        recovered, replay = run(root, "E3", "recovery", resume=True)
        assert len(replay) == 6 and recovered["epoch"] == reference["epoch"]
        assert all(
            torch.equal(v, recovered["model"][k]) for k, v in reference["model"].items()
        )
        assert [r["applied"] for r in replay[1::2]] == [
            r["applied"] for r in records[1::2]
        ]
        try:
            run(root, "E1", "wrong-seed", seed=25)
        except ValueError as exc:
            assert "reference" in str(exc)
        else:
            raise AssertionError("An incompatible reference was accepted")
    print(
        "Four global modes, paired dose, stale-safe accounting, signature and deterministic resume passed."
    )


if __name__ == "__main__":
    main()
