"""One synthetic GPU batch through the pinned upstream and adapted train functions."""

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import torch
from torch import nn

import train
from crish.runtime import arguments
from model import DINOHead

PIN = "f843f27195701b7c8c5713cfd17499ad67c86247"


def run(fp16=False, epochs=1):
    from types import ModuleType

    reference = ModuleType("reference_simgcd")
    source = subprocess.check_output(
        ["git", "show", PIN + ":train.py"], cwd=Path(__file__).resolve().parent
    )
    exec(compile(source, "upstream/train.py", "exec"), reference.__dict__)  # noqa: S102 - local upstream blob at fixed SHA
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parser.parse_args([])
    args.lr, args.momentum, args.weight_decay = 0.018, 0.9, 0.001
    args.epochs, args.n_views, args.sup_weight, args.memax_weight = epochs, 2, 0.35, 1.0
    args.warmup_teacher_temp_epochs, args.warmup_teacher_temp, args.teacher_temp = (
        0,
        0.07,
        0.04,
    )
    args.crish_warmup, args.crish_cap, args.run_seed = 0, 2, 2
    args.crish_every = 1
    args.train_classes, args.eval_funcs, args.print_freq = [0, 1, 2], ["v2"], 1
    args.logger = logging.getLogger()
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("reference", "E0", "E3", "E1", "E2", "E4"):
            torch.manual_seed(2)
            head = DINOHead(8, 4, hidden_dim=12, bottleneck_dim=6)
            model = nn.Sequential(nn.Linear(8, 8), head).cuda()
            x = torch.randn(48, 8)
            images = (x, x + 0.13 * torch.randn_like(x))
            ids = torch.arange(48)
            mask = ids < 24
            legal = (ids % 3).masked_fill(~mask, -1)
            batch = (images, legal, ids, mask[:, None])
            evaluation = [(x, ids % 4, ids)]
            folder = Path(tmp) / mode
            folder.mkdir()
            args.model_path = str(folder / "model.pt")
            args.fp16 = fp16
            args.crish_mode = mode
            args.crish_dose = (
                Path(tmp) / "E3" / "crish.jsonl" if mode in ("E1", "E2") else None
            )
            (reference.train if mode == "reference" else train.train)(
                model, [batch], None, evaluation, args
            )
            state = torch.load(args.model_path, map_location="cpu")
            if mode == "reference":
                initial = state
            elif mode == "E0":
                for k, v in initial["model"].items():
                    assert torch.equal(v, state["model"][k]), k
                for k, v in initial["optimizer"]["state"].items():
                    assert torch.equal(
                        v["momentum_buffer"],
                        state["optimizer"]["state"][k]["momentum_buffer"],
                    )
                print("E0 and upstream parameters and SGD buffers agree bit-for-bit")
            else:
                records = [
                    json.loads(line)
                    for line in (folder / "crish.jsonl").read_text().splitlines()
                ]
                for rec in records:
                    print(
                        mode,
                        rec["epoch"],
                        rec["counts"],
                        "applied",
                        len(rec["applied"]),
                        rec["stop"],
                        round(rec["elapsed_seconds"] * 1000, 2),
                        "ms",
                    )
                if not fp16 or epochs > 1:
                    assert any(rec["applied"] for rec in records), mode
                if mode in ("E3", "E4"):
                    assert any(rec["accepted"] for rec in records), mode


if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument("--fp16", action="store_true")
    options.add_argument("--epochs", type=int, choices=range(1, 17), default=1)
    args = options.parse_args()
    run(args.fp16, args.epochs)
