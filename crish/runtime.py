"""Experiment wiring for the official single-GPU trainer."""

import json
import math
import time
from pathlib import Path

import torch

from .repair import corrected_distill, select


def arguments(parser):
    parser.add_argument(
        "--crish_mode", choices=["E0", "E1", "E2", "E3", "E4"], default="E0"
    )
    for name, default in [
        ("warmup", 10),
        ("every", 5),
        ("fixed", 1),
        ("cap", 4),
        ("probe", 4),
        ("boundaries", 16),
        ("k", 5),
        ("seed", 1),
        ("evidence", 512),
        ("trials", 2),
        ("scoring", 256),
        ("probe_known", 32),
        ("probe_unknown", 64),
        ("confirmations", 8),
        ("encode_batch", 16),
    ]:
        parser.add_argument("--crish_" + name, type=int, default=default)
    parser.add_argument("--crish_gain", type=float, default=0.0)
    parser.add_argument("--crish_mass", type=float, default=0.2)
    parser.add_argument("--crish_seconds", type=float, default=120.0)
    parser.add_argument("--crish_dose", type=Path)
    parser.add_argument("--run_seed", type=int, default=None)


def global_defaults(parser):
    parser.set_defaults(crish_every=1, crish_cap=8, crish_seconds=300, crish_boundaries=64)


class Repairs:
    def __init__(self, args):
        self.args = args
        if (
            args.n_views != 2
            or min(
                args.crish_every,
                args.crish_cap,
                args.crish_fixed,
                args.crish_probe,
                args.crish_boundaries,
                args.crish_k,
            )
            < 1
            or args.crish_warmup < 0
            or not 0 < args.crish_mass <= 1
            or not math.isfinite(args.crish_seconds)
            or args.crish_seconds <= 0
            or not math.isfinite(args.crish_gain)
            or args.crish_gain < 0
        ):
            raise ValueError("Invalid two-view CRiSH settings")
        self.path = Path(args.model_path).parent / "crish.jsonl"
        self.signature = {
            k: v
            for k, v in vars(args).items()
            if k
            in (
                "dataset_name",
                "epochs",
                "batch_size",
                "run_seed",
                "sup_weight",
                "memax_weight",
                "grad_from_block",
                "lr",
                "weight_decay",
                "warmup_teacher_temp",
                "warmup_teacher_temp_epochs",
                "teacher_temp",
                "prop_train_labels",
                "use_ssb_splits",
                "transform",
                "fp16",
                "warmup_model_dir",
            )
            or k.startswith("crish_")
            and k not in ("crish_dose", "crish_mode")
        }
        self.quota = None
        if args.crish_dose:
            if args.crish_mode not in ("E1", "E2") or args.run_seed is None:
                raise ValueError("Use matched E1/E2 with an explicit paired run_seed")
            self.quota = {}
            for line in args.crish_dose.read_text().splitlines():
                r = json.loads(line)
                if (
                    r["mode"] != "E3"
                    or r["signature"] != self.signature
                    or r["epoch"] in self.quota
                ):
                    raise ValueError("Dose reference requires a matching E3 run")
                self.quota[r["epoch"]] = [a["kind"] for a in r["applied"]]
            if set(self.quota) != set(
                range(args.crish_warmup, args.epochs, args.crish_every)
            ):
                raise ValueError("Incomplete dose reference")
        elif args.crish_mode == "E1":
            raise ValueError("E1 requires --crish_dose from E3")
        self.path.touch(exist_ok=False)

    def due(self, epoch, batch):
        a = self.args
        return (
            batch == 0
            and epoch >= a.crish_warmup
            and (epoch - a.crish_warmup) % a.crish_every == 0
        )

    def loss(self, features, logits, ids, legal, layer, optimizer, criterion, epoch):
        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=False):
            actions, record = select(
                features,
                logits,
                ids,
                legal,
                layer,
                optimizer,
                criterion,
                epoch,
                self.args,
                None if self.quota is None else self.quota[epoch],
            )
        loss = corrected_distill(
            logits, logits.detach(), criterion, epoch, actions, self.args.crish_mass
        )
        if logits.is_cuda:
            torch.cuda.synchronize(logits.device)
        record.update(
            signature=self.signature, elapsed_seconds=time.perf_counter() - start
        )
        self.pending = record
        return loss

    def finish_step(self, optimizer_stepped):
        record = self.pending
        record["optimizer_stepped"] = bool(optimizer_stepped)
        record["applied"] = record["accepted"] if optimizer_stepped else []
        with self.path.open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        self.pending = None
        if (
            self.quota is not None
            and [a["kind"] for a in record["applied"]] != self.quota[record["epoch"]]
        ):
            raise RuntimeError(
                "AMP skipped a matched update; reject this run as a paired comparison"
            )
