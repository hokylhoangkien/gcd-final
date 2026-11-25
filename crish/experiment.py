"""Run identity, epoch-boundary recovery, and paired global dose ledgers."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def paired_settings(args):
    common = (
        "dataset_name",
        "epochs",
        "run_seed",
        "batch_size",
        "num_workers",
        "grad_from_block",
        "n_views",
        "transform",
        "prop_train_labels",
        "use_ssb_splits",
        "sup_weight",
        "memax_weight",
        "lr",
        "weight_decay",
        "momentum",
        "warmup_teacher_temp",
        "teacher_temp",
        "warmup_teacher_temp_epochs",
        "fp16",
        "warmup_model_dir",
        "split_digest",
        "backbone_digest",
    )
    return {
        k: v
        for k, v in vars(args).items()
        if k in common
        or (k.startswith("crish_") and k not in ("crish_mode", "crish_dose"))
    }


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_dose(args):
    if args.crish_dose is None or args.run_seed is None or args.crish_cap != 1:
        raise ValueError(
            "Global E1/E2 require a paired seed, cap 1 and a completed E3 ledger"
        )
    schedule = range(args.crish_warmup, args.epochs, args.crish_every)
    reference = {}
    with Path(args.crish_dose).open(encoding="utf-8") as stream:
        for epoch in schedule:
            refresh = json.loads(next(stream, "null"))
            exposure = json.loads(next(stream, "null"))
            if (
                not refresh
                or not exposure
                or refresh.get("event") != "refresh"
                or exposure.get("event") != "exposure"
                or refresh["epoch"] != epoch
                or exposure["epoch"] != epoch
                or refresh.get("mode") != "E3"
                or refresh.get("settings") != paired_settings(args)
            ):
                raise ValueError(
                    f"Incompatible or incomplete fixed E3 reference at epoch {epoch}"
                )
            types = [a["kind"] for a in refresh["accepted"]]
            dose = exposure["applied"]
            if (
                len(types) != exposure["accepted"]
                or len(types) > 1
                or any(k not in ("abstain", "move") for k in types)
                or any(
                    k not in types or type(v) is not int or v < 0
                    for k, v in dose.items()
                )
            ):
                raise ValueError(
                    f"Invalid E3 action or exposure count at epoch {epoch}"
                )
            reference[epoch] = (types, dose)
        if stream.read().strip():
            raise ValueError("Unexpected records after final E3 refresh")
    return reference


def load_checkpoint(path):
    # The suite reads its own trusted training checkpoint, including Python/NumPy RNG state.
    import inspect

    kwargs = (
        {"weights_only": False}
        if "weights_only" in inspect.signature(torch.load).parameters
        else {}
    )
    return torch.load(path, map_location="cpu", **kwargs)


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    py, np_state, cpu, gpu = state
    random.setstate(py)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(gpu)


def atomic_checkpoint(state, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def recover_ledger(path, committed_bytes):
    # Checkpoint owns the prefix. A killed refresh may have left a partial JSON line.
    if path.stat().st_size < committed_bytes:
        raise ValueError("Ledger shorter than committed checkpoint")
    with path.open("r+b") as stream:
        stream.truncate(committed_bytes)
