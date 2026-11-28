"""Two actual updated Cars batches per pair with the audited R2-100 refresh."""

import gc
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crish.experiment import file_digest, restore_rng
from dual.ablation import PAIRS, matched_step, pair_summary
from dual.ablation_benchmark import verified_snapshot
from dual.policy import Plan, Settings
from dual.train import optimizer_for
from model import DistillLoss
from relational.state import write_json
from relational.train import build


def pilot(pair, receipt):
    arms = PAIRS[pair]
    models, opts, plans = {}, {}, {}
    for arm in arms:
        m, loader, _, _, _, protocol = build()
        if not models:
            saved, digest = verified_snapshot(
                Path("dev_outputs/r2-audit/input/epoch-100.pt"), protocol["split"]
            )
            assert digest == receipt["verified_R2_100_sha256"]
        m.load_state_dict(saved["model"])
        m.train()
        opt = optimizer_for(m)
        opt.load_state_dict(saved["optimizer"])
        models[arm], opts[arm] = m, opt
        r = receipt["arms"][arm]
        plans[arm] = Plan(
            loader.dataset.labelled_dataset.uq_idxs,
            r["proposals"],
            r["budget"],
            Settings(),
            {"candidate_shortfall": r["candidate_shortfall"]},
        )
    restore_rng(saved["rng"])
    stream = iter(
        DataLoader(
            loader.dataset,
            sampler=loader.sampler,
            batch_size=128,
            drop_last=True,
            num_workers=0,
        )
    )
    counts = []
    for i in (1, 2):
        _, c = matched_step(
            models, opts, next(stream), DistillLoss(30, 200), 100, plans, i, micro=True
        )
        counts.append(c)
    return {
        "summary": pair_summary(plans, counts, 256),
        "batches": counts,
        "ledgers": {a: p.applied for a, p in plans.items()},
    }


def main():
    path = Path("dev_outputs/d1-ablation-ops/local/benchmark.json")
    sha = file_digest(path)
    assert sha == "135f9b40c2dc3096560fa589dd1cfca2ef7c0dd5090c9d725129bb417cbcf271"
    receipt = json.loads(path.read_text())
    result = {
        "refresh_receipt_sha256": sha,
        "snapshot": "R2-100",
        "batches_per_pair": 2,
    }
    for pair in PAIRS:
        result[pair] = pilot(pair, receipt)
        print(pair, result[pair]["summary"], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    write_json(Path("reports/D1_PAIR_PILOT.json"), result)


if __name__ == "__main__":
    main()
