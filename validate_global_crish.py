"""Focused checks for global selection, timeout-prefix retention and live exposure."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from crish.global_repair import Clock, Deadline, EpochRepairs, Proposal
from crish.runtime import arguments, global_defaults
from data.data_utils import MergedDataset
from model import DINOHead, DistillLoss


class Rows(Dataset):
    def __init__(self, x, labels, offset):
        self.x, self.labels, self.offset = x, labels, offset
        self.transform = lambda v: (v, v + 0.03 * torch.randn_like(v))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.labels[i], self.offset + i


def main():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    p = argparse.ArgumentParser()
    arguments(p)
    global_defaults(p)
    args = p.parse_args([])
    args.n_views, args.num_workers, args.crish_warmup = 2, 0, 0
    (
        args.crish_probe_known,
        args.crish_probe_unknown,
        args.crish_cap,
        args.crish_confirmations,
    ) = 4, 5, 3, 3
    args.crish_scoring, args.crish_evidence, args.crish_boundaries = 64, 128, 24
    args.memax_weight, args.sup_weight = 1, 0.35
    x = torch.randn(512, 12)
    ds = MergedDataset(
        Rows(x[:128], torch.arange(128) % 4, 1000),
        Rows(x[128:], torch.full((384,), -1), 1128),
    )
    head = DINOHead(12, 8, hidden_dim=16, bottleneck_dim=8)
    model = nn.Sequential(nn.Linear(12, 12), head)
    opt = torch.optim.SGD(model.parameters(), 0.06, momentum=0.9)
    crit = DistillLoss(0, 12)
    with tempfile.TemporaryDirectory() as tmp:
        args.model_path = str(Path(tmp) / "model.pt")
        repair = EpochRepairs(args)
        before = torch.get_rng_state().clone()
        params = [p.detach().clone() for p in model.parameters()]
        for epoch in range(7):
            info = repair.prepare(model, opt, ds, lambda v: v, crit, epoch)
            assert torch.equal(before, torch.get_rng_state())
            assert all(torch.equal(p, v) for p, v in zip(model.parameters(), params))
            assert info["snapshot_rows"] == 512 and len(info["score_ids"]) == 64
            folds = [i for fold in info["probe_ids"] for i in fold]
            assert len(folds) == len(set(folds)) and not set(folds).intersection(
                info["score_ids"] + info["evidence_ids"]
            )
            repair.close_epoch()
            if info["accepted"]:
                break
        assert info["accepted"], "fixture must reach genuine confirmation"
        check = Clock.check

        def budget(clock):
            if repair.plan:
                raise Deadline
            check(clock)

        with patch.object(Clock, "check", budget):
            info = repair.prepare(model, opt, ds, lambda v: v, crit, epoch)
        assert info["stop"] == "timeout" and len(repair.plan) == 1
        assert info["accepted"][0]["sample_id"] == next(iter(repair.plan))
        repair.close_epoch()
        # Explicit plans here exercise stale-source and skipped-step accounting independently of selection.
        for action in ("move", "abstain"):
            repair.epoch = 2
            repair.plan = {
                101: Proposal(101, action, 0, 1 if action == "move" else None)
            }
            repair.applied.clear()
            repair.applied_ids.clear()
            repair.exposure.clear()
            logits = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]] * 2, requires_grad=True
            )
            ids = torch.tensor([101, 101, 101])
            labeled = torch.zeros(3, dtype=torch.bool)
            loss = repair.loss(logits, ids, labeled, crit, 2)
            assert torch.isfinite(torch.autograd.grad(loss, logits)[0]).all()
            repair.finish_step(False)
            repair.loss(logits, ids, labeled, crit, 2)
            repair.finish_step(True)
            log = repair.close_epoch()
            assert log["applied"] == {action: 2} and log["applied_ids"] == [101]
            assert (
                log["exposure"]["stale_source"] == 2
                and log["exposure"]["amp_skipped_rows"] == 2
            )
            assert not repair.plan
    print(
        "Global folds/IDs, frozen state, natural confirmation, timeout prefix, persistence and stale guard passed."
    )


if __name__ == "__main__":
    main()
