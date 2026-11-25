"""Report final accuracy and realized exposure for completed Cars runs."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crish.experiment import load_checkpoint


def summarize(work, seed):
    rows = []
    for name in ("E4-adaptive", "E3-fixed", "E1-random", "E2-evidence"):
        folder = work / f"cars-{name}-s{seed}" / "checkpoints"
        state = load_checkpoint(folder / "model.pt")
        if state["epoch"] != 200 or state["identity"]["settings"]["run_seed"] != int(
            seed
        ):
            raise ValueError(f"Incomplete or incompatible checkpoint for {name}")
        accepted, applied = Counter(), Counter()
        mismatched = []
        with (folder / "crish-global.jsonl").open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["event"] == "refresh":
                    accepted.update(record["accepted_counts"])
                elif record["event"] == "exposure":
                    applied.update(record["applied"])
                    if not record.get("plan_matched", True) or not record.get(
                        "exposure_matched", True
                    ):
                        mismatched.append(record["epoch"])
        rows.append(
            {
                "mode": name,
                "seed": int(seed),
                "final_epoch": state["epoch"],
                "accuracy": state["accuracy"],
                "accepted": dict(accepted),
                "applied": dict(applied),
                "unmatched_epochs": mismatched,
            }
        )
        del state
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    print(json.dumps(summarize(args.work, args.seed), indent=2))
