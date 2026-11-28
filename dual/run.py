"""Generic Linux D1 stage controller with manual reviews at epochs 100 and 150."""

import argparse
import json
from pathlib import Path
from statistics import mean
from subprocess import CalledProcessError

from relational.export import export, verify
from relational.run import call
from relational.state import MILESTONES, append, checkpoints, write_json

AUDIT = Path("reports/r2_e0_baseline_audit.json")


def read_window(path, epoch):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["epoch"] for r in rows[:epoch]] == list(range(1, epoch + 1))
    return rows[epoch - 20 : epoch]


def review(folder, epoch):
    source = json.loads(AUDIT.read_text())
    rows = read_window(folder / "metrics.jsonl", epoch)
    dose = read_window(folder / "relationships.jsonl", epoch)
    e0 = source["trajectory"]["E0"][epoch - 20 : epoch]
    groups = {}
    for i, name in enumerate(("all", "old", "new")):
        y = [r[name] * 100 for r in rows]
        b = mean(r["accuracy"][i] * 100 for r in e0)
        x = [i - 9.5 for i in range(20)]
        groups[name] = {
            "latest": y[-1],
            "mean": mean(y),
            "e0_mean": b,
            "delta_pp": mean(y) - b,
            "slope_pp_per_epoch": sum(a * v for a, v in zip(x, y, strict=True))
            / sum(a * a for a in x),
        }
    ratios = {}
    for name in ("classifier", "block11"):
        g = [
            r["gradient_first_edited_batch"][name]["ratio"]
            for r in rows
            if r["gradient_first_edited_batch"]
            and r["gradient_first_edited_batch"][name]["ratio"] is not None
        ]
        ratios[name] = {"mean": mean(g) if g else None, "epochs": len(g)}
    return {
        "epoch": epoch,
        "first_epoch": epoch - 19,
        "accuracy": groups,
        "gradient_ratio": ratios,
        "dose": {
            k: sum(r[k] for r in dose) for k in ("rows", "proposed", "applied", "mass")
        },
        "mean_feature_variance": mean(
            [r["feature_variance"] for r in dose if "feature_variance" in r]
        )
        if any("feature_variance" in r for r in dose)
        else None,
        "historical_reference": {
            "E0": "matched protocol, different environment",
            "E3": source["training_summary"]["E3"]["last"],
            "E3-F": source["training_summary"]["E3-F"]["last"],
        },
    }


def publish(folder, status, geometry=True):
    write_json(folder / "status.json", status)
    path = folder / "geometry.json"
    if geometry:
        try:
            call(
                "relational.offline",
                [folder / status["checkpoint"], path],
                folder / "training.log",
            )
        except (
            OSError,
            RuntimeError,
            CalledProcessError,
        ) as error:
            write_json(
                path,
                {"status": "FAILED", "epoch": status["epoch"], "reason": str(error)},
            )
    else:
        write_json(path, {"status": "not_requested", "epoch": status["epoch"]})
    bundle = export(folder)
    verify(bundle)
    print(f"Verified archive {bundle} and checksum {bundle}.sha256", flush=True)


def drive(root, clip, micro, auto, decision):
    folder = root / "D1"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("training.log", "guards.jsonl"):
        (folder / name).touch()
    write_json(folder / "standalone.json", {"variant": "D1", "mode": "standalone"})
    request = folder / "operator-request.json"
    status = folder / "status.json"
    while True:
        files = checkpoints(folder)
        epoch = int(files[-1].stem[6:]) if files else 0
        prior = json.loads(status.read_text()) if status.exists() else {}
        if decision is not None and (
            prior.get("decision") != "PAUSED"
            or prior.get("epoch") != epoch
            or decision[1] != epoch
        ):
            raise ValueError(
                "Manual decision must name the currently paused checkpoint"
            )
        action = json.loads(request.read_text())["action"] if request.exists() else None
        assert action in (None, "PAUSE", "STOP")
        revisiting = prior.get("epoch") == epoch
        if revisiting and prior.get("decision") in ("STOPPED", "COMPLETE"):
            print(prior, flush=True)
            return
        if epoch and (
            epoch in MILESTONES
            or action
            or (revisiting and prior.get("decision") == "PAUSED")
        ):
            if epoch in (100, 150, 200):
                write_json(folder / f"review-{epoch}.json", review(folder, epoch))
            if epoch == 200:
                state = "COMPLETE"
            elif action == "STOP" or (decision and decision[0] == "STOP"):
                state = "STOPPED"
            elif action == "PAUSE":
                state = "PAUSED"
            elif (
                (decision and decision[0] == "CONTINUE")
                or (revisiting and prior.get("decision") == "CONTINUE")
                or auto
                and epoch in (100, 150)
                and not (revisiting and prior.get("reason") == "operator")
            ):
                state = "CONTINUE"
            else:
                state = "PAUSED"
            receipt = {
                "epoch": epoch,
                "variant": "D1",
                "checkpoint": files[-1].name,
                "decision": state,
                "reason": "operator"
                if action or (revisiting and prior.get("reason") == "operator")
                else "milestone",
            }
            append(folder / "guards.jsonl", receipt)
            publish(folder, receipt, geometry=epoch in MILESTONES)
            if action:
                request.unlink(missing_ok=True)
            if state != "CONTINUE":
                return
            decision = None
        args = [folder, next(e for e in MILESTONES if e > epoch), clip]
        if micro:
            args.append("--micro")
        call("dual.train", args, folder / "training.log")


def main():
    import fcntl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--micro", action="store_true")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--auto-continue", action="store_true")
    modes.add_argument("--continue-at", type=int)
    modes.add_argument("--stop-at", type=int)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    decision = (
        ("CONTINUE", a.continue_at)
        if a.continue_at is not None
        else (("STOP", a.stop_at) if a.stop_at is not None else None)
    )
    with (a.root / "D1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        drive(a.root, a.cache, a.micro, a.auto_continue, decision)


if __name__ == "__main__":
    main()
