"""Linux experiment controller. The training worker always uses the 200-epoch clock."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from relational.export import export
from relational.guard import evaluate
from relational.state import MILESTONES, append, checkpoints, write_json


def call(module, args, log):
    with log.open("a", encoding="utf8") as f:
        subprocess.run(
            [sys.executable, "-u", "-m", module, *map(str, args)],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def drive(root, variant, reference_through=None):
    if reference_through is not None and variant != "R1":
        raise ValueError("Only R1 may continue as an explicitly requested R2 reference")
    folder = root / variant
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "training.log"
    log.touch()
    guards = folder / "guards.jsonl"
    guards.touch()
    while True:
        files = checkpoints(folder)
        epoch = int(files[-1].stem.split("-")[1]) if files else 0
        if epoch == 200:
            decision = "COMPLETE"
            break
        if epoch in (100, 150) and variant != "R0":
            record = evaluate(root, variant, epoch)
            decision = record["decision"]
            override = (
                decision == "NO-GO"
                and reference_through is not None
                and reference_through > epoch
            )
            append(guards, dict(record, continue_as_reference=override))
            if decision == "BLOCKED" or (decision == "NO-GO" and not override):
                break
            if reference_through == epoch:
                decision = "REFERENCE TARGET REACHED"
                break
        limit = next(e for e in MILESTONES if e > epoch)
        call("relational.train", [variant, folder, limit], log)

    reference_only = any(
        json.loads(line).get("continue_as_reference")
        for line in guards.read_text().splitlines()
    )
    reason = record["reason"] if decision in ("BLOCKED", "NO-GO") else decision
    status = {
        "variant": variant,
        "epoch": epoch,
        "decision": decision,
        "reason": reason,
        "reference_only": reference_only,
        "exploratory": (variant != "R0"),
        "checkpoint": files[-1].name,
    }
    publish(folder, status)


def publish(folder, status):
    checkpoint = folder / status["checkpoint"]
    epoch = status["epoch"]
    log = folder / "training.log"
    write_json(folder / "status.json", status)
    # Geometry is downstream of the budget decision. A failed diagnostic must not strand the weights.
    try:
        geometry = folder / "geometry.json"
        cached = (
            json.loads(geometry.read_text())
            if geometry.exists() and geometry.stat().st_size
            else {}
        )
        if cached.get("epoch") != epoch or "groups" not in cached:
            call("relational.offline", [checkpoint, geometry], log)
    except subprocess.CalledProcessError as error:
        write_json(
            folder / "geometry.json",
            {"status": "FAILED", "epoch": epoch, "reason": str(error)},
        )
        status["geometry"] = "FAILED; see training.log and rerun sealed diagnostic"
        write_json(folder / "status.json", status)
    bundle = export(folder)
    print(json.dumps(status, indent=2), flush=True)
    print(f"Bundle: {bundle}\nChecksum: {bundle}.sha256", flush=True)
    print(
        f'scp -P "$SSH_PORT" "$SSH_HOST:{bundle}" "$SSH_HOST:{bundle}.sha256" .',
        flush=True,
    )
    print(f"python -m relational.export verify {bundle.name}", flush=True)


def standalone(root, *, auto=False, continue_at=None, stop_at=None):
    from relational.review import milestone

    folder = root / "R2"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "training.log").touch()
    guards = folder / "guards.jsonl"
    guards.touch()
    marker = folder / "standalone.json"
    write_json(marker, {"mode": "standalone", "variant": "R2"})
    state = folder / "status.json"
    request = folder / "operator-request.json"
    while True:
        files = checkpoints(folder)
        epoch = int(files[-1].stem[6:]) if files else 0
        prior = json.loads(state.read_text()) if state.exists() else {}
        last = prior.get("decision") if prior.get("epoch") == epoch else None
        manual = continue_at if continue_at is not None else stop_at
        if manual is not None and (manual != epoch or last != "PAUSED"):
            raise ValueError("Decision must name the currently paused checkpoint epoch")
        if last == "STOPPED":
            publish(folder, prior)
            return
        control = (
            json.loads(request.read_text())["action"] if request.exists() else None
        )
        if control not in (None, "PAUSE", "STOP"):
            raise ValueError("Invalid operator request")
        operator_pause = last == "PAUSED" and prior.get("reason") == "Operator request"
        if epoch and (epoch in MILESTONES or control or last == "PAUSED"):
            report = None
            if epoch in (100, 150):
                report = milestone(root, epoch)
                write_json(folder / f"review-{epoch}.json", report)
                print(json.dumps(report, indent=2), flush=True)
            if epoch == 200:
                decision, reason = "COMPLETE", "Completed 200 epochs"
            elif control or stop_at is not None:
                decision = (
                    "STOPPED" if control == "STOP" or stop_at is not None else "PAUSED"
                )
                reason = "Operator request"
            elif continue_at is not None or last == "CONTINUE":
                decision, reason = "CONTINUE", "Resume authorized for this checkpoint"
            elif (
                auto
                and report
                and not operator_pause
                and report["comparative"]["decision"] not in ("NO-GO", "BLOCKED")
                and all(
                    c["decision"] != "NO-GO"
                    for c in report["comparative"]["comparisons"].values()
                )
            ):
                decision, reason = "CONTINUE", "Explicit automatic continuation"
            else:
                decision, reason = (
                    "PAUSED",
                    (
                        "Operator request"
                        if operator_pause
                        else "Manual milestone review"
                    ),
                )
            status = {
                "variant": "R2",
                "mode": "standalone",
                "exploratory": True,
                "epoch": epoch,
                "decision": decision,
                "reason": reason,
                "checkpoint": files[-1].name,
                "comparative": report["comparative"]["decision"] if report else None,
            }
            append(guards, status)
            publish(folder, status)
            if control:
                request.unlink(missing_ok=True)
            if decision != "CONTINUE":
                return
            continue_at = stop_at = None
        limit = next(e for e in MILESTONES if e > epoch)
        call("relational.train", ["R2", folder, limit], folder / "training.log")


def main():
    import fcntl  # Linux session lock

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", choices=("R0", "R1", "R2"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference-through", type=int, choices=(150, 200))
    p.add_argument("--standalone", action="store_true")
    choice = p.add_mutually_exclusive_group()
    choice.add_argument("--auto-continue", action="store_true")
    choice.add_argument("--continue-at", type=int, metavar="PAUSED_EPOCH")
    choice.add_argument("--stop-at", type=int, metavar="PAUSED_EPOCH")
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{args.variant}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        independent = (
            args.standalone or (root / args.variant / "standalone.json").exists()
        )
        if independent:
            if args.variant != "R2" or args.reference_through is not None:
                p.error("Standalone mode requires R2 without --reference-through")
            standalone(
                root,
                auto=args.auto_continue,
                continue_at=args.continue_at,
                stop_at=args.stop_at,
            )
        elif (
            args.auto_continue
            or args.continue_at is not None
            or args.stop_at is not None
        ):
            p.error(
                "Manual decisions and automatic continuation require standalone mode"
            )
        else:
            drive(root, args.variant, args.reference_through)


if __name__ == "__main__":
    main()
