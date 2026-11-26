"""RunPod controller. Each child trains only to the next milestone on the 200-epoch clock."""

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
    write_json(folder / "status.json", status)
    # Geometry is downstream of the budget decision. A failed diagnostic must not strand the weights.
    try:
        call("relational.offline", [files[-1], folder / "geometry.json"], log)
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
        f'scp -P "$POD_PORT" "$POD_HOST:{bundle}" "$POD_HOST:{bundle}.sha256" .',
        flush=True,
    )
    print(f"python -m relational.export verify {bundle.name}", flush=True)


def main():
    import fcntl  # Linux RunPod session lock

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", choices=("R0", "R1", "R2"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference-through", type=int, choices=(150, 200))
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{args.variant}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        drive(root, args.variant, args.reference_through)


if __name__ == "__main__":
    main()
