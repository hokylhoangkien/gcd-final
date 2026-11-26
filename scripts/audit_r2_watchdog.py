"""Run a bounded offline audit subprocess with a persistent PID and status record."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("status", type=Path)
    p.add_argument("timeout_seconds", type=int)
    p.add_argument("module")
    p.add_argument("args", nargs="*")
    p.add_argument(
        "--after", type=Path, help="Wait for successful completion of another audit job"
    )
    a = p.parse_args()
    start = time.time()
    if a.after:
        a.status.write_text(
            json.dumps({"supervisor": os.getpid(), "waiting_for": str(a.after)})
        )
        while True:
            try:
                previous = json.loads(a.after.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                previous = {}
            if previous.get("exit_code") == 0:
                break
            if "exit_code" in previous or time.time() - start > a.timeout_seconds:
                a.status.write_text(
                    json.dumps({"exit_code": 1, "failed_dependency": str(a.after)})
                )
                sys.exit(1)
            time.sleep(20)
    with subprocess.Popen([sys.executable, "-u", "-m", a.module, *a.args]) as child:
        state = {
            "supervisor": os.getpid(),
            "pid": child.pid,
            "start_unix": start,
            "timeout_seconds": a.timeout_seconds,
        }
        a.status.write_text(json.dumps(state))
        try:
            child.wait(timeout=a.timeout_seconds)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            state["timed_out"] = True
        state.update(exit_code=child.returncode, seconds=time.time() - start)
        a.status.write_text(json.dumps(state))
        sys.exit(child.returncode)
