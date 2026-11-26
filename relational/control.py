"""Ask a live standalone worker to finish its epoch, checkpoint and exit cleanly."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def request(root, action):
    import fcntl

    folder = root / "R2"
    if not (folder / "standalone.json").is_file():
        raise ValueError("Expected an existing standalone R2 run")
    with (root / "R2.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError(
                "Controller is idle. For a paused run use --continue-at or --stop-at."
            )
        with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False) as stream:
            json.dump({"action": action}, stream)
            tmp = Path(stream.name)
        try:
            os.link(tmp, folder / "operator-request.json")
        finally:
            tmp.unlink()
    print(
        f"{action} requested. Follow the controller log for the committed epoch and verified download commands."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("pause", "stop"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    request(a.root, a.action.upper())
