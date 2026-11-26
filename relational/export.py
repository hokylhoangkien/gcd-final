"""Archive only named run artifacts and verify both member and download digests."""

import argparse
import hashlib
import io
import os
import tarfile
from pathlib import Path

from crish.experiment import file_digest
from relational.state import checkpoints

ARTIFACTS = (
    "config.json",
    "environment.json",
    "training.log",
    "metrics.jsonl",
    "relationships.jsonl",
    "guards.jsonl",
    "status.json",
    "geometry.json",
)


def digest_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def export(folder):
    files = [folder / name for name in ARTIFACTS] + checkpoints(folder)
    if not checkpoints(folder):
        raise ValueError("No committed checkpoint to export")
    output = folder.parent / "exports"
    output.mkdir(exist_ok=True)
    bundle = output / f"{folder.name}-{checkpoints(folder)[-1].stem}.tar"
    tmp = bundle.with_suffix(".tar.tmp")
    sums = []
    with tarfile.open(tmp, "w") as tar:
        for path in files:
            sums.append(f"{file_digest(path)}  {path.name}\n")
            tar.add(path, arcname=path.name, recursive=False)
        content = "".join(sums).encode()
        header = tarfile.TarInfo("SHA256SUMS")
        header.size = len(content)
        tar.addfile(header, io.BytesIO(content))
    os.replace(tmp, bundle)
    receipt = bundle.with_suffix(".tar.sha256")
    temp = receipt.with_suffix(".tmp")
    temp.write_text(f"{file_digest(bundle)}  {bundle.name}\n", encoding="ascii")
    os.replace(temp, receipt)
    return bundle


def verify(bundle):
    expected = bundle.with_suffix(".tar.sha256").read_text().split()[0]
    if file_digest(bundle) != expected:
        raise ValueError("Archive checksum mismatch")
    with tarfile.open(bundle, "r:") as tar:
        declared = tar.extractfile("SHA256SUMS").read().decode().splitlines()
        names = []
        for line in declared:
            digest, name = line.split("  ", 1)
            if (
                Path(name).name != name
                or digest_stream(tar.extractfile(name)) != digest
            ):
                raise ValueError(f"Member checksum mismatch: {name}")
            names.append(name)
        if sorted(tar.getnames()) != sorted(names + ["SHA256SUMS"]) or len(
            set(names)
        ) != len(names):
            raise ValueError("Unexpected archive contents")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("export", "verify"))
    p.add_argument("path", type=Path)
    a = p.parse_args()
    if a.action == "export":
        print(export(a.path))
    else:
        verify(a.path)
        print("All checksums match")
