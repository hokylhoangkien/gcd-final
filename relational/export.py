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


def export(folder, *, recovery=False):
    saved = checkpoints(folder)
    if not saved:
        raise ValueError("No committed checkpoint to export")
    names = ARTIFACTS
    if recovery:
        # A periodic recovery checkpoint can precede the first milestone review.
        names = tuple(
            name for name in names if name not in ("status.json", "geometry.json")
        )
    files = [folder / name for name in names] + (saved[-1:] if recovery else saved)
    for name in (
        "standalone.json",
        "review-100.json",
        "review-150.json",
        "review-200.json",
        "controller.log",
    ):
        if (folder / name).is_file():
            files.append(folder / name)
    output = folder.parent / "exports"
    output.mkdir(exist_ok=True)
    label = "recovery" if recovery else saved[-1].stem
    bundle = output / f"{folder.name}-{label}.tar"
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
