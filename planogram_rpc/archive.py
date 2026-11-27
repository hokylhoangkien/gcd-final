"""Download RPC once and read selected exemplar members without full extraction."""

from __future__ import annotations

import shutil
import time
import zipfile
from io import BytesIO
from pathlib import Path
from threading import Lock


def ensure_archive(path):
    path = Path(path)
    if not path.exists():
        # Low-level APIs are intentionally tied to kagglehub==0.3.13. The public
        # dataset_download(handle) API extracts the entire bundle and deletes ZIP.
        from kagglehub.clients import KaggleApiV1Client
        from kagglehub.handle import parse_dataset_handle
        from kagglehub.http_resolver import _build_dataset_download_url_path

        handle = parse_dataset_handle(
            "diyer22/retail-product-checkout-dataset/versions/5"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        # Observed v5 ZIP is 27,205,167,166 bytes. Reserve extra space for crops.
        already_downloaded = partial.stat().st_size if partial.exists() else 0
        if shutil.disk_usage(path.parent).free + already_downloaded < 30_000_000_000:
            raise OSError(
                "RPC ZIP preparation needs approximately 30 GB free including remaining ZIP bytes and crops. Check df -h or pass --archive on a larger volume; existing crops are preserved."
            )
        print(
            f"Downloading RPC v5 ZIP once to {partial} (~27 GB); existing product crops are preserved",
            flush=True,
        )
        for attempt in range(3):
            try:
                KaggleApiV1Client().download_file(
                    _build_dataset_download_url_path(handle, None), str(partial), handle
                )
                break
            except Exception as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status is not None and status < 500 and status not in (408, 429):
                    raise
                if attempt == 2:
                    raise
                # Keep .part; the pinned client attempts resume if the server
                # supports Range. Do not silently restart in an infinite loop.
                time.sleep(5 * (attempt + 1))
        with ArchiveSource(partial):
            pass  # Validate ZIP directory and annotation before publishing it.
        partial.replace(path)
    with ArchiveSource(path):
        pass
    return path


class ArchiveSource:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        self.lock = Lock()
        try:
            self.names = set(self.zip.namelist())
            self.annotation_member = self.member("instances_train2019.json")
        except Exception:
            self.zip.close()
            raise

    def member(self, relative):
        for name in ("retail_product_checkout/" + relative, relative):
            if name in self.names:
                return name
        raise FileNotFoundError(f"RPC ZIP is missing {relative}")

    def annotation_bytes(self):
        with self.lock:
            return self.zip.read(self.annotation_member)

    def open_image(self, filename):
        if Path(filename).name != filename:
            raise ValueError("Unsafe product filename")
        # Serialize archive seeks/decompression (including older Python ZIP
        # reader behavior), then let worker threads crop independent buffers.
        with self.lock:
            data = self.zip.read(self.member("train2019/" + filename))
        return BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.zip.close()
