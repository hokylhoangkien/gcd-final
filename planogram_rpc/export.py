"""Export the LAST trained backbone, without classifier/cluster identities."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from crish.experiment import load_checkpoint

from .assets import backbone
from .common import PREPROCESSING, digest, write_json


class UnitEmbedding(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, images):
        return F.normalize(self.encoder(images), dim=-1)


def export_model(encoder, destination):
    model = UnitEmbedding(encoder).cpu().eval()
    sample = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        traced = torch.jit.trace(model, sample, check_trace=True)
        # The deployed recognizer batches product crops, so verify another batch size.
        batch = torch.randn(2, 3, 224, 224)
        torch.testing.assert_close(traced(batch), model(batch), rtol=1e-4, atol=1e-5)
        if not torch.isfinite(traced(batch)).all():
            raise FloatingPointError("Nonfinite exported embeddings")
    temp = destination.with_suffix(".tmp")
    torch.jit.save(traced, str(temp))
    temp.replace(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("runs/rpc40-e3-s1"))
    p.add_argument("--assets", type=Path, default=Path("assets_local"))
    args = p.parse_args()
    checkpoint = args.run / "last.pt"
    saved = load_checkpoint(checkpoint)
    if saved["epoch"] != saved["identity"]["settings"]["epochs"]:
        raise ValueError(
            "Finish/resume training before exporting the deployment artifact"
        )
    encoder = backbone(args.assets)
    encoder.load_state_dict(
        {
            k.removeprefix("0."): v
            for k, v in saved["model"].items()
            if k.startswith("0.")
        },
        strict=True,
    )
    output = args.run / "export"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "embedding.ts"
    export_model(encoder, destination)
    checksum = digest(destination)
    contract = {
        "model_name": "simgcd_dino_vitb16_rpc",
        "model_version": "rpc-" + checksum[:16],
        "embedding_type": "IMAGE_FEATURE",
        "preprocessing_version": PREPROCESSING,
        "vector_dimension": 768,
        "artifact_sha256": checksum,
        "artifact": "embedding.ts",
        "checkpoint_sha256": digest(checkpoint),
        "epoch": saved["epoch"],
        "manifest_sha256": saved["identity"]["manifest_sha256"],
    }
    write_json(output / "manifest.json", contract)
    worker_env = {
        "RECOGNIZER_MODEL_NAME": contract["model_name"],
        "RECOGNIZER_MODEL_VERSION": contract["model_version"],
        "RECOGNIZER_EMBEDDING_TYPE": "IMAGE_FEATURE",
        "RECOGNIZER_PREPROCESSING_VERSION": PREPROCESSING,
        "RECOGNIZER_VECTOR_DIMENSION": 768,
    }
    backend_env = {
        "EMBEDDING_MODEL_NAME": contract["model_name"],
        "EMBEDDING_MODEL_VERSION": contract["model_version"],
        "EMBEDDING_TYPE": "IMAGE_FEATURE",
        "EMBEDDING_PREPROCESSING_VERSION": PREPROCESSING,
        "EMBEDDING_VECTOR_DIMENSION": 768,
    }
    for name, values in (
        ("worker.env.example", worker_env),
        ("backend.env.example", backend_env),
    ):
        (output / name).write_text(
            "\n".join(f"{k}={v}" for k, v in values.items()) + "\n"
        )
    print(
        f"Exported {destination}; 768-D normalized backbone embeddings. Rebuild all product references under the NEW contract."
    )


if __name__ == "__main__":
    main()
