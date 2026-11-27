"""Optional planogram BaseRecognizer adapter for the exported TorchScript backbone.

Copy into backend/ai-worker/recognizer/, then select it in worker.py as described
in PLANOGRAM_GPU.md. No Torch Hub download, training code or classifier is needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from core.config import config
from PIL import Image
from recognizer.base_recognizer import BaseRecognizer
from torchvision import transforms as T


class SimGCDRPCRecognizer(BaseRecognizer):
    def __init__(self, artifact_dir="models/simgcd_rpc", batch_size=16):
        self.artifact_dir = Path(artifact_dir)
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device(config.RECOGNIZER_DEVICE)
        if batch_size < 1:
            raise ValueError("Recognizer batch size must be positive")

    def load_model(self):
        manifest = json.loads((self.artifact_dir / "manifest.json").read_text())
        expected = {
            "model_name": config.RECOGNIZER_MODEL_NAME,
            "model_version": config.RECOGNIZER_MODEL_VERSION,
            "embedding_type": config.RECOGNIZER_EMBEDDING_TYPE,
            "preprocessing_version": config.RECOGNIZER_PREPROCESSING_VERSION,
            "vector_dimension": config.RECOGNIZER_VECTOR_DIMENSION,
        }
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("RPC artifact and configured embedding contract differ")
        if (
            manifest["preprocessing_version"] != "simgcd-rpc-center-224-v1"
            or manifest["vector_dimension"] != 768
        ):
            raise ValueError("Unsupported RPC preprocessing or vector dimension")
        artifact = self.artifact_dir / "embedding.ts"
        h = hashlib.sha256()
        with artifact.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != manifest["artifact_sha256"]:
            raise ValueError("RPC embedding artifact checksum mismatch")
        self.model = torch.jit.load(str(artifact), map_location=self.device).eval()
        self.transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def extract_embeddings(self, crops):
        if self.model is None:
            raise RuntimeError("Load the RPC recognizer first")
        result = []
        for start in range(0, len(crops), self.batch_size):
            images = []
            for crop in crops[start : start + self.batch_size]:
                if (
                    crop.ndim != 3
                    or crop.shape[2] != 3
                    or crop.dtype != np.uint8
                    or not crop.size
                ):
                    raise ValueError("Expected a nonempty RGB uint8 product crop")
                # Match the RPC product crop preprocessing: preserve the complete
                # pack with a white square pad before center crop/normalization.
                from PIL import ImageOps

                image = ImageOps.pad(
                    Image.fromarray(crop),
                    (256, 256),
                    method=Image.Resampling.BICUBIC,
                    color=(255, 255, 255),
                )
                images.append(self.transform(image))
            vectors = self.model(torch.stack(images).to(self.device)).float()
            if (
                vectors.ndim != 2
                or vectors.shape[1] != 768
                or not torch.isfinite(vectors).all()
            ):
                raise ValueError("RPC recognizer returned invalid embeddings")
            if not torch.allclose(
                vectors.norm(dim=1),
                torch.ones(len(vectors), device=self.device),
                atol=1e-4,
            ):
                raise ValueError("RPC embeddings must be L2-normalized")
            result.extend(vectors.cpu().tolist())
        return result
