"""Bounded CPU protocol tests; no network, dataset, weights or GPU required."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from crish.experiment import atomic_checkpoint, load_checkpoint, restore_rng, rng_state
from crish.global_repair import EpochRepairs
from model import DINOHead, DistillLoss
from planogram_rpc.archive import ArchiveSource, ensure_archive
from planogram_rpc.common import digest, image_path
from planogram_rpc.data import BalancedBatches, training_data
from planogram_rpc.export import export_model
from planogram_rpc.metrics import cluster_accuracy, distances, retrieval, ss_kmeans
from planogram_rpc.prepare import (
    RemoteFileMissing,
    assign_splits,
    crop_product,
    download_file,
    verify,
)
from planogram_rpc.prepare import main as prepare
from planogram_rpc.train import configure, objective, parser


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bulk_download_retry_preserves_partial_and_checks_disk(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("instances_train2019.json", b"{}")
        payload = buffer.getvalue()
        calls = []
        test = self

        class Client:
            def download_file(self, url, destination, handle):
                path = Path(destination)
                calls.append(destination)
                if len(calls) == 1:
                    path.write_bytes(b"partial")
                    raise ConnectionError("interrupted")
                test.assertEqual(path.read_bytes(), b"partial")
                path.write_bytes(payload)

        stubs = {
            name: types.ModuleType(name)
            for name in (
                "kagglehub",
                "kagglehub.clients",
                "kagglehub.handle",
                "kagglehub.http_resolver",
            )
        }
        stubs["kagglehub.clients"].KaggleApiV1Client = Client
        stubs["kagglehub.handle"].parse_dataset_handle = lambda value: value
        stubs["kagglehub.http_resolver"]._build_dataset_download_url_path = (
            lambda handle, path: "rpc-v5"
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, stubs),
            patch("planogram_rpc.archive.time.sleep"),
        ):
            path = Path(folder) / "rpc.zip"
            with patch(
                "planogram_rpc.archive.shutil.disk_usage",
                return_value=types.SimpleNamespace(free=100 * 2**30),
            ):
                self.assertEqual(ensure_archive(path), path)
            self.assertEqual(len(calls), 2)
            self.assertFalse(path.with_suffix(".zip.part").exists())
            with ArchiveSource(path) as archive:
                self.assertEqual(archive.annotation_bytes(), b"{}")
            with patch(
                "planogram_rpc.archive.shutil.disk_usage",
                return_value=types.SimpleNamespace(free=0),
            ):
                with self.assertRaisesRegex(OSError, "30 GB"):
                    ensure_archive(Path(folder) / "another.zip")
            self.assertEqual(len(calls), 2)

    def test_rpc_404_tries_alternate_layout(self):
        error = RuntimeError("not found")
        error.response = types.SimpleNamespace(status_code=404)
        module = types.ModuleType("kagglehub")
        from unittest.mock import Mock

        module.dataset_download = Mock(side_effect=[error, "/cached/product.jpg"])
        with patch.dict(sys.modules, {"kagglehub": module}):
            self.assertEqual(
                download_file("retail_product_checkout/train2019/a.jpg"),
                Path("/cached/product.jpg"),
            )
        self.assertEqual(
            [c.kwargs["path"] for c in module.dataset_download.call_args_list],
            ["retail_product_checkout/train2019/a.jpg", "train2019/a.jpg"],
        )

    def test_only_confirmed_404_becomes_missing_file(self):
        from unittest.mock import Mock

        module = types.ModuleType("kagglehub")
        for status in (401, 403, 404, 429, 503):
            error = RuntimeError(f"HTTP {status}")
            error.response = types.SimpleNamespace(status_code=status)
            module.dataset_download = Mock(side_effect=error)
            with (
                patch.dict(sys.modules, {"kagglehub": module}),
                patch("planogram_rpc.prepare.time.sleep"),
            ):
                if status == 404:
                    with self.assertRaises(RemoteFileMissing):
                        download_file("retail_product_checkout/train2019/a.jpg")
                    self.assertEqual(module.dataset_download.call_count, 2)
                else:
                    with self.assertRaises(RuntimeError):
                        download_file("retail_product_checkout/train2019/a.jpg")
                    self.assertEqual(
                        module.dataset_download.call_count,
                        1 if status in (401, 403) else 3,
                    )

    def test_defaults_preserve_requested_recipe(self):
        args = configure(parser().parse_args([]))
        self.assertEqual(
            (args.epochs, args.batch_size, args.lr, args.crish_mode),
            (200, 128, 0.1, "E3"),
        )
        self.assertEqual(
            (args.crish_warmup, args.crish_every, args.warmup_teacher_temp_epochs),
            (10, 1, 30),
        )

    def test_split_ids_and_hidden_labels(self):
        records = defaultdict(list)
        for c in range(4):
            records[c] = [
                {"source_id": str(i), "path": f"{c}-{i}.png"} for i in range(40)
            ]
        a = assign_splits(records, list(range(4)), 2, 1)
        self.assertEqual(a, assign_splits(records, list(range(4)), 2, 1))
        self.assertEqual(len({r["id"] for r in a}), len(a))
        manifest = {"images": a}
        dataset, _ = training_data("unused", manifest)
        self.assertTrue(all(r[1] == -1 for r in dataset.unlabelled_dataset.rows))
        self.assertTrue(all(0 <= r[1] < 2 for r in dataset.labelled_dataset.rows))
        self.assertEqual(len(dataset), 4 * 28)
        self.assertEqual(len(dataset.labelled_dataset), 28)

    def test_balanced_sampler_epoch_boundary_reproducibility(self):
        sampler = BalancedBatches(28, 84, 16, 1)
        initial = list(sampler)
        sampler.epoch = 7
        batches = list(sampler)
        self.assertNotEqual(initial, batches)
        restored = BalancedBatches(28, 84, 16, 1)
        restored.epoch = 7
        self.assertEqual(batches, list(restored))
        for batch in batches:
            self.assertEqual(sum(i < 28 for i in batch), 8)

    def test_ss_kmeans_known_constraints_without_novel_gold(self):
        torch.manual_seed(3)
        centers = torch.eye(4)
        x = torch.cat([centers[k] + 0.01 * torch.randn(20, 4) for k in range(4)])
        x = nn.functional.normalize(x, dim=1)
        legal = torch.full((80,), -1, dtype=torch.long)
        legal[:5] = 0
        legal[20:25] = 1
        result = ss_kmeans(x, legal, 4, seed=1, n_init=2, max_iter=20)
        self.assertTrue(torch.equal(result["labels"][legal >= 0], legal[legal >= 0]))
        metric = cluster_accuracy(
            np.repeat(np.arange(4), 20), result["labels"].numpy(), 2
        )
        self.assertEqual(metric["all"], 1.0)
        again = ss_kmeans(x, legal, 4, seed=1, n_init=2, max_iter=20)
        torch.testing.assert_close(result["centers"], again["centers"])
        self.assertTrue(torch.isfinite(distances(x, result["centers"])).all())

    def test_retrieval_margin_uses_second_product_not_second_image(self):
        # Both best images belong to SKU 0; comparing image top-2 would reject.
        g = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        metric, predictions = retrieval(
            g, [0, 0, 1], q, [0, 1, 0], known_count=1, threshold=0.35, margin=0.05
        )
        self.assertEqual(predictions[:2].tolist(), [0, 1])
        self.assertAlmostEqual(metric["coverage"], 2 / 3, places=5)

    def test_real_loss_backward_and_resume_next_update(self):
        torch.manual_seed(12)
        args = configure(parser().parse_args(["--device", "cpu", "--fp32"]))
        student = nn.Sequential(
            nn.Linear(8, 8), DINOHead(8, 4, hidden_dim=16, bottleneck_dim=8)
        )
        optimizer = torch.optim.SGD(student.parameters(), 0.03, momentum=0.9)
        criterion = DistillLoss(30, 200)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

        def step(model, opt):
            x = torch.randn(8, 8)
            batch = (
                (x, x + torch.randn_like(x) * 0.1),
                torch.tensor([0, 0, 1, 1, -1, -1, -1, -1]),
                torch.arange(8),
                torch.tensor([[1]] * 4 + [[0]] * 4),
            )
            opt.zero_grad()
            loss = objective(model, batch, criterion, 10, args)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                )
            )
            opt.step()

        step(student, optimizer)
        scheduler.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            atomic_checkpoint(
                {
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": rng_state(),
                },
                path,
            )
            step(student, optimizer)
            expected = copy.deepcopy(student.state_dict())
            saved = load_checkpoint(path)
            student.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            restore_rng(saved["rng"])
            step(student, optimizer)
            for name, tensor in expected.items():
                torch.testing.assert_close(
                    student.state_dict()[name], tensor, rtol=0, atol=0
                )

    def test_torchscript_export_dynamic_batch(self):
        encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 768)
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "embedding.ts"
            export_model(encoder, path)
            loaded = torch.jit.load(str(path))
            out = loaded(torch.randn(7, 3, 224, 224))
            self.assertEqual(tuple(out.shape), (7, 768))
            torch.testing.assert_close(out.norm(dim=1), torch.ones(7))

    def test_deployment_adapter_contract_hash_and_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 768)
            )
            export_model(encoder, root / "embedding.ts")
            manifest = {
                "model_name": "simgcd_dino_vitb16_rpc",
                "model_version": "rpc-fixture",
                "embedding_type": "IMAGE_FEATURE",
                "preprocessing_version": "simgcd-rpc-center-224-v1",
                "vector_dimension": 768,
                "artifact_sha256": digest(root / "embedding.ts"),
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            config = types.SimpleNamespace(
                RECOGNIZER_DEVICE="cpu",
                RECOGNIZER_MODEL_NAME=manifest["model_name"],
                RECOGNIZER_MODEL_VERSION=manifest["model_version"],
                RECOGNIZER_EMBEDDING_TYPE="IMAGE_FEATURE",
                RECOGNIZER_PREPROCESSING_VERSION=manifest["preprocessing_version"],
                RECOGNIZER_VECTOR_DIMENSION=768,
            )
            stubs = {
                "core": types.ModuleType("core"),
                "core.config": types.ModuleType("core.config"),
                "recognizer": types.ModuleType("recognizer"),
                "recognizer.base_recognizer": types.ModuleType(
                    "recognizer.base_recognizer"
                ),
            }
            stubs["core.config"].config = config
            stubs["recognizer.base_recognizer"].BaseRecognizer = object
            spec = importlib.util.spec_from_file_location(
                "rpc_adapter_fixture",
                Path(__file__).resolve().parents[1]
                / "deployment/simgcd_rpc_recognizer.py",
            )
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, stubs):
                spec.loader.exec_module(module)
                recognizer = module.SimGCDRPCRecognizer(root, batch_size=2)
                recognizer.load_model()
                vectors = recognizer.extract_embeddings(
                    [np.zeros((40, 30, 3), dtype=np.uint8)] * 3
                )
                self.assertEqual(np.asarray(vectors).shape, (3, 768))
                self.assertEqual(recognizer.extract_embeddings([]), [])
                with self.assertRaises(ValueError):
                    recognizer.extract_embeddings([np.zeros((4, 4), dtype=np.uint8)])
                config.RECOGNIZER_MODEL_VERSION = "wrong-contract"
                with self.assertRaisesRegex(ValueError, "contract"):
                    recognizer.load_model()
                config.RECOGNIZER_MODEL_VERSION = manifest["model_version"]
                (root / "embedding.ts").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    recognizer.load_model()

    def test_official_layout_prepare_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            (source / "train2019").mkdir()
            annotation = {"images": [], "annotations": [], "categories": []}
            for c in range(4):
                annotation["categories"].append({"id": c + 1, "name": f"sku-{c}"})
                for i in range(41):
                    key = c * 41 + i
                    filename = f"{key}.png"
                    array = np.random.default_rng(key).integers(
                        0, 256, (18, 24, 3), dtype=np.uint8
                    )
                    Image.fromarray(array).save(source / "train2019" / filename)
                    annotation["images"].append({"id": key, "file_name": filename})
                    annotation["annotations"].append(
                        {"image_id": key, "category_id": c + 1, "bbox": [0, 0, 24, 18]}
                    )
            (source / "instances_train2019.json").write_text(json.dumps(annotation))
            root = Path(folder) / "prepared"
            argv = [
                "prepare",
                "--source",
                str(source),
                "--root",
                str(root),
                "--classes",
                "4",
                "--per-class",
                "40",
                "--workers",
                "2",
            ]
            with patch("sys.argv", argv):
                prepare()
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(
                verify(root, manifest), {"train": 112, "val": 24, "test": 24}
            )
            with patch("sys.argv", argv):
                prepare()
            # One remote 404 must be replaced from the seeded reserve without
            # losing the class quota or redownloading already materialized crops.
            remote_root = Path(folder) / "remote-prepared"
            unavailable = manifest["images"][0]["source_file"]
            cached = next(
                r for r in manifest["images"] if r["source_file"] != unavailable
            )
            (remote_root / cached["path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / cached["path"], remote_root / cached["path"])

            def remote_file(name):
                if name.endswith("instances_train2019.json"):
                    return source / "instances_train2019.json"
                filename = Path(name).name
                if filename == unavailable:
                    raise RemoteFileMissing(filename)
                if filename == cached["source_file"]:
                    raise AssertionError("Existing crop was downloaded again")
                return source / "train2019" / filename

            remote_args = [
                "prepare",
                "--download-mode",
                "files",
                "--root",
                str(remote_root),
                "--classes",
                "4",
                "--per-class",
                "40",
                "--workers",
                "2",
            ]
            with (
                patch("sys.argv", remote_args),
                patch("planogram_rpc.prepare.download_file", side_effect=remote_file),
            ):
                prepare()
            remote_manifest = json.loads((remote_root / "manifest.json").read_text())
            self.assertEqual(
                verify(remote_root, remote_manifest),
                {"train": 112, "val": 24, "test": 24},
            )
            self.assertEqual(len(remote_manifest["excluded_downloads"]), 1)
            self.assertFalse(
                any(r["source_file"] == unavailable for r in remote_manifest["images"])
            )
            # Switching to bulk ZIP preserves class/image choices and existing
            # crops, while making ZERO per-image network calls.
            archive_path = Path(folder) / "rpc-v5.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for file in source.rglob("*"):
                    if file.is_file():
                        archive.write(
                            file,
                            "retail_product_checkout/"
                            + file.relative_to(source).as_posix(),
                        )
            zip_root = Path(folder) / "zip-prepared"
            cached_path = zip_root / cached["path"]
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / cached["path"], cached_path)
            before = cached_path.stat().st_mtime_ns
            zip_args = [
                "prepare",
                "--root",
                str(zip_root),
                "--archive",
                str(archive_path),
                "--classes",
                "4",
                "--per-class",
                "40",
                "--workers",
                "2",
            ]
            with (
                patch("sys.argv", zip_args),
                patch(
                    "planogram_rpc.prepare.download_file",
                    side_effect=AssertionError("Unexpected per-file network request"),
                ),
            ):
                prepare()
            self.assertEqual(
                json.loads((zip_root / "manifest.json").read_text()), manifest
            )
            self.assertEqual(cached_path.stat().st_mtime_ns, before)
            self.assertFalse((zip_root / "retail_product_checkout").exists())
            # Exercise the new image-only RPC dataset with the EXISTING global
            # CRISH selector and the real RPC training objective on CPU.
            args = configure(parser().parse_args(["--device", "cpu", "--fp32"]))
            args.model_path = str(Path(folder) / "fixture-last.pt")
            args.crish_cap = args.crish_confirmations = 1
            args.crish_probe_known, args.crish_probe_unknown = 4, 5
            args.crish_evidence, args.crish_scoring, args.crish_boundaries = 32, 16, 4
            args.crish_seconds = 10
            dataset, evaluate_transform = training_data(root, manifest)
            encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8)
            )
            model = nn.Sequential(
                encoder, DINOHead(8, 4, hidden_dim=16, bottleneck_dim=8)
            )
            optimizer = torch.optim.SGD(model.parameters(), 0.03, momentum=0.9)
            criterion = DistillLoss(30, 200)
            repairs = EpochRepairs(args)
            record = repairs.prepare(
                model, optimizer, dataset, evaluate_transform, criterion, 10
            )
            self.assertEqual(record["snapshot_rows"], len(dataset))
            sampler = BalancedBatches(
                len(dataset.labelled_dataset), len(dataset.unlabelled_dataset), 16, 1
            )
            loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
            loss = objective(model, next(iter(loader)), criterion, 10, args, repairs)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            repairs.finish_step(True)
            repairs.close_epoch()
            row = manifest["images"][0]
            Image.new("RGB", (256, 256)).save(root / row["path"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(root, manifest)

    def test_path_escape_and_bad_bbox_rejected(self):
        with self.assertRaises(ValueError):
            image_path("/tmp/rpc", "../secret")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.png"
            Image.new("RGB", (16, 16)).save(path)
            with self.assertRaises(ValueError):
                crop_product(path, [100, 100, 5, 5])

    def test_zip_layout_preference_and_completed_archive_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rpc.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("instances_train2019.json", b"root")
                archive.writestr("train2019/a.jpg", b"root-image")
            self.assertEqual(ensure_archive(path), path)  # No kagglehub/network needed.
            with ArchiveSource(path) as archive:
                self.assertEqual(archive.annotation_bytes(), b"root")
                with archive.open_image("a.jpg") as stream:
                    self.assertEqual(stream.read(), b"root-image")
                with self.assertRaises(ValueError):
                    archive.open_image("../outside.jpg")
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr(
                    "retail_product_checkout/instances_train2019.json", b"nested"
                )
            with ArchiveSource(path) as archive:
                self.assertEqual(archive.annotation_bytes(), b"nested")


if __name__ == "__main__":
    unittest.main()
