"""Audit the archived SimGCD E0 and replay locked frozen Cars diagnostics."""

import argparse
import ast
import copy
import hashlib
import itertools
import json
import re
import subprocess
import tarfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts import audit_dynamic_move_training as dynamic
from scripts import audit_frozen_readout as readout
from scripts import audit_g1 as g1
from scripts import audit_g4 as g4
from scripts import audit_g6 as g6
from scripts import audit_g6_confirm as heldout
from scripts import audit_oracle_move as audit
from scripts.audit_e3_proposals import align, json_native
from scripts.audit_g7 import pack
from util.cluster_and_log_utils import split_cluster_acc_v2

LOCAL = g1.WORK / "e0-baseline"
LOG = Path("../log")
REPORT = Path("reports/CRISH_E0_BASELINE_AUDIT.md")
OUT = Path("reports/r2_e0_baseline_audit.json")
SUMMARY = Path("reports/CRISH_RESEARCH_SUMMARY.md")
MATRIX = Path("reports/CRISH_EXPERIMENT_MATRIX.md")
E0_COMMIT = "aa7eec0c210f0ae0cc69d4f24d30eed3d3ff944d"
E3_COMMIT = "a63f29caa2be4355f6f977b1394a46ca2b4db752"
TAGS = ("E0-200", "R2-100", "R2-150")
GROUPS = ("all", "old", "new")
METRIC = re.compile(
    r"Train Accuracies: All (0\.\d+|1\.0000) \| Old (0\.\d+|1\.0000) \| New (0\.\d+|1\.0000)"
)
POLICIES = {"DINO": "DINO", "CLIP": "CLIP", "Confidence": "CLIP"}
SPLIT_FILES = (
    "data/stanford_cars.py",
    "data/get_datasets.py",
    "data/data_utils.py",
    "data/ssb_splits/scars_osr_splits.pkl",
    "data/augmentations/__init__.py",
    "model.py",
    "util/cluster_and_log_utils.py",
)


def require_git(commit):
    actual = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], text=True
    ).strip()
    assert actual == commit


def archive_receipt():
    folder = LOG / "E0"
    (bundle,) = list(folder.glob("*.tar.gz"))
    tree = folder / bundle.name.removesuffix(".tar.gz")
    actual = audit.file_digest(bundle)
    sidecars = [
        p
        for p in (folder / f"{bundle.name}.sha256", tree / "SHA256SUMS")
        if p.is_file()
    ]
    external = None
    if sidecars and sidecars[0].name.endswith(".sha256"):
        external = sidecars[0].read_text().split()[0].lower()
        assert actual == external
    local = {}
    for path in tree.rglob("*"):
        if path.is_file():
            assert path.name not in local
            local[path.name] = path
    members = {}
    with tarfile.open(bundle, "r:gz") as tar:
        for item in tar:
            if item.isdir():
                continue
            assert item.isfile() and Path(item.name).name not in members
            name = Path(item.name).name
            path = local[name]
            assert item.size == path.stat().st_size
            stream = tar.extractfile(item)
            with stream:
                value = hashlib.file_digest(stream, "sha256").hexdigest()
            assert value == audit.file_digest(path), name
            members[name] = value
    assert members.keys() == local.keys() and len(members) == 11
    assert local["exit_code.txt"].read_text().strip() == "0"
    assert local["commit.txt"].read_text().strip() == E0_COMMIT
    require_git(E0_COMMIT)
    subprocess.run(
        ["git", "diff", "--quiet", E0_COMMIT, "HEAD", "--", *SPLIT_FILES], check=True
    )
    patch = local["config.patch"].read_text()
    assert (
        patch.count("diff --git") == 1 and "diff --git a/config.py b/config.py" in patch
    )
    changed = [
        line[1:]
        for line in patch.splitlines()
        if line[:1] in ("+", "-") and not line.startswith(("+++", "---"))
    ]
    assert len(changed) == 14 and all(
        re.fullmatch(r"\w+_root\s*=\s*'[^']+'|herbarium_dataroot\s*=\s*'[^']+'", line)
        for line in changed
    )
    cfg = local["environment.txt"].read_text().splitlines()
    versions = dict(line.split("==", 1) for line in cfg if "==" in line)
    assert all(
        k in versions
        for k in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "pillow")
    )
    gpu = local["gpu.txt"].read_text()
    start = datetime.fromisoformat(local["start.txt"].read_text().strip())
    end = datetime.fromisoformat(local["end.txt"].read_text().strip())
    assert start < end
    receipt = {
        "archive": bundle.name,
        "archive_sha256": actual,
        "external_archive_digest_match": external is not None,
        "members_sha256": members,
        "member_count": len(members),
        "commit": E0_COMMIT,
        "config_changes": "dataset root paths only",
        "package_versions": versions,
        "gpu": re.search(r"NVIDIA GeForce RTX \d+", gpu).group(0),
        "driver": re.search(r"(?:KMD|Driver) Version:\s+([\d.]+)", gpu).group(1),
        "run_start": start.isoformat(),
        "run_end": end.isoformat(),
        "wall_seconds": (end - start).total_seconds(),
    }
    return local, receipt


def argument(log, k):
    hit = re.search(r"(?:\(|, )" + k + r"=(\[[^\]]*\]|'[^']*'|[^,)]*)", log)
    if hit is None:
        raise ValueError(f"E0 invocation missing {k}")
    return ast.literal_eval(hit.group(1))


def records(log, n):
    matches = list(METRIC.finditer(log))
    losses = list(re.finditer(r"Train Epoch: (\d+) Avg Loss: ([\d.]+)", log))
    assert len(matches) == len(losses) == n
    out = []
    for i, (score, loss) in enumerate(zip(matches, losses, strict=True), 1):
        assert int(loss[1]) == i - 1 and loss.end() < score.start()
        if i < n:
            assert score.end() < losses[i].start()
        out.append(
            {
                "epoch": i,
                "accuracy": [float(x) for x in score.groups()],
                "loss": float(loss[2]),
            }
        )
    return out


def trajectory(files):
    log = files["run.log"].read_text(encoding="utf8")
    log_train = files["log.txt"].read_text(encoding="utf8")
    e0 = records(log, 200)
    assert e0 == records(log_train, 200)
    assert [x["accuracy"] for x in e0] == [
        [float(v) for v in row]
        for row in METRIC.findall(files["metrics.txt"].read_text())
    ]
    assert len(files["metrics.txt"].read_text().splitlines()) == 200
    stamps = [
        datetime.fromisoformat(
            re.sub(r"\s+([+-]\d\d:\d\d)$", r"\1", line.split(" | ", 1)[0].strip())
        )
        for line in files["metrics.txt"].read_text().splitlines()
    ]
    assert all(a < b for a, b in itertools.pairwise(stamps))
    steps = re.findall(r"Epoch: \[(\d+)\]\[(\d+)/63\]", log)
    assert [(int(x), int(z)) for x, z in steps] == [
        (i, j) for i in range(200) for j in range(0, 63, 10)
    ]
    cfg = {
        "batch_size": 128,
        "num_workers": 8,
        "eval_funcs": ["v2"],
        "dataset_name": "scars",
        "warmup_model_dir": None,
        "prop_train_labels": 0.5,
        "use_ssb_splits": True,
        "grad_from_block": 11,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-5,
        "epochs": 200,
        "transform": "imagenet",
        "sup_weight": 0.35,
        "memax_weight": 1.0,
        "n_views": 2,
        "warmup_teacher_temp": 0.07,
        "teacher_temp": 0.04,
        "warmup_teacher_temp_epochs": 30,
        "fp16": False,
        "crish_mode": "E0",
        "run_seed": 1,
        "image_size": 224,
    }
    assert all(argument(log, k) == v for k, v in cfg.items())
    cfg["train_classes"] = argument(log, "train_classes")
    cfg["unlabeled_classes"] = argument(log, "unlabeled_classes")
    manifest = (LOG / "E3_MANIFEST.txt").read_text()
    assert (
        f"Commit: {E3_COMMIT}" in manifest
        and "Seed: 1" in manifest
        and "Epochs: 200" in manifest
    )
    require_git(E3_COMMIT)
    (e3_log,) = list(LOG.glob("e3-cars-*.log"))
    e3 = records(e3_log.read_text(encoding="utf8"), 200)
    r2 = json.loads(Path("reports/r2_audit_history.json").read_text())["epochs"]
    assert [x["epoch"] for x in r2] == list(range(1, 151))
    for row, x in zip(r2, e3[:150], strict=True):
        assert row["e3"] == x["accuracy"]
    e3f = Path("reports/E3F_RECOVERED_RESULTS.md").read_text()
    endpoint = re.search(
        r"Final All / Old / New \| ([\d.]+) / ([\d.]+) / ([\d.]+)", e3f
    )
    series = {
        "E0": e0,
        "E3": e3,
        "R2": [
            {
                "epoch": x["epoch"],
                "accuracy": [x[k] for k in ("all", "old", "new")],
                "loss": x["baseline_loss"],
            }
            for x in r2
        ],
        "E3-F": [
            {
                "epoch": 200,
                "accuracy": [float(v) / 100 for v in endpoint.groups()],
                "loss": None,
            }
        ],
    }
    return cfg, series


def context(files, cfg, receipt):
    model, loader, _, center, _, proto = audit.build()
    from types import SimpleNamespace

    from data.get_datasets import get_class_splits

    config = get_class_splits(
        SimpleNamespace(dataset_name="scars", use_ssb_splits=True)
    )
    assert list(config.train_classes) == cfg["train_classes"]
    assert list(config.unlabeled_classes) == cfg["unlabeled_classes"]
    ds = loader.dataset
    known = ds.labelled_dataset
    blind = ds.unlabelled_dataset
    ids = np.load(g1.WORK / "ids.npy")
    np.testing.assert_array_equal(blind.ids, ids)
    assert len(ids) == 6144 and len(known) == 2000
    r2 = json.loads(Path("reports/r2_audit_provenance.json").read_text())
    assert proto["split"] == r2["config"]["protocol"]["split"]
    keys = (
        "batch",
        "epochs",
        "sampler",
        "seed",
        "split",
        "backbone",
        "teacher",
        "dino_sha256",
        "dino_source",
    )
    receipt["local_split_sha256"] = proto["split"]
    receipt["local_train_ids_sha256"] = readout.digest(np.r_[known.uq_idxs, ids])
    receipt["local_unlabeled_ids_sha256"] = readout.digest(ids)
    receipt["local_protocol"] = {k: proto[k] for k in keys if k in proto}
    receipt["runtime"] = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    checks = {path: sha for sha, path in (line.split() for line in r2["members"])}
    start = torch.load(
        files["model.pt"], map_location="cpu", weights_only=True, mmap=True
    )
    assert start["epoch"] == 200 and set(start) == {"model", "optimizer", "epoch"}
    assert start["optimizer"]["state"]
    assert all(group["momentum"] == 0.9 for group in start["optimizer"]["param_groups"])
    assert sorted({g["weight_decay"] for g in start["optimizer"]["param_groups"]}) == [
        0.0,
        5e-5,
    ]
    assert all(
        abs(g["lr"] - 0.0001) < 1e-10 for g in start["optimizer"]["param_groups"]
    )
    receipt["checkpoint"] = {
        "epoch": start["epoch"],
        "keys": sorted(start),
        "sgd_lr": [g["lr"] for g in start["optimizer"]["param_groups"]],
        "optimizer_entries": len(start["optimizer"]["state"]),
    }
    states = {TAGS[0]: start["model"]}
    receipt["r2_checkpoint_sha256"] = {}
    for tag, e in zip(TAGS[1:], (100, 150), strict=True):
        file = g1.WORK / f"input/epoch-{e}.pt"
        digest = audit.file_digest(file)
        assert digest == checks[file.name]
        state = torch.load(file, map_location="cpu", weights_only=False, mmap=True)
        assert state["epoch"] == e
        states[tag] = state["model"]
        receipt["r2_checkpoint_sha256"][tag] = digest
    for state in states.values():
        assert set(state) == set(model.state_dict())
        for k, v in model[0].state_dict().items():
            if not k.startswith("blocks.11."):
                assert torch.equal(v.cpu(), state[f"0.{k}"]), k
    receipt["verified_common_frozen_prefix"] = True
    return model, ds, center, ids, states


@torch.inference_mode()
def extract(model, ds, center, ids, states, receipt, smoke):
    model.eval().requires_grad_(False)
    panel_record = json.loads(
        Path("reports/r2_oracle_move_trajectory.json").read_text()
    )
    panel = np.asarray(panel_record["panel_ids"])
    assert len(panel) == 588 and set(panel) <= set(ids)
    lookup = {int(k): i for i, k in enumerate(ids)}
    tails, weights = {}, {}
    for tag, state in states.items():
        tail = copy.deepcopy(model[0].blocks[-1])
        tail.load_state_dict({k: state[f"0.blocks.11.{k}"] for k in tail.state_dict()})
        tails[tag] = tail
        v, g = (state[f"1.last_layer.weight_{k}"].cuda() for k in ("v", "g"))
        weights[tag] = F.normalize(v, dim=1) * g

    def feature(tag, tokens):
        return F.normalize(
            torch.cat([model[0].norm(tails[tag](t.cuda()))[:, 0] for t in tokens]),
            dim=1,
        )

    def use(prefix, indices, dest, tag):
        offset = 0
        for t in prefix:
            end = offset + len(t)
            dest[indices[offset:end]] = feature(tag, [t])
            offset = end
        assert offset == len(indices)

    e0 = torch.empty((len(ids), 768), device="cuda")
    panel_cache = torch.load(
        g1.WORK / "oracle-move/step0-panel-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert (
        panel_cache["panel_ids"] == panel.tolist()
        and panel_cache["checkpoint_sha256"] == panel_record["checkpoint_sha256"]
    )
    slot_rows = np.asarray([lookup[k] for k in panel])
    use(
        (piece for batch in panel_cache["tokens"] for piece in batch),
        slot_rows,
        e0,
        TAGS[0],
    )
    reference = dynamic.LOCAL / "reference-prefix.npy"
    metadata = dynamic.LOCAL / "reference-prefix.json"
    if reference.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        proof = json.loads(dynamic.OUT.read_text())["provenance"]
        assert (
            info["identity"]["prefix_checkpoint_sha256"]
            == panel_record["checkpoint_sha256"]
        )
        assert info["sha256"] == proof["reference_prefix_sha256"]
        assert audit.file_digest(reference) == info["sha256"]
        ref = np.asarray([lookup[k] for k in info["identity"]["ids"]])
        assert set(ref) == set(range(len(ids))) - set(slot_rows)
        bank = np.load(reference, mmap_mode="r")
        assert (
            list(bank.shape) == info["identity"]["shape"] and bank.dtype == np.float32
        )
        limit = 8 if smoke else len(ref)
        use(
            (
                torch.from_numpy(bank[0, i : min(limit, i + 8)].copy())
                for i in range(0, limit, 8)
            ),
            ref[:limit],
            e0,
            TAGS[0],
        )
        receipt["reuse"] = {
            "unlabeled_prefix_rows": len(ids),
            "reference_prefix_sha256": info["sha256"],
        }
    else:
        blind = copy.copy(ds.unlabelled_dataset)
        blind.transform = center
        ref = np.setdiff1d(np.arange(len(ids)), slot_rows)
        if smoke:
            ref = ref[:8]
        offset = 0
        from torch.utils.data import Subset

        for images, legal, keys in DataLoader(
            Subset(blind, ref.tolist()), batch_size=16
        ):
            rows = ref[offset : offset + len(images)]
            np.testing.assert_array_equal(ids[rows], keys.numpy())
            assert (legal == -1).all()
            use(audit.prefix(model[0], images), rows, e0, TAGS[0])
            offset += len(images)
        assert len(ref) == offset
        receipt["reuse"] = {
            "unlabeled_prefix_rows": len(panel),
            "encoded_nonpanel_rows": len(ref),
        }
    if smoke:
        assert torch.isfinite(e0[slot_rows]).all()
        print("Real E0 center features from verified frozen-prefix cache", flush=True)
        return None

    hist = json.loads(g6.OUT.read_text())
    p = g1.WORK / "features.npy"
    assert audit.file_digest(p) == hist["dinov1_cache_sha256"]
    saved = np.load(p, mmap_mode="r")
    assert saved.shape == (3, 3, len(ids), 768)
    x = {
        TAGS[0]: e0,
        **{
            tag: F.normalize(torch.from_numpy(saved[i, 0].copy()).cuda(), dim=1)
            for i, tag in enumerate(TAGS[1:], 1)
        },
    }
    labeled = copy.copy(ds.labelled_dataset)
    labeled.transform = center
    features, legal, keys = {a: [] for a in TAGS}, [], []
    for images, y, keys_batch in DataLoader(labeled, batch_size=16):
        assert (y >= 0).all()
        tokens = audit.prefix(model[0], images)
        for a in TAGS:
            features[a].append(feature(a, tokens).cpu())
        legal.extend(y.tolist())
        keys.extend(keys_batch.tolist())
    np.testing.assert_array_equal(keys, labeled.uq_idxs)
    assert not np.intersect1d(keys, ids).size
    logits = {a: (x[a] @ weights[a].T).cpu().numpy() for a in TAGS}
    for a, e in zip(TAGS[1:], (100, 150), strict=True):
        np.testing.assert_array_equal(
            logits[a].argmax(1),
            g4.snapshot(e, json.loads(g6.G1_REPORT.read_text()))["slots"],
        )
    rest = np.flatnonzero(~np.isin(ids, panel))
    fit = np.r_[keys, ids[rest]]
    lock_x = {
        a: torch.cat([torch.cat(features[a]).cuda(), x[a][rest], x[a][slot_rows]])
        for a in TAGS
    }
    classifiers = {a: (lock_x[a] @ weights[a].T).argmax(1).cpu().numpy() for a in TAGS}
    labels = np.r_[legal, np.full(len(rest), -1)]
    print("E0, R2-100 and R2-150 feature banks verified", flush=True)
    return x, logits, lock_x, classifiers, labels, fit, panel


def confirmed_manifest(ids):
    report = json.loads(heldout.OUT.read_text())
    assert tuple(report["coverage_percent"]) == heldout.COVERAGES
    assert report["primary"]["coverage"] == 10
    assert report["seeds"]["random"] == heldout.RANDOM_SEED
    assert audit.file_digest(g6.G1_REPORT) == report["g1_sha256"]
    records = {}
    for e in (100, 150):
        path = heldout.LOCAL / f"frozen-{e}.npz"
        assert audit.file_digest(path) == report["snapshots"][str(e)]["frozen_sha256"]
        with np.load(path, allow_pickle=False) as obj:
            np.testing.assert_array_equal(ids, obj["ids"])
            records[e] = {k: obj[k] for k in ("q", "bank", "pairs")}
    latest = records[150]
    for k in ("q", "bank"):
        np.testing.assert_array_equal(latest[k], records[100][k])
    np.testing.assert_array_equal(
        latest["pairs"][:, :256], records[100]["pairs"][:, :256]
    )
    q, bank = latest["q"], latest["bank"]
    assert (
        len(np.unique(np.r_[q, bank])) == len(ids)
        and len(q) == 1024
        and len(bank) == 5120
    )
    assert report["query_ids"] == ids[q].tolist()
    clip = heldout.checked_cache(
        g6.LOCAL / "clip-image.npz", report["feature_sha256"]["CLIP"], ids
    )
    return q, bank, latest["pairs"], clip, report


@torch.inference_mode()
def policies(ids, q, bank, common_pairs, di, clip, logits, random_report):
    slots = logits.argmax(1)
    inv = np.full(len(ids), -1)
    inv[bank] = np.arange(len(bank))
    pairs = inv[common_pairs]
    assert (pairs >= 0).all()
    out = {"ids": ids, "q": q, "bank": bank, "slots": slots, "pairs": common_pairs}
    for name, x in (("DINO", di), ("CLIP", clip)):
        screen = pack(x[q] @ x[bank].T, q, bank, slots, pairs)
        out.update({f"{name}_{k}": screen[k] for k in g4.FIELDS})
    for k in g4.FIELDS:
        out[f"Confidence_{k}"] = out[f"CLIP_{k}"]
    p = match_prob(logits[q], 0.04)
    out["Confidence_wrong"] = 1 - p.max(1)
    g4.freeze_flags(out, POLICIES)
    rng = np.random.default_rng(heldout.RANDOM_SEED)
    repeats = random_report["random_draws"]
    random_order = np.stack([rng.permutation(len(q)) for _ in range(repeats)])
    for name, retriever in POLICIES.items():
        dest = out[f"{retriever}_dest"][:, 0]
        eligible = (dest >= 0) & (dest != slots[q])
        order = random_order[eligible[random_order]].reshape(
            repeats, int(eligible.sum())
        )
        for cover in heldout.COVERAGES:
            k = int(eligible[out[f"{name}_flag_{cover}"]].sum())
            out[f"{name}_random_{cover}"] = order[:, :k]
    return out


def match_prob(logits, temp):
    z = logits / temp
    probs = np.exp(z - z.max(1, keepdims=True))
    return probs / probs.sum(1, keepdims=True)


@torch.inference_mode()
def freeze(ids, data, receipt):
    x, logits, fits, classifiers, legal, fit, panel = data
    q, bank, pairs, clip, ref = confirmed_manifest(ids)
    all_outputs = {}
    for tag in TAGS:
        out = policies(ids, q, bank, pairs, x[tag], clip, logits[tag], ref)
        out["feature"] = x[tag].cpu().numpy()
        out["logits"] = logits[tag]
        out["fit_feature"] = fits[tag].cpu().numpy()
        out["fit_ids"], out["fit_legal"], out["panel"] = fit, legal, panel
        out["classifier"] = classifiers[tag][None]
        for name in ("spherical", "semi"):
            out[name] = np.stack(
                [
                    readout.spherical(fits[tag], len(fit), legal, s, name == "semi")
                    for s in readout.SEEDS
                ]
            )
        path = LOCAL / f"frozen-{tag}.npz"
        np.savez(path, **out)
        all_outputs[tag] = audit.file_digest(path)
        print("Frozen policies and legal readouts", tag, flush=True)
    receipt["frozen_sha256"] = all_outputs
    receipt["benchmark"] = {
        "reference_sha256": audit.file_digest(heldout.OUT),
        "query_ids_sha256": readout.digest(ids[q]),
        "bank_ids_sha256": readout.digest(ids[bank]),
        "uniform_plus_R150_hard_pairs_sha256": readout.digest(pairs),
        "clip_sha256": ref["feature_sha256"]["CLIP"],
        "query_n": len(q),
        "bank_n": len(bank),
        "coverage_percent": ref["coverage_percent"],
        "readout_panel_n": len(panel),
        "readout_fit_n": len(fit),
        "readout_seeds": readout.SEEDS,
    }
    audit.write_json(LOCAL / "frozen-provenance.json", json_native(receipt))


def bias(p, truth, aligned):
    freq = np.bincount(p["slots"], minlength=196)
    prob = freq[freq > 0] / freq.sum()
    mapping = np.array([aligned.get(i, -1) for i in range(196)])[p["slots"]]
    confidence = match_prob(p["logits"], 0.04).max(1)
    groups = {}
    for g, mask in (("all", truth >= 0), ("old", truth < 98), ("new", truth >= 98)):
        x = mapping[mask]
        groups[g] = {
            "n": len(x),
            "mapped_old_rate": float(((x >= 0) & (x < 98)).mean()),
            "mapped_new_rate": float((x >= 98).mean()),
            "unmatched_rate": float((x < 0).mean()),
            "mean_teacher_confidence": float(confidence[mask].mean()),
        }
    return {
        "active_slots": int((freq > 0).sum()),
        "largest_slot_rows": int(freq.max()),
        "entropy_effective_slots": float(np.exp(-(prob * np.log(prob)).sum())),
        "slot_counts": freq.tolist(),
        "groups": groups,
    }


def sealed_fit(p, truth):
    ids, panel, legal = (p[k] for k in ("fit_ids", "panel", "fit_legal"))
    yf, yp = truth[ids], truth[panel]
    assert np.array_equal(yf[legal >= 0], legal[legal >= 0])
    x = torch.from_numpy(p["fit_feature"]).cuda()
    result = {}
    for name, rows in {
        "classifier": p["classifier"],
        "spherical": p["spherical"],
        "semi": p["semi"],
        **readout.oracle(x, yf),
    }.items():
        stats = [readout.evaluate(row, yf, yp) for row in rows]
        result[name] = {"per_seed": stats, "estimate": readout.reduce_rows(stats)}
    return result


def seal(receipt, series):
    from scripts.audit_r2_evidence import sealed_labels

    for tag, digest in receipt["frozen_sha256"].items():
        path = LOCAL / f"frozen-{tag}.npz"
        assert audit.file_digest(path) == digest
    truth = sealed_labels()
    out = {}
    for tag in TAGS:
        path = LOCAL / f"frozen-{tag}.npz"
        with np.load(path, allow_pickle=False) as f:
            p = dict(f)
        ids, slots = (p[k] for k in ("ids", "slots"))
        y = truth[ids]
        observed = split_cluster_acc_v2(y, slots, y < 98)
        run, epoch = ("E0", 200) if tag == TAGS[0] else ("R2", int(tag.split("-")[1]))
        recorded = series[run][epoch - 1]["accuracy"]
        delta = np.asarray(observed) - recorded
        matches = bool(np.all(np.abs(delta) <= (0.00005001 if run == "E0" else 1e-10)))
        aligned, info = align(
            ids.tolist(), dict(zip(ids.tolist(), slots.tolist(), strict=True)), truth
        )
        panel, _ = audit.assess_features(
            torch.from_numpy(p["feature"]).cuda(), slots, y, aligned
        )
        detection, bank_map = g4.sealed(p, truth, POLICIES, baseline="DINO")
        # Stable history is unavailable for E0 and not used for this three-snapshot comparison.
        out[tag] = {
            "v2": list(observed),
            "v2_recorded": recorded,
            "v2_matches_log": matches,
            "v2_minus_log_pp": (100 * delta).tolist(),
            "geometry": panel["geometry"],
            "official_alignment": info,
            "prediction_bias": bias(p, y, aligned),
            "readouts": sealed_fit(p, truth),
            "detection": detection,
            "bank_alignment": bank_map,
            "correction": heldout.corrections(
                p, truth, policies=POLICIES, baseline="Confidence"
            ),
        }
        print("Sealed v2", tag, observed, "matches log", matches, flush=True)
    return out


def accuracy(values):
    return " / ".join(f"{v * 100:.2f}" for v in values)


def summary_rows(series):
    out = {}
    for k, rows in series.items():
        peak = max(rows, key=lambda x: x["accuracy"][0])
        end = rows[-1]
        out[k] = {
            "validated_records": len(rows) if k != "E3-F" else 0,
            "last_epoch": end["epoch"],
            "last": end["accuracy"],
            "peak": peak if k != "E3-F" else None,
            "last20": [
                float(x)
                for x in np.asarray([r["accuracy"] for r in rows[-20:]]).mean(0)
            ]
            if k != "E3-F"
            else None,
        }
    return out


def comparisons(series):
    out = {}
    for mode in ("E3", "R2"):
        out[mode] = [
            {
                "epoch": row["epoch"],
                "e0_minus_run_pp": [
                    100 * (a - b)
                    for a, b in zip(e0["accuracy"], row["accuracy"], strict=True)
                ],
                "e0_loss": e0["loss"],
                "other_reported_loss": row["loss"],
            }
            for e0, row in zip(
                series["E0"][: len(series[mode])], series[mode], strict=True
            )
        ]
    return out


def docs(r):
    meta = r["provenance"]
    final = r["frozen"][TAGS[0]]
    confirmed = [r["trajectory"][k][-1] for k in ("E0", "E3", "E3-F", "R2")]
    external = (
        "The supplied SHA-256 sidecar matches the tarball."
        if meta["external_archive_digest_match"]
        else "The archive has no external SHA-256 sidecar. The recorded digest identifies the local tarball, and all members match the supplied extracted copies."
    )
    lines = [
        "# SimGCD E0 baseline audit on Stanford Cars",
        "",
        (
            f"E0 epoch 200 measured transductive v2 **{accuracy(final['v2'])}% All / Old / New** "
            "on the 6,144 unlabeled training rows. "
            f"The archived metric is {accuracy(final['v2_recorded'])}%."
        ),
        "",
        (
            f"Archive SHA-256 `{meta['archive_sha256']}`. {external} "
            f"All {meta['member_count']} regular members matched byte-for-byte by digest. "
            f"The commit `{E0_COMMIT}` resolves locally. Its archived patch changes dataset roots only. "
            "The checkpoint marks completed epoch 200 and contains model and SGD state."
        ),
        "",
        (
            f"The archived run used {meta['gpu']}, driver {meta['driver']}, PyTorch {meta['package_versions']['torch']} "
            f"and torchvision {meta['package_versions']['torchvision']}. The audit ran on {meta['runtime']['gpu']} "
            f"with PyTorch {meta['runtime']['torch']}. Recorded launch arguments match the seed-1 Cars "
            "SSB protocol, ViT-B/16, batch 128, 200 epochs and v2. The archived class order and the "
            "commit's split, preprocessing and evaluation code match the audit. The regenerated local "
            "split and ID ordering match the verified R2 split fingerprint and CLIP bank. All frozen "
            "backbone tensors, including final LayerNorm, agree across checkpoints. The E0 archive "
            "does not include a remote image-ID list or image-content hashes."
        ),
        "",
        (
            "All 200 metric lines match the full stdout and trainer log, in strict epoch and timestamp order. "
            "E3 has a complete 200-epoch log. R2 has 150 logged epochs. E3-F retains only manually recovered "
            "aggregates, so the comparison uses its reported endpoint and gives no epochwise curve. "
            "The R2 loss in JSON is the recorded base objective, with its relational term reported separately in the source audit."
        ),
        "",
        "| Run | Completed epoch | Validated epoch records | Endpoint All / Old / New (%) | Last-20 mean All / Old / New (%) |",
        "|---|---:|---:|---|---|",
    ]
    for run, last in zip(("E0", "E3", "E3-F", "R2"), confirmed, strict=True):
        s = r["training_summary"][run]
        lines.append(
            f"| {run} | {last['epoch']} | {s['validated_records']} | {accuracy(last['accuracy'])} | "
            f"{accuracy(s['last20']) if s['last20'] is not None else '—'} |"
        )
    lines += [
        "",
        "| Epoch | E0 All / Old / New (%) | E3 All / Old / New (%) | R2 All / Old / New (%) |",
        "|---|---|---|---|",
    ]
    for e in (1, 20, 50, 100, 150, 200):
        triplets = [
            accuracy(r["trajectory"][k][e - 1]["accuracy"])
            if len(r["trajectory"][k]) >= e
            else "—"
            for k in ("E0", "E3", "R2")
        ]
        lines.append(f"| {e} | " + " | ".join(triplets) + " |")
    lines += [
        "",
        "## Frozen checkpoint geometry and prediction bias",
        "",
        (
            "The v2 rows use the whole transductive split. Geometry uses normalized final CLS, kNN-5 "
            "and leave-one-out class centroid contrast. Slot bias uses full-split alignment. "
            "Reported classifier misrouting is conditional on the sample's true Old/New group."
        ),
        "",
        "| Snapshot | v2 All / Old / New (%) | Matches log | kNN-5 All / Old / New | Margin All / Old / New | Old→New / New→Old | Active / effective slots |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, x in r["frozen"].items():
        geometry = [
            " / ".join(f"{x['geometry'][g][k]:.4f}" for g in GROUPS)
            for k in ("purity5", "centroid_margin")
        ]
        b = x["prediction_bias"]
        mis = [
            b["groups"][g][stat]
            for g, stat in (("old", "mapped_new_rate"), ("new", "mapped_old_rate"))
        ]
        lines.append(
            f"| {tag} | {accuracy(x['v2'])} | {'Yes' if x['v2_matches_log'] else 'No'} | {geometry[0]} | {geometry[1]} | "
            f"{accuracy(mis)} | {b['active_slots']} / {b['entropy_effective_slots']:.1f} |"
        )
    lines += [
        "",
        "## Frozen readouts",
        "",
        (
            "Readouts follow the earlier fit protocol and seeds 1, 7 and 19. The same 588-ID panel "
            "is held out from fitting. All other images form the fit set. Legal-labeled images anchor "
            "semisupervised K-means. Oracle centroid and linear ridge use privileged fit labels only "
            "inside sealed diagnostics. Fit alignment uses non-panel labels, while panel rematching is descriptive."
        ),
        "",
        "| Snapshot / readout | Fit aligned All / Old / New (%) | Panel rematched All / Old / New (%) |",
        "|---|---|---|",
    ]
    for tag, x in r["frozen"].items():
        for method, m in x["readouts"].items():
            aligned = [
                accuracy([m["estimate"][g][k] for g in ("All", "Old", "New")])
                for k in ("fit", "rematch")
            ]
            lines.append(f"| {tag} / {method} | {aligned[0]} | {aligned[1]} |")
    lines += [
        "",
        "## Locked held-out retrieval and correction",
        "",
        (
            "DINO and CLIP use the same 1,024 G6-confirm queries, 5,120 bank IDs and fixed R2-150 "
            "hard-neighbor set. The bank supplies Hungarian alignment. Risks are alternative-slot minus "
            "current-slot support. The confidence baseline flags lowest max teacher probability at temperature .04 "
            "and uses the same CLIP destinations as the CLIP arm. Coverage budgets stay at 10, 20 and 40 percent, "
            "with ID tie-breaking. Random controls draw from eligible queries with the same retriever and number "
            "of edits. Paired pointwise intervals condition on the fixed bank and flags."
        ),
        "",
        "| Snapshot / detector | Error AUROC All / Old / New | Destination recall@1/3/5 (All) | Recall@1/3/5 (New) |",
        "|---|---|---|---|",
    ]
    for tag, x in r["frozen"].items():
        for name in POLICIES:
            d = x["detection"][name]
            auc = " / ".join(
                f"{d[g]['detection_auc']:.3f}" for g in ("All", "Old", "New")
            )
            recalls = [
                " / ".join(f"{d[g]['destination_recall'][k]:.3f}" for k in (1, 3, 5))
                for g in ("All", "New")
            ]
            lines.append(f"| {tag} / {name} | {auc} | {recalls[0]} | {recalls[1]} |")
    lines += [
        "",
        "| Snapshot / policy at 10% | Edits | Repairs / harmful / wrong-to-wrong | Δ All vs KEEP (pp) | Δ All vs confidence (pp, 95% CI) | Δ All vs random (pp, 95% CI) |",
        "|---|---:|---|---:|---|---|",
    ]

    def interval(row):
        if row is None:
            return "—"
        lo, hi = row["ci95_pp"]
        return f"{row['pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    for tag, x in r["frozen"].items():
        for name, c in x["correction"][10]["All"].items():
            lines.append(
                f"| {tag} / {name} | {c['edits']} | {c['correct_repairs']} / {c['harmful_edits']} / {c['wrong_to_wrong']} | "
                f"{c['vs_keep']['pp']:+.2f} | {interval(c['vs_confidence'])} | {interval(c.get('vs_matched_random'))} |"
            )
    lines += [
        "",
        (
            "The frozen E0-200 and R2-100/150 comparisons are descriptive across different training epochs and trajectories. "
            "The held-out queries were reused from a previous Cars analysis. The offline assignment changes use sealed "
            "labels for scoring and do not measure training effects. Oracle readouts also require privileged labels."
        ),
        "",
        f"Audit runtime {r['seconds']:.1f} seconds.",
        "",
        "[Complete trajectories, package versions, checksums, subgroup metrics and intervals](r2_e0_baseline_audit.json)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")
    research_summary(r)
    experiment_matrix(r)


def research_summary(r):
    z = r["frozen"][TAGS[0]]
    rows = r["training_summary"]
    ox = z["correction"][10]["All"]
    values = [
        "# CRiSH research summary",
        "",
        (
            f"The newly reproduced E0 transductive v2 baseline is **{accuracy(z['v2'])}% All / Old / New** "
            "on Stanford Cars at epoch 200. The checkpoint and full training log were audited. "
            f"The historical E3 endpoint is {accuracy(rows['E3']['last'])}%, and the manually recovered E3-F endpoint "
            f"is {accuracy(rows['E3-F']['last'])}%. R2 ended at checkpoint 150 with {accuracy(rows['R2']['last'])}%. "
            "The current evidence covers a single seed and different adaptation and training conditions."
        ),
        "",
        (
            f"On the reused G6-confirm panel at 10% coverage, frozen E0 CLIP correction changes aligned All "
            f"accuracy by {ox['CLIP']['vs_keep']['pp']:+.2f} pp. The confidence-selected policy with the same "
            f"CLIP retriever changes it by {ox['Confidence']['vs_keep']['pp']:+.2f} pp. "
            "The full report gives matched-random contrasts and paired uncertainty. "
            "These are sealed offline assignment changes. The controlled continuation diagnostics separately "
            "track exposure, feature response and panel transfer."
        ),
        "",
        "- [E0 baseline audit and frozen geometry](CRISH_E0_BASELINE_AUDIT.md)",
        "- [Experiment matrix](CRISH_EXPERIMENT_MATRIX.md)",
        "- [R2 mechanistic audit](CRISH_RELATIONAL_MECHANISTIC_AUDIT.md)",
        "- [Locked held-out CLIP confirmation](CRISH_VLM_HELDOUT_CONFIRMATION.md)",
        "- [Gradient conflict audit](CRISH_GRADIENT_CONFLICT.md)",
        "- [Hard and soft CLIP continuation](CRISH_SOFT_TARGET_TRAINING.md)",
    ]
    SUMMARY.write_text("\n".join(values) + "\n", encoding="utf8")


def experiment_matrix(r):
    links = {
        "E0": "[Archive and frozen audit](CRISH_E0_BASELINE_AUDIT.md)",
        "E3": "[Global verifier](CRISH_SIMGCD_GLOBAL.md)",
        "E3-F": "[Manual aggregates only](E3F_RECOVERED_RESULTS.md)",
        "R2": "[R2 mechanistic audit](CRISH_RELATIONAL_MECHANISTIC_AUDIT.md)",
    }
    rows = [
        "# CRiSH experiment matrix",
        "",
        "| Training run | Seed | Observed endpoint | All / Old / New (%) | Evidence |",
        "|---|---:|---:|---|---|",
    ]
    for k, s in r["training_summary"].items():
        rows.append(
            f"| {k} | 1 | {s['last_epoch']} | {accuracy(s['last'])} | {links[k]} |"
        )
    rows += [
        "",
        "| Diagnostic | Starting checkpoint or feature bank | Comparison |",
        "|---|---|---|",
        "| [Frozen E0 benchmark](CRISH_E0_BASELINE_AUDIT.md) | E0-200, R2-100/150 | v2, geometry, readouts, DINO/CLIP and matched offline correction |",
        "| [G6 confirmation](CRISH_VLM_HELDOUT_CONFIRMATION.md) | R2-100/150 | Locked detector/retriever policies and fixed budgets |",
        "| [Oracle trajectory](CRISH_ORACLE_MOVE_FIVE_EPOCHS.md) | R2-100 | Oracle MOVE, KEEP and matched random |",
        "| [CLIP continuation](CRISH_CLIP_REPAIR_TRAINING.md) | R2-150 | Direct MOVE, KEEP and matched random |",
        "| [MOVE strength](CRISH_MOVE_STRENGTH_TRAINING.md) | R2-150 | Paired mass .2/.8 and random |",
        "| [Early continuation](CRISH_SUSTAINED_MOVE_TRAINING.md) | R2-100 | Fixed proposals through epoch 105 |",
        "| [Dynamic refresh](CRISH_DYNAMIC_MOVE_TRAINING.md) | Shared epoch 102 | Dynamic versus fixed with observed exposure |",
        "| [Soft targets](CRISH_SOFT_TARGET_TRAINING.md) | Shared epoch 102 | Matched hard and top-three soft MOVE |",
        "",
        (
            "Training endpoints and frozen checkpoints record different comparisons. E3-F lacks raw trajectory files. "
            "The frozen benchmark compares checkpoint-conditioned geometry and cannot establish a causal method effect."
        ),
    ]
    MATRIX.write_text("\n".join(rows) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    smoke = parser.parse_args().smoke
    start = time.monotonic()
    LOCAL.mkdir(exist_ok=True)
    files, meta = archive_receipt()
    cfg, series = trajectory(files)
    meta["launch_arguments"] = cfg
    model, ds, center, ids, states = context(files, cfg, meta)
    print(
        "Verified E0 archive, environment, split, checkpoint and 200 epochs", flush=True
    )
    data = extract(model, ds, center, ids, states, meta, smoke)
    if smoke:
        return
    freeze(ids, data, meta)
    result = {
        "provenance": meta,
        "trajectory": series,
        "training_summary": summary_rows(series),
        "aligned_epoch_differences": comparisons(series),
        "frozen": seal(meta, series),
    }
    result["seconds"] = time.monotonic() - start
    docs(result)
    audit.write_json(OUT, json_native(result))


if __name__ == "__main__":
    main()
