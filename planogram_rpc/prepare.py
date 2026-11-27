"""Download only selected RPC exemplar images, or read a local official dataset.

Nothing downloads at import. Kaggle credentials, if required, stay in Kaggle's
normal environment/config. No full-dataset download or checkout-image download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from .common import digest, image_path, write_json

HANDLE = "diyer22/retail-product-checkout-dataset/versions/5"


def download_file(name):
    import kagglehub

    for attempt in range(3):
        try:
            return Path(kagglehub.dataset_download(HANDLE, path=name))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def candidates(annotation):
    """Accept exactly one annotated product per exemplar; never checkout scenes."""
    by_image = defaultdict(list)
    for item in annotation["annotations"]:
        if not item.get("iscrowd", False):
            by_image[item["image_id"]].append(item)
    groups = defaultdict(list)
    for image in sorted(annotation["images"], key=lambda i: str(i["id"])):
        anns = by_image[image["id"]]
        if len(anns) != 1:
            continue
        ann = anns[0]
        filename = str(image["file_name"])
        if Path(filename).name != filename or not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            raise ValueError("Expected a safe exemplar image filename")
        groups[int(ann["category_id"])].append(
            {
                "source_id": str(image["id"]),
                "source_file": filename,
                "bbox": ann["bbox"],
                "source_category": int(ann["category_id"]),
            }
        )
    return groups


def crop_product(path, bbox):
    with Image.open(path) as original:
        # Official COCO bboxes are in the stored image coordinate frame.
        image = original.convert("RGB")
        x, y, w, h = map(float, bbox)
        if w <= 0 or h <= 0:
            raise ValueError("Nonpositive product bbox")
        margin = 0.05 * max(w, h)
        box = (
            max(0, int(x - margin)),
            max(0, int(y - margin)),
            min(image.width, int(x + w + margin + 0.999)),
            min(image.height, int(y + h + margin + 0.999)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Product bbox does not intersect the image")
        image = image.crop(box)
        # Preserve the whole pack; do not stretch aspect ratio or cut the label.
        return ImageOps.pad(
            image, (256, 256), method=Image.Resampling.BICUBIC, color=(255, 255, 255)
        )


def assign_splits(records, classes, known_count, seed):
    rng = random.Random(seed)
    result = []
    for target, category in enumerate(classes):
        rows = sorted(records[category], key=lambda r: r["source_id"])
        rng.shuffle(rows)
        n_eval = max(2, int(len(rows) * 0.15))
        train_count = len(rows) - 2 * n_eval
        if train_count < 8:
            raise ValueError("Too few distinct images for train/validation/test")
        for index, record in enumerate(rows):
            split = (
                "train"
                if index < train_count
                else "val"
                if index < train_count + n_eval
                else "test"
            )
            # Exactly half the known-class training images expose labels.
            legal = (
                target < known_count and split == "train" and index < train_count // 2
            )
            result.append(dict(record, target=target, split=split, labeled=legal))
    # Stable global IDs across all subsets and transforms.
    return [dict(row, id=i) for i, row in enumerate(result)]


def verify(root, manifest):
    rows = manifest["images"]
    ids, hashes, paths = set(), set(), set()
    counts = Counter()
    for row in rows:
        if row["id"] in ids or row["sha256"] in hashes or row["path"] in paths:
            raise ValueError("Duplicate image ID, decoded pixels or path in manifest")
        ids.add(row["id"])
        hashes.add(row["sha256"])
        paths.add(row["path"])
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        if not 0 <= row["target"] < len(manifest["classes"]):
            raise ValueError("Invalid target")
        if row["labeled"] and (
            row["split"] != "train" or row["target"] >= manifest["known_count"]
        ):
            raise ValueError("Illegal novel/evaluation label exposed for training")
        with Image.open(image_path(root, row["path"])) as im:
            actual = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"Image checksum differs: {row['path']}")
        counts[(row["target"], row["split"])] += 1
    for target in range(len(manifest["classes"])):
        if any(counts[target, split] < 2 for split in ("train", "val", "test")):
            raise ValueError("Every class requires train, validation and test images")
    return {
        s: sum(v for (_, split), v in counts.items() if split == s)
        for s in ("train", "val", "test")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/rpc40"))
    parser.add_argument(
        "--source", type=Path, help="Optional local official RPC folder"
    )
    parser.add_argument("--classes", type=int, default=40)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.classes < 4 or args.classes % 2 or args.per_class < 40 or args.workers < 1:
        parser.error("Use an even class count >=4, >=40 images/class and >=1 worker")
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "manifest.json"
    settings = {"classes": args.classes, "per_class": args.per_class, "seed": args.seed}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not args.verify_only and manifest["settings"] != settings:
            raise ValueError("Dataset settings differ; use a new output directory")
        print("Verified", verify(args.root, manifest), flush=True)
        return
    if args.verify_only:
        raise FileNotFoundError(manifest_path)
    source = args.source
    if source:
        annotation_path = source / "instances_train2019.json"
        if not annotation_path.is_file():
            source = source / "retail_product_checkout"
            annotation_path = source / "instances_train2019.json"
    else:
        annotation_path = download_file("instances_train2019.json")
    annotation = json.loads(annotation_path.read_text())
    groups = candidates(annotation)
    eligible = sorted(k for k, v in groups.items() if len(v) >= args.per_class)
    if len(eligible) < args.classes:
        raise ValueError("Not enough eligible product classes; reduce --per-class")
    rng = random.Random(args.seed)
    # RPC IDs are grouped by product family: randomize known/novel class order.
    classes = rng.sample(eligible, args.classes)
    # Reserve extra candidates so exact duplicates can be replaced without losing quota.
    plans = {}
    for category in classes:
        rows = list(groups[category])
        rng.shuffle(rows)
        plans[category] = rows
    selected = defaultdict(list)
    seen = set()
    print(
        f"Preparing {args.classes} classes x {args.per_class} product crops; resumable cache, no full RPC archive",
        flush=True,
    )

    def materialize(row):
        target = (
            args.root
            / "images"
            / f"{row['source_category']:03d}"
            / (row["source_file"] + ".png")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            path = (
                source / "train2019" / row["source_file"]
                if source
                else download_file(
                    "retail_product_checkout/train2019/" + row["source_file"]
                )
            )
            image = crop_product(path, row["bbox"])
            temp = target.with_suffix(".tmp")
            image.save(temp, format="PNG")
            temp.replace(target)
        with Image.open(target) as image:
            checksum = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        return dict(row, path=target.relative_to(args.root).as_posix(), sha256=checksum)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for category in classes:
            # Small ordered windows bound downloads and preserve reproducible selection.
            offset = 0
            while offset < len(plans[category]):
                window = plans[category][
                    offset : offset
                    + min(args.workers, args.per_class - len(selected[category]))
                ]
                for row in pool.map(materialize, window):
                    if row["sha256"] not in seen:
                        seen.add(row["sha256"])
                        selected[category].append(row)
                if len(selected[category]) == args.per_class:
                    break
                offset += len(window)
            if len(selected[category]) != args.per_class:
                raise ValueError(
                    f"Class {category} lacks enough unique crops; use a new root with smaller --per-class"
                )
            print(
                f"Class {category}: {len(selected[category])} unique images", flush=True
            )
    names = {int(c["id"]): c["name"] for c in annotation["categories"]}
    manifest = {
        "version": 1,
        "source": HANDLE,
        "annotation_sha256": digest(annotation_path),
        "settings": settings,
        "known_count": len(classes) // 2,
        "classes": [
            {"target": i, "source_category": k, "name": names[k]}
            for i, k in enumerate(classes)
        ],
        "images": assign_splits(selected, classes, len(classes) // 2, args.seed),
        "protocol": "custom exemplar holdout; 70/15/15 per SKU; exact decoded crop duplicates excluded; not the official RPC checkout benchmark",
    }
    counts = verify(args.root, manifest)
    write_json(manifest_path, manifest)
    print("Ready", counts, "manifest_sha256", digest(manifest_path), flush=True)


if __name__ == "__main__":
    main()
