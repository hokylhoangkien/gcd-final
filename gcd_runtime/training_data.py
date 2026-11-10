"""Materialize training-only image rows with no hidden-target storage.

Official split construction is unchanged. Only the legally labeled side's target
array is read; unlabeled rows are copied as (path, -1, ID) without reading targets.
No reference to the original annotation-bearing dataset/label transform survives.
"""
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageRow:
    path: str
    target: int
    sample_id: int


class ImageRows(Dataset):
    def __init__(self, rows, transform):
        self.rows = tuple(rows)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from torchvision.datasets.folder import default_loader
        row = self.rows[index]
        image = default_loader(row.path)
        return self.transform(image) if self.transform is not None else image, row.target, row.sample_id


class ArrayRows(Dataset):
    """Own only RGB pixels, sanitized targets and IDs; never the source dataset."""
    def __init__(self, pixels, rows, transform):
        pixels = np.asarray(pixels)
        if pixels.ndim != 4 or pixels.shape[1:] != (32, 32, 3) or pixels.dtype != np.uint8:
            raise ValueError('Expected CIFAR-100 RGB pixel arrays')
        self.pixels = np.array(pixels, copy=True)
        self.pixels.flags.writeable = False
        self.rows, self.transform = tuple(rows), transform
        if len(self.pixels) != len(self.rows):
            raise ValueError('Pixel/target/ID lengths differ')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        image, row = Image.fromarray(self.pixels[index]), self.rows[index]
        return self.transform(image) if self.transform is not None else image, row.target, row.sample_id


class SanitizedMerged(Dataset):
    def __init__(self, labeled, unlabeled):
        self.labelled_dataset, self.unlabelled_dataset = labeled, unlabeled
        ids = [r.sample_id for part in (labeled, unlabeled) for r in part.rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate training IDs across membership partitions')
        self.sample_ids = tuple(ids)

    def __len__(self):
        return len(self.labelled_dataset) + len(self.unlabelled_dataset)

    def __getitem__(self, index):
        labeled = index < len(self.labelled_dataset)
        part = self.labelled_dataset if labeled else self.unlabelled_dataset
        offset = index if labeled else index-len(self.labelled_dataset)
        image, target, sample_id = part[offset]
        return image, target, sample_id, np.array([int(labeled)])


def sanitize_training(merged, dataset_name='scars'):
    from .datasets import image_paths, raw_targets, dataset_spec
    labeled, unlabeled = merged.labelled_dataset, merged.unlabelled_dataset
    if dataset_name == 'cifar100':
        # Only the legally labeled side's targets/target_transform are accessed.
        targets = raw_targets(labeled, dataset_name)
        if len(targets) != len(labeled.uq_idxs):
            raise ValueError('Labeled target/ID lengths differ')
        rows = []
        for target, sample_id in zip(targets, labeled.uq_idxs):
            legal = int(target)
            if labeled.target_transform is not None:
                legal = int(labeled.target_transform(legal))
            if not 0 <= legal < 80:
                raise ValueError('CIFAR-100 supervised target outside known classes')
            rows.append(ImageRow('', legal, int(sample_id)))
        other = [ImageRow('', -1, int(sample_id)) for sample_id in unlabeled.uq_idxs]
        return SanitizedMerged(ArrayRows(labeled.data, rows, labeled.transform),
                               ArrayRows(unlabeled.data, other, unlabeled.transform))
    paths, targets = image_paths(labeled, dataset_name), raw_targets(labeled, dataset_name)
    if not len(paths) == len(targets) == len(labeled.uq_idxs):
        raise ValueError('Labeled path/target/ID lengths differ')
    rows = []
    for path, target, sample_id in zip(paths, targets, labeled.uq_idxs):
        legal = int(target)-dataset_spec(dataset_name).target_offset
        if labeled.target_transform is not None:
            legal = int(labeled.target_transform(legal))
        rows.append(ImageRow(str(path), legal, int(sample_id)))
    # Do not access unlabeled.target or retain its transform (which maps classes).
    other_paths = image_paths(unlabeled, dataset_name)
    if len(other_paths) != len(unlabeled.uq_idxs):
        raise ValueError('Unlabeled path/ID lengths differ')
    other = [ImageRow(str(path), -1, int(sample_id)) for path, sample_id in zip(other_paths, unlabeled.uq_idxs)]
    return SanitizedMerged(ImageRows(rows, labeled.transform), ImageRows(other, unlabeled.transform))


def sanitize_cars_training(merged):
    """Compatibility alias for the original Cars-only adapter."""
    return sanitize_training(merged, 'scars')
