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
    from .datasets import image_paths, raw_targets
    labeled, unlabeled = merged.labelled_dataset, merged.unlabelled_dataset
    paths, targets = image_paths(labeled, dataset_name), raw_targets(labeled, dataset_name)
    if not len(paths) == len(targets) == len(labeled.uq_idxs):
        raise ValueError('Labeled path/target/ID lengths differ')
    rows = []
    for path, target, sample_id in zip(paths, targets, labeled.uq_idxs):
        legal = int(target)-1
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
