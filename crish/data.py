"""Image-only view of the official, already constructed unlabeled training split."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageOnly(Dataset):
    def __init__(self, dataset, name):
        self.ids = dataset.uq_idxs.copy()
        self.transform = dataset.transform
        self.loader = getattr(dataset, "loader", None)
        if name in ("cifar10", "cifar100"):
            self.images = dataset.data
            self.loader = Image.fromarray
        elif name == "scars":
            self.images = list(dataset.data)
        elif name == "cub":
            root = Path(dataset.root) / dataset.base_folder
            self.images = [str(root / path) for path in dataset.data.filepath]
        elif name in ("aircraft", "herbarium_19", "imagenet_100", "imagenet_1k"):
            self.images = [s[0] for s in dataset.samples]
        else:
            raise ValueError("Unsupported image-only dataset")
        if len(self.images) != len(self.ids):
            raise ValueError("Unaligned training image IDs")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.transform(self.loader(self.images[index])), -1, self.ids[index]
