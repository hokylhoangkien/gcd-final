"""Dataset layout adapters; method/loss code is shared across supported datasets."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder: str
    split_file: str
    split_sha256: str
    num_classes: int
    metadata: tuple


DATASETS = {
    'scars': DatasetSpec('scars', 'cars', 'scars_osr_splits.pkl',
        '28d471f0fb9e2c773e1255b8b447e542974f6617952203056a0635c638513d39', 196,
        ('devkit/cars_train_annos.mat', 'devkit/cars_test_annos_withlabels.mat')),
    'cub': DatasetSpec('cub', 'cub', 'cub_osr_splits.pkl',
        '48bdcaa05603d932037738d3df776ab500945e9b7aeff85f4668305bf71bd1e7', 200,
        ('CUB_200_2011/images.txt', 'CUB_200_2011/image_class_labels.txt',
         'CUB_200_2011/train_test_split.txt', 'CUB_200_2011/attributes/image_attribute_labels.txt')),
}


def dataset_spec(name='scars'):
    if name not in DATASETS:
        raise ValueError(f'Unsupported dataset: {name}; choose scars or cub')
    return DATASETS[name]


def image_paths(dataset, name):
    dataset_spec(name)
    if name == 'scars':
        return tuple(map(str, dataset.data))
    # Access no target column, attribute array or target transform here. The
    # unlabeled training adapter calls only this path/ID accessor.
    base = (Path(dataset.root)/dataset.base_folder).resolve()
    paths = tuple((base/str(relative)).resolve() for relative in dataset.data['filepath'])
    if any(not path.is_relative_to(base) for path in paths):
        raise ValueError('CUB image path escapes the images directory')
    return tuple(map(str, paths))


def raw_targets(dataset, name):
    """Native one-based targets; only legal labeled training rows or audits."""
    dataset_spec(name)
    return dataset.target if name == 'scars' else dataset.data['target'].tolist()


def validate_cub_metadata(root):
    """Reject ambiguous native joins before the upstream loader constructs splits."""
    base = Path(root)/'CUB_200_2011'
    tables = []
    for filename in ('images.txt', 'image_class_labels.txt', 'train_test_split.txt'):
        rows = {}
        for line in (base/filename).read_text().splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f'Invalid CUB metadata row: {filename}')
            key = int(fields[0])
            if key <= 0 or key in rows:
                raise ValueError(f'Duplicate/invalid CUB image ID: {filename}')
            rows[key] = fields[1]
        tables.append(rows)
    images, targets, split = tables
    if not images or images.keys() != targets.keys() or images.keys() != split.keys():
        raise ValueError('CUB metadata image IDs differ')
    image_root = (base/'images').resolve()
    paths = [(image_root/path).resolve() for path in images.values()]
    if len(set(paths)) != len(paths) or any(not p.is_relative_to(image_root) or not p.is_file() for p in paths):
        raise ValueError('Duplicate, missing or unsafe CUB image paths')
    if any(not 1 <= int(v) <= 200 for v in targets.values()) or any(v not in ('0', '1') for v in split.values()):
        raise ValueError('Invalid CUB class or train/test flag')
    if not (base/'attributes/image_attribute_labels.txt').is_file():
        raise ValueError('Missing CUB attributes/image_attribute_labels.txt required by upstream')


def configure_modules(config_module):
    from data import cub, stanford_cars
    # Native loaders cache roots at import. Refresh for repeated probes/tests
    # without reloading modules or changing any sample ordering/split function.
    cub.cub_root = config_module.cub_root
    stanford_cars.car_root = config_module.car_root

    def offline_cub(self):
        validate_cub_metadata(self.root)
        if not self._check_integrity():
            raise RuntimeError('CUB inputs missing/corrupt; automatic download/extraction disabled. Prepare CUB_200_2011 locally.')
    cub.CustomCub2011._download = offline_cub


def whole_training_dataset(name, root):
    if name == 'scars':
        from data.stanford_cars import CarsDataset
        return CarsDataset(train=True, data_dir=str(root))
    dataset_spec(name)
    from data.cub import CustomCub2011
    return CustomCub2011(root=str(root), train=True, download=False)
