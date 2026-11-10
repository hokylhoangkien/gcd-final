"""Dataset layout adapters; method/loss code is shared across supported datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder: str
    split_file: Optional[str]
    split_sha256: Optional[str]
    num_classes: int
    metadata: tuple
    target_offset: int = 1


DATASETS = {
    'scars': DatasetSpec('scars', 'cars', 'scars_osr_splits.pkl',
        '28d471f0fb9e2c773e1255b8b447e542974f6617952203056a0635c638513d39', 196,
        ('devkit/cars_train_annos.mat', 'devkit/cars_test_annos_withlabels.mat')),
    'cub': DatasetSpec('cub', 'cub', 'cub_osr_splits.pkl',
        '48bdcaa05603d932037738d3df776ab500945e9b7aeff85f4668305bf71bd1e7', 200,
        ('CUB_200_2011/images.txt', 'CUB_200_2011/image_class_labels.txt',
         'CUB_200_2011/train_test_split.txt', 'CUB_200_2011/attributes/image_attribute_labels.txt')),
    'cifar100': DatasetSpec('cifar100', 'cifar100', None, None, 100,
        ('cifar-100-python/train', 'cifar-100-python/test', 'cifar-100-python/meta'), 0),
    'aircraft': DatasetSpec('aircraft', 'aircraft/fgvc-aircraft-2013b', 'aircraft_osr_splits.pkl',
        '2a2481307cec7821721d6a73c1a3ca77db7e657649e043d1a83aa7dd81a5a194', 100,
        ('data/images_variant_trainval.txt', 'data/images_variant_test.txt'), 0),
}


def dataset_spec(name='scars'):
    if name not in DATASETS:
        raise ValueError(f'Unsupported dataset: {name}; choose {", ".join(DATASETS)}')
    return DATASETS[name]


def image_paths(dataset, name):
    dataset_spec(name)
    if name == 'cifar100':
        raise ValueError('CIFAR-100 stores pixel arrays; use sample IDs, not invented image paths')
    if name == 'scars':
        return tuple(map(str, dataset.data))
    if name == 'aircraft':
        base = (Path(dataset.root)/'data/images').resolve()
        # Index path only: never unpack/read an unlabeled sample's target.
        paths = tuple(Path(row[0]).resolve() for row in dataset.samples)
        if any(not path.is_relative_to(base) for path in paths):
            raise ValueError('Aircraft image path escapes the images directory')
        return tuple(map(str, paths))
    # Access no target column, attribute array or target transform here. The
    # unlabeled training adapter calls only this path/ID accessor.
    base = (Path(dataset.root)/dataset.base_folder).resolve()
    paths = tuple((base/str(relative)).resolve() for relative in dataset.data['filepath'])
    if any(not path.is_relative_to(base) for path in paths):
        raise ValueError('CUB image path escapes the images directory')
    return tuple(map(str, paths))


def raw_targets(dataset, name):
    """Native targets; use target_offset, only legal labeled rows or audits."""
    dataset_spec(name)
    if name == 'cifar100':
        return dataset.targets
    if name == 'aircraft':
        return [row[1] for row in dataset.samples]
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
    # Keep the pinned loader's download=True calls offline, including on missing
    # or corrupt CIFAR inputs. Existing CUB/Cars-only environments are unchanged.
    if hasattr(config_module, 'cifar_100_root'):
        from data import cifar
        cifar.cifar_100_root = config_module.cifar_100_root

        def offline_cifar(self):
            if not self._check_integrity():
                raise RuntimeError('CIFAR-100 inputs missing/corrupt; automatic download disabled. Prepare cifar100/cifar-100-python/{train,test,meta}.')
        cifar.CustomCIFAR100.download = offline_cifar
    if hasattr(config_module, 'aircraft_root'):
        from data import fgvc_aircraft
        fgvc_aircraft.aircraft_root = config_module.aircraft_root

        def offline_aircraft(self):
            validate_aircraft_metadata(self.root)
            if not self._check_exists():
                raise RuntimeError('Aircraft inputs missing; automatic download disabled')
        fgvc_aircraft.FGVCAircraft.download = offline_aircraft


def validate_aircraft_metadata(root):
    """Match native variant indexing in both pools before any loader/download.

    The source indexes sorted raw class strings separately in trainval/test;
    preserve that mapping, including detection of inconsistent line endings.
    """
    root = Path(root)
    base = (root/'data/images').resolve()
    pools, mappings = {}, {}
    for split in ('trainval', 'test'):
        path = root/f'data/images_variant_{split}.txt'
        if not path.is_file():
            raise ValueError(f'Missing Aircraft metadata: {path.name}; automatic download disabled')
        ids, variants = set(), set()
        for line in path.read_text().splitlines(keepends=True):
            parts = line.split(' ', 1)
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].strip():
                raise ValueError('Invalid Aircraft variant metadata row')
            sample, variant = parts
            if sample in ids:
                raise ValueError('Duplicate Aircraft image ID')
            image = (base/(sample+'.jpg')).resolve()
            if not image.is_relative_to(base) or not image.is_file():
                raise ValueError('Missing or unsafe Aircraft image')
            ids.add(sample)
            variants.add(variant)
        if not ids or len(variants) != 100 or len({v.strip() for v in variants}) != 100:
            raise ValueError('Aircraft requires 100 consistent variant classes in each pool')
        pools[split], mappings[split] = ids, sorted(variants)
    if pools['trainval'] & pools['test']:
        raise ValueError('Aircraft trainval/test image ID overlap')
    if mappings['trainval'] != mappings['test']:
        raise ValueError('Aircraft trainval/test variant class mappings differ')


def whole_training_dataset(name, root):
    if name == 'scars':
        from data.stanford_cars import CarsDataset
        return CarsDataset(train=True, data_dir=str(root))
    dataset_spec(name)
    if name == 'cifar100':
        from data.cifar import CustomCIFAR100
        return CustomCIFAR100(root=str(root), train=True, download=False)
    if name == 'aircraft':
        from data.fgvc_aircraft import FGVCAircraft
        return FGVCAircraft(root=str(root), split='trainval', download=False)
    from data.cub import CustomCub2011
    return CustomCub2011(root=str(root), train=True, download=False)
