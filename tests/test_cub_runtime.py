"""Shared runtime with the actual pinned CUB loader and synthetic images."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from audit.protocol import compare_worker, upstream_reference
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.datasets import validate_cub_metadata
from gcd_runtime.readiness import audit_dataset
from gcd_runtime.training_data import sanitize_training
from scripts.check_integrity import check_split


class CubRuntimeTests(TestCase):
    def test_profiles_and_pinned_split(self):
        expected, _ = upstream_reference('cub')
        self.assertEqual([expected[k] for k in ('hyper_max_weight', 'unsupervised_smoothing', 'c', 'cr')], [1., 1., .1, 1.5])
        split = check_split(ROOT/'upstream/HypCD/data/ssb_splits/cub_osr_splits.pkl', 'cub')
        self.assertEqual((split['known_count'], split['novel_count']), (100, 100))
        for method in ('baseline', 'crish'):
            for suffix in ('', '_smoke'):
                cfg = load_config(ROOT/f'configs/cub_{method}{suffix}.json')
                self.assertEqual(compare_worker(cfg, smoke=bool(suffix))['differences'], {})

    def test_actual_loader_membership_and_sanitized_targets(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'cub/CUB_200_2011'
            (base/'images').mkdir(parents=True)
            (base/'attributes').mkdir()
            images, targets, split, attributes = [], [], [], []
            for sample in range(1, 601):
                filename = f'{sample}.png'
                Image.new('RGB', (2, 2)).save(base/'images'/filename)
                images.append(f'{sample} {filename}\n')
                targets.append(f'{sample} {(sample-1)//3+1}\n')
                split.append(f'{sample} {int(sample % 3 != 0)}\n')
                attributes.append(f'{sample} 1 1\n')
            for filename, rows in [('images.txt', images), ('image_class_labels.txt', targets),
                                   ('train_test_split.txt', split), ('attributes/image_attribute_labels.txt', attributes)]:
                (base/filename).write_text(''.join(rows))
            with patch.dict('os.environ', {'GCD_DATA_ROOT': directory}):
                counts = audit_dataset({'dataset_name': 'cub'})
                self.assertEqual(counts, dict(labeled=100, unlabeled=300, disjoint_test=200, known_classes=100, novel_classes=100))
                from data.get_datasets import get_datasets, get_class_splits
                args = get_class_splits(SimpleNamespace(dataset_name='cub', use_ssb_splits=True, prop_train_labels=.5))
                merged, _, _, _ = get_datasets('cub', None, None, args)
                safe = sanitize_training(merged, 'cub')
                for i in range(len(safe.labelled_dataset)):
                    actual = safe.labelled_dataset[i]
                    native = merged.labelled_dataset[i]
                    self.assertEqual(actual[1:], native[1:])
                self.assertTrue(all(r.target == -1 for r in safe.unlabelled_dataset.rows))
                self.assertEqual(len(set(safe.sample_ids)), 400)
                self.assertEqual(safe.unlabelled_dataset[0][0].size, (2, 2))
                self.assertFalse(hasattr(safe.unlabelled_dataset, 'data'))
                (base/'attributes/image_attribute_labels.txt').unlink()
                from data.cub import CustomCub2011
                with patch('data.cub.download_url', side_effect=AssertionError('No download allowed')):
                    with self.assertRaisesRegex(ValueError, 'attributes'):
                        CustomCub2011(str(base.parent), download=True)
            (base/'images.txt').write_text(''.join(images)+images[0])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                validate_cub_metadata(base.parent)

    def test_unlabeled_adapter_never_reads_hidden_columns(self):
        import pandas as pd
        class PathsOnly:
            def __getitem__(self, key):
                if key != 'filepath':
                    raise AssertionError('Hidden target read')
                return ['hidden.png']
        class Unlabeled:
            root, base_folder = '/tmp', 'images'
            data, uq_idxs, transform = PathsOnly(), [8], None
            @property
            def target_transform(self):
                raise AssertionError('Hidden transform read')
        labeled = SimpleNamespace(root='/tmp', base_folder='images', data=pd.DataFrame({'filepath': ['legal.png'], 'target': [151]}),
                                  uq_idxs=[2], transform=None, target_transform=lambda target: {150: 0}[target])
        safe = sanitize_training(SimpleNamespace(labelled_dataset=labeled, unlabelled_dataset=Unlabeled()), 'cub')
        self.assertEqual([(r.target, r.sample_id) for part in (safe.labelled_dataset, safe.unlabelled_dataset) for r in part.rows], [(0, 2), (-1, 8)])
