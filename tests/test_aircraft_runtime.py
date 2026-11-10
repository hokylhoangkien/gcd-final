"""Pinned Aircraft loader with synthetic images; no real-data/GPU results."""
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from gcd_runtime.config import ROOT
from gcd_runtime.datasets import validate_aircraft_metadata
from gcd_runtime.training_data import sanitize_training
from scripts.check_integrity import check_split


def make_fixture(directory):
    from PIL import Image
    root = Path(directory)/'aircraft/fgvc-aircraft-2013b'
    (root/'data/images').mkdir(parents=True)
    rows = dict(trainval=[], test=[])
    for variant in range(100):
        for sample in range(6):
            image_id = f'{variant*6+sample+1:07d}'
            Image.new('RGB', (4, 4), (variant, sample, 20)).save(root/f'data/images/{image_id}.jpg')
            rows['trainval' if sample < 4 else 'test'].append(f'{image_id} Variant {variant:03d}\n')
    for split, values in rows.items():
        (root/f'data/images_variant_{split}.txt').write_text(''.join(values))
    return root


class AircraftRuntimeTests(unittest.TestCase):
    def test_official_pinned_split_and_safe_native_offset(self):
        path = ROOT/'upstream/SelEx/data/ssb_splits/aircraft_osr_splits.pkl'
        split = check_split(path, 'aircraft')
        self.assertEqual((split['known_count'], split['novel_count']), (50, 50))
        self.assertEqual(split['novel_groups'], dict(Hard=13, Medium=17, Easy=20))
        class PathOnly:
            def __getitem__(self, key):
                if key != 0: raise AssertionError('Hidden Aircraft target read')
                return '/tmp/aircraft-fixture/data/images/hidden.jpg'
        class Unlabeled:
            root = '/tmp/aircraft-fixture'
            samples, uq_idxs, transform = [PathOnly()], [9], None
            @property
            def target_transform(self): raise AssertionError('Hidden transform read')
        labeled = SimpleNamespace(root='/tmp/aircraft-fixture',
            samples=[('/tmp/aircraft-fixture/data/images/legal.jpg', 73)],
            uq_idxs=[2], transform=None, target_transform=lambda x: {73: 0}[x])
        safe = sanitize_training(SimpleNamespace(labelled_dataset=labeled, unlabelled_dataset=Unlabeled()), 'aircraft')
        self.assertEqual(safe.labelled_dataset.rows[0].target, 0)
        self.assertEqual(safe.unlabelled_dataset.rows[0].target, -1)
        self.assertEqual(safe.sample_ids, (2, 9))

    def test_metadata_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fixture(tmp)
            validate_aircraft_metadata(root)
            path = root/'data/images_variant_trainval.txt'
            text = path.read_text()
            path.write_text(text+text.splitlines(keepends=True)[0])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                validate_aircraft_metadata(root)

    def test_metadata_rejects_cross_pool_overlap_and_mapping_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fixture(tmp)
            path = root/'data/images_variant_test.txt'
            original = path.read_text()
            path.write_text(original.replace('0000005', '0000001', 1))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                validate_aircraft_metadata(root)
            path.write_text(original.replace('Variant 000', 'Another variant 000'))
            with self.assertRaisesRegex(ValueError, 'mappings differ'):
                validate_aircraft_metadata(root)

    def test_metadata_rejects_missing_images_bad_ids_and_inconsistent_class_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fixture(tmp)
            path = root/'data/images_variant_test.txt'
            original = path.read_text()
            path.write_text(original.replace('0000005', '../escape', 1))
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                validate_aircraft_metadata(root)
            path.write_text(original.rstrip('\n'))
            with self.assertRaisesRegex(ValueError, 'consistent'):
                validate_aircraft_metadata(root)
            path.write_text(original)
            (root/'data/images/0000005.jpg').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                validate_aircraft_metadata(root)

    def test_actual_pinned_aircraft_loader_audit_labels_and_cpu_backward(self):
        code = r'''
import os,sys
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gcd_runtime.config import ROOT,load_config
from gcd_runtime.readiness import audit_dataset
from gcd_runtime.training_data import sanitize_training
from tests.test_aircraft_runtime import make_fixture
from selex_runtime.reference import configure_environment,definitions
from selex_runtime.trainer import make_args
os.environ['GCD_DATA_ROOT']=sys.argv[1]
root=make_fixture(sys.argv[1])
cfg=load_config(ROOT/'configs/selex_aircraft_baseline_200.json')
configure_environment(Path(sys.argv[1]),cfg)
counts=audit_dataset(cfg)
assert counts==dict(labeled=100,unlabeled=300,disjoint_test=200,known_classes=50,novel_classes=50),counts
from data.get_datasets import get_class_splits,get_datasets
from data.augmentations import get_transform
from data.fgvc_aircraft import FGVCAircraft
args=get_class_splits(make_args(cfg))
assert len(args.train_classes)==len(args.unlabeled_classes)==50
transform,evaluation=get_transform('imagenet',224,args)
class TwoViews:
 def __call__(self,image): return [transform(image),transform(image)]
merged,test,_,_=get_datasets('aircraft',TwoViews(),evaluation,args)
safe=sanitize_training(merged,'aircraft')
for i in range(len(safe.labelled_dataset)):
 assert safe.labelled_dataset[i][1:]==merged.labelled_dataset[i][1:]
assert all(row.target==-1 for row in safe.unlabelled_dataset.rows)
assert len(set(safe.sample_ids))==400
views,targets,ids,mask=next(iter(DataLoader(safe,batch_size=4,num_workers=0)))
assert views[0].shape==(4,3,224,224)
model=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((2,2)),torch.nn.Flatten(),torch.nn.Linear(12,8))
features=torch.stack([F.normalize(model(view),dim=-1) for view in views],dim=1)
loss=definitions()['SupConLoss']()(features,labels=targets)
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
with patch('urllib.request.urlopen',side_effect=AssertionError('No download permitted')):
 try: FGVCAircraft(root=str(root/'missing'),split='trainval',download=True)
 except ValueError as error: assert 'automatic download disabled' in str(error)
 else: raise AssertionError('Missing dataset must fail offline')
print('Pinned Aircraft trainval/test, zero-based targets, SSB anchors, image transforms/backward and offline failure passed')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT,
                                    capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
