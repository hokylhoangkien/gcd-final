"""Herbarium19 native ImageFolder/split integration; synthetic CPU evidence only."""
import copy
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from gcd_runtime.config import ROOT, load_config
from gcd_runtime.datasets import validate_herbarium_layout
from gcd_runtime.training_data import sanitize_training
from scripts.check_integrity import check_split, read_class_split
from selex_runtime.config import validate_config
from selex_runtime.reference import SOURCE_ROOT


def make_fixture(directory):
    from PIL import Image
    root = Path(directory)/'herbarium_19'
    image = io.BytesIO()
    Image.new('RGB', (3, 3), 'green').save(image, format='PNG')
    for pool in ('small-train', 'small-validation'):
        for cls in range(683):
            folder = root/pool/f'{cls:05d}'
            folder.mkdir(parents=True)
            count = 8+cls%3 if pool == 'small-train' else 1
            for row in range(count):
                (folder/f'{row}.png').write_bytes(image.getvalue())
    return root


class HerbariumRuntimeTests(unittest.TestCase):
    def test_split_pin_numpy_old_labels_and_extended_matched_profiles(self):
        path = SOURCE_ROOT/'data/ssb_splits/herbarium_19_class_splits.pkl'
        split = check_split(path, 'herbarium_19')
        self.assertEqual((split['known_count'], split['novel_count']), (341, 342))
        native = read_class_split(path, 'herbarium_19')
        self.assertEqual(len(set(native['Old']) | set(native['New'])), 683)
        a, b = [load_config(ROOT/f'configs/selex_herbarium_19_{arm}_200.json')
                for arm in ('baseline', 'crish')]
        self.assertFalse(a['crish']['enabled'])
        self.assertTrue(b['crish']['enabled'])
        a, b = copy.deepcopy(a), copy.deepcopy(b)
        for c in (a, b):
            self.assertEqual(c['epochs'], 200)
            self.assertEqual(c['ss_eval_every_epochs'], 1)
            self.assertFalse(c['population_policy']['enabled'])
            self.assertFalse(c['pair_reliability']['enabled'])
            source = runpy.run_path(str(SOURCE_ROOT/'config.py'))['dataset_configs']['herbarium_19']
            for key in ('unbalanced', 'temperature', 'grad_from_block', 'unsupervised_smoothing'):
                self.assertEqual(c[key], source[key])
            c.pop('method')
            c['crish'].pop('enabled')
        self.assertEqual(a, b)
        bad = load_config(ROOT/'configs/selex_herbarium_19_crish_200.json')
        with self.assertRaisesRegex(ValueError, 'unbalanced'):
            validate_config(dict(bad, unbalanced=False))

    def test_missing_dataset_and_independent_imagefolder_mapping_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Missing'):
                validate_herbarium_layout(tmp)
            root = make_fixture(tmp)
            validate_herbarium_layout(root)
            (root/'small-validation/00000').rename(root/'small-validation/renamed')
            with self.assertRaisesRegex(ValueError, 'mappings differ'):
                validate_herbarium_layout(root)

    def test_sanitization_never_reads_hidden_target_columns(self):
        class PathOnly:
            def __getitem__(self, key):
                if key != 0:
                    raise AssertionError('Read hidden target')
                return '/tmp/herbarium-fixture/small-train/00000/u.png'
        class Hidden:
            root = '/tmp/herbarium-fixture/small-train'
            samples, uq_idxs, transform = [PathOnly()], [21], None
            @property
            def targets(self): raise AssertionError('Read hidden targets')
            @property
            def target_transform(self): raise AssertionError('Read hidden transform')
        labeled = SimpleNamespace(root=Hidden.root,
            samples=[(Hidden.root+'/00001/l.png', 113)], uq_idxs=[3],
            transform=None, target_transform=lambda label: {113: 0}[label])
        safe = sanitize_training(SimpleNamespace(labelled_dataset=labeled, unlabelled_dataset=Hidden()), 'herbarium_19')
        self.assertEqual(safe.sample_ids, (3, 21))
        self.assertEqual(safe.labelled_dataset.rows[0].target, 0)
        self.assertEqual(safe.unlabelled_dataset.rows[0].target, -1)

    def test_native_loader_audit_and_real_image_cpu_backward(self):
        code = r'''
import os,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from gcd_runtime.config import ROOT,load_config
from gcd_runtime.readiness import audit_dataset
from gcd_runtime.datasets import raw_targets,whole_training_dataset
from gcd_runtime.training_data import sanitize_training
from tests.test_herbarium_runtime import make_fixture
from selex_runtime.reference import configure_environment,definitions
from selex_runtime.trainer import make_args
os.environ['GCD_DATA_ROOT']=sys.argv[1]
root=make_fixture(sys.argv[1])
cfg=load_config(ROOT/'configs/selex_herbarium_19_crish_200.json')
configure_environment(Path(sys.argv[1]),cfg)
counts=audit_dataset(cfg)
assert (counts['known_classes'],counts['novel_classes'],counts['disjoint_test'])==(341,342,683),counts
from data.get_datasets import get_class_splits,get_datasets
args=get_class_splits(make_args(cfg))
whole=whole_training_dataset('herbarium_19',root)
known_count=sum(t in args.train_classes for t in raw_targets(whole,'herbarium_19'))
assert counts['labeled']==int(.5*known_count)
assert counts['labeled']+counts['unlabeled']==len(whole)
assert len(set(np.bincount(whole.targets)))>1  # fixture is deliberately imbalanced
class TwoViews:
 def __call__(self,image):
  x=ToTensor()(image)
  return [x,x+.001]
merged,test,_,_=get_datasets('herbarium_19',TwoViews(),ToTensor(),args)
safe=sanitize_training(merged,'herbarium_19')
assert len(set(safe.sample_ids))==len(whole)
for i in range(len(safe.labelled_dataset)):
 assert safe.labelled_dataset[i][1:]==merged.labelled_dataset[i][1:]
assert all(row.target==-1 for row in safe.unlabelled_dataset.rows)
views,targets,ids,mask=next(iter(DataLoader(safe,batch_size=8,num_workers=0)))
model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(27,8))
features=torch.nn.functional.normalize(torch.stack([model(view) for view in views],dim=1),dim=-1)
loss=definitions('cpu')['SupConLoss']()(features,labels=targets)
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
print('Native Herbarium19 split/loader/audit/sanitization and synthetic image CPU backward passed')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT,
                                    capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_cli_uses_local_200_and_correct_smoke(self):
        for flags, method in ((['--variant', 'local'], 'crish'), (['--baseline'], 'baseline')):
            result = subprocess.run(['bash', 'scripts/train_selex.sh', '--dataset', 'herbarium_19',
                                     '--dry-run', *flags], cwd=ROOT,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            objects, text = [], result.stdout.strip()
            while text:
                value, end = json.JSONDecoder().raw_decode(text)
                objects.append(value)
                text = text[end:].lstrip()
            self.assertEqual(objects[0]['method'], method)
            self.assertTrue(objects[0]['unbalanced'])
            self.assertEqual(objects[0]['discrete_repair'], method == 'crish')
            for stage, epochs in zip(objects[1:], (1, 200)):
                self.assertEqual(stage['config']['epochs'], epochs)
                self.assertEqual(stage['config']['dataset_name'], 'herbarium_19')
                self.assertTrue(stage['config']['unbalanced'])

    def test_hierarchy_passes_native_unbalanced_capacity_and_legal_anchors(self):
        code = r'''
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader,TensorDataset
from gcd_runtime.config import ROOT,load_config
from selex_runtime.reference import configure_environment
from selex_runtime.trainer import make_args
from selex_runtime.hierarchy import extract_state
cfg=load_config(ROOT/'configs/selex_herbarium_19_crish_200.json')
configure_environment(Path(sys.argv[1]),cfg)
from methods.clustering import faster_mix_k_means_pytorch as fitter
args=make_args(cfg)
args.num_labeled_classes=args.num_unlabeled_classes=2
images=torch.eye(4).repeat(2,1)
membership=torch.tensor([True,False,True,False,False,False,False,False])
targets=torch.tensor([0,-1,1,-1,-1,-1,-1,-1])
calls=[]
class Cluster:
 def __init__(self,**kwargs): calls.append(kwargs)
 def fit_mix(self,u,l,legal):
  assert legal.tolist()==[0,1]
  self.labels_=torch.tensor([0,1,0,1,2,3,2,3])
  self.cluster_centers_=torch.eye(4)
model=torch.nn.Linear(4,4,bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(4))
loader=DataLoader(TensorDataset(images,targets,torch.arange(8),membership[:,None]),batch_size=8)
for unbalanced in (True,False):
 args.unbalanced=unbalanced
 runtime=SimpleNamespace(epoch=0)
 with patch.object(fitter,'K_Means',Cluster): state=extract_state(model,loader,args,runtime)
 assert state.known_targets.tolist()==[0,1]
 assert state.labels[0][membership].tolist()==[0,1]
 assert calls[-1]['cluster_size']==(None if unbalanced else 2)
print('Hierarchy unbalanced capacity, legacy balanced capacity, row restoration and legal-only anchors passed')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT,
                                    capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
