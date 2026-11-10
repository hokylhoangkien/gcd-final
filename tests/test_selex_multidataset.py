"""Bounded synthetic checks of the three-dataset, 200-epoch SelEx protocol."""
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gcd_runtime.config import ROOT, load_config
from gcd_runtime.readiness import audit_cifar_membership
from gcd_runtime.training_data import sanitize_training
from scripts.check_integrity import check_split
from selex_runtime.config import validate_config
from selex_runtime.reference import SOURCE_ROOT
from selex_runtime.rng import (capture_rng, restore_rng, rng_digest, train_plain_due,
    create_request, begin_plain_worker, finish_plain_worker, commit_plain_result)
from selex_runtime.trainer import SelExRuntime, evaluate_plain_schedule


class SelExMultiDatasetTests(unittest.TestCase):
    def test_matched_profiles_keep_source_settings_except_declared_budget(self):
        source = runpy.run_path(str(SOURCE_ROOT/'config.py'))['dataset_configs']
        for dataset in ('scars', 'cub', 'aircraft', 'cifar100'):
            arms = [load_config(ROOT/f'configs/selex_{dataset}_{arm}_200.json')
                    for arm in ('baseline', 'crish')]
            baseline, proposed = map(copy.deepcopy, arms)
            self.assertEqual(baseline.pop('method'), 'baseline')
            self.assertEqual(proposed.pop('method'), 'crish')
            self.assertEqual(baseline, proposed)
            self.assertEqual(baseline['epochs'], 200)
            self.assertEqual(baseline['dino'], 'v2')
            for key in ('temperature', 'unsupervised_smoothing', 'grad_from_block'):
                self.assertEqual(baseline[key], source[dataset][key])
            self.assertEqual(baseline['max_steps_per_epoch'], 0)
        for dataset, counts in (('scars', (98, 98)), ('cub', (100, 100)), ('aircraft', (50, 50)), ('cifar100', (80, 20))):
            path = SOURCE_ROOT/f'data/ssb_splits/{dataset}_osr_splits.pkl'
            report = check_split(path, dataset)
            self.assertEqual((report['known_count'], report['novel_count']), counts)

    def test_invalid_dataset_and_report_schedule_fail(self):
        cfg = load_config(ROOT/'configs/selex_scars_baseline_200.json')
        for change in (dict(dataset_name='typo'), dict(train_report_interval=0),
                       dict(train_report_interval=True), dict(epochs=0)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(dict(cfg, **change))

    def test_cli_supported_dry_runs_match_full_and_smoke_dataset(self):
        decoder = json.JSONDecoder()
        for dataset in ('scars', 'cub', 'aircraft', 'cifar100'):
            for arm in ('baseline', 'crish'):
                command = ['bash', 'scripts/train_selex.sh', '--dataset', dataset, '--dry-run', '--seed', '2']
                if arm == 'baseline':
                    command.append('--baseline')
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                rest, objects = result.stdout.strip(), []
                while rest:
                    value, end = decoder.raw_decode(rest)
                    objects.append(value)
                    rest = rest[end:].lstrip()
                self.assertEqual(objects[0]['dataset_name'], dataset)
                self.assertEqual(objects[0]['epochs'], 200)
                self.assertEqual(len(objects), 3)
                for stage, epochs in zip(objects[1:], (1, 200)):
                    cfg = stage['config']
                    self.assertEqual((cfg['dataset_name'], cfg['method'], cfg['seed']), (dataset, arm, 2))
                    self.assertEqual(cfg['epochs'], epochs)
                    self.assertNotIn('c', cfg)

    def test_cifar_unlabeled_targets_are_unreadable_and_pixels_are_owned(self):
        class Unlabeled:
            data = np.ones((2, 32, 32, 3), dtype=np.uint8)
            uq_idxs, transform = [3, 7], None
            @property
            def targets(self):
                raise AssertionError('Hidden labels read')
            @property
            def target_transform(self):
                raise AssertionError('Hidden label transform read')
        labeled = SimpleNamespace(data=np.zeros((2, 32, 32, 3), dtype=np.uint8),
            targets=[0, 79], uq_idxs=[1, 4], target_transform=lambda x: x, transform=None)
        original = Unlabeled()
        safe = sanitize_training(SimpleNamespace(labelled_dataset=labeled, unlabelled_dataset=original), 'cifar100')
        self.assertEqual([r.target for r in safe.labelled_dataset.rows], [0, 79])
        self.assertEqual([r.target for r in safe.unlabelled_dataset.rows], [-1, -1])
        self.assertEqual(safe.unlabelled_dataset[0][0].size, (32, 32))
        original.data[:] = 99
        self.assertEqual(np.asarray(safe.unlabelled_dataset[0][0])[0, 0, 0], 1)
        self.assertFalse(safe.unlabelled_dataset.pixels.flags.writeable)
        self.assertFalse(hasattr(safe.unlabelled_dataset, 'targets'))
        labeled.targets[0] = 80
        with self.assertRaisesRegex(ValueError, 'known classes'):
            sanitize_training(SimpleNamespace(labelled_dataset=labeled, unlabelled_dataset=original), 'cifar100')
        labeled.targets[0] = 0
        original.uq_idxs = [1, 7]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            sanitize_training(SimpleNamespace(labelled_dataset=labeled, unlabelled_dataset=original), 'cifar100')

    def test_cifar_audit_rejects_overlap_and_bad_alignment(self):
        def pixels(ids):
            rgb = np.repeat((np.asarray(ids) % 251).astype(np.uint8)[:, None], 3, axis=1)
            return np.broadcast_to(rgb[:, None, None, :], (len(ids), 32, 32, 3))
        ids = np.arange(50000)
        truth = np.repeat(np.arange(100), 500)
        class Pool(SimpleNamespace):
            def __len__(self): return len(self.targets)
        whole = Pool(data=pixels(ids), targets=truth, uq_idxs=ids, train=True)
        test = Pool(data=pixels(np.arange(10000)), targets=np.repeat(np.arange(100), 100),
                    uq_idxs=np.arange(10000), train=False)
        labeled_ids = np.arange(0, 40000, 2)
        unlabeled_ids = np.setdiff1d(ids, labeled_ids)
        def subset(index):
            return Pool(data=pixels(index), targets=truth[index], uq_idxs=index.copy())
        merged = SimpleNamespace(labelled_dataset=subset(labeled_ids), unlabelled_dataset=subset(unlabeled_ids))
        args = SimpleNamespace(train_classes=range(80), unlabeled_classes=range(80, 100))
        with patch('gcd_runtime.readiness.whole_training_dataset', return_value=whole):
            result = audit_cifar_membership(merged, test, args, Path('/unused'))
            self.assertEqual((result['labeled'], result['unlabeled'], result['disjoint_test']), (20000, 30000, 10000))
            merged.unlabelled_dataset.uq_idxs[0] = labeled_ids[0]
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                audit_cifar_membership(merged, test, args, Path('/unused'))
            merged.unlabelled_dataset = subset(unlabeled_ids)
            merged.labelled_dataset.targets[0] = 99
            with self.assertRaisesRegex(RuntimeError, 'alignment'):
                audit_cifar_membership(merged, test, args, Path('/unused'))

    def test_fresh_pinned_cifar_split_and_offline_adapter(self):
        # Exercise the actual source split/merge functions with tiny RGB fixtures.
        code = r'''
import sys, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.training_data import sanitize_training
from selex_runtime.reference import configure_environment, definitions
from selex_runtime.trainer import make_args
cfg=load_config(ROOT/'configs/selex_cifar100_baseline_200.json')
configure_environment(Path(sys.argv[1]),cfg)
from data.cifar import CustomCIFAR100
from data.get_datasets import get_class_splits,get_datasets
args=get_class_splits(make_args(cfg))
assert list(args.train_classes)==list(range(80))
assert list(args.unlabeled_classes)==list(range(80,100))
with patch.object(CustomCIFAR100,'_check_integrity',return_value=False):
 try: CustomCIFAR100(root=sys.argv[1],download=True)
 except RuntimeError as e: assert 'automatic download disabled' in str(e)
 else: raise AssertionError('Missing inputs should fail offline')
class TwoViews:
 def __call__(self,image):
  tensor=ToTensor()(image)
  return [tensor,tensor+.001]
def fixture_init(self,root,transform=None,train=True,**unused):
 self.root,self.transform,self.target_transform,self.train=root,transform,None,train
 self.targets=np.repeat(np.arange(100),6 if train else 1).tolist()
 self.data=np.zeros((len(self.targets),32,32,3),dtype=np.uint8)
 self.uq_idxs=np.arange(len(self.targets))
with patch.object(CustomCIFAR100,'__init__',fixture_init):
 merged,test,eval_unlabeled,_=get_datasets('cifar100',TwoViews(),ToTensor(),args)
 safe=sanitize_training(merged,'cifar100')
 assert len(safe.labelled_dataset)==240 and len(safe.unlabelled_dataset)==360
 assert len(set(safe.sample_ids))==600 and len(test)==100
 for i in range(len(safe.labelled_dataset)):
  assert safe.labelled_dataset[i][1:]==merged.labelled_dataset[i][1:]
 assert all(row.target==-1 for row in safe.unlabelled_dataset.rows)
 views,targets,ids,mask=next(iter(DataLoader(safe,batch_size=4,num_workers=0)))
 model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3072,8))
 features=torch.stack([model(view) for view in views],dim=1)
 loss=definitions()['SupConLoss']()(features,labels=targets)
 loss.backward()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
print('Pinned CIFAR split/merge, sanitized RGB CPU backward and offline failure passed')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT,
                                    capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_epoch_200_plain_schedule_order_rng_and_metric_free_handoff(self):
        cfg = load_config(ROOT/'configs/selex_scars_baseline_200.json')
        for epoch, expected in ((0, False), (198, False), (199, True), (399, True)):
            self.assertEqual(train_plain_due(cfg, epoch), expected)
        state = capture_rng()
        try:
            calls = []
            def evaluate(model, dataset, args, runtime, **kwargs):
                calls.append(kwargs.get('pool', 'test'))
                # Different draws make reversing/skipping pools observable.
                torch.rand(2 if kwargs.get('pool') else 3)
                np.random.rand(2 if kwargs.get('pool') else 3)
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp)/'model.pt'
                checkpoint.write_bytes(b'synthetic snapshot')
                runtime = SimpleNamespace(run_dir=Path(tmp), config=cfg, evaluation_epoch=199,
                                          evaluated_checkpoint=checkpoint)
                from gcd_runtime.readiness import sha256
                request = create_request(runtime, 199, sha256(checkpoint))
                with patch('selex_runtime.trainer.evaluate_plain', side_effect=evaluate):
                    evaluate_plain_schedule(None, 'test', 'train', None, runtime)
                expected = rng_digest()
                self.assertEqual(calls, ['train_unlabeled', 'test'])
                calls.clear()
                begin_plain_worker(runtime)
                with patch('selex_runtime.trainer.evaluate_plain', side_effect=evaluate):
                    evaluate_plain_schedule(None, 'test', 'train', None, runtime)
                finish_plain_worker(runtime, request)
                torch.rand(7)
                commit_plain_result(runtime, request)
                self.assertEqual(rng_digest(), expected)
                runtime.evaluation_epoch = 198
                calls.clear()
                with patch('selex_runtime.trainer.evaluate_plain', side_effect=evaluate):
                    evaluate_plain_schedule(None, 'test', 'train', None, runtime)
                self.assertEqual(calls, ['test'])
        finally:
            restore_rng(state)

    def test_epoch_200_failed_monitor_resume_replays_both_pools(self):
        cfg = load_config(ROOT/'configs/selex_scars_baseline_200.json')
        cfg.update(device='cpu', synthetic_fixture=True, report_best_old_test=False)
        state = capture_rng()
        try:
            model, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                runtime = SelExRuntime(a, cfg)
                runtime.plain_rng_pending = True
                runtime.save(199, model, head, optimizer, scheduler)
                def evaluate(model, dataset, args, runtime, **kwargs):
                    torch.rand(2 if kwargs.get('pool') else 3)
                with patch('selex_runtime.trainer.evaluate_plain', side_effect=evaluate):
                    evaluate_plain_schedule(model, None, None, None,
                        SimpleNamespace(config=cfg, evaluation_epoch=199))
                expected = rng_digest()
                resumed = SelExRuntime(b, cfg, str(Path(a)/'resume.pt'))
                def recover(runtime, epoch, model):
                    runtime.evaluation_epoch = epoch
                    evaluate_plain_schedule(model, None, None, None, runtime)
                    runtime.plain_rng_pending = False
                with patch('selex_runtime.monitoring.monitor_plain', side_effect=recover), \
                     patch('selex_runtime.trainer.evaluate_plain', side_effect=evaluate) as monitored:
                    self.assertEqual(resumed.restore(model, head, optimizer, scheduler), 200)
                    self.assertEqual(monitored.call_count, 2)
                self.assertEqual(expected, rng_digest())
                self.assertTrue((Path(b)/'backbone_final.pt').is_file())
        finally:
            restore_rng(state)

    def test_fresh_selex_cub_loader_audit_and_sanitization(self):
        code = r'''
import os,sys
from pathlib import Path
from PIL import Image
from gcd_runtime.config import ROOT,load_config
from gcd_runtime.readiness import audit_dataset
from gcd_runtime.training_data import sanitize_training
from selex_runtime.reference import configure_environment
from selex_runtime.trainer import make_args
os.environ['GCD_DATA_ROOT']=sys.argv[1]
base=Path(sys.argv[1])/'cub/CUB_200_2011'
(base/'images').mkdir(parents=True)
(base/'attributes').mkdir()
tables={name:[] for name in ('images.txt','image_class_labels.txt','train_test_split.txt','attributes/image_attribute_labels.txt')}
for sample in range(1,601):
 filename=f'{sample}.png'
 Image.new('RGB',(2,2)).save(base/'images'/filename)
 tables['images.txt'].append(f'{sample} {filename}\n')
 tables['image_class_labels.txt'].append(f'{sample} {(sample-1)//3+1}\n')
 tables['train_test_split.txt'].append(f'{sample} {int(sample%3!=0)}\n')
 tables['attributes/image_attribute_labels.txt'].append(f'{sample} 1 1\n')
for filename,rows in tables.items(): (base/filename).write_text(''.join(rows))
cfg=load_config(ROOT/'configs/selex_cub_baseline_200.json')
configure_environment(Path(sys.argv[1]),cfg)
counts=audit_dataset(cfg)
assert counts==dict(labeled=100,unlabeled=300,disjoint_test=200,known_classes=100,novel_classes=100),counts
from data.get_datasets import get_class_splits,get_datasets
args=get_class_splits(make_args(cfg))
merged,_,_,_=get_datasets('cub',None,None,args)
safe=sanitize_training(merged,'cub')
for i in range(len(safe.labelled_dataset)):
 assert safe.labelled_dataset[i][1:]==merged.labelled_dataset[i][1:]
assert all(row.target==-1 for row in safe.unlabelled_dataset.rows)
assert len(set(safe.sample_ids))==400
print('Pinned SelEx CUB audit and sanitized labels passed')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT,
                                    capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_epoch_200_monitor_requires_both_reports_before_rng_commit(self):
        from selex_runtime.monitoring import monitor_plain
        cfg = load_config(ROOT/'configs/selex_scars_baseline_200.json')
        with tempfile.TemporaryDirectory() as tmp:
            runtime = SimpleNamespace(run_dir=Path(tmp), config=cfg)
            model = torch.nn.Linear(2, 2)
            def test_report_only(*unused, **kwargs):
                folder = Path(tmp)/'test_evaluations'
                folder.mkdir()
                (folder/'epoch_0200.json').write_text('{}')
            with patch('selex_runtime.monitoring.subprocess.run', side_effect=test_report_only), \
                 patch('selex_runtime.rng.commit_plain_result') as commit:
                with self.assertRaisesRegex(RuntimeError, 'train-plain'):
                    monitor_plain(runtime, 199, model)
                commit.assert_not_called()
            self.assertTrue(runtime.plain_rng_pending)


if __name__ == '__main__':
    unittest.main()
