"""Synthetic image evidence for RP2K preparation, leakage boundary and CLI."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from gcd_runtime.config import ROOT, load_config
from gcd_runtime.rp2k import (prepare, prepare_smoke, audit, digest, load_manifest,
                              read_rows, class_splits, datasets, resolve_image)
from scripts.prepare_rp2k import extract, plan
from scripts.export_retail_backbone import export


def fixture(root, classes=4, count=8):
    from PIL import Image
    root = Path(root)
    source = root/'raw/all'
    for pool, samples in (('train', count), ('test', 2)):
        for cls in range(classes):
            directory = source/pool/f'SKU-{cls:03d}'
            directory.mkdir(parents=True)
            for sample in range(samples):
                Image.new('RGB', (4, 4), (cls, sample, 10 if pool == 'train' else 200)).save(directory/f'{sample}.png')
    return source


class RP2KTests(unittest.TestCase):
    def test_deterministic_split_labels_ids_and_exact_duplicate_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'rp2k'
            source = fixture(root)
            duplicate = source/'test/SKU-000/duplicate.png'
            duplicate.write_bytes((source/'train/SKU-000/0.png').read_bytes())
            manifest = prepare(root, source)
            report = audit(root, rehash=True)
            self.assertEqual(manifest['removed_duplicates'], {'train->test': 1})
            self.assertEqual((report['known_classes'], report['novel_classes']), (2, 2))
            self.assertEqual((report['labeled'], report['unlabeled'], report['disjoint_test']), (8, 24, 8))
            public, truth = read_rows(root, 'train.jsonl'), read_rows(root, 'train_evaluation.jsonl')
            self.assertEqual(sum(r['target'] == -1 for r in public), 24)
            self.assertEqual([r['sample_id'] for r in public], [r['sample_id'] for r in truth])
            other = Path(tmp)/'copy'
            second = prepare(other, fixture(other))
            self.assertEqual(manifest['classes'], second['classes'])
            self.assertEqual([(r['sample_id'], r['target']) for r in public],
                             [(r['sample_id'], r['target']) for r in read_rows(other, 'train.jsonl')])
            with self.assertRaises(FileExistsError): prepare(root, source)

    def test_conflicting_duplicates_and_ineligible_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = fixture(root, classes=5)
            for p in (source/'test/SKU-004').glob('*.png'): p.unlink()
            manifest = prepare(root, source)
            self.assertEqual(manifest['class_count'], 4)
            self.assertEqual(manifest['excluded_classes'][0]['sku'], 'SKU-004')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = fixture(root)
            (source/'test/SKU-001/conflict.png').write_bytes((source/'train/SKU-000/0.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'conflicting'): prepare(root, source)

    def test_manifest_index_image_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root, fixture(root))
            expected = digest(root/'manifest.json')
            path = root/'train.jsonl'
            original = path.read_text()
            path.write_text(original+'\n')
            with self.assertRaisesRegex(ValueError, 'index fingerprint'): load_manifest(root, expected)
            path.write_text(original)
            with self.assertRaisesRegex(ValueError, 'manifest fingerprint'): load_manifest(root, '0'*64)
            row = read_rows(root, 'train.jsonl')[0]
            resolve_image(root, row['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'image changed'): audit(root)
            with self.assertRaisesRegex(ValueError, 'escapes'): resolve_image(root, '../outside.png')

    def test_training_cannot_open_withheld_truth_and_cpu_backward(self):
        import torch
        from torch.utils.data import DataLoader
        from torchvision.transforms import ToTensor
        from gcd_runtime.training_data import sanitize_training
        from selex_runtime.reference import SOURCE_ROOT, definitions
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, GCD_DATA_ROOT=tmp):
            root = Path(tmp)/'rp2k'
            prepare(root, fixture(root))
            args = SimpleNamespace(dataset_name='rp2k', rp2k_manifest_sha256=digest(root/'manifest.json'))
            class_splits(args)
            # A training constructor must still work when evaluation truth is
            # physically unavailable, not just when it promises to ignore it.
            for name in ('train_evaluation.jsonl', 'test.jsonl'):
                (root/name).rename(root/(name+'.withheld'))
            class TwoViews:
                def __call__(self, image):
                    x = ToTensor()(image)
                    return [x, x+.001]
            with patch.dict(sys.modules):
                sys.path.insert(0, str(SOURCE_ROOT))
                try:
                    merged, test, other, _ = datasets(TwoViews(), ToTensor(), args)
                    safe = sanitize_training(merged, 'rp2k')
                finally:
                    sys.path.remove(str(SOURCE_ROOT))
            self.assertIsNone(test); self.assertIsNone(other)
            self.assertTrue(all(row.target == -1 for row in safe.unlabelled_dataset.rows))
            views, labels, _, mask = next(iter(DataLoader(safe, batch_size=8, num_workers=0)))
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 8))
            features = torch.stack([torch.nn.functional.normalize(model(v), dim=-1) for v in views], dim=1)
            loss = definitions('cpu')['SupConLoss']()(features, labels=labels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_smoke_plan_config_and_cli_freeze_separate_data_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, GCD_DATA_ROOT=tmp):
            root = Path(tmp)/'rp2k'
            prepare(root, fixture(root, classes=16, count=48))
            smoke = prepare_smoke(root)
            self.assertEqual(smoke['counts']['train'], 16*48)
            cfg_path = Path(tmp)/'plan/crish.json'
            result = plan(root, cfg_path)
            cfg = load_config(cfg_path)
            self.assertFalse(result['training_started'])
            self.assertEqual(cfg['batch_size'], 32)
            self.assertTrue(cfg['unbalanced'])
            self.assertTrue(cfg['crish']['enabled'])
            self.assertFalse(cfg['population_policy']['enabled'])
            self.assertFalse(cfg['pair_reliability']['enabled'])
            self.assertNotEqual(cfg['rp2k_manifest_sha256'], cfg['rp2k_smoke_manifest_sha256'])
            env = dict(os.environ, SELEX_PYTHON=sys.executable)
            command = ['bash', 'scripts/train_selex.sh', '--config', str(cfg_path), '--variant', 'local', '--dry-run']
            proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
            text, stages = proc.stdout.strip(), []
            while text:
                item, end = json.JSONDecoder().raw_decode(text)
                stages.append(item); text = text[end:].lstrip()
            self.assertEqual([s['config']['rp2k_subdir'] for s in stages[1:]], ['rp2k/smoke', 'rp2k'])
            self.assertEqual([s['config']['epochs'] for s in stages[1:]], [1, 200])
            with self.assertRaises(FileExistsError): plan(root, cfg_path)

    def test_extract_rejects_zip_escape_and_existing_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/'downloads/rp2k_dataset.zip'
            archive.parent.mkdir()
            with zipfile.ZipFile(archive, 'w') as zipped: zipped.writestr('../outside.txt', 'escape')
            with self.assertRaisesRegex(ValueError, 'Unsafe'): extract(root)
            self.assertFalse((root/'raw').exists())
            (root/'raw').mkdir()
            with self.assertRaises(FileExistsError): extract(root)

    def test_readiness_and_evaluation_dispatch_in_fresh_process(self):
        code = r'''
import os,sys,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from tests.test_rp2k_runtime import fixture
from gcd_runtime.config import ROOT,load_config
from gcd_runtime.rp2k import prepare,digest
from gcd_runtime.readiness import inspect_inputs,audit_dataset
from selex_runtime.reference import configure_environment
from selex_runtime.data import get_class_splits,get_datasets
from selex_runtime.trainer import make_args,SelExRuntime
root=Path(sys.argv[1])
os.environ['GCD_DATA_ROOT']=str(root/'datasets')
os.environ['GCD_WEIGHTS_ROOT']=str(root/'weights')
data=root/'datasets/rp2k'
prepare(data,fixture(data))
(root/'weights').mkdir()
(root/'weights/dinov2_vitb14_reg4_pretrain.pth').write_bytes(b'fingerprint-only-not-real-weights')
cfg=json.loads((ROOT/'configs/selex_rp2k_crish_200.json').read_text())
cfg['rp2k_manifest_sha256']=digest(data/'manifest.json')
cfg['rp2k_smoke_manifest_sha256']='b'*64
config=root/'config.json';config.write_text(json.dumps(cfg));cfg=load_config(config)
configure_environment(root,cfg)
report=inspect_inputs(cfg)
assert report['split']['kind']=='custom_rp2k_gcd_not_ssb'
counts=audit_dataset(cfg)
assert (counts['known_classes'],counts['novel_classes'])==(2,2),counts
args=get_class_splits(make_args(cfg));args.rp2k_evaluation=True
merged,test,unlabeled,_=get_datasets('rp2k',None,None,args)
assert len(test)==8 and len(unlabeled)==24
assert all(target>=0 for _,target in merged.unlabelled_dataset.samples)
args.rp2k_evaluation=False
training,none,_,_=get_datasets('rp2k',None,None,args)
assert none is None and all(t==-1 for _,t in training.unlabelled_dataset.samples)
runtime=SelExRuntime(root,cfg)
with patch('selex_runtime.monitoring.monitor_plain',side_effect=AssertionError('Test monitor must not run')):
 runtime.monitor_epoch(0,object())
cfg['rp2k_manifest_sha256']=None
try: inspect_inputs(cfg)
except RuntimeError as exc: assert 'manifest-pinned' in str(exc)
else: raise AssertionError('Unpinned retail data was accepted')
print('RP2K readiness, withheld evaluation and no per-epoch test monitor passed')
'''
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT,
                                  capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)

    def test_export_contract_and_reject_incomplete_or_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'; run.mkdir()
            cfg = json.loads((ROOT/'configs/selex_rp2k_crish_200.json').read_text())
            cfg['rp2k_manifest_sha256'] = 'a'*64
            cfg['rp2k_smoke_manifest_sha256'] = 'b'*64
            (run/'config.json').write_text(json.dumps(cfg))
            (run/'backbone_final.pt').write_bytes(b'packaging-fixture-not-model-weights')
            (run/'metadata.json').write_text(json.dumps(dict(status='failed')))
            with self.assertRaisesRegex(ValueError, 'not completed'): export(run, Path(tmp)/'bad')
            (run/'metadata.json').write_text(json.dumps(dict(status='completed', git_sha='fixture')))
            contract = export(run, Path(tmp)/'export')
            self.assertEqual(contract['vectorDimension'], 768)
            self.assertIsNone(contract['recognition_threshold'])
            self.assertEqual(contract['modelVersion'], digest(run/'backbone_final.pt'))
            cfg['max_steps_per_epoch'] = 2
            (run/'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'smoke'): export(run, Path(tmp)/'smoke')


if __name__ == '__main__':
    unittest.main()
