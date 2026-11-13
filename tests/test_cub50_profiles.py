"""Isolated CUB 50 profiles match pinned corrected SelEx base settings."""
import ast
import copy
import json
import runpy
import subprocess
import sys
import unittest

import torch

from gcd_runtime.config import ROOT, load_config
from selex_runtime.reference import SOURCE_ROOT, COMMIT


class Cub50ProfilesTests(unittest.TestCase):
    def test_pair_differs_only_in_local_intervention(self):
        base = load_config(ROOT/'configs/selex_cub_baseline_50.json')
        local = load_config(ROOT/'configs/selex_cub_crish_50.json')
        for cfg, method in ((base, 'baseline'), (local, 'crish')):
            self.assertEqual(cfg['method'], method)
            self.assertEqual(cfg['crish']['enabled'], method == 'crish')
            self.assertFalse(cfg['population_policy']['enabled'])
            self.assertFalse(cfg['pair_reliability']['enabled'])
            self.assertEqual(cfg['ss_eval_every_epochs'], 1)
            self.assertTrue(cfg['report_best_old_test'])
            self.assertEqual(cfg['upstream_sha'], COMMIT)
            self.assertEqual(cfg['dataset_name'], 'cub')
            self.assertEqual(cfg['epochs'], 50)
            self.assertEqual(cfg['max_steps_per_epoch'], 0)
        a, b = copy.deepcopy(base), copy.deepcopy(local)
        a.pop('method'); b.pop('method')
        a['crish'].pop('enabled'); b['crish'].pop('enabled')
        self.assertEqual(a, b)

    def test_base_hyperparameters_match_corrected_cub_source(self):
        cfg = load_config(ROOT/'configs/selex_cub_crish_50.json')
        source = runpy.run_path(str(SOURCE_ROOT/'config.py'))['dataset_configs']['cub']
        for key in ('epochs', 'temperature', 'unsupervised_smoothing', 'grad_from_block'):
            self.assertEqual(cfg[key], source[key])
        tree = ast.parse((SOURCE_ROOT/'methods/contrastive_training/contrastive_training.py').read_text())
        defaults = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args
                    and isinstance(node.args[0], ast.Constant)):
                for keyword in node.keywords:
                    if keyword.arg == 'default':
                        try:
                            defaults[node.args[0].value] = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):
                            pass
        for key in ('batch_size', 'lr', 'momentum', 'weight_decay', 'sup_con_weight',
                    'num_workers', 'train_report_interval'):
            self.assertEqual(cfg[key], defaults['--'+key])
        self.assertEqual((cfg['dino'], cfg['mlp_out_dim'], cfg['level_interval']), ('v2', 65536, 10))
        self.assertEqual((cfg['ss_n_init'], cfg['ss_max_iter']), (10, 10))

    def test_cosine_matches_upstream_schedule_over_all_50_steps(self):
        # Execute only the scheduler constructor from pinned upstream, without
        # importing its training/data/GPU code or fabricating a benchmark.
        tree = ast.parse((SOURCE_ROOT/'methods/contrastive_training/contrastive_training.py').read_text())
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        statement = next(n for n in train.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'exp_lr_scheduler' for t in n.targets))
        from types import SimpleNamespace
        cfg = load_config(ROOT/'configs/selex_cub_crish_50.json')
        pa, pb = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
        oa = torch.optim.SGD([pa], lr=cfg['lr'])
        ob = torch.optim.SGD([pb], lr=cfg['lr'])
        ours = torch.optim.lr_scheduler.CosineAnnealingLR(oa, T_max=cfg['epochs'], eta_min=cfg['lr']*1e-3)
        scope = dict(optimizer=ob, args=SimpleNamespace(**cfg), lr_scheduler=torch.optim.lr_scheduler)
        exec(compile(ast.Module(body=[statement], type_ignores=[]), '<upstream scheduler>', 'exec'), scope)
        for _ in range(50):
            self.assertAlmostEqual(oa.param_groups[0]['lr'], ob.param_groups[0]['lr'], places=12)
            oa.step(); ob.step(); ours.step(); scope['exp_lr_scheduler'].step()
        self.assertAlmostEqual(oa.param_groups[0]['lr'], .0001, places=12)

    def test_cli_modes_use_50_config_and_matching_smoke(self):
        for method, flags in (('crish', ['--variant', 'local']), ('baseline', ['--baseline'])):
            path = ROOT/f'configs/selex_cub_{method}_50.json'
            result = subprocess.run([sys.executable, '-m', 'selex_runtime.cli', '--dataset', 'cub',
                                     '--config', str(path), *flags, '--dry-run'],
                                    capture_output=True, text=True, check=True, timeout=15)
            plans, rest = [], result.stdout.strip()
            while rest:
                plan, end = json.JSONDecoder().raw_decode(rest)
                plans.append(plan); rest = rest[end:].strip()
            self.assertEqual(plans[0]['epochs'], 50)
            self.assertEqual(plans[0]['discrete_repair'], method == 'crish')
            configs = [p['config'] for p in plans[1:]]
            self.assertEqual([c['epochs'] for c in configs], [1, 50])
            self.assertEqual([c['ss_eval_every_epochs'] for c in configs], [0, 1])
            self.assertTrue(all(c['dataset_name'] == 'cub' for c in configs))

    def test_50_profile_preserves_other_200_hyperparameters(self):
        old = load_config(ROOT/'configs/selex_cub_crish_200.json')
        new = load_config(ROOT/'configs/selex_cub_crish_50.json')
        self.assertEqual(old['epochs'], 200)
        self.assertEqual({k for k in old if old[k] != new[k]},
                         {'epochs', 'ss_eval_every_epochs', 'max_runtime_seconds',
                          'crish', 'population_policy'})
        a, b = copy.deepcopy(old['crish']), copy.deepcopy(new['crish'])
        a.pop('enabled'); b.pop('enabled')
        self.assertEqual(a, b)
