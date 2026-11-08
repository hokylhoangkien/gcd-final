"""Integrated strict B0/CRiSH training fixtures; no images/weights/GPU needed."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gcd_runtime.config import ROOT, load_config
from gcd_runtime.checkpoint import EpochRuntime
from crish.config import CrishSettings
from crish.smoke import FeatureFixture, crish_method_probe
from crish.loss import hypselex_loss
from gcd_runtime.worker import configure_environment
from gcd_runtime.training_data import sanitize_cars_training
from crish.ids import rows_for_ids
from crish.lookahead import clone_snapshot, exact_sgd_comparison
from crish.probe import LegalProbe, ProbeWeights
from crish.rebuild import prototype_radii, rebuild
from crish.candidates import Action
from crish.state import SupervisionState


def configure_trainer():
    configure_environment(ROOT/'runs/probe-placeholder')
    from gcd_runtime import trainer
    return trainer


def args_fixture(epochs):
    return SimpleNamespace(lr=.002, momentum=.9, weight_decay=5e-5, epochs=epochs, c=.1,
          n_views=2, temperature=1., contrast_unlabel_only=False, strategy='zero_one',
          unsupervised_smoothing=.5, sup_con_weight=.35, hyper_start_epoch=0,
          hyper_end_epoch=200, hyper_max_weight=1., distance='euclidean', cluster_momentum=1.,
          num_labeled_classes=2, num_unlabeled_classes=2)


def state_from(model, dataset, pseudo):
    with torch.no_grad():
        features = F.normalize(model(dataset.features), dim=-1).cpu()
    levels = (pseudo, pseudo // 2)
    centers = tuple(torch.stack([features[y == g].mean(0) for g in y.unique()]) for y in levels)
    return SupervisionState(torch.tensor(dataset.sample_ids), features, dataset.labeled,
                            dataset.targets[dataset.labeled], levels, centers,
                            tuple(prototype_radii(p) for p in centers),
                            tuple(torch.zeros(len(dataset), dtype=torch.bool) for _ in levels))


def training_config(method='crish', enabled=True):
    cfg = load_config(ROOT/'configs/crish_smoke.json')
    cfg.update(method=method, epochs=2, device='cpu', synthetic_fixture=True,
               max_steps_per_epoch=2, mlp_out_dim=16, checkpoint_selection='final_epoch',
               test_every_epochs=0, final_ss_evaluation=False)
    cfg['crish'] = asdict(CrishSettings(enabled=enabled, warmup_epochs=0, refresh_every=1,
              evidence_samples=20, evidence_trials=1, k=2, boundaries=1, max_targets=1,
              max_accepted=1, min_support=1, min_edges=1, train_batch=8,
              probe_known=2, probe_unlabeled=2, max_protected_shift=1.,
              max_refresh_seconds=20, save_partitions=True))
    return cfg


def run_fixture(directory, cfg, *, seed=71, poison=-1, resume=None, stop=False):
    trainer = configure_trainer()
    from hyptorch.nn import ToPoincare
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(15)
    raw = torch.randn(40, 4, generator=generator)
    pseudo = torch.arange(40) % 4
    labeled = torch.zeros(40, dtype=torch.bool)
    labeled[[0, 1, 4, 5, 8, 9, 12, 13]] = True
    dataset = FeatureFixture(raw, labeled, pseudo[labeled])
    dataset.targets[~labeled] = poison
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = torch.nn.Linear(4, 8)
    head = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Dropout(.1),
                               ToPoincare(c=.1, clip_r=2., riemannian=False))
    class StopAtEpochZero(EpochRuntime):
        def save(self, epoch, *args, **kwargs):
            super().save(epoch, *args, **kwargs)
            if epoch == 0:
                raise InterruptedError('Bounded synthetic checkpoint interruption')
    runtime = (StopAtEpochZero if stop else EpochRuntime)(directory, cfg, resume)
    args = args_fixture(cfg['epochs'])
    trainer.args, trainer.device, trainer._runtime = args, torch.device('cpu'), runtime
    def extract(*unused, **kwargs):
        model.eval()
        snapshot = state_from(model, dataset, pseudo)
        runtime.hierarchy_state = snapshot
        return snapshot.ids.numpy(), snapshot.labels[0], snapshot.prototypes, snapshot.labels, {}
    with patch.object(trainer, 'extract_labeled_protos', side_effect=extract), \
         patch.object(trainer, 'evaluate', side_effect=AssertionError('Oracle reached training')):
        trainer.train(head, model, loader, loader, args)
    return copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict())


class CrishTrainConfigTests(unittest.TestCase):
    def test_b0_and_crish_only_differ_by_method(self):
        for suffix in ('', '_smoke'):
            b0 = load_config(ROOT/f'configs/baseline{suffix}.json')
            crish = load_config(ROOT/f'configs/crish{suffix}.json')
            self.assertEqual({key for key in b0 if b0[key] != crish[key]}, {'method'})
            self.assertNotIn('reliability', crish)
            self.assertEqual(crish['training_protocol'], 'crish_v2_1')

    def test_invalid_or_mixed_config_fails_before_training(self):
        original = json.loads((ROOT/'configs/crish.json').read_text())
        changes = [dict(training_protocol='legacy'), dict(align_cluster_rows=False),
                   dict(test_every_epochs=0), dict(method='proposed'), dict(reliability={}),
                   dict(crish=dict(warmup_epochs=-1)), dict(max_steps_per_epoch=-2)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'invalid.json'
            for values in changes:
                path.write_text(json.dumps(dict(original, **values)))
                with self.subTest(values=values), self.assertRaises(ValueError):
                    load_config(path)

    def test_single_entrypoint_selects_crish_then_smoke_and_full(self):
        result = subprocess.run(['bash', 'scripts/train_crish.sh', '--dry-run'], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"selected_method": "crish"', result.stdout)
        self.assertIn('"selected_profile": "final"', result.stdout)
        self.assertIn('configs/crish_final_smoke.json', result.stdout)
        self.assertIn('configs/crish_final.json', result.stdout)
        self.assertNotIn('"method": "proposed"', result.stdout)


class TrainingDataTests(unittest.TestCase):
    def test_sanitizer_never_reads_unlabeled_targets_or_retains_source(self):
        labeled = SimpleNamespace(data=['a.jpg', 'b.jpg'], target=[6, 9], uq_idxs=np.array([40, 7]),
                                  target_transform=lambda value: {5: 0, 8: 1}[value], transform=None)
        class Unlabeled:
            data = ['c.jpg', 'd.jpg']
            uq_idxs = [99, 15]
            transform = None
            @property
            def target(self):
                raise AssertionError('Hidden targets accessed')
            @property
            def target_transform(self):
                raise AssertionError('Hidden class mapping retained')
        source = SimpleNamespace(labelled_dataset=labeled, unlabelled_dataset=Unlabeled())
        clean = sanitize_cars_training(source)
        self.assertEqual([r.target for r in clean.labelled_dataset.rows], [0, 1])
        self.assertEqual([r.target for r in clean.unlabelled_dataset.rows], [-1, -1])
        self.assertEqual(clean.sample_ids, (40, 7, 99, 15))
        self.assertNotIn('target_transform', vars(clean.unlabelled_dataset))
        self.assertNotIn(source, list(vars(clean).values()))

    def test_actual_hssk_capture_skips_oracle_metrics_and_aligns_sparse_ids(self):
        trainer = configure_trainer()
        images = torch.eye(8)
        predictions = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3])
        ids = torch.tensor([110, 7, 99, 15, 42, 66, 80, 12])
        membership = torch.tensor([[1]]*4+[[0]]*4)
        class Cluster:
            calls = []
            def __init__(self, k, **kwargs):
                self.k = k
            def fit_mix(self, unlabeled, labeled, targets):
                self.calls.append(targets.clone())
                rows = torch.cat([labeled, unlabeled]).argmax(1)
                self.labels_ = predictions[rows] if self.k == 4 else (predictions[rows] >= 2).long()
                self.cluster_centers_ = torch.stack([torch.cat([labeled, unlabeled])[self.labels_ == g].mean(0) for g in range(self.k)])
        class Coarse:
            def __init__(self, n_clusters, **kwargs):
                pass
            def fit(self, features):
                self.labels_ = np.zeros(len(features), dtype=int)
                return self
        captured = []
        with tempfile.TemporaryDirectory() as tmp:
            runtime = EpochRuntime(tmp, training_config('baseline'))
            args = SimpleNamespace(train_classes=[0, 1], num_labeled_classes=2,
                                   num_unlabeled_classes=2, unbalanced=False, eval_funcs=['v2'])
            with patch.object(trainer, '_runtime', runtime, create=True), \
                 patch.object(trainer, 'device', torch.device('cpu'), create=True), \
                 patch.object(trainer, 'SemiSupKMeans', Cluster), patch.object(trainer, 'KMeans', Coarse), \
                 patch.object(trainer, 'log_accs_from_preds', side_effect=AssertionError('Hidden labels reached metrics')):
                for poison in (-999, 888888):
                    trainer.extract_labeled_protos(torch.nn.Identity(),
                        [([images], torch.tensor([0, 1, 0, 1]+[poison]*4), ids, membership)], args)
                    captured.append(runtime.hierarchy_state)
        self.assertEqual(captured[0].ids.tolist(), ids.tolist())
        self.assertEqual(captured[0].known_targets.tolist(), [0, 1, 0, 1])
        self.assertTrue(torch.equal(captured[0].labels[1][:4], torch.zeros(4, dtype=torch.long)))
        for first, second in zip(captured[0].labels+captured[0].prototypes, captured[1].labels+captured[1].prototypes):
            self.assertTrue(torch.equal(first, second))


class OriginalDINOHeadTests(unittest.TestCase):
    def test_weight_norm_clone_exact_step_preserves_live_attributes_and_momentum(self):
        configure_trainer()
        from models.vision_transformer import DINOHead
        torch.manual_seed(42)
        head = DINOHead(in_dim=3, out_dim=8, hidden_dim=16, bottleneck_dim=4)
        opt = torch.optim.SGD(head.parameters(), lr=.001, momentum=.9, weight_decay=.01)
        known, other = torch.randn(2, 2, 3), torch.randn(3, 2, 3)
        head(known.flatten(0, 1)).square().mean().backward()
        opt.step()
        with torch.no_grad():
            teacher = head(other.flatten(0, 1)).reshape(3, 2, -1)
        probe = LegalProbe(known_ids=torch.tensor([100, 101]), known_views=known,
                known_targets=torch.tensor([0, 1]), unlabeled_ids=torch.tensor([200, 201, 202]),
                unlabeled_views=other, teacher_views=teacher, proposal_ids=torch.arange(10),
                legal_labels={100: 0, 101: 1}, weights=ProbeWeights(1., 1., .1, .1))
        parameters = tuple(p for p in head.parameters() if p.requires_grad)
        q = torch.autograd.grad(probe.loss(head), parameters)
        before = copy.deepcopy(head.state_dict())
        derived = head.last_layer.weight
        clone, cloned_opt = clone_snapshot(head, opt)
        self.assertIsNot(clone.last_layer.weight_v, head.last_layer.weight_v)
        self.assertIn(clone.last_layer.weight_v, cloned_opt.state)
        actual = exact_sgd_comparison(head, opt, q, q, probe)
        self.assertEqual(actual['gain'], 0.)
        self.assertIs(head.last_layer.weight, derived)
        for key in before:
            self.assertTrue(torch.equal(head.state_dict()[key], before[key]))

    def test_real_dino_head_full_loss_and_repair_scorer_probe(self):
        cfg = training_config()
        cfg['weight_decay'] = 5e-5
        result = crish_method_probe(cfg, 'cpu', backbone_dim=8, hidden_dim=32)
        self.assertEqual(result['status'], 'synthetic_strict_training_probe_passed')
        self.assertGreater(result['scoring_attempts'], 0)
        self.assertEqual(result['trained_epochs'], 0)


class StrictLossTests(unittest.TestCase):
    def test_full_loss_and_backbone_head_gradients_match_actual_source(self):
        trainer = configure_trainer()
        from audit.baseline_trace import baseline_oracle
        from hyptorch.nn import ToPoincare
        args = args_fixture(2)
        trainer.args, trainer.device = args, torch.device('cpu')
        torch.manual_seed(10)
        raw = torch.randn(12, 4)
        labels = torch.arange(12) % 4
        labeled = torch.tensor([True, True]+[False]*10)
        dataset = FeatureFixture(raw, labeled, labels[labeled])
        model = torch.nn.Linear(4, 8)
        head = torch.nn.Sequential(torch.nn.Linear(8, 16), ToPoincare(c=.1, clip_r=2., riemannian=False))
        state = state_from(model, dataset, labels)
        ids = state.ids[:8]
        features = model(torch.cat([raw[:8], raw[:8]+.01]))
        result = hypselex_loss(head, features, ids, labeled[:8], labels[:8][labeled[:8]], state, args, 1)
        original = baseline_oracle()
        original['args'] = args
        # Independent backbone forward for the source oracle: the two backward
        # comparisons must not share freed intermediate backbone activations.
        flat = head(model(torch.cat([raw[:8], raw[:8]+.01])))
        paired = torch.stack(flat.chunk(2), dim=1)
        cf = original['confusion_oracle'](features.detach(), [x[:8] for x in state.labels], state.prototypes, state.radii)
        branches = []
        for curvature in (0., .1):
            criterion = original['SupConLoss'](hyper_c=curvature)
            logits, targets, smooth = original['info_nce_logits'](flat, cf, args, hyper_c=curvature)
            use = original['LabelSmoothingLoss']()(logits, targets, smooth, .5)
            sse = criterion(paired[labeled[:8]], labels=labels[:8][labeled[:8]])
            for level in range(2):
                sse = sse + criterion(paired[:, :, :16//2**(level+1)], labels=state.labels[level][:8]) / 2**(level+1)
            branches.append(.65*use+.35*sse/2)
        weight = (1-(args.hyper_start_epoch-1))/(args.hyper_end_epoch-args.hyper_start_epoch)
        expected = (1-weight)*branches[0]+weight*branches[1]
        torch.testing.assert_close(result['loss'], expected, rtol=0, atol=0)
        parameters = tuple(model.parameters())+tuple(head.parameters())
        a, b = torch.autograd.grad(result['loss'], parameters), torch.autograd.grad(expected, parameters)
        for actual, source in zip(a, b):
            torch.testing.assert_close(actual, source, rtol=0, atol=0)


class StrictTrainingLoopTests(unittest.TestCase):
    def test_b0_and_disabled_crish_updates_match_without_oracle_metrics(self):
        with tempfile.TemporaryDirectory() as b0, tempfile.TemporaryDirectory() as disabled:
            first = run_fixture(b0, training_config('baseline'))
            second = run_fixture(disabled, training_config('crish', enabled=False))
            for a, b in zip(first, second):
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)
            for folder in (b0, disabled):
                rows = [json.loads(line) for line in (Path(folder)/'metrics.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row['All'] is None and row['Old'] is None and row['New'] is None for row in rows))
                self.assertTrue((Path(folder)/'resume.pt').is_file())
                self.assertTrue((Path(folder)/'backbone_final.pt').is_file())

    def test_enabled_path_scores_actions_and_is_hidden_label_invariant(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_fixture(first, training_config(), poison=123456)
            b = run_fixture(second, training_config(), poison=-98765)
            events = [json.loads(line) for line in (Path(first)/'crish.jsonl').read_text().splitlines()]
            self.assertTrue(any(event.get('gains') for event in events))
            self.assertTrue(all(event['lookahead_scope'] == 'projector' for event in events))
            for x, y in zip(a, b):
                for key in x:
                    self.assertTrue(torch.equal(x[key], y[key]), key)

    def test_enabled_epoch_resume_reproduces_uninterrupted_updates(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as interrupted, tempfile.TemporaryDirectory() as resumed:
            cfg = training_config()
            expected = run_fixture(full, cfg)
            with self.assertRaises(InterruptedError):
                run_fixture(interrupted, cfg, stop=True)
            actual = run_fixture(resumed, cfg, seed=991, resume=str(Path(interrupted)/'resume.pt'))
            for a, b in zip(expected, actual):
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_training_consumes_an_accepted_rebuilt_snapshot(self):
        # Controlled intervention tests the wiring into actual SGD, separately
        # from whether the real legal probe should accept this toy intervention.
        def accepted(engine, state, *args, **kwargs):
            row = int((~state.labeled).nonzero()[0])
            source = int(state.labels[0][row])
            target = (source+1) % len(state.prototypes[0])
            return rebuild(state, Action('move', int(state.ids[row]), 0, source, target))[0]
        with tempfile.TemporaryDirectory() as baseline, tempfile.TemporaryDirectory() as changed:
            first = run_fixture(baseline, training_config('baseline'))
            with patch('crish.engine.CrishEngine.prepare', new=accepted):
                second = run_fixture(changed, training_config())
            self.assertTrue(any(not torch.equal(a[key], b[key]) for a, b in zip(first, second) for key in a))
            rows = [json.loads(line) for line in (Path(changed)/'metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['repairs'] == 1 for row in rows))

    def test_oracle_metric_rejection_happens_before_checkpoint_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = EpochRuntime(tmp, training_config())
            with self.assertRaises(TypeError):
                runtime.save(0, None, None, None, None, all_acc=.99)
            self.assertFalse((Path(tmp)/'resume.pt').exists())
