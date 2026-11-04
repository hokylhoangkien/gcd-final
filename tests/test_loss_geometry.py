import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from gcd_runtime.checkpoint import EpochRuntime
from gcd_runtime.training_diagnostics import loss_geometry


class LossGeometryTests(unittest.TestCase):
    def test_membership_split_conflict_and_endpoint_weights(self):
        features = torch.ones(4, 2, requires_grad=True)
        # Labeled rows 0,2 agree; unlabeled rows 1,3 oppose.
        angle = features.sum()
        distance = (features * torch.tensor([1., -1., 1., -1.])[:, None]).sum()
        mask = torch.tensor([True, False])
        result = loss_geometry(angle, distance, features, mask, .75)
        self.assertAlmostEqual(result['groups']['labeled']['cosine'], 1)
        self.assertAlmostEqual(result['groups']['unlabeled']['cosine'], -1)
        self.assertAlmostEqual(result['groups']['all']['cosine'], 0)
        self.assertAlmostEqual(result['groups']['all']['distance_norm_share'], .75)
        self.assertEqual(loss_geometry(angle, distance, features, mask, 0.)['groups']['all']['weighted_distance_norm'], 0)
        self.assertIsNone(features.grad)

    def test_rng_and_parameter_update_are_identical_with_diagnostics(self):
        torch.manual_seed(17)
        model = torch.nn.Linear(3, 4)
        inputs = torch.arange(12).reshape(4, 3).float()/12
        labeled = torch.tensor([True, False])
        results = []
        for logging in (False, True):
            model.zero_grad()
            z = model(inputs)
            a, d = z.square().sum(), (z-.2).square().sum()
            rng = torch.get_rng_state().clone()
            python_rng = random.getstate()
            if logging:
                loss_geometry(a, d, z, labeled, .6)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(python_rng, random.getstate())
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            (.4*a+.6*d).backward()
            results.append([p.detach() - .01*p.grad for p in model.parameters()])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*results)))

    def test_runtime_sampling_zero_gradient_and_empty_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = EpochRuntime(tmp, {'training_diagnostics_every': 2, 'synthetic_fixture': True})
            z = torch.ones(4, 2, requires_grad=True)
            mask = torch.zeros(2, dtype=torch.bool)
            for batch in (0, 1, 2):
                runtime.begin_batch(4, batch)
                runtime.log_loss_geometry(z.sum(), (z*0).sum(), z, mask, 1.)
            rows = [json.loads(s) for s in (Path(tmp)/'loss_geometry.jsonl').read_text().splitlines()]
        self.assertEqual([r['batch'] for r in rows], [0, 2])
        self.assertEqual(rows[0]['groups']['labeled']['rows'], 0)
        self.assertIsNone(rows[0]['groups']['all']['cosine'])
        self.assertIsNone(rows[0]['groups']['all']['distance_norm_share'])
        self.assertEqual(rows[0]['status'], 'synthetic_fixture')

    def test_invalid_membership_and_nonfinite_loss(self):
        z = torch.ones(4, 2, requires_grad=True)
        for mask in (torch.ones(2), torch.ones(3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                loss_geometry(z.sum(), z.sum(), z, mask, .5)
        with self.assertRaises(FloatingPointError):
            loss_geometry(z.sum()*float('nan'), z.sum(), z, torch.ones(2, dtype=torch.bool), .5)
