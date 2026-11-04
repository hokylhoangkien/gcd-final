from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from audit.baseline_trace import baseline_oracle
from crish.candidates import Action
from crish.logging import AuditTrace
from crish.lookahead import gradient_gain
from crish.rebuild import confusion_factor, pseudo_sse_loss, rebuild

from .fixtures import state_fixture, backbone_views


class FiniteTests(unittest.TestCase):
    def test_zero_confusion_mass_fails_before_nan_without_mutation(self):
        state = state_fixture()
        saved = state.copy()
        with self.assertRaisesRegex(ValueError, 'zero/nonfinite'):
            confusion_factor(state, state.ids, torch.zeros_like(backbone_views(state)))
        self.assertTrue(torch.equal(state.prototypes[0], saved.prototypes[0]))
        with self.assertRaises(ValueError):
            replace(state, features=state.features * float('nan'))

    def test_all_unlabeled_items_abstain_with_identity_positives_finite(self):
        state = state_fixture()
        for row in (~state.labeled).nonzero().flatten():
            state, _ = rebuild(state, Action('abstain', int(state.ids[row]), 0,
                                            int(state.labels[0][row]), version=state.version))
        oracle = baseline_oracle()
        from hyptorch.nn import ToPoincare
        raw = torch.randn(8, 2, 6, generator=torch.Generator().manual_seed(93), requires_grad=True)
        features = ToPoincare(c=.1, clip_r=2., riemannian=False)(raw)
        self.assertTrue((features.norm(dim=-1) < .1**-.5).all())
        loss = pseudo_sse_loss(oracle['SupConLoss'](hyper_c=.1), features, state, state.ids, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_zero_changed_mass_is_not_fake_normalized_evidence(self):
        gain = gradient_gain([torch.ones(2)], [torch.ones(2)], [torch.zeros(2)], changed_mass=0.)
        self.assertIsNone(gain['per_mass'])
        with self.assertRaises(ValueError):
            gradient_gain([torch.ones(2)], [torch.ones(3)], [torch.ones(2)], changed_mass=1.)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            huge = torch.full((2,), 1e300, dtype=torch.float64)
            gradient_gain([huge], [huge], [torch.zeros(2)], changed_mass=1.)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            gradient_gain([torch.ones(2)], [torch.ones(2)], [torch.zeros(2)], changed_mass=1e-320)

    def test_trace_measures_cost_and_refuses_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.jsonl'
            trace = AuditTrace(path)
            with trace.measure('rebuild'):
                sum(range(100))
            trace.append(dict(status='keep'))
            original = path.read_text()
            self.assertGreaterEqual(json.loads(original)['measured_seconds']['rebuild'], 0.)
            with self.assertRaises(ValueError):
                trace.append(dict(gain=float('nan')))
            self.assertEqual(path.read_text(), original)
