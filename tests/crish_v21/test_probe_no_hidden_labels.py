import inspect
import unittest

import torch

from crish.probe import LegalProbe, ProbeWeights

from .fixtures import probe_fixture


class ProbeSafetyTests(unittest.TestCase):
    def kwargs(self):
        return dict(known_ids=torch.tensor([100, 101]), known_views=torch.zeros(2, 2, 3),
                    known_targets=torch.tensor([0, 1]), unlabeled_ids=torch.tensor([200, 201]),
                    unlabeled_views=torch.zeros(2, 2, 3), teacher_views=torch.ones(2, 2, 3),
                    proposal_ids=torch.tensor([0, 1]), legal_labels={100: 0, 101: 1},
                    weights=ProbeWeights(1., 1., .1, .5))

    def test_hidden_labels_loaders_and_callbacks_are_not_probe_inputs(self):
        params = inspect.signature(LegalProbe).parameters
        self.assertNotIn('hidden_labels', params)
        for extra in ('hidden_labels', 'unlabeled_targets', 'loader', 'callback'):
            with self.assertRaises(TypeError):
                LegalProbe(**self.kwargs(), **{extra: torch.tensor([-123, 999])})
        # A changing external oracle store cannot affect a probe that never reads it.
        model = torch.nn.Linear(3, 3)
        probe = LegalProbe(**self.kwargs())
        oracle_store = torch.tensor([0, 1])
        before = probe.loss(model).detach()
        oracle_store.fill_(987654)
        self.assertTrue(torch.equal(before, probe.loss(model).detach()))

    def test_ids_disjoint_registry_correct_and_teacher_detached(self):
        for change in (dict(proposal_ids=torch.tensor([200])),
                       dict(known_targets=torch.tensor([1, 0])),
                       dict(unlabeled_ids=torch.tensor([100, 201]))):
            args = dict(self.kwargs(), **change)
            with self.assertRaises(ValueError):
                LegalProbe(**args)
        teacher = torch.ones(2, 2, 3, requires_grad=True)
        probe = LegalProbe(**dict(self.kwargs(), teacher_views=teacher))
        teacher.data.zero_()
        self.assertTrue(probe.teacher_views.eq(1).all())
        self.assertFalse(probe.teacher_views.requires_grad)
        self.assertIsNone(teacher.grad)

    def test_separate_confirmation_folds_and_finite_component_gradients(self):
        model = torch.nn.Linear(3, 4)
        probe = probe_fixture(model)
        other = probe_fixture(model, offset=1000)
        probe.assert_disjoint(other)
        with self.assertRaises(ValueError):
            probe.assert_disjoint(probe)
        probe.loss(model).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(set(probe.components(model)), {'known', 'view', 'variance'})

    def test_model_callback_hooks_are_rejected_before_they_execute(self):
        model = torch.nn.Linear(3, 4)
        probe = probe_fixture(model)
        seen = []
        handle = model.register_forward_hook(lambda *args: seen.append('external callback'))
        try:
            with self.assertRaisesRegex(ValueError, 'callback hooks'):
                probe.loss(model)
            self.assertEqual(seen, [])
        finally:
            handle.remove()
