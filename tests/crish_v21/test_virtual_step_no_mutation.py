import copy
import random
import unittest

import numpy as np
import torch

from crish.lookahead import exact_sgd_comparison, gradient_gain

from .fixtures import probe_fixture


def assert_recursive_equal(test, first, second):
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_recursive_equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_recursive_equal(test, a, b)
    else:
        test.assertEqual(first, second)


class VirtualStepTests(unittest.TestCase):
    def fixture(self):
        with torch.random.fork_rng():
            torch.manual_seed(8)
            model = torch.nn.Linear(3, 4, bias=False)
        return model, probe_fixture(model)

    def test_first_order_sign_matches_tiny_plain_sgd_on_legal_probe(self):
        model, probe = self.fixture()
        model.eval()
        q = torch.autograd.grad(probe.loss(model), tuple(model.parameters()))
        zero = tuple(torch.zeros_like(g) for g in q)
        gain = gradient_gain(q, q, zero, changed_mass=1.)
        opt = torch.optim.SGD(model.parameters(), lr=1e-5)
        exact = exact_sgd_comparison(model, opt, zero, q, probe)
        self.assertGreater(gain['raw'], 0)
        self.assertGreater(exact['gain'], 0)
        self.assertAlmostEqual(exact['gain'], opt.param_groups[0]['lr'] * gain['raw'], delta=2e-6)
        self.assertAlmostEqual(gain['cosine'], 1.)

    def test_exact_momentum_weight_decay_step_matches_independent_recurrence(self):
        model, probe = self.fixture()
        opt = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, nesterov=True, weight_decay=.02)
        for p in model.parameters():
            p.grad = torch.full_like(p, .1)
        opt.step()
        params = tuple(model.parameters())
        q = torch.autograd.grad(probe.loss(model), params)
        zero = tuple(torch.zeros_like(g) for g in q)
        before = copy.deepcopy((model.state_dict(), opt.state_dict()))
        grads_before = [p.grad.clone() for p in params]
        mode = model.training
        rng, py, numpy = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        observed = exact_sgd_comparison(model, opt, zero, q, probe)
        expected = []
        for gradients in (zero, q):
            independent = copy.deepcopy(model).eval()
            with torch.no_grad():
                for live, virtual, grad in zip(params, independent.parameters(), gradients):
                    d = grad + .02 * live
                    buffer = .9 * opt.state[live]['momentum_buffer'] + d
                    virtual.copy_(live - .001 * (d + .9 * buffer))
                expected.append(float(probe.loss(independent)))
        self.assertAlmostEqual(observed['gain'], expected[0] - expected[1], places=7)
        assert_recursive_equal(self, (model.state_dict(), opt.state_dict()), before)
        self.assertEqual(model.training, mode)
        for actual, original in zip(params, grads_before):
            self.assertTrue(torch.equal(actual.grad, original))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(py, random.getstate())
        np.testing.assert_equal(numpy, np.random.get_state())

    def test_no_action_and_invalid_gradients_leave_model_optimizer_unchanged(self):
        model, probe = self.fixture()
        opt = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        q = torch.autograd.grad(probe.loss(model), tuple(model.parameters()))
        before = copy.deepcopy((model.state_dict(), opt.state_dict()))
        self.assertEqual(exact_sgd_comparison(model, opt, q, q, probe)['gain'], 0.)
        with self.assertRaises(ValueError):
            exact_sgd_comparison(model, opt, q, [q[0] * float('nan')], probe)
        with self.assertRaises(TypeError):
            exact_sgd_comparison(model, opt, q, q, lambda m: torch.tensor(0.))
        assert_recursive_equal(self, (model.state_dict(), opt.state_dict()), before)
