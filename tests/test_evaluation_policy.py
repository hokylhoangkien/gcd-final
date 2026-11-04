"""Final evaluation cannot perturb training RNG or module modes."""
import random
import unittest
import numpy as np
import torch
from gcd_runtime.evaluation_policy import evaluation_context


class EvaluationIsolationTests(unittest.TestCase):
    def test_rng_modes_and_exception_restoration(self):
        model = torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.2))
        model[0].eval()  # Deliberately mixed submodule modes must survive.
        states = [(m,m.training) for m in model.modules()]
        torch.manual_seed(3); np.random.seed(3); random.seed(3)
        before = torch.get_rng_state().clone(),np.random.get_state(),random.getstate()
        cfg = dict(evaluation_rng='isolated',seed=8)
        outputs = []
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with evaluation_context(cfg,5,'test',model):
                    model.eval()
                    outputs.append((torch.rand(2),np.random.rand(),random.random()))
                    raise RuntimeError('fixture')
            self.assertTrue(torch.equal(torch.get_rng_state(),before[0]))
            self.assertTrue(np.array_equal(np.random.get_state()[1],before[1][1]))
            self.assertEqual(np.random.get_state()[2:],before[1][2:])
            self.assertEqual(random.getstate(),before[2])
            self.assertTrue(all(m.training==mode for m,mode in states))
        self.assertTrue(torch.equal(outputs[0][0],outputs[1][0]))
        self.assertEqual(outputs[0][1:],outputs[1][1:])
