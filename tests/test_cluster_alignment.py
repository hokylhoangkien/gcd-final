import unittest
import torch
from gcd_runtime.cluster_alignment import restore_input_order


class ClusterAlignmentTests(unittest.TestCase):
    def test_interleaved_rows_and_prefix_identity(self):
        grouped = torch.tensor([10, 30, 50, 20, 40])
        mask = torch.tensor([True, False, True, False, True])
        self.assertEqual(restore_input_order(grouped, mask).tolist(), [10, 20, 30, 40, 50])
        for mask in (torch.tensor([True, True, False, False, False]),
                     torch.ones(5, dtype=torch.bool), torch.zeros(5, dtype=torch.bool)):
            self.assertTrue(torch.equal(restore_input_order(grouped, mask), grouped))
        with self.assertRaises(ValueError):
            restore_input_order(grouped, torch.ones(5))
        with self.assertRaises(ValueError):
            restore_input_order(grouped, torch.ones(4, dtype=torch.bool))
