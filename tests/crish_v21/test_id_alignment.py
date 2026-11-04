import unittest

import numpy as np
import torch

from crish.ids import rows_for_ids


class IDAlignmentTests(unittest.TestCase):
    def test_shuffled_sparse_ids_and_repeated_queries(self):
        ids, query = [40, 7, 91, 15], [91, 40, 91, 7]
        self.assertEqual(rows_for_ids(ids, query).tolist(), [2, 0, 2, 1])
        with self.assertRaises(IndexError):
            np.argsort(ids)[query]  # actual source formula fails on this valid ID set

    def test_source_formula_valid_only_for_full_zero_based_permutation(self):
        ids, query = [2, 0, 3, 1], [1, 2, 0]
        self.assertEqual(rows_for_ids(ids, query).tolist(), np.argsort(ids)[query].tolist())

    def test_duplicate_missing_fractional_and_ambiguous_ids_fail(self):
        for reference, query in [([1, 1], [1]), ([1, 2], [9]), ([1., 2.2], [1]), ([float(2**53)], [1])]:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                rows_for_ids(reference, query)
        self.assertEqual(rows_for_ids([40., 7.], [7]).tolist(), [1])

    def test_row_permutation_equivariance(self):
        generator = torch.Generator().manual_seed(4)
        ids = torch.tensor([5, 100, 17, 80])
        labels = torch.tensor([3, 0, 2, 1])
        perm = torch.randperm(4, generator=generator)
        self.assertTrue(torch.equal(labels[rows_for_ids(ids, [17, 100])],
                                    labels[perm][rows_for_ids(ids[perm], [17, 100])]))
