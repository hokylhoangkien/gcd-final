import unittest

import torch

from crish.partition_audit import audit_stack, nesting_violation
from crish.reliability import blockwise_euclidean_knn, edge_score, sparse_evidence
from crish.stability import match_partition, partition_stability


class PartitionTests(unittest.TestCase):
    def test_non_nested_stack_measured_without_coercion(self):
        a, b = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
        saved = a.clone()
        result = audit_stack([a, b])
        self.assertEqual(result['adjacent_violation'], [.5])
        self.assertFalse(result['nested'])
        self.assertTrue(torch.equal(a, saved))
        self.assertEqual(nesting_violation(a * 3 + 7, b * 8 + 1), .5)
        self.assertEqual(nesting_violation(a, torch.zeros(4)), 0.)

    def test_stability_permutation_and_unmatched_clusters(self):
        a = torch.tensor([0, 0, 1, 1, 2, 2])
        result = partition_stability(a, [a * 4 + 7, a * 5 + 9])
        self.assertTrue(all(v == 1 for v in result['stability'].values()))
        merged = match_partition(a, torch.zeros_like(a))
        self.assertEqual(sum(x is None for x in merged['mapping'].values()), 2)
        self.assertAlmostEqual(merged['cluster_coverage'], 1/3)
        self.assertAlmostEqual(merged['sample_coverage'], 1/3)


class EvidenceTests(unittest.TestCase):
    def test_sparse_unknown_is_distinct_from_measured_zero(self):
        ids = torch.tensor([10, 20, 30, 40])
        a = torch.tensor([[1], [0], [3], [2]])
        b = torch.tensor([[1], [2], [1], [2]])
        evidence = sparse_evidence(ids, [(a, b)])
        self.assertEqual(edge_score(evidence, 10, 20), 0.)
        self.assertIsNone(edge_score(evidence, 10, 40))
        self.assertEqual(edge_score(evidence, 20, 10), 0.)
        self.assertLessEqual(len(evidence), 2 * len(ids))
        self.assertTrue(all(0 <= e.joint <= 1 and 0 <= e.cross_view <= 1 for e in evidence.values()))
        self.assertEqual(edge_score(sparse_evidence(ids, [(a, a)]), 10, 20), 1.)

    def test_block_knn_and_invalid_neighbors(self):
        x = torch.tensor([[0.], [.1], [3.], [3.1]])
        small = blockwise_euclidean_knn(x, 1, block_size=1)
        self.assertTrue(torch.equal(small, blockwise_euclidean_knn(x, 1, block_size=3)))
        self.assertEqual(small.flatten().tolist(), [1, 0, 3, 2])
        for neighbors in [torch.tensor([[0], [0], [0], [0]]), torch.tensor([[8], [0], [0], [0]])]:
            with self.assertRaises(ValueError):
                sparse_evidence(torch.arange(4), [(neighbors, neighbors)])
        self.assertEqual(sparse_evidence(torch.arange(4), [(torch.empty(4, 0, dtype=torch.long),)*2]), {})
