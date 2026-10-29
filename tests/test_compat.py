import time
import unittest
from unittest.mock import patch
from gcd_runtime.compat import kmeans


class BoundedClusteringTests(unittest.TestCase):
    def test_nonconverging_call_times_out(self):
        def never_finishes(**kwargs):
            time.sleep(.2)
        with patch('kmeans_pytorch.kmeans', never_finishes):
            with self.assertRaises(TimeoutError):
                kmeans(timeout_seconds=.02)

    def test_invalid_deadline(self):
        for value in (0,-1,float('inf')):
            with self.assertRaises(ValueError):
                kmeans(timeout_seconds=value)
