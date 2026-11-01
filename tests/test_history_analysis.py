"""Protect offline protocol alignment, sampling and resume-chain summaries."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.analyze_reliability_history import SS, aggregate, load_runs, read_jsonl


class HistoryTests(unittest.TestCase):
    def test_truncated_last_record_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'a.jsonl'
            p.write_text('{"a":1}\n{"a":')
            self.assertEqual(read_jsonl(p), [{'a': 1}])
            p.write_text('{"a":1}\nbroken\n')
            with self.assertRaises(ValueError):
                read_jsonl(p)

    def test_pair_weighting_handles_empty_snapshot(self):
        rows = [dict(eligible_pairs=n, effective_weight_mass=n*w,
                     downweighted_fraction=d, fallback_anchor_fraction=0.5)
                for n, w, d in [(100, .5, .8), (10, 1., 0.), (0, 0., None)]]
        result = aggregate(rows)
        self.assertAlmostEqual(result['pair_weight_mean'], 60/110)
        self.assertAlmostEqual(result['downweighted_fraction'], 80/110)
        self.assertIsNone(aggregate(rows[-1:])['pair_weight_mean'])

    def test_metric_timing_protocol_and_duplicate_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / n for n in ('a', 'b')]
            for p, epoch, protocol in [(roots[0], 25, SS), (roots[1], 199, 'plain_kmeans_train_unlabeled')]:
                p.mkdir()
                (p / 'config.json').write_text('{"method":"proposed"}')
                record = dict(epoch=epoch, protocol=protocol, All=.8, Old=.9, New=.7)
                (p / 'metrics.jsonl').write_text(json.dumps(record)+'\n')
            rows, _, _ = load_runs(roots)
            self.assertEqual([r['completed_updates_at_measurement'] for r in rows], [25, 200])
            with self.assertRaisesRegex(ValueError, 'Duplicate epoch'):
                load_runs([roots[0], roots[0]])
