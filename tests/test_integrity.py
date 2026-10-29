import tempfile
import unittest
from pathlib import Path
from scripts.check_integrity import ROOT, check_split, check_membership


class IntegrityTests(unittest.TestCase):
    def test_official_partition(self):
        report = check_split(ROOT / 'upstream/HypCD/data/ssb_splits/scars_osr_splits.pkl')
        self.assertEqual(report['known_count'] + report['novel_count'], 196)

    def test_leakage_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.jpg').touch()
            (root / 'b.jpg').touch()
            rows = [dict(id=0, path='a.jpg', partition='labeled', label=1),
                    dict(id=1, path='b.jpg', partition='unlabeled', label=2)]
            self.assertEqual(check_membership(rows, {1}, root), {'labeled': 1, 'unlabeled': 1})
            for key, value in [('id', 0), ('path', 'a.jpg'), ('path', 'missing.jpg'),
                               ('partition', 'test'), ('label', -1)]:
                changed = [rows[0], dict(rows[1], **{key: value})]
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    check_membership(changed, {1}, root)
            with self.assertRaises(ValueError):
                check_membership(rows, {2}, root)

    def test_changed_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'split.pkl'
            path.write_bytes(b'invalid')
            with self.assertRaises(ValueError):
                check_split(path)


if __name__ == '__main__':
    unittest.main()
