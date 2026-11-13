"""Temporal audit must not conflate epoch indices, protocols or causality."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.analyze_crish_interventions import audit, PROTOCOL


def ss(epoch, values, purpose='monitor_only', protocol=PROTOCOL):
    return dict(epoch=epoch, epoch_number=epoch+1, status='observed', unit='fraction',
                protocol=protocol, purpose=purpose, **dict(zip(('all_acc', 'old_acc', 'new_acc'), values)))


def event(epoch=10, kind='move', target=5):
    d = dict(kind=kind, sample_id=123, level=0, source=4, target=target, version=0,
             status='accepted', changed_global_pseudo_pairs=118)
    g = dict(d, exact_discovery=dict(gain=.00008, known_increase=-.00007),
             confirmation=dict(gain=.00006, known_increase=-.00005))
    return dict(epoch=epoch, status='completed', accepted=1, candidate_count=2,
                seconds=70., decisions=[d], gains=[g])


def write(directory, name, records):
    (Path(directory)/name).write_text(''.join(json.dumps(r)+'\n' for r in records))


class InterventionAuditTests(unittest.TestCase):
    def test_zero_based_action_matches_display_epoch_11_not_10(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, 'crish.jsonl', [event()])
            write(tmp, 'epoch_metrics.jsonl', [ss(9, (.8,.9,.7)), ss(10,(.79,.89,.69)), ss(11,(.81,.91,.71))])
            report = audit(tmp)
            e = report['events'][0]
            self.assertEqual(e['epoch_number'], 11)
            self.assertEqual(e['before_epoch_number'], 10)
            self.assertAlmostEqual(e['during_minus_before_all_acc_pp'], -1.)
            self.assertAlmostEqual(e['next_minus_before_all_acc_pp'], 1.)
            self.assertEqual(report['moves_with_temporal_all_drop'], [10])
            self.assertEqual(e['confirmation_gain'], .00006)
            self.assertEqual(e['attribution'], 'temporal_association_not_causal_effect')
            self.assertIn('not_observed', report['no_move_counterfactual'])

    def test_plain_and_best_reports_cannot_replace_scheduled_ss(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, 'crish.jsonl', [event()])
            write(tmp, 'epoch_metrics.jsonl', [ss(9,(.8,.9,.7)), ss(10,(.79,.89,.69)),
                ss(10,(1.,1.,1.),purpose='best_old_test'),
                ss(10,(1.,1.,1.),protocol='plain_kmeans_disjoint_test')])
            (Path(tmp)/'evaluation_best_ss_all.json').write_text(json.dumps(ss(10,(1.,1.,1.))))
            e = audit(tmp)['events'][0]
            self.assertEqual(e['during_all_acc_percent'], 79.)
            self.assertIsNone(e['next_all_acc_percent'])

    def test_action_identity_includes_target_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = event()
            wrong = dict(r['gains'][0], target=6, exact_discovery=dict(gain=999))
            r['gains'].append(wrong)
            write(tmp, 'crish.jsonl', [r])
            e = audit(tmp)['events'][0]
            self.assertEqual(e['discovery_gain'], .00008)
            self.assertIsNone(e['during_minus_before_all_acc_pp'])

    def test_keep_fallback_and_abstain_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = dict(status='completed', accepted=0, decisions=[], gains=[])
            write(tmp, 'crish.jsonl', [dict(base, epoch=10),
                dict(base, epoch=15, status='fallback_keep', reason='deadline'),
                event(20, 'abstain', None)])
            r = audit(tmp)
            self.assertEqual([x['group'] for x in r['refreshes']],
                             ['keep_epoch','fallback_epoch','abstain_only_epoch'])
            self.assertEqual(len(r['events']), 1)
            self.assertEqual(r['moves_with_temporal_all_drop'], [])

    def test_conflicting_ss_or_missing_gain_is_visible_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, 'crish.jsonl', [event()])
            write(tmp, 'epoch_metrics.jsonl', [ss(10,(.8,.9,.7)),ss(10,(.9,.9,.7))])
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                audit(tmp)
            write(tmp, 'epoch_metrics.jsonl', [])
            write(tmp, 'crish.jsonl', [dict(event(), gains=[])])
            with self.assertRaisesRegex(ValueError, 'matching scored'):
                audit(tmp)

    def test_cli_exports_without_modifying_source_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'; run.mkdir()
            write(run, 'crish.jsonl', [event()])
            write(run, 'epoch_metrics.jsonl', [ss(9,(.8,.9,.7)),ss(10,(.79,.89,.69))])
            before = {p.name:p.read_bytes() for p in run.iterdir()}
            out = Path(tmp)/'out.csv'
            tool = Path(__file__).resolve().parents[1]/'scripts/analyze_crish_interventions.py'
            result = subprocess.run([sys.executable,str(tool),str(run),'--output',str(out)],
                                    capture_output=True,text=True,check=True,timeout=10)
            self.assertIn('NOT OBSERVED', result.stdout)
            self.assertTrue(out.is_file())
            self.assertTrue(out.with_suffix('.json').is_file())
            self.assertEqual(before,{p.name:p.read_bytes() for p in run.iterdir()})
