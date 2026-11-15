import unittest
import json

import torch

from crish.candidates import Action, conflicts
from crish.policy import DecisionEvidence, sequential_repair

from .fixtures import state_fixture


class SequentialRepairTests(unittest.TestCase):
    def run_policy(self, state, actions, scorer, **kwargs):
        return sequential_repair(state, actions, scorer, max_accepted=3,
                                 threshold=0., agreement=1., known_tolerance=0., **kwargs)

    def test_shared_prototypes_and_cross_level_use_conflicts(self):
        self.assertTrue(conflicts(Action('move', 22, 0, 2, 3), Action('move', 66, 0, 2, 1)))
        self.assertTrue(conflicts(Action('move', 22, 0, 2, 3), Action('move', 12, 1, 1, 0)))
        self.assertFalse(conflicts(Action('keep'), Action('move', 22, 0, 2, 3)))

    def test_remaining_action_is_rebuilt_and_rescored_after_acceptance(self):
        state = state_fixture()
        actions = [Action('move', 22, 0, 2, 3), Action('move', 91, 0, 0, 1)]
        calls = []
        def scorer(current, trial, action):
            calls.append((current.version, action.sample_id))
            gain = 1. if current.version == 0 else -2.
            return DecisionEvidence((gain,), gain, 0., 1.)
        final, trace = self.run_policy(state, actions, scorer)
        self.assertEqual(final.version, 1)
        self.assertEqual(sum(r['status'] == 'accepted' for r in trace), 1)
        self.assertEqual([version for version, _ in calls], [0, 0, 1])
        self.assertEqual(state.labels[0][4], 2)

    def test_independent_positive_gains_can_be_jointly_harmful(self):
        theta = torch.tensor([1.])
        delta = torch.tensor([-1.5])
        objective = lambda x: x.square().sum() / 2
        self.assertLess(objective(theta + delta), objective(theta))
        self.assertGreater(objective(theta + 2 * delta), objective(theta))

    def test_shared_cap_allows_zero_or_multiple_evidence_selected_acceptances(self):
        # Controlled scores test stopping semantics, not learned score quality.
        for beneficial_rounds in range(4):
            with self.subTest(beneficial_rounds=beneficial_rounds):
                state = state_fixture()
                actions = [Action('abstain', 91, 0, 0), Action('abstain', 22, 0, 2),
                           Action('abstain', 80, 0, 3)]
                confirmations = []

                def scorer(current, trial, action):
                    gain = 1. if current.version < beneficial_rounds else -1.
                    return DecisionEvidence((gain,), 0., 0., 1.)

                def confirmer(current, trial, action):
                    confirmations.append((current.version, action.sample_id))
                    return dict(gain=1., known_increase=0.)

                final, trace = self.run_policy(state, actions, scorer, confirmer=confirmer)
                accepted = [r for r in trace if r['status'] == 'accepted']
                self.assertEqual(len(accepted), beneficial_rounds)
                self.assertEqual(final.version, beneficial_rounds)
                self.assertEqual(final.abstained[0].sum().item(), beneficial_rounds)
                self.assertEqual([v for v, _ in confirmations], list(range(beneficial_rounds)))
                self.assertEqual(len({sample for _, sample in confirmations}), beneficial_rounds)
                self.assertFalse(state.abstained[0].any())

    def test_failed_later_confirmation_preserves_earlier_commit_and_stops(self):
        state = state_fixture()
        actions = [Action('abstain', 91, 0, 0), Action('abstain', 22, 0, 2),
                   Action('abstain', 80, 0, 3)]
        confirmations = []

        def confirmer(current, trial, action):
            confirmations.append(current.version)
            return dict(gain=1. if current.version == 0 else -1., known_increase=0.)

        final, trace = self.run_policy(state, actions,
            lambda *args: DecisionEvidence((1.,), 0., 0., 1.), confirmer=confirmer)
        self.assertEqual(final.version, 1)
        self.assertEqual(final.abstained[0].sum().item(), 1)
        self.assertEqual(sum(r['status'] == 'accepted' for r in trace), 1)
        self.assertEqual(confirmations, [0, 1])
        self.assertTrue(any(r.get('reason') == 'Independent confirmation failed; refresh stopped'
                            for r in trace))
        self.assertFalse(state.abstained[0].any())

    def test_empty_source_and_stale_initial_actions_do_not_get_committed(self):
        state = state_fixture()
        actions = [Action('move', 22, 0, 2, 3), Action('move', 66, 0, 2, 3),
                   Action('move', 91, 0, 0, 1, version=99)]
        final, trace = self.run_policy(state, actions, lambda *args: DecisionEvidence((1.,), 1., 0., 1.))
        self.assertEqual(final.version, 1)
        self.assertTrue(any('Stale' in r.get('reason', '') for r in trace))
        self.assertTrue(any('empty' in r.get('reason', '') for r in trace))

    def test_failed_scorer_cannot_mutate_live_supervision(self):
        state = state_fixture()
        before = state.copy()
        def failing(current, trial, action):
            current.labels[0].zero_()
            trial.prototypes[0].fill_(torch.nan)
            raise ValueError('Rejected virtual proposal')
        final, trace = self.run_policy(state, [Action('move', 22, 0, 2, 3)], failing)
        self.assertIs(final, state)
        self.assertEqual(trace[0]['status'], 'failed')
        for actual, expected in zip(state.labels + state.prototypes, before.labels + before.prototypes):
            self.assertTrue(torch.equal(actual, expected))

    def test_confirmation_failure_does_not_select_a_runner_up(self):
        state = state_fixture()
        actions = [Action('move', 22, 0, 2, 3), Action('move', 91, 0, 0, 1)]
        def scorer(current, trial, action):
            return (DecisionEvidence((2.,), -1., 0., 1.) if action.sample_id == 22
                    else DecisionEvidence((1.,), 1., 0., 1.))
        final, trace = self.run_policy(state, actions, scorer)
        self.assertIs(final, state)
        self.assertEqual(sum(r['status'] == 'accepted' for r in trace), 0)

    def test_confirmation_callback_only_queries_top_discovery_candidate(self):
        state = state_fixture()
        queried = []
        actions = [Action('move', 22, 0, 2, 3), Action('move', 91, 0, 0, 1)]
        def scorer(current, trial, action):
            return DecisionEvidence((2. if action.sample_id == 22 else 1.,), 0., 0., 1.)
        def confirmer(current, trial, action):
            queried.append(action.sample_id)
            return dict(gain=-1., known_increase=0.)
        final, _ = self.run_policy(state, actions, scorer, confirmer=confirmer)
        self.assertIs(final, state)
        self.assertEqual(queried, [22])

    def test_confirmation_known_guard_zero_mass_and_nonfinite_rejections(self):
        state = state_fixture()
        for evidence in (DecisionEvidence((1.,), -1., 0., 1.),
                         DecisionEvidence((1.,), 1., .01, 1.),
                         DecisionEvidence((1.,), 1., 0., 0.),
                         DecisionEvidence((float('nan'),), 1., 0., 1.),
                         DecisionEvidence((1., -1.), 1., 0., 1.)):
            final, trace = self.run_policy(state, [Action('move', 22, 0, 2, 3)], lambda *args: evidence)
            self.assertIs(final, state)
            self.assertEqual(trace[0]['status'], 'rejected')
            json.dumps(trace, allow_nan=False)
