"""Confirmation must replay the exact Move selected on the discovery fold."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from crish.candidates import Action
from crish.config import CrishSettings
from crish.engine import CrishEngine
from crish.ids import rows_for_ids
from tests.crish_v21.fixtures import state_fixture


def test_multiple_move_targets_confirm_the_selected_gradient_and_record():
    state = state_fixture()
    # The first destination wins discovery; the second is scored last. Both
    # have support and leave a legal anchor in the source cluster.
    actions = [Action('move', 91, 0, 0, target) for target in (2, 3)]
    pool = torch.tensor([2, 3])
    folds = [(torch.tensor([0]), torch.tensor([4, 5])),
             (torch.tensor([1]), torch.tensor([6, 7]))]
    settings = CrishSettings(warmup_epochs=0, evidence_samples=3,
        evidence_trials=1, k=1, boundaries=1, max_targets=2, min_support=1,
        train_batch=2, probe_known=1, probe_unlabeled=2, max_refresh_seconds=20)
    torch.manual_seed(408)
    head = torch.nn.Linear(3, 4)
    optimizer = torch.optim.SGD(head.parameters(), lr=.01, momentum=.9)

    def views(model, dataset, ids, device, deadline):
        features = state.features[rows_for_ids(state.ids, ids)].to(device)
        return torch.stack([features, features + .02], dim=1)

    def objective(head, features, ids, membership, targets, current, args, epoch):
        # Distinct real autograd gradients for Keep and each destination.
        coefficient = int(current.labels[0][2]) + 1
        return {'loss': coefficient * head(features).square().mean()}

    scored_gradients = []

    def comparison(model, opt, keep, proposed, probe):
        scored_gradients.append(tuple(g.detach().clone() for g in proposed))
        # Top discovery gain belongs to target 2, not the last scored target 3.
        gain = 2. if len(scored_gradients) == 1 else 1.
        return dict(gain=gain, known_increase=0.)

    drift = dict(sse_levels=[dict(level=0, protected_coefficient_shift_max=0.,
                                 coefficient_half_l1_mean=1.)])
    with tempfile.TemporaryDirectory() as directory:
        runtime = SimpleNamespace(run_dir=Path(directory),
            config=dict(seed=1, method='crish', crish=asdict(settings)))
        with patch('crish.engine._draw_context', return_value=(folds, pool)), \
             patch('crish.engine.cached_views', side_effect=views), \
             patch('crish.engine._shortlist', return_value=(actions, {91: [(15, 1.)]})), \
             patch('crish.engine.propose_actions', return_value=actions), \
             patch('crish.engine.target_drift', return_value=drift), \
             patch('crish.engine.exact_sgd_comparison', side_effect=comparison):
            result = CrishEngine(runtime, objective=objective).prepare(
                state, torch.nn.Identity(), head, optimizer, object(),
                SimpleNamespace(unsupervised_smoothing=.75, strategy='zero_one'), 0)
        record = json.loads((Path(directory)/'crish.jsonl').read_text())

    assert len(scored_gradients) == 3  # two discovery calls, one confirmation
    assert any(not torch.equal(a, b) for a, b in zip(*scored_gradients[:2]))
    for actual, selected in zip(scored_gradients[2], scored_gradients[0]):
        torch.testing.assert_close(actual, selected, rtol=0, atol=0)
    assert result.version == 1 and int(result.labels[0][2]) == 2
    assert record['accepted'] == 1
    assert [entry['target'] for entry in record['gains']] == [2, 3]
    assert record['gains'][0]['confirmation'] is not None
    assert record['gains'][1]['confirmation'] is None
