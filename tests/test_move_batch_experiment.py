from collections import Counter
import json

import pytest
import torch

from crish.candidates import Action
from crish.rebuild import rebuild
from scripts.move_batch_experiment import active_plan, changed_pairs, fixture, seeded_views, validate_plan
from tests.crish_v21.fixtures import state_fixture


def snapshots():
    before = state_fixture()
    action = Action('move', 91, 0, 0, 2)
    after, _ = rebuild(before, action)
    return before, after, action


def test_real_masks_lose_old_positive_and_add_new_at_exact_level():
    before, after, action = snapshots()
    lost, added = changed_pairs(before, after, [91, 40, 22, 66], 0)
    assert lost == {(40, 91)} and added == {(22, 91), (66, 91)}
    assert changed_pairs(before, after, [91, 40, 22, 66], 1) == (set(), set())


def test_permutation_preserves_multiplicities_batch_sizes_and_each_batch_quota():
    before, after, action = snapshots()
    original = [[91, 7, 15, 12], [40, 22, 66, 80]]
    current, disabled = active_plan(original, before, after, action)
    assert current == original and disabled['status'] == 'disabled'
    proposed, info = active_plan(original, before, after, action, enabled=True)
    validate_plan(original, proposed, before)
    assert info['status'] == 'placed'
    lost, added = changed_pairs(before, after, proposed[info['anchor_batch']], 0)
    assert lost and added
    assert original == [[91, 7, 15, 12], [40, 22, 66, 80]]


@pytest.mark.parametrize('plan', [[[7, 15], [40, 22]], [[91, 7], [40, 22]], [[91, 7, 15], [12, 80, 66]]])
def test_missing_anchor_or_peers_or_slots_falls_back_without_adding_samples(plan):
    before, after, action = snapshots()
    proposed, info = active_plan(plan, before, after, action, enabled=True)
    assert proposed == plan and info['status'] != 'placed'


def test_repeated_occurrences_do_not_double_count_original_image_pairs():
    before, after, _ = snapshots()
    assert changed_pairs(before, after, [91, 91, 40, 22, 22], 0) == ({(40, 91)}, {(22, 91)})


def test_seeded_two_views_match_independent_of_sampling_order():
    from crish.smoke import FeatureFixture
    before, _, _ = snapshots()
    data = FeatureFixture(before.features, before.labeled, before.known_targets)
    data.sample_ids = tuple(before.ids.tolist())
    state = torch.get_rng_state().clone()
    first = seeded_views(data, 91, 10, 42)
    seeded_views(data, 22, 10, 42)
    second = seeded_views(data, 91, 10, 42)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(state, torch.get_rng_state())


def test_exactly_one_epoch_per_arm_with_shared_move_and_no_fake_acceptance(tmp_path):
    result = fixture(tmp_path/'ab')
    a, b = result['results']
    assert result['status'] == 'completed'
    assert not result['provenance']['accepted_move_evidence']
    assert a['steps'] == b['steps'] == 2
    assert a['move_id'] == b['move_id'] == 91 and a['level'] == b['level'] == 0
    assert a['changed_batches'] == a['lost_pairs'] == a['added_pairs'] == 0
    assert b['changed_batches'] == 1 and b['lost_pairs'] >= 1 and b['added_pairs'] >= 1
    assert a['All'] is b['All'] is None
    assert (tmp_path/'ab'/'locked_input.pt').is_file()
    with pytest.raises(FileExistsError):
        fixture(tmp_path/'ab')
