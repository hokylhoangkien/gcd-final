from pathlib import Path
import signal
from unittest.mock import patch

import pytest

from scripts.stop_after_checkpoint import stop_after_checkpoint


def test_only_stops_after_new_valid_checkpoint_and_delivers_pending_term():
    events = []
    with patch('scripts.stop_after_checkpoint.worker_identity', return_value='same'), \
         patch('scripts.stop_after_checkpoint.latest_epoch', side_effect=[2, 3]), \
         patch('scripts.stop_after_checkpoint.validate_checkpoint', return_value=3) as validate, \
         patch('scripts.stop_after_checkpoint.os.killpg', side_effect=lambda pid, sig: events.append((pid, sig))):
        assert stop_after_checkpoint(Path('/fixture/run'), 4321) == 3
    assert events == [(4321, signal.SIGSTOP), (4321, signal.SIGTERM), (4321, signal.SIGCONT)]
    validate.assert_called_once_with(Path('/fixture/run'), 3)


def test_invalid_checkpoint_continues_original_without_termination():
    events = []
    with patch('scripts.stop_after_checkpoint.worker_identity', return_value='same'), \
         patch('scripts.stop_after_checkpoint.latest_epoch', side_effect=[2, 3]), \
         patch('scripts.stop_after_checkpoint.validate_checkpoint', side_effect=ValueError('ahead')), \
         patch('scripts.stop_after_checkpoint.os.killpg', side_effect=lambda pid, sig: events.append(sig)):
        with pytest.raises(ValueError, match='ahead'):
            stop_after_checkpoint(Path('/fixture/run'), 4321)
    assert events == [signal.SIGSTOP, signal.SIGCONT]


def test_changed_process_identity_never_receives_signal():
    with patch('scripts.stop_after_checkpoint.worker_identity', side_effect=['old', 'reused']), \
         patch('scripts.stop_after_checkpoint.latest_epoch', return_value=2), \
         patch('scripts.stop_after_checkpoint.os.killpg') as kill:
        with pytest.raises(ValueError, match='identity'):
            stop_after_checkpoint(Path('/fixture/run'), 4321)
    kill.assert_not_called()
