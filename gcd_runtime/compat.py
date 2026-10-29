"""Pinned PyPI k-means adapter with a per-call wall-time deadline.

The display-only tqdm_flag is omitted for PyPI 0.3. Numerical operations are
unchanged. A nonconverging call fails rather than consuming the whole run budget.
"""
import inspect
import math
import signal
import threading
import time


def kmeans(*args, tqdm_flag=True, timeout_seconds=120, **kwargs):
    from kmeans_pytorch import kmeans as original
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('kmeans timeout must be finite and positive')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Bounded kmeans must run in the main worker thread')
    if 'tqdm_flag' in inspect.signature(original).parameters:
        kwargs['tqdm_flag'] = tqdm_flag
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def deadline(signum, frame):
        raise TimeoutError(f'kmeans exceeded {timeout_seconds}s; checkpoint retained, inspect empty clusters/convergence')

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, min(timeout_seconds, previous_timer[0]) if previous_timer[0] else timeout_seconds)
    try:
        result = original(*args, **kwargs)
        import torch
        if not torch.isfinite(result[1]).all():
            raise FloatingPointError('Non-finite kmeans centers')
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            remaining = max(1e-6, previous_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])
