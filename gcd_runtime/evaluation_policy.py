"""Final evaluation RNG isolation, independent of benchmark labels."""
from contextlib import contextmanager
import random
import zlib


@contextmanager
def evaluation_context(config, epoch, kind, model):
    """Keep parent training RNG and per-submodule modes intact, even on error.

    Applies to this runtime's Python/NumPy/Torch RNGs and non-persistent eval
    DataLoader workers. Separate Generator objects/persistent workers need their
    own explicit state handling if introduced later.
    """
    import numpy as np
    import torch
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(module, module.training) for module in model.modules()]
    # No Python hash(): evaluation seed is stable across processes and schedules.
    seed = (config['seed'] * 1000003 + epoch * 97 + zlib.crc32(kind.encode('utf-8'))) % (2**32)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        for module, training in modes:
            module.training = training
