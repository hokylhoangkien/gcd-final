"""Source-stage seeding and metric-free RNG handoff for bounded evaluators."""
import ast
import hashlib
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from .reference import SOURCE_ROOT, verify_reference


def upstream_policy(config):
    return config.get('evaluation_rng') == 'upstream'


def seed_training(seed):
    """Exact source main calls; do not add Python/cudnn seeding here."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_ss_setup():
    """Execute upstream standalone SS seed_torch(0), BEFORE dataset creation."""
    verify_reference()
    path = SOURCE_ROOT/'project_utils/general_utils.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'seed_torch')
    scope = dict(torch=torch, np=np, random=random, os=os)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    scope['seed_torch'](0)


def capture_rng():
    return dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    validate_state(state)
    if torch.cuda.is_available() and len(state['cuda_rng']) != torch.cuda.device_count():
        raise ValueError('RNG handoff requires the same visible CUDA devices')
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def validate_state(state):
    if set(state) != {'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'}:
        raise ValueError('Only RNG state may be returned by an evaluator')


def rng_digest(state=None):
    state = capture_rng() if state is None else state
    validate_state(state)
    digest = hashlib.sha256(repr(state['python_rng']).encode())
    numpy = state['numpy_rng']
    digest.update(numpy[0].encode())
    digest.update(numpy[1].tobytes())
    digest.update(repr(numpy[2:]).encode())
    digest.update(state['torch_rng'].cpu().numpy().tobytes())
    for value in state['cuda_rng']:
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def atomic_bundle(path, bundle):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(bundle, temporary)
    os.replace(temporary, path)


def create_request(runtime, epoch, checkpoint_sha256):
    state = capture_rng()
    request = dict(format='selex_plain_rng_v1', request_id=uuid.uuid4().hex,
        epoch=epoch, checkpoint_sha256=checkpoint_sha256, config=runtime.config,
        before_sha256=rng_digest(state), rng=state)
    atomic_bundle(runtime.run_dir/'plain_rng_input.pt', request)
    return request


def begin_plain_worker(runtime):
    from gcd_runtime.readiness import sha256
    request = torch.load(runtime.run_dir/'plain_rng_input.pt', map_location='cpu', weights_only=False)
    if (request.get('format') != 'selex_plain_rng_v1'
            or request['epoch'] != runtime.evaluation_epoch
            or request['config'] != runtime.config
            or request['checkpoint_sha256'] != sha256(Path(runtime.evaluated_checkpoint))
            or request['before_sha256'] != rng_digest(request['rng'])):
        raise ValueError('Plain evaluation RNG request does not match this snapshot')
    restore_rng(request['rng'])  # discard worker setup's extra random draws
    return request


def finish_plain_worker(runtime, request):
    state = capture_rng()
    # Deliberately no predictions, targets or metrics in the return channel.
    result = dict(format='selex_plain_rng_result_v1', request_id=request['request_id'],
        epoch=request['epoch'], checkpoint_sha256=request['checkpoint_sha256'],
        before_sha256=request['before_sha256'], after_sha256=rng_digest(state), rng=state)
    atomic_bundle(runtime.run_dir/'plain_rng_output.pt', result)


def commit_plain_result(runtime, request):
    result = torch.load(runtime.run_dir/'plain_rng_output.pt', map_location='cpu', weights_only=False)
    if (set(result) != {'format','request_id','epoch','checkpoint_sha256','before_sha256','after_sha256','rng'}
            or result['format'] != 'selex_plain_rng_result_v1'
            or any(result[key] != request[key] for key in
                   ('request_id','epoch','checkpoint_sha256','before_sha256'))
            or result['after_sha256'] != rng_digest(result['rng'])):
        raise ValueError('Stale, mismatched or non-RNG evaluator return bundle')
    restore_rng(result['rng'])
