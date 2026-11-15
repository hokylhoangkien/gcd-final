import json
from unittest.mock import patch

import pytest
import torch

from gcd_runtime.readiness import sha256
from scripts.change_ss_cadence import prepare
from selex_runtime.selection import BEST_NAME, read_best
from selex_runtime.trainer import SelExRuntime
from tests.test_selex_selection import config, report


def source_run(path):
    path.mkdir()
    cfg = config()
    cfg['ss_eval_every_epochs'] = 10
    (path/'config.json').write_text(json.dumps(cfg))
    model, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=.01, momentum=.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    head(model(torch.ones(2, 2))).square().mean().backward()
    opt.step(); scheduler.step()
    report(path, cfg, 0, .6, .5, model)
    runtime = SelExRuntime(path, cfg)
    runtime.plain_rng_pending = False
    runtime.save(0, model, head, opt, scheduler)
    return cfg, model, head, opt, scheduler


def test_migrates_only_cadence_preserves_source_and_resumes_with_best_history(tmp_path):
    src, out, resumed = tmp_path/'source', tmp_path/'migration', tmp_path/'resumed'
    cfg, model, head, opt, scheduler = source_run(src)
    hashes = {p.name: sha256(p) for p in src.iterdir() if p.is_file()}
    original = torch.load(src/'resume.pt', weights_only=False)
    manifest = prepare(src, out, stopped_check=lambda source: None)
    target = json.loads((out/'config.json').read_text())
    assert {k for k in cfg if cfg[k] != target[k]} == {'ss_eval_every_epochs'}
    assert target['ss_eval_every_epochs'] == 1
    assert manifest['completed_epochs'] == 1
    assert hashes == {p.name: sha256(p) for p in src.iterdir() if p.is_file()}
    assert read_best(out/BEST_NAME, target)['selection']['epoch'] == 0
    migrated = torch.load(out/'resume.pt', weights_only=False)
    for key in ('model', 'head', 'torch_rng'):
        a, b = original[key], migrated[key]
        if isinstance(a, dict):
            assert all(torch.equal(a[k], b[k]) for k in a)
        else:
            assert torch.equal(a, b)
    assert original['optimizer']['param_groups'] == migrated['optimizer']['param_groups']
    for index, state in original['optimizer']['state'].items():
        assert torch.equal(state['momentum_buffer'], migrated['optimizer']['state'][index]['momentum_buffer'])
    assert original['scheduler'] == migrated['scheduler']
    resumed.mkdir()
    runtime = SelExRuntime(resumed, target, str(out/'resume.pt'))
    with patch('gcd_runtime.monitoring.monitor_epoch') as evaluate:
        assert runtime.restore(model, head, opt, scheduler) == 1
    evaluate.assert_called_once_with(runtime, 0, model)
    assert (resumed/'test_evaluations'/'epoch_0001.json').is_file()


def test_refuses_active_worker_before_writing_and_existing_output(tmp_path):
    src = tmp_path/'source'
    source_run(src)
    def active(source):
        raise RuntimeError('Source run still has worker')
    with pytest.raises(RuntimeError, match='still has worker'):
        prepare(src, tmp_path/'out', stopped_check=active)
    assert not (tmp_path/'out').exists()
    with pytest.raises(ValueError, match='new directory'):
        prepare(src, src, stopped_check=lambda source: None)


def test_refuses_different_runtime_or_config(tmp_path):
    src = tmp_path/'source'
    source_run(src)
    state = torch.load(src/'resume.pt', weights_only=False)
    state['code_fingerprint'] = {'different_runtime': 'no'}
    torch.save(state, src/'resume.pt')
    with pytest.raises(ValueError, match='do not match'):
        prepare(src, tmp_path/'out', stopped_check=lambda source: None)
    assert not (tmp_path/'out').exists()
