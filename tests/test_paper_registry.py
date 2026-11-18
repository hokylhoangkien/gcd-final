"""Final-only provenance, archival read-only import and offline semantic audit."""
import copy
import json
import zipfile
from pathlib import Path

import pytest

from scripts.paper_registry import inspect_run, export
from scripts.paper_posthoc import analyze


def make_run(path):
    path.mkdir()
    cfg = dict(epochs=200, seed=1, dataset_name='scars', method='crish')
    final = dict(status='observed', purpose='final', checkpoint_selection='final_epoch', epoch=199,
        seed=1, unit='fraction', matching='hungarian_v2_global',
        protocol='balanced_ss_kmeans_train_unlabeled', all_acc=.75, old_acc=1., new_acc=.5,
        n_unlabeled=4, n_old=2, n_new=2)
    refresh = dict(epoch=10, status='completed', candidate_count=2, candidate_samples=2,
        accepted=1, applied_actions=1, seconds=2., decisions=[dict(kind='move', sample_id=1,
        level=0, source=0, target=1, version=0, status='accepted')], gains=[dict(kind='move',
        sample_id=1, level=0, source=0, target=1, version=0,
        exact_discovery=dict(gain=1.), confirmation=dict(gain=.5))])
    for name, obj in [('config', cfg), ('evaluation_last', final), ('metadata',dict(run_id=path.name))]:
        (path/(name+'.json')).write_text(json.dumps(obj))
    (path/'crish.jsonl').write_text(json.dumps(refresh)+'\n')
    (path/'metrics.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e, seconds=s))
                                    for e,s in [(9, 1.), (10, 3.)])+'\n')
    return cfg, final, refresh


def test_registry_zip_parity_dose_and_no_fabricated_std(tmp_path):
    directory = tmp_path/'one'
    make_run(directory)
    archive = tmp_path/'run.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        for path in directory.iterdir():
            zipped.write(path, 'one/'+path.name)
    a, rr, actions = inspect_run(directory)
    b, _, _ = inspect_run(archive)
    assert a['all_percent'] == b['all_percent'] == 75.
    assert a['local_accepted'] == 1 and rr[0]['not_applied'] == 1
    assert a['mean_epoch_seconds_with_refresh'] == 3.
    assert a['mean_epoch_seconds_without_refresh'] == 1.
    result = export([directory], tmp_path/'export')
    assert not result['groups'][0]['paper_ready']
    assert 'all_percent' not in result['groups'][0]  # incomplete provenance
    with pytest.raises(FileExistsError):
        export([directory], tmp_path/'export')


@pytest.mark.parametrize('field,value', [('epoch', 10), ('purpose','best'),
    ('selected_using_evaluation_labels',True), ('seed',2), ('status','synthetic_fixture')])
def test_registry_rejects_nonfinal_or_oracle_selected(tmp_path, field, value):
    directory = tmp_path/'one'
    _, final, _ = make_run(directory)
    final[field] = value
    (directory/'evaluation_last.json').write_text(json.dumps(final))
    row, _, _ = inspect_run(directory)
    assert row['status'] == 'NOT_VERIFIED_FINAL' and row['all_percent'] is None


def test_registry_checks_weighted_accuracy_and_duplicate_import(tmp_path):
    directory = tmp_path/'one'
    _, final, _ = make_run(directory)
    with pytest.raises(ValueError, match='Duplicate run ID'):
        export([directory, directory], tmp_path/'export')
    final['all_acc'] = .9
    (directory/'evaluation_last.json').write_text(json.dumps(final))
    with pytest.raises(ValueError, match='weighted'):
        inspect_run(directory)


def test_posthoc_requires_full_population_and_has_explicit_denominators(tmp_path):
    directory = tmp_path/'one'
    make_run(directory)
    snapshot = dict(run_id='one', epoch=10, level=0, sample_ids=[1,2,3,4], true_labels=[1,0,1,1],
                    clusters=[0,0,1,1], labeled=[False]*4)
    path = tmp_path/'snapshots.jsonl'
    path.write_text(json.dumps(snapshot)+'\n')
    report = analyze(directory, path)
    assert report['move_target_majority_accuracy'] == dict(numerator=1, denominator=1, fraction=1.)
    assert report['discovery_confirmation_sign_agreement']['denominator'] == 1
    assert report['abstain_source_error']['fraction'] is None
    snapshot['labeled'][0] = True
    path.write_text(json.dumps(snapshot)+'\n')
    with pytest.raises(ValueError, match='full unlabeled'):
        analyze(directory, path)


def test_missing_log_is_unknown_not_zero(tmp_path):
    directory = tmp_path/'one'
    make_run(directory)
    (directory/'crish.jsonl').unlink()
    row, _, _ = inspect_run(directory)
    assert row['action_log_status'] == 'MISSING' and row['local_accepted'] is None
    assert row['mean_epoch_seconds_without_refresh'] is None


def test_tar_import_and_snapshot_join(tmp_path):
    import tarfile
    import numpy as np
    from scripts.prepare_posthoc_snapshots import export as join
    directory = tmp_path/'one'
    make_run(directory)
    archive = tmp_path/'one.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for p in directory.iterdir():
            handle.add(p, arcname='runs/one/'+p.name)
    assert inspect_run(archive)[0]['local_accepted'] == 1
    np.savez(directory/'partitions_epoch_0010.npz', ids=np.array([1,2,3,4]),
             labeled=np.zeros(4,dtype=bool), level_0=np.array([0,0,1,1]))
    labels = tmp_path/'labels.csv'
    labels.write_text('sample_id,true_label\n1,1\n2,0\n3,1\n4,1\n')
    snapshots = tmp_path/'offline.jsonl'
    join(directory, labels, snapshots)
    assert analyze(directory, snapshots)['move_target_majority_accuracy']['fraction'] == 1.
    with pytest.raises(FileExistsError):
        join(directory, labels, snapshots)


def test_archival_plan_preserves_observed_cadence_and_only_requests_full(tmp_path):
    from scripts.prepare_paper_plan import prepare
    from gcd_runtime.config import ROOT, load_config
    directory=tmp_path/'one'
    directory.mkdir()
    cfg=load_config(ROOT/'configs/selex_scars_local_adaptive_200.json')
    cfg['ss_eval_every_epochs']=1
    (directory/'config.json').write_text(json.dumps(cfg))
    archive=tmp_path/'one.zip'
    with zipfile.ZipFile(archive,'w') as handle:
        handle.write(directory/'config.json','one/config.json')
    manifest=prepare(archive,tmp_path/'plan',[2,3],['full'])
    assert len(manifest['entries'])==2
    for entry in manifest['entries']:
        actual=load_config(tmp_path/'plan'/entry['config'])
        assert actual['ss_eval_every_epochs']==1 and actual['crish_ablation']=='full'
