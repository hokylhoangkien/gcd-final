"""Bounded offline checkpoint evidence extraction for the Phase-0 two-view gate.

Uses sanitized training images and re-extracts a partition at the supplied locked
checkpoint. No SGD step, probe, dose selection, oracle or evaluation label read.
The existing trainer setup is reused with its train-loop entry replaced by a
capture callback in this diagnostic process only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def capture(state,model,dataset,*,device,seed,epoch,batch_size,num_workers):
    from crish.evidence_v2 import evidence_for_epoch
    from crish.global_policy import epoch_arrays
    settings=SimpleNamespace(knn_k=10,nonreciprocal_weight=.5,rank_block=256,evidence_views=2,train_batch=batch_size)
    two,one=evidence_for_epoch(state,model,dataset,settings,device=device,seed=seed,epoch=epoch,
        batch_size=batch_size,num_workers=num_workers)
    arrays=epoch_arrays(state,state,two)
    for l,r in enumerate(two.reliability):
        arrays[f'reliability_{l}']=r.numpy()
        arrays[f'single_view_reliability_{l}']=one.reliability[l].numpy()
    return arrays


def worker(config,checkpoint,output,epoch):
    import numpy as np
    import torch
    from unittest.mock import patch
    from gcd_runtime.config import load_config
    from gcd_runtime.readiness import sha256
    from gcd_runtime.selection import load_evaluation_checkpoint
    from selex_runtime.reference import configure_environment
    cfg=load_config(config)
    if cfg.get('baseline_family')!='selex' or not 0<=epoch<cfg['epochs']:
        raise ValueError('SelEx config and an epoch within its training schedule required')
    configure_environment(output,cfg)
    from selex_runtime import trainer
    calls=[]
    def extract(head,model,train_loader,merge_loader,args,runtime,*,objective,extractor):
        calls.append(1)
        model.load_state_dict(load_evaluation_checkpoint(checkpoint,cfg,runtime))
        model.eval();runtime.epoch=epoch
        with torch.no_grad():
            state=extractor(model,merge_loader,args)
            arrays=capture(state,model,train_loader.dataset,device=next(model.parameters()).device,
                seed=cfg['seed'],epoch=epoch,batch_size=cfg['batch_size'],num_workers=cfg['num_workers'])
        np.savez_compressed(output/f'crish_g_epoch_{epoch:04d}.npz',**arrays)
        (output/'evidence_metadata.json').write_text(json.dumps(dict(status='observed_diagnostic',
            kind='offline_locked_checkpoint_two_view_evidence',sgd_steps=0,uses_unlabeled_truth=False,
            partition='reextracted_at_checkpoint_not_historical_partition',epoch=epoch,
            checkpoint_sha256=sha256(checkpoint),config_sha256=sha256(config),seed=cfg['seed']),indent=2)+'\n')
    with patch.object(trainer,'train_strict',extract):
        trainer.run(cfg,output,'train')
    if calls!=[1]:raise RuntimeError('Evidence capture did not replace exactly one train-loop call')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epoch',type=int,default=199)
    p.add_argument('--timeout',type=int,default=1800);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.epoch<0 or a.timeout<=0:p.error('Nonnegative epoch and positive timeout required')
    if not a.checkpoint.is_file() or not a.config.is_file():p.error('Existing checkpoint and config required')
    if a.worker:worker(a.config.resolve(),a.checkpoint.resolve(),a.output,a.epoch)
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(a.config.resolve()),
            '--checkpoint',str(a.checkpoint.resolve()),'--output',str(a.output.resolve()),'--epoch',str(a.epoch)],
            cwd=ROOT,check=True,timeout=a.timeout)
