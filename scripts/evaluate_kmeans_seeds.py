"""Offline-only repeated SS evaluation of one locked checkpoint; no training.

Each seed runs the existing trainer/evaluate_ss in a fresh bounded worker.
Only fit_mix's RNG entry is overridden, after dataset setup and feature extraction.
Seed 0 is tagged standard_single_seed, not asserted equal to a historical
upstream-before-dataset run (its NumPy stream may have advanced at setup).
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def summarize(rows):
    out={}
    for key in ('all_acc','old_acc','new_acc'):
        values=[float(r[key]) for r in rows]
        out[key]=dict(median=statistics.median(values),mean=statistics.mean(values),
                      std=statistics.stdev(values) if len(values)>1 else None,min=min(values),max=max(values))
    return out


def worker(config,checkpoint,seed,output):
    import random
    import numpy as np
    import torch
    from unittest.mock import patch
    from gcd_runtime.config import load_config
    from selex_runtime.reference import configure_environment
    cfg=load_config(config)
    configure_environment(output.parent,cfg)
    from selex_runtime import trainer
    from methods.clustering.faster_mix_k_means_pytorch import K_Means
    original=K_Means.fit_mix
    def fit(self,*a,**kw):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        return original(self,*a,**kw)
    class Capture(trainer.SelExRuntime):
        def evaluation(self,metrics):
            metrics=dict(metrics,clustering_seed=seed,evaluation_seed=seed,
                seed_stage='immediately_before_fit',unit='fraction',status='observed',
                standard_single_seed=seed==0,checkpoint=str(checkpoint))
            output.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    with patch.object(K_Means,'fit_mix',fit),patch.object(trainer,'SelExRuntime',Capture):
        trainer.run(cfg,output.parent,'evaluate',str(checkpoint))


def evaluate(config,checkpoint,seeds=(0,1,2,3,4),timeout=7200):
    config,checkpoint=Path(config).resolve(),Path(checkpoint).resolve()
    if not config.is_file() or not checkpoint.is_file():raise ValueError('Existing config and checkpoint required')
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('Distinct nonnegative clustering seeds required')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='crish-ss-seeds-') as d:
        for seed in seeds:
            out=Path(d)/f'seed_{seed}.json'
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--config',str(config),
                '--checkpoint',str(checkpoint),'--worker-seed',str(seed),'--output',str(out)],
                cwd=ROOT,check=True,timeout=timeout)
            rows.append(json.loads(out.read_text()))
    with checkpoint.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
    if digest is None:
        h=hashlib.sha256()
        with checkpoint.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        digest=h.hexdigest()
    return dict(status='observed',kind='fixed_checkpoint_multiseed_ss',checkpoint_sha256=digest,
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),unit='fraction',
                seed_stage='immediately_before_fit',per_seed=rows,summary=summarize(rows),
                primary='median_5_clustering_seeds' if list(seeds)==[0,1,2,3,4] else 'custom_seed_diagnostic',
                historical_seed0_parity='not_assumed_for_upstream_setup_stream')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--worker-seed',type=int,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; choose a new path')
    if a.timeout<=0:p.error('Positive timeout required')
    if a.worker_seed is not None:worker(a.config,a.checkpoint,a.worker_seed,a.output)
    else:
        result=evaluate(a.config,a.checkpoint,a.seeds,a.timeout)
        with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
