"""Prepare matched per-seed configs and commands; never starts training."""
import argparse
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gcd_runtime.config import ROOT, load_config

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
parser.add_argument('--output',type=Path,default=ROOT/'configs/generated')
args=parser.parse_args()
if any(seed<0 for seed in args.seeds) or len(set(args.seeds))!=len(args.seeds):
    parser.error('Seeds must be distinct nonnegative integers')
args.output.mkdir(parents=True,exist_ok=True)
for variant,name in [('B0','baseline'),('B1','neighborhood'),('B2','crossview'),('B3','proposed')]:
    for seed in args.seeds:
        cfg=load_config(ROOT/f'configs/cars_{name}_dino_v1.json')
        cfg['seed']=seed
        path=args.output/f'{variant}_seed{seed}.json'
        payload=json.dumps(cfg,indent=2)+'\n'
        if path.exists() and path.read_text()!=payload:
            raise SystemExit(f'Preserved existing different config: {path}')
        path.write_text(payload)
        print(f'# {variant} seed={seed}')
        print(shlex.join(['.venv/bin/python','-m','gcd_runtime.launch','train','--config',str(path)]))
