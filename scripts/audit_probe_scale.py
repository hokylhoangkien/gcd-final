"""Read-only probe-scale audit for CRiSH local and CRiSH-G v1/v2 (stdlib only)."""
import argparse
import json
import math
from pathlib import Path
import statistics


def pearson(x,y):
    if len(x)<2:return None
    a,b=statistics.mean(x),statistics.mean(y)
    num=sum((u-a)*(v-b) for u,v in zip(x,y))
    den=(sum((u-a)**2 for u in x)*sum((v-b)**2 for v in y))**.5
    return num/den if den else None


def analyze(paths):
    shares={c:[] for c in ('known','view','variance')};proxy=[];exact=[]
    moves=zero=0;seen=0
    def compare(row,weight):
        a,b=row.get('keep'),row.get('action')
        if not a or not b:return
        values={c:abs(weight.get(c,1.)*(a[c]-b[c])) for c in shares if c in a and c in b}
        total=sum(values.values())
        if total:
            for c,v in values.items():shares[c].append(v/total)
    for raw in paths:
        path=Path(raw);path=path/'crish.jsonl' if path.is_dir() else path
        cfgpath=path.parent/'config.json'
        cfg=json.loads(cfgpath.read_text()) if cfgpath.is_file() else {}
        weight={c:cfg.get('crish',{}).get(f'probe_{c}_weight',.1 if c=='variance' else 1.) for c in shares}
        for line in path.read_text().splitlines():
            if not line.strip():continue
            record=json.loads(line);seen+=1
            for g in record.get('gains',[]):
                comparison=g.get('exact_discovery') or {}
                compare(comparison,weight)
                p=(g.get('proxy') or {}).get('raw');e=comparison.get('gain')
                if p is not None and e is not None and math.isfinite(p) and math.isfinite(e):proxy.append(p);exact.append(e)
                if g.get('kind')=='move' or (g.get('action') or {}).get('kind')=='move':
                    drift=g.get('target_drift') or {};v=drift.get('use_target_half_l1_mean')
                    if v is not None:moves+=1;zero+=v==0
            for g in record.get('candidates',[]):
                d=g.get('discovery') or {}
                if 'mean' in d:
                    values={c:abs(d['mean'][c]) for c in shares};total=sum(values.values())
                    if total:
                        for c,v in values.items():shares[c].append(v/total)
                else:compare(d,weight)
    return dict(kind='offline_probe_scale_audit',records=seen,
        share_definition='absolute weighted component difference / sum absolute component differences; normalized_v1 uses raw unweighted component means',
        median_component_share={c:statistics.median(v) if v else None for c,v in shares.items()},
        proxy_exact_pearson=pearson(proxy,exact),proxy_exact_pairs=len(proxy),
        same_sign_fraction=sum((x>0)==(y>0) and (x<0)==(y<0) for x,y in zip(proxy,exact))/len(proxy) if proxy else None,
        move_zero_use_drift=dict(numerator=zero,denominator=moves,fraction=zero/moves if moves else None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.paths)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
