"""Explicit diagnostic-only annotation reader; never imported by ordinary runs.

Not a proposed method: privileged leave-one-out error targeting measures only
this particular fixed-dose diagnostic's performance, not a proven global bound.
"""
from collections import Counter
import csv
from pathlib import Path
import torch


def load_oracle_errors(state,settings):
    if not settings.diagnostic_oracle or not settings.oracle_labels_csv:
        raise ValueError('Oracle requires diagnostic_oracle=true and an explicit CSV')
    truth={}
    with Path(settings.oracle_labels_csv).open(newline='') as handle:
        for row in csv.DictReader(handle):
            i,y=int(row['sample_id']),int(row['true_label'])
            if i in truth or y<0:raise ValueError('Duplicate/missing oracle labels')
            truth[i]=y
    if any(int(i) not in truth for i in state.ids):raise ValueError('Oracle CSV lacks training IDs')
    values=[truth[int(i)] for i in state.ids]
    if any(values[i]!=int(y) for i,y in zip(state.labeled.nonzero().flatten(),state.known_targets)):
        raise ValueError('Oracle CSV does not match legal anchor mapping')
    out=[]
    for labels in state.labels:
        groups={}
        for c,y in zip(labels.tolist(),values):groups.setdefault(c,Counter())[y]+=1
        wrong=torch.zeros(len(labels),dtype=torch.bool)
        for i in (~state.labeled).nonzero().flatten().tolist():
            counts=groups[int(labels[i])].copy();counts[values[i]]-=1
            order=[(n,y) for y,n in counts.items() if n>0];order.sort(reverse=True)
            if order and (len(order)==1 or order[0][0]>order[1][0]):wrong[i]=order[0][1]!=values[i]
        out.append(wrong)
    return tuple(out)
