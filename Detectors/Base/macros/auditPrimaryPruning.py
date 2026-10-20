#!/usr/bin/env python3
"""Audit existing selected-track caches; no original CSV reads or model changes."""
import argparse
import json
from pathlib import Path
import numpy as np
from primaryPruningCache import load_cache


def rates(y, s, cut):
    reject = np.isfinite(s) & (s >= 0) & (s <= 1) & (s.astype(float) >= cut)
    hit, free = y == 0, y == 1
    return dict(rows=len(y), hit_trees=int(hit.sum()), hit_free_trees=int(free.sum()),
                lost=int((reject & hit).sum()), rejected_hit_free=int((reject & free).sum()),
                hit_loss=float((reject & hit).sum()/hit.sum()) if hit.any() else None,
                hit_free_rejection=float((reject & free).sum()/free.sum()) if free.any() else None)


def select(y, s, budget):
    valid = np.isfinite(s) & (s >= 0) & (s <= 1)
    idx = np.flatnonzero(valid)
    idx = idx[np.argsort(-s[idx], kind='stable')]
    if not len(idx) or not (y == 0).any() or not (y == 1).any():
        raise ValueError('Calibration requires both truth classes and valid scores')
    end = np.r_[np.flatnonzero(np.diff(s[idx])), len(idx)-1]
    fp = np.cumsum(y[idx] == 0)[end]
    tp = np.cumsum(y[idx] == 1)[end]
    eligible = np.flatnonzero(fp/(y == 0).sum() <= budget)
    if not len(eligible):
        return float(np.nextafter(1., np.inf))
    best = eligible[np.argmax(tp[eligible])]
    return float(s[idx[end[best]]])


def audit(path):
    y,s,g,meta,f = load_cache(path)
    cuts = meta.get('details', {}).get('configured_thresholds', [])
    report = dict(cache=str(path), metadata=meta, events=len(np.unique(g)), rows=len(y),
                  invalid=int((~(np.isfinite(s)&(s>=0)&(s<=1))).sum()), leave_one_event_out=[])
    for budget in [0., 0.0001, 0.001]:
        folds=[]
        for event in np.unique(g):
            test=g==event
            cut=select(y[~test],s[~test],budget)
            folds.append(dict(event=meta['event_keys'][int(event)], threshold=cut,
                              calibration=rates(y[~test],s[~test],cut), test=rates(y[test],s[test],cut)))
        lost=sum(a['test']['lost'] for a in folds)
        rejected=sum(a['test']['rejected_hit_free'] for a in folds)
        report['leave_one_event_out'].append(dict(calibration_loss_budget=budget,folds=folds,
            pooled_hit_loss=lost/int((y==0).sum()), pooled_hit_free_rejection=rejected/int((y==1).sum()),
            lost=lost,rejected_hit_free=rejected))
    report['interpretation']='Diagnostic event cross-validation; budgets are exploratory, not approved deployment limits. Correlated tracks and six events do not establish rare-loss safety. No species veto or threshold is deployed.'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for i,path in enumerate(a.cache):
        report=audit(path)
        target=a.out/('audit_%d.json'%i)
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(path, flush=True)
        for r in report['leave_one_event_out']:
            print('budget',r['calibration_loss_budget'],'held-out hit-free rejection',r['pooled_hit_free_rejection'],'hit loss',r['pooled_hit_loss'],'lost',r['lost'],flush=True)

if __name__=='__main__': main()
