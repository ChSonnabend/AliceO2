#!/usr/bin/env python3
"""Join birth-time ONNX scores to complete transport-subtree hit truth.

Requires numpy, uproot, scikit-learn, matplotlib. Run on shadow-mode *_Kine.root
files only, produced with Stack.pruneKine=false. One model per simulation run.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def subtree_masks(mothers, masks, n_inputs):
    """Leaf-to-root propagation; input particles are independent transport roots.

    Generator-history mother links between input particles do not represent
    transport ancestry and must not combine separately simulated root labels.
    """
    mothers = np.asarray(mothers, dtype=np.int64).copy()
    masks = np.asarray(masks, dtype=np.uint32).copy()
    n = len(mothers)
    if len(masks) != n or not 0 <= n_inputs <= n:
        raise ValueError('Invalid event sizes/input-root count')
    mothers[:n_inputs] = -1
    if np.any(mothers >= n) or np.any(mothers < -1):
        raise ValueError('Invalid mother index; cannot establish complete truth')
    valid = mothers >= 0
    degree = np.bincount(mothers[valid], minlength=n)
    leaves = np.flatnonzero(degree == 0)
    visited = 0
    while len(leaves):
        visited += len(leaves)
        children = leaves[mothers[leaves] >= 0]
        parents = mothers[children]
        np.bitwise_or.at(masks, parents, masks[children])
        parents, count = np.unique(parents, return_counts=True)
        degree[parents] -= count
        leaves = parents[degree[parents] == 0]
    if visited != n:
        raise ValueError('Cycle in transport ancestry')
    return masks


def rates(truth, score, threshold):
    reject = np.isfinite(score) & (score >= 0) & (score <= 1) & (score.astype(np.float64) >= threshold)
    positive = truth == 1
    tp, fp = int(np.sum(reject & positive)), int(np.sum(reject & ~positive))
    npos, nneg = int(positive.sum()), int((~positive).sum())
    return dict(rows=len(truth), avoidable_trees=npos, hit_trees=nneg,
                correctly_rejected=tp, wrongly_rejected=fp,
                avoidable_rejection_rate=tp / npos if npos else None,
                hit_tree_loss_rate=fp / nneg if nneg else None,
                rejected_fraction=float(reject.mean()) if len(reject) else None)


def operating_curve(truth, score):
    # Invalid scores always KEEP. They remain in the operational denominators.
    valid = np.isfinite(score) & (score >= 0) & (score <= 1)
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(-score[indices], kind='stable')]
    ordered = score[indices].astype(np.float64)
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered)-1] if len(ordered) else np.array([],dtype=int)
    positives = np.cumsum(truth[indices] == 1)
    negatives = np.cumsum(truth[indices] == 0)
    cuts = np.r_[np.nextafter(1., np.inf), ordered[ends]]
    tpr = np.r_[0., positives[ends] / max(1, np.sum(truth == 1))]
    fpr = np.r_[0., negatives[ends] / max(1, np.sum(truth == 0))]
    return fpr, tpr, cuts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs='+', required=True, help='Kinematics ROOT files from one model in shadow mode')
    p.add_argument('--out', required=True)
    p.add_argument('--target-hit-loss', type=float, default=0., help='Calibration bound on fraction of hit-producing trees rejected')
    p.add_argument('--calibration-fraction', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-events', type=int, default=-1, help='Smoke check only; default scans all')
    a = p.parse_args()
    if not 0 <= a.target_hit_loss <= 1 or not 0 < a.calibration_fraction < 1:
        p.error('Invalid fraction')
    import uproot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    out = Path(a.out);out.mkdir(parents=True,exist_ok=True)
    names = ['MCTrack.mMotherTrackId','MCTrack.mProp','MCTrack.mTransportModelScore',
             'MCTrack.mTransportModelShadow','MCEventHeader.FairMCEventHeader.fNPrim']
    truths, scores, groups = [], [], []
    event_count = 0; event_keys = []
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['input_file','event_id','track_id','is_simulation_root','score','valid_score','can_avoid_subtree','subtree_hit_mask'])
        for filename in a.input:
            filename=str(Path(filename).resolve())
            with uproot.open(filename) as root:
                tree=root['o2sim']
                missing=set(names)-set(tree.keys())
                if missing: raise ValueError(f'{filename}: missing {missing}; rebuild and run shadow mode first')
                for event, arrays in enumerate(tree.iterate(names,step_size=1,library='ak')):
                    if a.max_events >= 0 and event_count >= a.max_events: break
                    event_count += 1
                    m=np.asarray(arrays[names[0]][0]);prop=np.asarray(arrays[names[1]][0],dtype=np.uint32)
                    score=np.asarray(arrays[names[2]][0],dtype=np.float32);shadow=np.asarray(arrays[names[3]][0],dtype=bool)
                    n_inputs=int(arrays[names[4]][0])
                    if np.any((score != -1) & ~shadow) or np.any((prop & (1<<30)) != 0):
                        raise ValueError('Pruned/inhibited tracks detected: ROC truth requires a fully transported shadow run')
                    ids=np.flatnonzero(shadow)
                    if not len(ids): continue
                    masks=subtree_masks(m,(prop>>7)&((1<<22)-1),n_inputs)
                    y=(masks[ids]==0).astype(np.int8);s=score[ids]
                    g=len(event_keys);event_keys.append([filename,event])
                    truths.append(y);scores.append(s);groups.append(np.full(len(ids),g,dtype=np.int32))
                    for track,label,value in zip(ids,y,s):
                        writer.writerow([filename,event,int(track),int(track<n_inputs),float(value),int(np.isfinite(value) and 0<=value<=1),int(label),int(masks[track])])
                    print(f'{Path(filename).name} event {event}: {len(ids)} scored tracks, {int(y.sum())} hit-free',flush=True)
    if not truths: raise ValueError('No scored shadow tracks found')
    y=np.concatenate(truths);s=np.concatenate(scores);g=np.concatenate(groups)
    valid=np.isfinite(s)&(s>=0)&(s<=1)
    report=dict(events=len(event_keys),scored_tracks=len(y),valid_scores=int(valid.sum()),invalid_scores_kept=int((~valid).sum()),
                truth='Class 1: no recorded detector hit on this track or any transport descendant',
                scope='All scored tracks; secondaries included only if enabled during simulation',
                caution='Tree-loss rates are neither individual-hit losses nor CPU savings. Zero observed loss is not a guarantee.')
    fig, ax=plt.subplots(figsize=(7,5))
    # Randomize whole events. Calibration and test never share an event.
    if len(event_keys) >= 2:
        order=np.random.default_rng(a.seed).permutation(len(event_keys))
        ncal=min(len(order)-1,max(1,int(len(order)*a.calibration_fraction)))
        cal=np.isin(g,order[:ncal]);test=~cal
        report['calibration_events']=[event_keys[i] for i in order[:ncal]]
        report['test_events']=[event_keys[i] for i in order[ncal:]]
        if len(np.unique(y[cal]))==2 and len(np.unique(y[test]))==2:
            fpr,tpr,cuts=operating_curve(y[cal],s[cal])
            allowed=np.flatnonzero(fpr<=a.target_hit_loss)
            best=allowed[np.argmax(tpr[allowed])]
            cut=float(cuts[best])
            report.update(threshold=cut,target_hit_loss=a.target_hit_loss,threshold_selection_split='calibration',
                          calibration=rates(y[cal],s[cal],cut),test=rates(y[test],s[test],cut))
        else:
            report['threshold_not_selected']='Both classes must occur in both event splits; collect more events'
        scopes=[('Calibration',cal),('Held-out test',test)]
    else:
        report['threshold_not_selected']='Need at least two events; one event cannot provide independent test data'
        scopes=[('Descriptive only',np.ones(len(y),dtype=bool))]
    for label, selected in scopes:
        fpr,tpr,cuts=operating_curve(y[selected],s[selected])
        np.savetxt(out/(label.lower().replace(' ','_')+'_threshold_scan.csv'),np.column_stack([cuts,fpr,tpr]),delimiter=',',header='threshold,hit_tree_loss_rate,avoidable_rejection_rate',comments='')
        ax.plot(100*fpr,100*tpr,label=label)
        usable=selected&valid
        if len(np.unique(y[usable]))==2:
            report[label+'_valid_score_auc']=float(roc_auc_score(y[usable],s[usable]))
            rf,rt,rc=roc_curve(y[usable],s[usable],drop_intermediate=False)
            np.savetxt(out/(label.lower().replace(' ','_')+'_roc_valid_scores.csv'),np.column_stack([rc,rf,rt]),delimiter=',',header='threshold,false_positive_rate,true_positive_rate',comments='')
    ax.set(xlabel='Hit-producing trees wrongly rejected (%)',ylabel='Hit-free trees rejected (%)',title='Shadow-mode threshold trade-off')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'roc.png',dpi=160);fig.savefig(out/'roc.pdf');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
