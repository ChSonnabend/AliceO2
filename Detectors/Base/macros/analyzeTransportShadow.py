#!/usr/bin/env python3
"""Join birth-time ONNX scores to complete transport-subtree hit truth.

Requires numpy, uproot, scikit-learn, matplotlib. Run on shadow-mode *_Kine.root
files only, produced with Stack.pruneKine=false. One model per simulation run.
"""
import argparse
import csv
import json
import sys
import time
from csvProgress import count_csv_rows, million_rows
from pathlib import Path
import numpy as np
from primaryPruningCache import FEATURES, save_cache, load_cache


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


def read_score_csv(files, scope, chunksize=250000, max_rows=None, features=None):
    """Read O2's merged CSV in chunks; metrics need no kinematics ROOT files."""
    import pandas as pd
    truths, scores, groups, event_keys = [], [], [], []
    event_map = {}; models = set(); schemas = set(); paths = set(); total_rows = 0
    columns = ['csv_schema_version','model','event_id','model_evaluated','valid_score',
               'score','can_avoid_subtree','is_simulation_root','is_hitless_tree_root',
               'shadow_mode','configured_threshold','would_reject','detector_contribution_version','own_contribution_mask','subtree_contribution_mask','protection_mask','effective_would_reject']
    reference_thresholds = set(); reference_counts = dict(correctly_rejected=0,wrongly_rejected=0,rows=0)
    effective_counts = dict(correctly_rejected=0,wrongly_rejected=0,rows=0,protected=0)
    for filename in files:
        filename = str(Path(filename).resolve())
        if filename in paths: raise ValueError('Duplicate CSV input: '+filename)
        paths.add(filename)
        total_file_rows = count_csv_rows(filename, max_rows)
        next_progress = 1000000
        started = time.monotonic()
        file_rows = file_selected = 0
        print(f'[read] {filename} (scope={scope}, chunk={chunksize:,})', file=sys.stderr, flush=True)
        wanted = set(columns) | (set(FEATURES) if features is not None else set())
        for frame in pd.read_csv(filename,usecols=lambda name: name in wanted,chunksize=chunksize,nrows=max_rows,float_precision="round_trip"):
            total_rows += len(frame)
            file_rows += len(frame)
            if not frame['shadow_mode'].eq(1).all() or not frame['csv_schema_version'].isin([1,2]).all():
                raise ValueError('Expected versioned full-simulation shadow CSV')
            schemas.update(map(int,frame['csv_schema_version'].unique()))
            if len(schemas)!=1: raise ValueError('Do not mix legacy hit-mask and detector-contribution truth')
            if frame['csv_schema_version'].eq(2).all():
                if not frame['detector_contribution_version'].eq(1).all():
                    raise ValueError('Missing detector contribution truth')
                frame['own_hit_mask'] = frame['own_contribution_mask']
                frame['subtree_hit_mask'] = frame['subtree_contribution_mask']
            models.update(frame['model'].unique())
            if len(models)>1: raise ValueError('Evaluate one model at a time; CSV inputs contain different models')
            for name in ['model_evaluated','valid_score','can_avoid_subtree','is_simulation_root','is_hitless_tree_root']:
                if not frame[name].isin([0,1]).all(): raise ValueError('Invalid binary field '+name)
            frame=frame[frame.model_evaluated.eq(1)]
            if scope=='roots': frame=frame[frame.is_simulation_root.eq(1)]
            elif scope=='deduplicated': frame=frame[frame.can_avoid_subtree.eq(0)|frame.is_hitless_tree_root.eq(1)]
            file_selected += len(frame)
            now = time.monotonic()
            if file_rows >= next_progress:
                elapsed = now-started
                print(f'[read] scanned {file_rows:,}/{total_file_rows:,} rows ({100*file_rows/max(1,total_file_rows):.2f}%); retained {file_selected:,} scored {scope} tracks; '
                      f'{elapsed:.0f}s elapsed; {file_rows/max(elapsed, .001):,.0f} rows/s', file=sys.stderr, flush=True)
                next_progress = (file_rows//1000000+1)*1000000
            if frame.empty: continue
            if features is not None:
                for name in FEATURES:
                    dtype = np.float64 if name in ('pdg', 'n_descendants') else np.float32
                    values = frame[name].to_numpy(dtype=dtype) if name in frame else np.full(len(frame), np.nan, dtype=dtype)
                    features.setdefault(name, []).append(values)
            y=frame.can_avoid_subtree.to_numpy(dtype=np.int8);score=frame.score.to_numpy(dtype=np.float32)
            valid=np.isfinite(score)&(score>=0)&(score<=1)
            if not np.array_equal(valid,frame.valid_score.to_numpy(dtype=bool)):
                raise ValueError('CSV valid_score contradicts the saved score')
            g=np.empty(len(frame),dtype=np.int32)
            for event in frame.event_id.unique():
                key=(filename,int(event))
                if key not in event_map:
                    event_map[key]=len(event_keys);event_keys.append(list(key))
                g[frame.event_id.to_numpy()==event]=event_map[key]
            truths.append(y);scores.append(score);groups.append(g)
            for cut in frame.configured_threshold.unique():
                if np.isfinite(cut) and cut>=0: reference_thresholds.add(float(cut))
            cuts=frame.configured_threshold.to_numpy(dtype=np.float64)
            expected=np.where(cuts>=0,(valid&(score.astype(np.float64)>=cuts)).astype(int),-1)
            if not np.array_equal(expected,frame.would_reject.to_numpy()):
                raise ValueError('CSV would_reject contradicts the configured threshold and score')
            selected=frame.would_reject.to_numpy()==1
            reference_counts['correctly_rejected']+=int(np.sum(selected&(y==1)))
            reference_counts['wrongly_rejected']+=int(np.sum(selected&(y==0)))
            reference_counts['rows']+=len(frame)
            if 'protection_mask' in frame:
                protection=frame.protection_mask.to_numpy(dtype=np.int32)
                effective_decision=np.where(expected<0,-1,((expected==1)&(protection==0)).astype(int))
                effective=effective_decision==1
                if not np.array_equal(effective_decision,frame.effective_would_reject.to_numpy()):
                    raise ValueError('Protection guard contradicts effective decision')
                effective_counts['rows']+=len(frame)
                effective_counts['protected']+=int((protection!=0).sum())
                effective_counts['correctly_rejected']+=int((effective&(y==1)).sum())
                effective_counts['wrongly_rejected']+=int((effective&(y==0)).sum())
        print(f'[read done] {filename}: {file_rows:,}/{total_file_rows:,} rows scanned (100.00%); {file_selected:,} scored {scope} tracks retained; '
              f'{time.monotonic()-started:.1f}s elapsed', file=sys.stderr, flush=True)
    details=dict(csv_rows=total_rows,model=next(iter(models),None),
                 truth_schema="detector-contribution.v1" if schemas=={2} else "legacy-hit-mask",
                 configured_thresholds=sorted(reference_thresholds),configured_cut_counts=reference_counts,
                 guard_effective_cut_counts=effective_counts,
                 score_metrics_interpretation='ROC and configured_operating_point are model-only; guard_effective_cut_counts applies deterministic protections')
    return truths,scores,groups,event_keys,details


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', nargs='+', help='Kinematics ROOT files from one model in shadow mode')
    source.add_argument('--csv', nargs='+', help='O2-generated *_transport_scores.csv files from one model')
    source.add_argument('--cache', help='Previously saved selected_tracks.npz; no CSV access')
    p.add_argument('--scope',choices=['all','roots','deduplicated'],default='all',
                   help='All scored tracks, input roots only, or omit nested hit-free descendants')
    p.add_argument('--out', required=True)
    p.add_argument('--target-hit-loss', type=float, default=0., help='Calibration bound on fraction of hit-producing trees rejected')
    p.add_argument('--calibration-fraction', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-events', type=int, default=-1, help='Smoke check only; default scans all')
    p.add_argument('--csv-chunk-size', type=int, default=250000,
                   help='CSV input rows per chunk; only selected scores/truth are retained for exact ROC')
    limit = p.add_mutually_exclusive_group()
    limit.add_argument('--max-csv-rows', type=int, help='Partial sample: read at most this many data rows per CSV, possibly part of an event')
    limit.add_argument('--max-million-rows', dest='max_csv_rows', type=million_rows, help='Maximum million input rows per CSV (e.g. 10 or 0.5)')
    a = p.parse_args(argv)
    if a.csv_chunk_size <= 0 or (a.max_csv_rows is not None and a.max_csv_rows <= 0):
        p.error('CSV row limits must be positive')
    if not a.csv and a.max_csv_rows is not None:
        p.error('--max-csv-rows applies only to CSV input')
    if not 0 <= a.target_hit_loss <= 1 or not 0 < a.calibration_fraction < 1:
        p.error('Invalid fraction')
    if a.input:
        import uproot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from primaryPruningPlotStyle import apply_style, cutoff_colors, LOSS_TARGETS
    apply_style()
    from sklearn.metrics import roc_curve, roc_auc_score
    out = Path(a.out);out.mkdir(parents=True,exist_ok=True)
    names = ['MCTrack.mMotherTrackId','MCTrack.mProp','MCTrack.mTransportModelScore',
             'MCTrack.mTransportModelShadow','MCEventHeader.FairMCEventHeader.fNPrim',
             'MCTrack.mDetectorContributionMask','MCTrack.mDetectorContributionVersion']
    truths, scores, groups = [], [], []
    event_count = 0; event_keys = []
    csv_details={}
    feature_chunks = {}
    features = {}
    cached_meta = None
    if a.cache:
        if a.max_events != -1: p.error('--max-events does not apply to cached input')
        y, s, g, cached_meta, features = load_cache(a.cache)
        if cached_meta['scope'] != a.scope: p.error('Cache scope differs from requested scope')
        truths, scores, groups = [y], [s], [g]
        event_keys = cached_meta['event_keys']
        csv_details = cached_meta['details']
        print(f'[cache] Loaded {len(y):,} selected tracks from {a.cache}', file=sys.stderr, flush=True)
    elif a.csv:
        if a.max_events != -1: p.error('--max-events applies only to ROOT input')
        truths,scores,groups,event_keys,csv_details=read_score_csv(a.csv,a.scope,a.csv_chunk_size,a.max_csv_rows,feature_chunks)
    else:
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
                        if not np.all(np.asarray(arrays[names[6]][0])==1):
                            raise ValueError('Missing contribution truth: regenerate complete unpruned data')
                        masks=subtree_masks(m,np.asarray(arrays[names[5]][0],dtype=np.uint32),n_inputs)
                        selected=shadow.copy()
                        if a.scope=='roots': selected[n_inputs:]=False
                        elif a.scope=='deduplicated':
                            mothers=m.copy();mothers[:n_inputs]=-1
                            parent_hit=np.ones(len(mothers),dtype=bool)
                            has_parent=mothers>=0;parent_hit[has_parent]=masks[mothers[has_parent]]!=0
                            selected &= (masks!=0)|parent_hit
                        ids=np.flatnonzero(selected)
                        if not len(ids): continue
                        y=(masks[ids]==0).astype(np.int8);s=score[ids]
                        g=len(event_keys);event_keys.append([filename,event])
                        truths.append(y);scores.append(s);groups.append(np.full(len(ids),g,dtype=np.int32))
                        for track,label,value in zip(ids,y,s):
                            writer.writerow([filename,event,int(track),int(track<n_inputs),float(value),int(np.isfinite(value) and 0<=value<=1),int(label),int(masks[track])])
                        print(f'{Path(filename).name} event {event}: {len(ids)} scored tracks, {int(y.sum())} hit-free',flush=True)
    if not truths: raise ValueError('No scored shadow tracks found')
    print('[analysis] Computing ROC curves and event-separated threshold validation...', file=sys.stderr, flush=True)
    y=np.concatenate(truths);s=np.concatenate(scores);g=np.concatenate(groups)
    if feature_chunks:
        features = {name: np.concatenate(parts) for name, parts in feature_chunks.items()}
        features = {name: values for name, values in features.items() if np.isfinite(values).any()}
    del truths, scores, groups, feature_chunks
    metadata = cached_meta or dict(version=1, scope=a.scope, event_keys=event_keys, details=csv_details,
                                   max_csv_rows_per_file=a.max_csv_rows,
                                   partial_input_smoke_test=a.max_csv_rows is not None or a.max_events != -1)
    cache_path = out/'selected_tracks.npz'
    if not a.cache or Path(a.cache).resolve() != cache_path.resolve():
        save_cache(cache_path, y, s, g, metadata, features)
    print(f'[cache saved] {cache_path.resolve()}', file=sys.stderr, flush=True)
    valid=np.isfinite(s)&(s>=0)&(s<=1)
    report=dict(events=len(event_keys),scored_tracks=len(y),valid_scores=int(valid.sum()),invalid_scores_kept=int((~valid).sum()),
                truth='Class 1: no recorded detector contribution in the transport subtree (CSV v1 uses legacy hit masks only)',
                scope=a.scope,
                caution='Tree-loss rates are neither individual-hit losses nor CPU savings. Zero observed loss is not a guarantee.')
    report.update(csv_details)
    report['selected_tracks_cache']=str(cache_path.resolve())
    report['cached_physics_columns']=sorted(features)
    if a.cache:
        report['max_csv_rows_per_file']=metadata.get('max_csv_rows_per_file')
        report['partial_input_smoke_test']=metadata.get('partial_input_smoke_test',False)
    if a.csv:
        report['csv_chunk_size']=a.csv_chunk_size
        report['max_csv_rows_per_file']=a.max_csv_rows
        report['partial_input_smoke_test']=a.max_csv_rows is not None
    if len(csv_details.get('configured_thresholds',[]))==1:
        report['configured_operating_point']=rates(y,s,csv_details['configured_thresholds'][0])
    fig, ax=plt.subplots(figsize=(16,10))
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
            report['operating_points'] = []
            for target in LOSS_TARGETS:
                allowed = np.flatnonzero(fpr <= target)
                index = allowed[np.argmax(tpr[allowed])]
                threshold = float(cuts[index])
                report['operating_points'].append(dict(target_hit_loss=target, threshold=threshold,
                    calibration=rates(y[cal],s[cal],threshold), test=rates(y[test],s[test],threshold)))
            with (out/'operating_points.csv').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['target_hit_loss','threshold','split','hit_tree_loss_rate','avoidable_rejection_rate','wrongly_rejected','correctly_rejected','hit_trees','avoidable_trees'])
                for point in report['operating_points']:
                    for split in ['calibration','test']:
                        r=point[split]
                        writer.writerow([point['target_hit_loss'],point['threshold'],split,r['hit_tree_loss_rate'],r['avoidable_rejection_rate'],r['wrongly_rejected'],r['correctly_rejected'],r['hit_trees'],r['avoidable_trees']])

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
    for color,point in zip(cutoff_colors(len(report.get('operating_points',[]))),report.get('operating_points',[])):
        for split,marker in [('calibration','o'),('test','X')]:
            r=point[split]
            ax.scatter(100*r['hit_tree_loss_rate'],100*r['avoidable_rejection_rate'],color=color,marker=marker,s=140,
                       label=f"{100*point['target_hit_loss']:g}% target" if split=='calibration' else None,zorder=5)
    ax.set(xlabel='Hit-producing trees wrongly rejected (%)',ylabel='Hit-free trees rejected (%)')
    # Circles: calibration; crosses: event-held-out evaluation at the same cuts.
    if report.get('operating_points'):
        ax.plot([],[],'ko',label='Calibration cut')
        ax.plot([],[],'kX',label='Held-out cut')
    ax.set_xlim(0,6)
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'roc.png',dpi=160);fig.savefig(out/'roc.pdf');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'[done] Results written to {out.resolve()}', file=sys.stderr, flush=True)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
