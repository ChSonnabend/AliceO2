#!/usr/bin/env python3
"""Extra descriptive primary QA from selected_tracks.npz; never opens original CSVs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from primaryPruningCache import load_cache


def run_qa(cache, output, threshold=None, operating_points=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from primaryPruningPlotStyle import apply_style, cutoff_colors
    apply_style()
    operating_points = operating_points or []
    colors = cutoff_colors(len(operating_points))
    y, score, event, meta, features = load_cache(cache)
    if meta['scope'] != 'roots':
        raise ValueError('Primary QA requires a roots cache')
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., np.inf)):
        raise ValueError('Threshold must be in [0, nextafter(1,+inf)]')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(score) & (score >= 0) & (score <= 1)
    reject = valid & (score.astype(np.float64) >= threshold) if threshold is not None else np.zeros(len(y), dtype=bool)
    report = dict(rows=len(y), events=len(meta['event_keys']), valid_scores=int(valid.sum()),
                  invalid_scores_kept=int((~valid).sum()), threshold=threshold,
                  partial_input_smoke_test=meta.get('partial_input_smoke_test', False),
                  physics_columns=sorted(features),
                  interpretation='Descriptive, unweighted QA of all cached scored input roots, including calibration events. '
                                 'These plots do not constitute held-out validation. Invalid scores always KEEP. '
                                 'Rejection rates count root trees, not individual hits or measured CPU savings.')
    if valid.any():
        report['valid_score_brier']=float(np.mean((score[valid].astype(np.float64)-y[valid])**2))
    def savefig(fig, name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    def table(name, labels, ids):
        ids = np.asarray(ids)
        take = (ids >= 0) & (ids < len(labels))
        counts = {}
        for key, mask in [('rows',np.ones(len(y),dtype=bool)),('valid',valid),('hit_free',y==1),
                          ('hit_producing',y==0),('rejected',reject),
                          ('correctly_rejected',reject&(y==1)),('wrongly_rejected',reject&(y==0))]:
            counts[key]=np.bincount(ids[take&mask],minlength=len(labels))
        rows=[]
        for i,label in enumerate(labels):
            row=dict(category=str(label), **{key:int(value[i]) for key,value in counts.items()})
            row['invalid']=row['rows']-row['valid']
            for key,num,den in [('valid_fraction','valid','rows'),('rejected_fraction','rejected','rows'),
                                ('avoidable_rejection_rate','correctly_rejected','hit_free'),
                                ('hit_tree_loss_rate','wrongly_rejected','hit_producing')]:
                row[key]=row[num]/row[den] if row[den] and (threshold is not None or key=='valid_fraction') else None
            if threshold is None:
                for key in ['rejected','correctly_rejected','wrongly_rejected']:row[key]=None
            rows.append(row)
        if rows:
            with (out/(name+'.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        return rows
    def multi_table(name, labels, ids):
        if not operating_points: return
        ids=np.asarray(ids);take=(ids>=0)&(ids<len(labels))
        free=np.bincount(ids[take&(y==1)],minlength=len(labels))
        hit=np.bincount(ids[take&(y==0)],minlength=len(labels))
        records=[];fig,axes=plt.subplots(2,1,figsize=(16,10),sharex=True)
        x=np.arange(len(labels))
        for color,point in zip(colors,operating_points):
            rejected=valid&(score.astype(float)>=point['threshold'])
            tp=np.bincount(ids[take&rejected&(y==1)],minlength=len(labels))
            fp=np.bincount(ids[take&rejected&(y==0)],minlength=len(labels))
            tr=np.divide(tp,free,out=np.full(len(labels),np.nan),where=free>0)
            fr=np.divide(fp,hit,out=np.full(len(labels),np.nan),where=hit>0)
            label=f"{100*point['target_hit_loss']:g}% target"
            axes[0].plot(x,100*tr,'o-',color=color,label=label)
            axes[1].plot(x,100*fr,'o-',color=color)
            for i,category in enumerate(labels):
                records.append(dict(category=str(category),target_hit_loss=point['target_hit_loss'],threshold=point['threshold'],
                    hit_free=int(free[i]),hit_producing=int(hit[i]),correctly_rejected=int(tp[i]),wrongly_rejected=int(fp[i]),
                    avoidable_rejection_rate=float(tr[i]),hit_tree_loss_rate=float(fr[i])))
        axes[0].set_ylabel('Hit-free trees rejected (%)');axes[1].set_ylabel('Hit-producing trees rejected (%)')
        axes[0].legend(ncol=3)
        ticks=np.unique(np.linspace(0,len(labels)-1,min(len(labels),12)).astype(int))
        axes[1].set_xticks(ticks);axes[1].set_xticklabels([str(labels[i]) for i in ticks],rotation=60,ha='right')
        for ax in axes: ax.grid(alpha=.2)
        savefig(fig,name+'_cutoffs')
        with (out/(name+'_cutoffs.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=records[0]);writer.writeheader();writer.writerows(records)

    def category_plot(name, rows, indices=None):
        if indices is not None:rows=[rows[i] for i in indices]
        if not rows:return
        fig, axes=plt.subplots(2,1,figsize=(16,10),sharex=True)
        x=np.arange(len(rows));axes[0].bar(x,[r['rows'] for r in rows],label='All scored roots')
        axes[0].bar(x,[r['invalid'] for r in rows],label='Invalid scores');axes[0].set_ylabel('Roots');axes[0].legend()
        for key,label in [('valid_fraction','Valid score fraction'),('avoidable_rejection_rate','Hit-free trees rejected'),
                          ('hit_tree_loss_rate','Hit-producing trees rejected')]:
            values=[np.nan if r[key] is None else 100*r[key] for r in rows]
            axes[1].plot(x,values,'o-',label=label,markersize=3)
        axes[1].set_ylabel('Percent');axes[1].legend();axes[1].grid(alpha=.2)
        ticks=np.unique(np.linspace(0,len(rows)-1,min(len(rows),12)).astype(int))
        axes[1].set_xticks(ticks);axes[1].set_xticklabels([rows[i]['category'] for i in ticks],rotation=60,ha='right')
        savefig(fig,name)
    print(f'[extra QA] {len(y):,} cached roots; writing score, event and physics QA to {out}',file=sys.stderr,flush=True)
    fig, axes=plt.subplots(1,2,figsize=(16,10))
    for label,mask in [('Hit-producing subtree',(y==0)&valid),('Hit-free subtree',(y==1)&valid)]:
        if mask.any():
            axes[0].hist(score[mask],bins=np.linspace(0,1,51),histtype='step',label=label)
            axes[1].hist(score[mask],bins=np.linspace(0,1,51),histtype='step',density=True,label=label)
    for ax in axes:
        ax.set_xlabel('Predicted hit-free probability');ax.legend();ax.grid(alpha=.2)
        for color,point in zip(colors,operating_points):
            ax.axvline(point['threshold'],color=color,ls='--',label=f"{100*point['target_hit_loss']:g}% target")
        if threshold is not None and not operating_points:ax.axvline(threshold,color='black',ls='--',label='Threshold')
        ax.legend()
    axes[0].set_ylabel('Valid-score roots');axes[1].set_ylabel('Density');savefig(fig,'score_distributions')
    # Reliability is assessed only on valid predictions; invalid counts are separate.
    ids=np.full(len(y),-1,dtype=np.int32);ids[valid]=np.minimum((score[valid]*20).astype(int),19)
    n=np.bincount(ids[valid],minlength=20)
    sums=np.bincount(ids[valid],weights=score[valid],minlength=20)
    positives=np.bincount(ids[valid],weights=y[valid],minlength=20)
    nonzero=n>0;means=sums[nonzero]/n[nonzero];observed=positives[nonzero]/n[nonzero]
    with (out/'reliability.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['lower','upper','rows','mean_score','observed_hit_free_fraction'])
        for i in np.flatnonzero(nonzero):writer.writerow([i/20,(i+1)/20,int(n[i]),sums[i]/n[i],positives[i]/n[i]])
    fig,ax=plt.subplots(figsize=(16,10));ax.plot([0,1],[0,1],'k--',label='Ideal calibration')
    ax.plot(means,observed,'o-',label='Observed valid scores');ax.set(xlabel='Mean predicted hit-free probability',ylabel='Observed hit-free fraction')
    ax.legend();ax.grid(alpha=.2);savefig(fig,'probability_calibration')
    if nonzero.any():report['valid_score_binned_calibration_error']=float(np.sum(n[nonzero]*np.abs(means-observed))/n.sum())
    event_labels=[f'{i}: event {key[1]}' for i,key in enumerate(meta['event_keys'])]
    rows=table('by_event',event_labels,event)
    multi_table('by_event',event_labels,event)
    # Cap plot size only; CSV always contains every event.
    category_plot('by_event',rows,list(range(min(len(rows),50))))
    report['event_plot_first_n']=min(len(rows),50)
    if 'pdg' in features:
        pdg=features['pdg'];finite=np.isfinite(pdg)
        labels,inv=np.unique(pdg[finite].astype(np.int64),return_inverse=True)
        ids=np.full(len(y),-1,dtype=np.int32);ids[finite]=inv
        rows=table('by_pdg',labels,ids)
        top=np.argsort([-r['rows'] for r in rows])[:20]
        category_plot('by_pdg',rows,top)
        remap=np.full(len(labels),-1,dtype=int);remap[top]=np.arange(len(top))
        plot_ids=np.full(len(ids),-1,dtype=int);good=ids>=0;plot_ids[good]=remap[ids[good]]
        multi_table('by_pdg',labels[top],plot_ids)
    variables={}
    if all(name in features for name in ['px','py','pz']):
        px,py,pz=(features[name].astype(np.float64) for name in ['px','py','pz'])
        pt=np.hypot(px,py);momentum=np.hypot(pt,pz)
        eta=np.full(len(y),np.nan);good=(pt>0)&np.isfinite(pt)&np.isfinite(pz)
        eta[good]=np.arcsinh(pz[good]/pt[good])
        momentum_edges=np.r_[0,np.geomspace(.001,1000,13),np.inf]
        variables.update(pt_GeV=(pt,momentum_edges),p_GeV=(momentum,momentum_edges),
                         pseudorapidity=(eta,np.r_[-np.inf,np.arange(-10,11,1),np.inf]))
    if 'n_descendants' in features:
        variables['descendants']=(features['n_descendants'],np.array([0,1,2,5,10,50,100,500,1000,10000,np.inf]))
    if 'vx' in features and 'vy' in features:
        variables['vertex_radius_cm']=(np.hypot(features['vx'],features['vy']),np.r_[0,np.geomspace(.001,1000,10),np.inf])
    if 'vz' in features:
        variables['vertex_z_cm']=(features['vz'],np.r_[-np.inf,np.linspace(-30,30,13),np.inf])
    for name,(values,edges) in variables.items():
        ids=np.searchsorted(edges,values,side='right')-1
        ids[~np.isfinite(values)]=-1
        labels=[f'[{lo:g}, {hi:g})' for lo,hi in zip(edges[:-1],edges[1:])]
        rows=table('by_'+name,labels,ids);category_plot('by_'+name,rows)
        multi_table('by_'+name,labels,ids)
        report[name+'_nonfinite_excluded']=int((~np.isfinite(values)).sum())
    report['operating_points']=operating_points
    (out/'qa_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'[extra QA done] {out.resolve()}',file=sys.stderr,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--threshold',type=float,help='Optional fixed cut for descriptive rejection QA; no retuning')
    a=p.parse_args();run_qa(a.cache,a.out,a.threshold)


if __name__=='__main__':main()
