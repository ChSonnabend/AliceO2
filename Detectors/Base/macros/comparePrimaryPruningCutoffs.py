#!/usr/bin/env python3
"""Compare loss-budget cutoffs using pooled calibration caches, without CSV reads."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from primaryPruningCache import load_cache,save_cache
from analyzeTransportShadow import operating_curve,rates
from primaryPruningPlotStyle import apply_style,cutoff_colors,LOSS_TARGETS
from plotPrimaryPruningQA import run_qa


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--skip-physics-qa',action='store_true',help='Regenerate comparison/tables without the per-model physics figures')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    apply_style();fig,ax=plt.subplots(figsize=(16,10));records=[];reports=[]
    colors=cutoff_colors(len(LOSS_TARGETS))
    model_colors=plt.get_cmap('plasma')(np.linspace(.1,.8,3))
    for model,linestyle,model_color in zip(['bdt','nn_deep64_tanh','transformer_light'],['-','--',':'],model_colors):
        paths=sorted(a.base.glob(f'20EV_{model}_calibration_seed*_shadow_roots/primary_qa_calibration/selected_tracks.npz'))
        if len(paths)!=2:raise ValueError(f'Expected exactly two calibration caches for {model}, got {paths}')
        datasets=[load_cache(path) for path in paths]
        y=np.concatenate([d[0] for d in datasets]);s=np.concatenate([d[1] for d in datasets])
        offset=0;groups=[];event_keys=[]
        for _,_,g,meta,_ in datasets:
            groups.append(g+offset);event_keys.extend(meta['event_keys']);offset+=len(meta['event_keys'])
        features={name:np.concatenate([d[4][name] for d in datasets]) for name in set.intersection(*(set(d[4]) for d in datasets))}
        meta=dict(version=1,scope='roots',event_keys=event_keys,details=dict(model=model),partial_input_smoke_test=False)
        target=a.out/model;target.mkdir(exist_ok=True)
        cache=target/'selected_tracks.npz';save_cache(cache,y,s,np.concatenate(groups),meta,features)
        fpr,tpr,cuts=operating_curve(y,s);points=[]
        ax.plot(100*fpr[fpr>0],100*tpr[fpr>0],color=model_color,linestyle=linestyle,linewidth=3,label=model)
        for color,budget in zip(colors,LOSS_TARGETS):
            eligible=np.flatnonzero(fpr<=budget);best=eligible[np.argmax(tpr[eligible])];cut=float(cuts[best]);r=rates(y,s,cut)
            point=dict(target_hit_loss=budget,threshold=cut,calibration=r)
            points.append(point);records.append(dict(model=model,target_hit_loss=budget,threshold=cut,**r))
            ax.scatter(100*r['hit_tree_loss_rate'],100*r['avoidable_rejection_rate'],color=color,s=120,zorder=5)
        report=dict(model=model,events=len(event_keys),rows=len(y),source_caches=[str(p) for p in paths],operating_points=points,
                    interpretation='All points fitted on pooled calibration seeds; descriptive calibration performance only. Freeze each cut before independent validation. Invalid scores stay KEEP. Loss denominator is hit-bearing root trees, not all roots or hits.')
        (target/'operating_points.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        if not a.skip_physics_qa:
            run_qa(cache,target/'extra_qa',operating_points=points)
    ax.set(xscale='log',xlim=(0.0001,6),xlabel='Hit-producing trees wrongly rejected (%)',ylabel='Hit-free trees rejected (%)')
    handles,labels=ax.get_legend_handles_labels()
    handles.extend(Line2D([],[],color=c,marker='o',linestyle='None',label=f'{100*b:g}% target') for c,b in zip(colors,LOSS_TARGETS))
    ax.legend(handles=handles,ncol=2);ax.grid(which='both',alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.out/('model_cutoff_comparison.'+ext),dpi=150)
    plt.close(fig)
    with (a.out/'operating_points.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    (a.out/'operating_points.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('Completed pooled calibration cutoff comparison:',a.out,flush=True)

if __name__=='__main__':main()
