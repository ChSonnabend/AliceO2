#!/usr/bin/env python3
"""Stream root CSVs and audit reconstructible selected-feature domain limits."""
import argparse, collections, csv, hashlib, json, math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--model-json',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); m=json.loads(a.model_json.read_text())
    limits={k:(lo,hi) for k,lo,hi in zip(m['selected_features'],m['feature_min'],m['feature_max'])}
    counts=collections.Counter(); combos=collections.Counter(); events={}; wrong=[]
    reasons_range={};rows=invalid=unexplained=valid_violations=0
    before=a.csv.stat()
    with a.csv.open(newline='') as stream:
        for r in csv.DictReader(stream):
            rows+=1
            if r['model_evaluated']!='1': continue
            px,py,pz,vx,vy,vz,t=map(float,[r[k] for k in ['px','py','pz','vx','vy','vz','t_ns']])
            pt=math.hypot(px,py)
            values=dict(pt=pt,eta=math.asinh(pz/pt) if pt else math.inf,vx=vx,vy=vy,vz=vz,r_xy=math.hypot(vx,vy),t_ns=t)
            reasons=[k for k,v in values.items() if k in limits and not limits[k][0]<=v<=limits[k][1]]
            if r['valid_score']!='1':
                invalid+=1;counts.update(reasons);combos.update([','.join(reasons) or 'unexplained'])
                if not reasons: unexplained+=1
                event=events.setdefault(r['event_id'],collections.Counter());event.update(reasons)
                for k in reasons:
                    low,high=reasons_range.get(k,(values[k],values[k]))
                    reasons_range[k]=(min(low,values[k]),max(high,values[k]))
            elif reasons: valid_violations+=1
            if r['would_reject']=='1' and r['can_avoid_subtree']=='0': wrong.append(r)
    after=a.csv.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise RuntimeError('CSV changed during audit')
    report=dict(csv=str(a.csv),csv_bytes=after.st_size,rows=rows,model_json=str(a.model_json),
                model_json_sha256=hashlib.sha256(a.model_json.read_bytes()).hexdigest(),
                invalid=invalid,explained_by=counts,reason_combinations=combos,unexplained=unexplained,
                valid_scores_violating_checked_limits=valid_violations,events=events,violating_value_ranges=reasons_range,
                checked_limits={k:limits[k] for k in values if k in limits},wrongly_rejected=wrong,
                caveat='Checks features reconstructible from CSV; cannot exclude additional simultaneous violations of other model inputs. CSV decimal rounding may affect exact boundary comparisons.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['rows','invalid','explained_by','unexplained','valid_scores_violating_checked_limits','violating_value_ranges']}))

if __name__=='__main__': main()
