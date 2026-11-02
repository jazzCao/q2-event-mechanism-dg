"""Recompute the reported ten-run summaries. Standard library only; no GPU."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]


def rec_metrics(cm):
    n=len(cm); support=[sum(row) for row in cm]; predicted=[sum(row[j] for row in cm) for j in range(n)]
    return {'recording_acc':sum(cm[j][j] for j in range(n))/sum(support),
            'recording_macro_f1':statistics.fmean(2*cm[j][j]/(support[j]+predicted[j]) if support[j]+predicted[j] else 0. for j in range(n)),
            'recording_ba':statistics.fmean(cm[j][j]/support[j] for j in range(n) if support[j])}


def recompute(data):
    groups=defaultdict(lambda:defaultdict(list)); identities=set()
    for row in data['rows']:
        ident=(row['unit'],row['role'],row['fold'])
        if ident in identities:raise ValueError('duplicated metric role: '+str(ident))
        identities.add(ident)
        metrics=dict(row['metrics'])
        cm=metrics.get('recording_confusion',metrics.get('confusion'))
        if cm is not None:
            recalculated=rec_metrics(cm)
            for key,val in recalculated.items():
                if key in metrics and abs(metrics[key]-val)>1e-10:raise ValueError('saved recording metric differs from its confusion matrix')
                metrics[key]=val
        # PCR's original seeds have recording metrics only; never fabricate
        # missing window metrics or call the new-seed subset ten repetitions.
        keys=('recording_ba','recording_acc','recording_macro_f1') if row['family']=='PCR' else ('acc','macro_f1','recording_ba','recording_acc','recording_macro_f1')
        for key in keys:
            if key in metrics:
                groups[(row['family'],row['recipe'],row['protocol'],row['role'],key)][row['seed']].append(float(metrics[key]))
    result={}
    for key,by_seed in groups.items():
        if set(by_seed)!=set(range(10)):raise ValueError('ten real seed records required: '+str(key))
        lengths={len(v) for v in by_seed.values()}
        if len(lengths)!=1:raise ValueError('unequal task coverage across seeds')
        v=[100*statistics.fmean(by_seed[s]) for s in range(10)]
        result[key]={'mean_percent':statistics.fmean(v),'std_pp':statistics.stdev(v),'per_seed_percent':v,'tasks':next(iter(lengths))}
    errors=[]
    for old in data['summary']:
        key=tuple(old[k] for k in ('family','recipe','protocol','role','metric'))
        new=result[key]
        errors.extend(abs(new[k]-old[k]) for k in ('mean_percent','std_pp'))
        if new['tasks']!=old['tasks']:raise ValueError('task count mismatch')
        errors.extend(abs(a-b) for a,b in zip(new['per_seed_percent'],old['per_seed_percent']))
    if max(errors)>1e-10:raise ValueError('summary arithmetic mismatch')
    return result,max(errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/statistics')
    args=parser.parse_args()
    path=ROOT/'evidence/ten_run_results.json.gz'
    data=json.loads(gzip.decompress(path.read_bytes()))
    summaries,error=recompute(data)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'summary.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['family','recipe','dataset','checkpoint_policy','metric','mean_percent','sample_sd_pp','tasks_per_run','runs'])
        for key,value in sorted(summaries.items()):writer.writerow([*key,value['mean_percent'],value['std_pp'],value['tasks'],10])
    contrasts=[]
    datasets=('pu_clock4','phm09_w3','jnu_speed')
    for policy in ('last','selected'):
        for control in ('current/A0_STRONG_IN_MIXUP','current/F0_UNEDITED_BANK','current/Q2_FIXED_IMPULSE','current/A0_IMPULSE_ONLY','current/M0_GLOBAL_AFFINE_MOMENTS','baseline/DCORAL_SOURCE_ONLY'):
            diff=[statistics.fmean(summaries['MDG','current/Q2',d,policy,'acc']['per_seed_percent'][s]-summaries['MDG',control,d,policy,'acc']['per_seed_percent'][s] for d in datasets) for s in range(10)]
            contrasts.append({'policy':policy,'control':control,'mean_delta_pp':statistics.fmean(diff),'sd_delta_pp':statistics.stdev(diff),'positive_runs':sum(x>0 for x in diff),'per_seed_delta_pp':diff})
    result={'status':'PASS','rows':len(data['rows']),'original_summary_rows_checked':len(data['summary']),'all_derived_summaries':len(summaries),'maximum_error':error,'aggregation':'tasks equally within each seed; mean/sample SD of ten seed scores; datasets equally only for pooled contrasts','new_training':0,'new_predictions':0,'contrasts':contrasts}
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='contrasts'}))


if __name__=='__main__':main()
