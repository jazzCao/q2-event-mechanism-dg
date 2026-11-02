"""Ten-run means/sample SDs; keep old archives and all checkpoint policies."""
from collections import defaultdict
from pathlib import Path
import csv
import io
import json
import sys
from tenrepeat_source_metadata import ROOT,OUT,read,sha,need,write

def old_rows():
    inventory=read(OUT/'MASTER_INVENTORY_V1.json')
    for p,h in inventory['input_sha256'].items():need(sha(p)==h,'unchanged previously audited metric archive '+p)
    rows=[]
    for name,prefix in (('ress_event_component_attribution/SCREEN_RESULT_V1.json','current/'),('ress_mocd_complete/METRICS.json','archived/'),('ress_q2_exposure_full/SCREEN_RESULT_V1.json','exposure/')):
        for r in read(ROOT/'results/wfdg'/name)['rows']:
            if prefix=='exposure/' and r['arm'] not in ('T0_HALF_EXPOSURE','M0_EXPECTED_ENERGY_SHRINK'):continue
            recipe=prefix+r['arm'];unit='MDG/'+recipe+'/'+r['protocol']+'/'+r['fold']+'/s'+str(r['seed'])
            rows.append(dict(unit=unit,family='MDG',recipe=recipe,protocol=r['protocol'],fold=r['fold'],seed=r['seed'],role=r['role'],metrics=r['metrics'],origin='UNCHANGED_CLOSED_THREE_SEED_ARCHIVE'))
    for r in read(ROOT/'results/wfdg/ress_event_full_validation/postclosure_pcr_comparison/COMPARISON_V1.json')['rows']:
        if not r['method'].startswith(('closed_pcr/NATIVE40/','closed_pcr/SHORT_PHYSICAL_WIDTH/')):continue
        rows.append(dict(unit='PCR/'+r['method']+'/pu_clock4/'+r['fold']+'/s'+str(r['seed']),family='PCR',recipe=r['method'],protocol='pu_clock4',fold=r['fold'],seed=r['seed'],role=r['role'],metrics=r['metrics'],origin='UNCHANGED_CLOSED_THREE_SEED_ARCHIVE'))
    for r in read(ROOT/'results/wfdg/ress_jnu_sdg/EVALUATION_V1.json')['rows']:
        rows.append(dict(unit='SDG/'+r['arm']+'/jnu_speed/single_'+r['source']+'/s'+str(r['seed']),family='SDG',recipe=r['arm'],protocol='jnu_speed',fold=r['source']+'_to_'+r['target'],seed=r['seed'],role=r['policy'],metrics=r['metrics'],origin='UNCHANGED_CLOSED_THREE_SEED_ARCHIVE'))
    need(len(rows)==3330,'all3330 old frozen metric roles retained')
    return rows

def summaries(rows):
    import numpy as np
    from scipy.stats import t
    group=defaultdict(list)
    for r in rows:group[r['family'],r['recipe'],r['protocol'],r['role']].append(r)
    result=[]
    for (family,recipe,protocol,role),values in sorted(group.items()):
        tasks=sorted({r['fold'] for r in values});keys={(r['fold'],r['seed']) for r in values}
        need(len(values)==len(keys) and keys=={(task,seed) for task in tasks for seed in range(10)},'exact ten runs per task, no duplicates or selected seeds')
        fields=('recording_acc','recording_macro_f1','recording_ba') if family=='PCR' else ('acc','macro_f1','recording_ba')
        for field in fields:
            per_seed=[float(np.mean([r['metrics'][field] for r in values if r['seed']==seed])) for seed in range(10)]
            a=np.asarray(per_seed)*100;mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,9)*sd/np.sqrt(10))
            result.append(dict(family=family,recipe=recipe,protocol=protocol,role=role,metric=field,n=10,tasks=len(tasks),mean_percent=mean,std_pp=sd,mean_ci95=[mean-half,mean+half],per_seed_percent=a.tolist(),formatted=f'{mean:.2f} ± {sd:.2f}'))
    return result

def run():
    new=read(OUT/'evaluation/RESULT_V1.json');need(new['status']=='ALL_NEW_TARGET_ROLES_COMPLETE' and len(new['rows'])==7770,'all new frozen target evaluations complete')
    rows=old_rows()+[dict(r,origin='NEW_FIXED_SEEDS_3_TO_9') for r in new['rows']]
    need(len(rows)==11100,'all11100 complete last/selected target roles')
    inventory=read(OUT/'MASTER_INVENTORY_V1.json');need({r['unit'] for r in rows}=={u['key'] for u in inventory['units']},'all5460 actual semantic model units exactly represented')
    stats=summaries(rows);need(len(stats)==396,'all66 recipe/dataset groups ×2policies ×3metrics')
    taskstats=summaries([dict(r,protocol=r['protocol']+'/'+r['fold']) for r in rows]);need(len(taskstats)==3330,'all555 task/recipe groups ×2policies ×3metrics')
    # Independent scalar mean/SD arithmetic on all reported aggregate rows.
    import math
    maximum=0.
    for v in stats+taskstats:
        a=v['per_seed_percent'];mean=sum(a)/10;sd=math.sqrt(sum((x-mean)**2 for x in a)/9)
        maximum=max(maximum,abs(mean-v['mean_percent']),abs(sd-v['std_pp']))
    need(maximum<1e-10,'independent all-row mean and unbiased sample SD')
    result=dict(status='TEN_REPETITIONS_COMPLETE',seeds=list(range(10)),models=5460,new_models=3822,reused_models=1638,target_metric_roles=11100,
        rows=rows,summaries=stats,task_summaries=taskstats,independent_mean_std_max_error=maximum,
        definition='First task-equal within a dataset separately for each seed, then ten-run mean and sample SD (ddof=1). PCR uses recording-level accuracy/F1. Last and source-selected policies stay separate.',
        new_result_sha256=sha(OUT/'evaluation/RESULT_V1.json'),metric_archive_reuse='Exact closed metric file hashes revalidated; old training and old three-seed reports retained unchanged; no best-seed selection.')
    write(OUT/'FULL_TEN_RUN_RESULTS_V1.json',result)
    buf=io.StringIO(newline='');fields=['family','recipe','protocol','role','metric','n','tasks','mean_percent','std_pp','formatted'];w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in stats)
    with (OUT/'MEAN_PLUS_MINUS_STD_V1.csv').open('x',encoding='utf-8-sig',newline='') as f:f.write(buf.getvalue())
    lines=['% Generated only after all ten fixed seeds completed. Values are percent.','\\begin{tabular}{llllr}','Recipe & Dataset & Policy & Metric & Mean $\\pm$ SD \\\\','\\hline']
    esc=lambda x:x.replace('_','\\_')
    for r in stats:lines.append(' & '.join([esc(r['recipe']),esc(r['protocol']),r['role'],esc(r['metric']),f'{r["mean_percent"]:.2f} $\\pm$ {r["std_pp"]:.2f}'])+' \\\\')
    lines.append('\\end{tabular}')
    with (OUT/'MEAN_PLUS_MINUS_STD_V1.tex').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    write(OUT/'FINAL_AUDIT_V1.json',dict(status='PASS',seed_coverage=list(range(10)),models=5460,new_models=3822,old_models=1638,metric_roles=11100,summary_rows=396,task_metric_summaries=3330,independent_scalar_checks=2*(396+3330),maximum_error=maximum,results_sha256=sha(OUT/'FULL_TEN_RUN_RESULTS_V1.json')))
    print('TEN_REPETITIONS_ALL_MANUSCRIPT_COMPLETE_5460_MODELS',flush=True)

def test():
    rows=[]
    for seed in range(10):
        for fold in ('a','b'):
            v=(seed+(0 if fold=='a' else 2))/100
            rows.append(dict(family='MDG',recipe='fixture',protocol='fake',fold=fold,seed=seed,role='last',metrics=dict(acc=v,macro_f1=v,recording_ba=v)))
    result=summaries(rows);need(all(abs(r['mean_percent']-5.5)<1e-12 and abs(r['std_pp']-3.0276503540974917)<1e-12 for r in result),'known two-task equal mean and sample SD fixture')
    for bad in (rows[:-1],rows+rows[:1]):
        try:summaries(bad)
        except AssertionError:pass
        else:raise AssertionError('incomplete or duplicate repeat coverage accepted')
    write(OUT/'SUMMARY_CPU_TESTS_V1.json',dict(status='PASS',known_mean=5.5,known_sample_sd=3.0276503540974917,missing_seed_rejected=True,duplicate_rejected=True,real_target_reads=0,new_neural_actions=0))
    print('TEN_REPEAT_SUMMARY_CPU_PASS',flush=True)

if __name__=='__main__':{'run':run,'test':test}[sys.argv[1]]()
