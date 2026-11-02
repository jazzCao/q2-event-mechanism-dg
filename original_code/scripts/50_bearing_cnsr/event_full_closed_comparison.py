"""Post-closure recipe comparison using audited JSON only; no array/model IO."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'results/wfdg/ress_event_full_validation'
OUT=PARENT/'postclosure_comparison'
OLD=ROOT/'results/wfdg/ress_mocd_complete'
FIELDS=('acc','macro_f1','recording_ba')
CURRENT=('A0_STRONG_IN_MIXUP','F0_UNEDITED_BANK','A1_MATCHED_E0','Q2_FIXED_IMPULSE','Q2')
AUTH='01a1715e-74fe-7eb2-8a68-306e29cb5cf3'


def need(value,message):
    if not value:raise ValueError(message)


def read(path):
    path=Path(path)
    need(path.suffix=='.json' and ROOT in path.resolve().parents,'repository JSON metadata only')
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def code_hashes():return {str(q):sha(q) for q in (Path(__file__).resolve(),Path(__file__).with_name('event_full_closed_comparison_test.py').resolve())}


def require_closed(parent=PARENT):
    names=('SCREEN_RESULT_V1.json','AUDIT_V1.json','CLOSED_RECEIPT_V1.json','CLOSED_POSTCHECK_V1.json')
    need(all((parent/n).is_file() for n in names),'parent not fully closed; no result or baseline metrics may be loaded')
    post=read(parent/names[3]);closed=read(parent/names[2]);audit=read(parent/names[1])
    need(post['status']=='PASS_ALL810_ROLES_PRESERVED_P0_PREFIX_EFFECTIVE_LIMITS_AND_CLOCKS' and closed['execution']=='COMPLETE','actual independent parent closure')
    h=sha(parent/names[0])
    need(h==post['result_sha256']==audit['result_sha256']==closed['files_sha256'][names[0]],'same independently audited parent result')
    return post,closed


def validate_metrics(v):
    def stats(cm):
        k=len(cm);need(k>0 and all(len(row)==k for row in cm),'square confusion matrix')
        need(all(type(n) is int and n>=0 for row in cm for n in row),'nonnegative integer confusion counts')
        support=[sum(row) for row in cm];pred=[sum(cm[i][j] for i in range(k)) for j in range(k)]
        total=sum(support);need(total>0,'nonempty confusion matrix')
        return (sum(cm[i][i] for i in range(k))/total,
                fmean(2*cm[i][i]/(support[i]+pred[i]) if support[i]+pred[i] else 0. for i in range(k)),
                fmean(cm[i][i]/support[i] for i in range(k) if support[i]),total)
    w=stats(v['window_confusion']);r=stats(v['recording_confusion'])
    need(len(v['window_confusion'])==len(v['recording_confusion']),'same declared label set')
    for field,value in zip(FIELDS,(w[0],w[1],r[2])):
        need(type(v[field]) in (int,float) and math.isfinite(v[field]) and abs(v[field]-value)<1e-12,'independent confusion arithmetic '+field)
    need(v['windows']==w[3] and v['recordings']==r[3],'confusion denominators')


def index_rows(rows,cases,methods):
    keys={(v['task']['protocol'],v['task']['fold'],v['seed']) for v in cases.values()}
    wanted={(*k,a,role) for k in keys for a in methods for role in ('last','selected')}
    lookup={}
    for row in rows:
        key=(row['protocol'],row['fold'],row['seed'],row['arm'],row['role'])
        need(key not in lookup and key in wanted,'duplicate or undeclared comparison role')
        validate_metrics(row['metrics']);lookup[key]=row['metrics']
    need(set(lookup)==wanted,'complete methods/tasks/seeds/policies required')
    return lookup


def summarize(index,cases,methods,policy,seed=None):
    values={}
    for arm in methods:
        tasks=defaultdict(list)
        for v in cases.values():
            if seed is not None and v['seed']!=seed:continue
            p,f,s=v['task']['protocol'],v['task']['fold'],v['seed']
            tasks[p,f].append(index[p,f,s,arm,policy])
        means={k:{f:fmean(v[f] for v in rows) for f in FIELDS} for k,rows in tasks.items()}
        protocols=sorted({k[0] for k in means})
        per={p:{f:fmean(v[f] for (pp,_),v in means.items() if pp==p) for f in FIELDS} for p in protocols}
        values[arm]=dict(per_protocol=per,dataset_equal={f:fmean(v[f] for v in per.values()) for f in FIELDS},
                        task_equal={f:fmean(v[f] for v in means.values()) for f in FIELDS})
    return values


def without_target(cases,protocol,target):
    return {k:v for k,v in cases.items() if (v['task']['protocol'],v['task']['target'])!=(protocol,target)}


def difference(a,b):
    return {w:{f:a[w][f]-b[w][f] for f in FIELDS} for w in ('dataset_equal','task_equal')}


def prepare():
    tests=read(OUT/'CPU_TESTS_V1.json');code=code_hashes()
    need(tests['status']=='PASS' and tests['code_sha256']==code and not tests['torch_imported'],'complete actual stdlib tests')
    bridge_path=PARENT/'CLOSED_BASELINE_METADATA_BRIDGE_V1.json';bridge=read(bridge_path)
    need(bridge['status']=='PASS_CLOSED_AUTHOR_FAMILY_METADATA_BRIDGE_ONLY' and len(bridge['cases'])==81,'existing complete eligibility bridge')
    plan=dict(kind='POSTCLOSURE_JSON_ONLY_FIXED_RECIPE_COMPARISON',candidate='current/Q2',
        scientific_fingerprint=bridge['scientific_fingerprint'],methods=['current/'+a for a in CURRENT]+['archived_recipe/'+a for a in bridge['existing_methods']],
        cases=bridge['cases'],code_sha256=code,metadata_sha256={**bridge['metadata_sha256'],str(bridge_path):sha(bridge_path)},
        primary='last',secondary='selected',primary_weight='dataset_equal',supplemental_weight='task_equal',
        uncertainty='all three seed means and sample SD; no window-as-independent p-value',
        robustness='leave out every protocol/target identity with all its source-scenario tasks and seeds',
        added_models=0,added_optimizer_updates=0,added_predictions=0,array_checkpoint_or_raw_reads=0,
        analysis_seconds=1200,requires_actual_parent_closure=True,automatic_launch_from_current_coordinator=False,
        interpretation='Complete recipes, not single-operator ablations; preserve original full-scope gate and every unfavorable row.',
        at_unix=time.time())
    need(len(plan['methods'])==15,'five current plus ten archived recipe rows')
    plan['fingerprint']=canonical(plan);save(OUT/'PLAN_V1.json',plan)
    print('POSTCLOSURE_PLAN_READY_81_CASES_15_ROWS_NO_NEURAL_ACTIONS',flush=True)


def verify(plan):
    need(canonical({k:v for k,v in plan.items() if k!='fingerprint'})==plan['fingerprint'],'fixed comparison plan')
    need(code_hashes()==plan['code_sha256'],'same tested comparison implementation')
    for path,h in plan['metadata_sha256'].items():need(sha(path)==h,'unchanged eligible metadata '+path)


def activate():
    require_closed();plan=read(OUT/'PLAN_V1.json');verify(plan)
    save(OUT/'ACTUAL_GRANT_V1.json',dict(authorization='standing_authorization',human_message_id=AUTH,
        fingerprint=plan['fingerprint'],at_unix=time.time(),deadline_unix=time.time()+plan['analysis_seconds'],
        models=0,optimizer_updates=0,predictions=0,array_checkpoint_or_raw_reads=0))
    print('POSTCLOSURE_ZERO_NEURAL_SCOPE_ACTIVATED',flush=True)


def run():
    post,closed=require_closed();plan=read(OUT/'PLAN_V1.json');verify(plan);grant=read(OUT/'ACTUAL_GRANT_V1.json')
    need(grant['authorization']=='standing_authorization' and grant['human_message_id']==AUTH and grant['fingerprint']==plan['fingerprint'] and time.time()<grant['deadline_unix'],'actual separately activated finite zero-neural scope')
    result=read(PARENT/'SCREEN_RESULT_V1.json');old=read(OLD/'METRICS.json');receipt=read(OLD/'FINAL_EXECUTION_RECEIPT.json')
    need(result['fingerprint']==plan['scientific_fingerprint'] and result['execution']=='COMPLETE','actual current result')
    need(result['status']==closed['scientific']==post['scientific'],'original scientific terminal retained')
    need(sha(OLD/'METRICS.json')==receipt['file_hashes']['METRICS.json'],'unchanged previously audited baseline metrics')
    rows=[]
    for prefix,data in (('current/',result),('archived_recipe/',old)):
        rows.extend(dict(protocol=r['protocol'],fold=r['fold'],seed=r['seed'],arm=prefix+r['arm'],role=r['role'],metrics=r['metrics']) for r in data['rows'])
    cases,methods=plan['cases'],plan['methods'];index=index_rows(rows,cases,methods);q=plan['candidate'];panels={}
    targets=sorted({(v['task']['protocol'],v['task']['target']) for v in cases.values()})
    for policy in ('last','selected'):
        aggregate=summarize(index,cases,methods,policy)
        per_seed={str(s):summarize(index,cases,methods,policy,s) for s in (0,1,2)}
        sd={a:{f:stdev(per_seed[str(s)][a]['dataset_equal'][f] for s in (0,1,2)) for f in FIELDS} for a in methods}
        leave={p+'/'+t:summarize(index,without_target(cases,p,t),methods,policy) for p,t in targets}
        contrasts={a:dict(aggregate=difference(aggregate[q],aggregate[a]),
            per_protocol={p:{f:aggregate[q]['per_protocol'][p][f]-aggregate[a]['per_protocol'][p][f] for f in FIELDS} for p in aggregate[q]['per_protocol']},
            per_seed={s:difference(v[q],v[a]) for s,v in per_seed.items()},
            leave_one_target={t:difference(v[q],v[a]) for t,v in leave.items()}) for a in methods if a!=q}
        panels[policy]=dict(aggregate=aggregate,per_seed=per_seed,three_seed_sample_SD=sd,contrasts=contrasts,
            compared_recipe_panel_best={f:aggregate[q]['dataset_equal'][f]+1e-12>=max(v['dataset_equal'][f] for v in aggregate.values()) for f in FIELDS})
    for policy,key in (('last','primary'),('selected','secondary')):
        for arm,value in result[key].items():
            actual=panels[policy]['aggregate']['current/'+arm]
            for w in ('dataset_equal','task_equal'):
                for f in FIELDS:need(abs(actual[w][f]-value[w][f])<1e-12,'original current aggregate reproduced exactly')
    need(time.time()<grant['deadline_unix'],'finite postclosure analysis clock')
    output=dict(execution='COMPLETE',original_scientific_status=result['status'],candidate=q,panels=panels,
        recipe_rows=15,case_count=81,source_tasks=27,metric_roles=len(rows),target_identities=len(targets),
        global_SOTA_established=False,publication_ready=False,new_neural_actions=0,scientific_payload_reads=0,
        independent_replicate_count_claim=None,plan_fingerprint=plan['fingerprint'],
        parent_result_sha256=sha(PARENT/'SCREEN_RESULT_V1.json'),baseline_metrics_sha256=sha(OLD/'METRICS.json'),at_unix=time.time())
    save(OUT/'COMPARISON_V1.json',output)
    print('POSTCLOSURE_COMPARISON_COMPLETE',result['status'],len(rows),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','activate','run'));args=parser.parse_args()
    if args.mode=='test':
        import unittest
        import event_full_closed_comparison_test as tests
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        save(OUT/'CPU_TESTS_V1.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,code_sha256=code_hashes(),at_unix=time.time()))
        need(result.wasSuccessful() and 'torch' not in sys.modules and 'numpy' not in sys.modules,'actual stdlib-only comparison tests')
    elif args.mode=='prepare':prepare()
    elif args.mode=='activate':activate()
    else:run()
