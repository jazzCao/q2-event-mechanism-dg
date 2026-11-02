"""Independent stdlib-only postcheck: full result arithmetic, timing and ledger.

Does not import the training entry or read a feature/probability/checkpoint payload.
Run inside the finite coordinator after child wait returned and closeout sealed.
"""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/wfdg/ress_rate_impulse_screen'
ARMS=('A0_STRONG_IN_MIXUP','A1_MATCHED_E0','C0_RATE_IMPULSE','G0_GLOBAL_GAIN','G1_EVENT_GAIN','N0_INVERSE_AMPLITUDE','F0_UNEDITED_BANK')
FIELDS=('acc','macro_f1','recording_ba')


def need(ok,why):
    if not ok:raise ValueError(why)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values=list(values);need(bool(values),'nonempty declared comparison');return sum(values)/len(values)


def stats(window,recording):
    k=len(window);need(k>0 and len(recording)==k,'same number of classes')
    for matrix in (window,recording):
        need(all(len(row)==k and all(type(v) is int and v>=0 for v in row) for row in matrix),'square nonnegative integer confusion')
        need(sum(map(sum,matrix))>0,'nonempty confusion')
    support=[sum(row) for row in window];predicted=[sum(row[j] for row in window) for j in range(k)]
    rs=[sum(row) for row in recording]
    return dict(acc=sum(window[i][i] for i in range(k))/sum(support),
        macro_f1=mean(2*window[i][i]/(support[i]+predicted[i]) if support[i]+predicted[i] else 0. for i in range(k)),
        recording_ba=mean(recording[i][i]/rs[i] for i in range(k) if rs[i]))


def aggregate(rows,policy,seed=None):
    result={}
    for arm in ARMS:
        tasks=defaultdict(list)
        for row in rows:
            if row['arm']==arm and row['role']==policy and (seed is None or row['seed']==seed):
                tasks[(row['protocol'],row['fold'])].append(stats(row['metrics']['window_confusion'],row['metrics']['recording_confusion']))
        tm={task:{f:mean(v[f] for v in values) for f in FIELDS} for task,values in tasks.items()}
        protocols=sorted({p for p,_ in tm})
        pp={p:{f:mean(v[f] for (pr,_),v in tm.items() if pr==p) for f in FIELDS} for p in protocols}
        result[arm]=dict(per_protocol=pp,dataset_equal={f:mean(v[f] for v in pp.values()) for f in FIELDS},
            task_equal={f:mean(v[f] for v in tm.values()) for f in FIELDS})
    return result


def gate(primary,per_seed):
    candidate=ARMS[2];checks=[]
    def add(name,delta,bound,strict=False):
        checks.append(dict(name=name,delta=delta,bound=bound,passed=delta>bound+1e-12 if strict else delta+1e-12>=bound))
    for anchor in ARMS[:2]:
        for weight in ('dataset_equal','task_equal'):
            add(anchor+'/'+weight+'/acc',primary[candidate][weight]['acc']-primary[anchor][weight]['acc'],.003)
        for metric in ('macro_f1','recording_ba'):
            add(anchor+'/dataset_equal/'+metric,primary[candidate]['dataset_equal'][metric]-primary[anchor]['dataset_equal'][metric],0.)
        for seed,value in per_seed.items():
            add(anchor+'/seed'+seed,value[candidate]['dataset_equal']['acc']-value[anchor]['dataset_equal']['acc'],0.,True)
    for protocol,value in primary[candidate]['per_protocol'].items():
        add(protocol+'/stronger_anchor',value['acc']-max(primary[a]['per_protocol'][protocol]['acc'] for a in ARMS[:2]),-.005)
    n=len(checks)
    for control in ARMS[3:]:
        add(control+'/specificity',primary[candidate]['dataset_equal']['acc']-primary[control]['dataset_equal']['acc'],.001)
    effect=all(v['passed'] for v in checks[:n]);specific=all(v['passed'] for v in checks[n:])
    return ('CANDIDATE_SCREEN_FAILED' if not effect else 'GAIN_WITHOUT_RATE_IMPULSE_SPECIFICITY' if not specific else 'DEVELOPMENT_RATE_IMPULSE_SIGNAL'),checks


def close(a,b,path='value'):
    need(type(a) is type(b) or isinstance(a,(float,int)) and isinstance(b,(float,int)),path+' type')
    if isinstance(a,dict):
        need(a.keys()==b.keys(),path+' keys')
        for k in a:close(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list):
        need(len(a)==len(b),path+' length')
        for i,(v,w) in enumerate(zip(a,b)):close(v,w,path+'/'+str(i))
    elif isinstance(a,float):need(abs(a-b)<1e-12,path+' arithmetic')
    else:need(a==b,path+' equality')


def run():
    result=read(OUT/'SCREEN_RESULT_V1.json');lock=read(OUT/'NEURAL_LOCK_V1.json');receipt=read(OUT/'CLOSED_RECEIPT_V1.json')
    audit=read(OUT/'AUDIT_V1.json');grant=read(OUT/'NEURAL_ACTUAL_GRANT_V1.json');exit_result=read(OUT/'NEURAL_PRODUCER_EXIT_V1.json')
    need(exit_result['exit_code']==0 and exit_result['wait_returned'] is True,'actual child wait/exit, not observation timeout')
    source_exit=read(OUT/'SOURCE_PRODUCER_EXIT_V1.json')
    need(source_exit['exit_code']==0 and source_exit['wait_returned'] is True,'actual completed source child')
    need(receipt['execution']=='COMPLETE' and audit['status']=='PASS' and result['execution']=='COMPLETE','three actual completion receipts')
    for filename,digest in receipt['files_sha256'].items():need(sha(OUT/filename)==digest,'closed receipt '+filename)
    need(audit['result_sha256']==sha(OUT/'SCREEN_RESULT_V1.json'),'audited result binding')
    canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    need(lock['fingerprint']==canonical({k:v for k,v in lock.items() if k!='fingerprint'}),'full lock identity')
    for path,digest in {**lock['input_sha256'],**lock['code_sha256']}.items():need(sha(path)==digest,'exact locked code/input '+path)
    rows=result['rows'];actual=[(r['protocol'],r['fold'],r['seed'],r['arm'],r['role']) for r in rows]
    expected={(c['task']['protocol'],c['task']['fold'],c['seed'],a,policy) for c in lock['cases'].values() for a in ARMS for policy in ('last','selected')}
    need(len(actual)==len(expected)==140 and set(actual)==expected,'complete100 roles, no duplication or task omission')
    for row in rows:
        direct=stats(row['metrics']['window_confusion'],row['metrics']['recording_confusion'])
        for f in FIELDS:need(abs(row['metrics'][f]-direct[f])<1e-12,'all140 role confusion arithmetic')
    primary=aggregate(rows,'last');secondary=aggregate(rows,'selected');per_seed={str(s):aggregate(rows,'last',s) for s in (0,1)}
    close(primary,result['primary']);close(secondary,result['secondary']);close(per_seed,result['per_seed'])
    route,checks=gate(primary,per_seed);need(route==result['status']==receipt['scientific'],'independent fixed scientific decision')
    close(checks,result['checks'])
    dbpath=Path(receipt['database']['path']);need(sha(dbpath)==receipt['database']['sha256'],'closed database unchanged')
    db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        counts=dict(db.execute('SELECT k,n FROM counters'));meta={k:json.loads(v) for k,v in db.execute('SELECT k,v FROM meta')}
        need('DONE' in meta and meta['owner']['state']=='DONE' and db.execute('SELECT count(*) FROM finished').fetchone()[0]==50,'actual50 completed native chains')
        need(db.execute('SELECT count(*),sum(done) FROM steps').fetchone()==(100150,100150),'actual100150 paid and completed updates')
        need(db.execute('SELECT count(*) FROM artifacts').fetchone()[0]==805,'all495checkpoint/310probability receipts')
        need(counts==result['counts']==receipt['counts'],'same authoritative final resource counts')
        for field,n in counts.items():need(n<=lock['limits'][field],'finite resource '+field)
        for field in ('formal_updates','qualification_updates','training_forward_windows','source_prediction_windows','target_prediction_windows','archive_probability_reads'):
            need(counts[field]==lock['forecast'][field],'complete forecast '+field)
        start=meta['neural_start']['at_unix'];end=meta['NEURAL_END']['at_unix'];freeze=meta['SOURCE_FREEZE']['at_unix']
        need(grant['at_unix']<=start<=freeze<=end<=exit_result['at_unix'],'actual grant/start/freeze/end/exit order')
        need(end-start<=lock['campaign']['neural_seconds'],'original neural clock, not only update time')
        need(receipt['at_unix']<datetime.fromisoformat(lock['campaign']['deadline']).timestamp(),'closure before original total deadline')
        need(all(json.loads(detail)['at_unix']>=freeze for path,detail in db.execute("SELECT path,detail FROM artifacts WHERE kind='probability'") if 'target_' in path),'every actual target artifact after50-source freeze')
    finally:db.close()
    source=read(OUT/'SOURCE_BANKS_V1.json');source_lock=read(OUT/'SOURCE_LOCK_V1.json')
    source_db=sqlite3.connect((OUT/'SOURCE_CONTROL.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        need(source_db.execute("SELECT v FROM meta WHERE k='DONE'").fetchone() and
            source_db.execute("SELECT count(*) FROM units WHERE state='DONE'").fetchone()[0]==2160 and
            source_db.execute("SELECT count(*) FROM units WHERE state!='DONE'").fetchone()[0]==0,'all2150 source groups and10banks finished')
        source_counts=dict(source_db.execute('SELECT k,n FROM counters'))
        need(source_counts==source['counts'],'complete source counters from durable journal')
        for k,n in source_counts.items():need(n<=source_lock['limits'][k],'source finite quota '+k)
        for k in ('source_payload_decodes','source_payload_bytes'):need(source_counts[k]==source_lock['forecast'][k],'complete original source IO '+k)
        need(len(source['groups'])==2150 and len(source['banks'])==10,'complete source set coverage')
    finally:source_db.close()
    output=dict(status='PASS_ALL140_METRIC_ROLES_METADATA_COUNTERS_AND_TIMING',scientific=route,at=datetime.now().astimezone().isoformat(),
        result_sha256=sha(OUT/'SCREEN_RESULT_V1.json'),checks=len(checks),target_metric_roles=140,counts=counts,source_counts=source_counts,
        new_scientific_payload_reads=0,new_neural_actions=0,new_FFTs=0,
        boundary='independent stdlib recomputation from complete saved confusion JSON and SQLite; does not re-infer signals or identify counterfactual labels')
    with (OUT/'CLOSED_POSTCHECK_V1.json').open('x',encoding='utf8',newline='\n') as f:json.dump(output,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
