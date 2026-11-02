"""Independent stdlib all-role/confusion/clock/accounting closure checker."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
import event_full_validation_plan as p
import event_full_validation_execution as e
from check_rate_impulse_screen import stats,close,mean,FIELDS


def aggregate(rows,policy):
    out={}
    for arm in p.ARMS:
        by=defaultdict(list)
        for r in rows:
            if r['arm']==arm and r['role']==policy:by[r['protocol'],r['fold']].append(stats(r['metrics']['window_confusion'],r['metrics']['recording_confusion']))
        tasks={k:{f:mean(v[f] for v in values) for f in FIELDS} for k,values in by.items()}
        protocols={ds:{f:mean(v[f] for (d,_),v in tasks.items() if d==ds) for f in FIELDS} for ds in sorted({d for d,_ in tasks})}
        out[arm]=dict(per_protocol=protocols,dataset_equal={f:mean(v[f] for v in protocols.values()) for f in FIELDS},task_equal={f:mean(v[f] for v in tasks.values()) for f in FIELDS})
    return out


def run():
    out=p.OUT;r=p.read(out/'SCREEN_RESULT_V1.json');closed=p.read(out/'CLOSED_RECEIPT_V1.json');lock=e.read(out/e.NN_NAME);source=p.read(out/'SOURCE_LOCK_V1.json')
    for name,h in closed['files_sha256'].items():p.need(p.sha(out/name)==h,'immutable closed '+name)
    for spec in (source,lock):
        p.need(p.canonical({k:v for k,v in spec.items() if k!='fingerprint'})==spec['fingerprint'],'full original source/neural identity')
        for path,h in {**spec['code_sha256'],**spec['input_sha256']}.items():p.need(p.sha(path)==h,'code/input '+path)
    p.need(r['execution']=='COMPLETE' and p.read(out/'AUDIT_V1.json')['result_sha256']==p.sha(out/'SCREEN_RESULT_V1.json'),'complete actual probability audit')
    from event_full_validation_decision import decide
    for k,v in decide(r['rows'],list(lock['cases'].values()),aggregate).items():close(v,r[k],k)
    p.need(sum(x['origin']=='NEW' for x in r['rows'])==720 and sum(x['origin']=='ARCHIVE' for x in r['rows'])==90,'360 new/45 archive identities preserved')
    for row in r['rows']:
        for f,v in stats(row['metrics']['window_confusion'],row['metrics']['recording_confusion']).items():
            if f in FIELDS:p.need(abs(row['metrics'][f]-v)<1e-12,'all810 independent confusion metrics')
    banks=p.read(out/'SOURCE_BANKS_V1.json')
    p.need(len(banks['banks'])==81 and len(banks['archive_input_proofs'])==15 and all(v['status']=='EXACT_ORIGINAL_SOURCE_INPUTS' for v in banks['archive_input_proofs'].values()),'all source banks and archive input equality')
    for label,key,spec,result in (('SOURCE','source_database',source,banks),('NEURAL','database',lock,r)):
        end=p.read(out/(label+'_PRODUCER_EXIT_V1.json'));p.need(end['exit_code']==0 and end['wait_returned'],'actual '+label+' native exit0')
        b=closed[key];p.need(p.sha(b['path'])==b['sha256'],'closed '+label+' ledger bytes')
        db=sqlite3.connect(Path(b['path']).resolve().as_uri()+'?mode=ro',uri=True)
        try:
            counts=dict(db.execute('select k,n from counters'));meta={k:json.loads(v) for k,v in db.execute('select k,v from meta')}
            p.need('DONE' in meta and 'STOPPED' not in meta and counts==result['counts'],'complete '+label+' accounting')
            for k,v in counts.items():p.need(v<=spec['limits'][k],'within bound '+label+'/'+k)
            for k,v in spec['forecast'].items():
                if k in spec['limits']:p.need(counts.get(k)==v,'exact whole work '+label+'/'+k)
            if label=='SOURCE':
                p.need(db.execute('select state,count(*) from units group by state').fetchall()==[('DONE',3453)],'all3372 groups/81 banks')
                p.need(banks['source_end']-banks['source_start']<=spec['campaign']['source_seconds'],'original cumulative source clock')
            else:
                p.need(db.execute('select count(*),sum(done) from steps').fetchone()==(720150,720150),'all actual paid/done CUDA steps')
                p.need(db.execute('select count(*) from finished').fetchone()[0]==360 and db.execute('select count(*) from artifacts').fetchone()[0]==5145,'all360 new models/3285 states/1860 probabilities')
                start=meta['neural_start']['at_unix'];freeze=meta['SOURCE_FREEZE']['at_unix'];neural_end=meta['NEURAL_END']['at_unix']
                p.need(p.read(out/'NEURAL_ACTUAL_GRANT_V1.json')['at_unix']<=start<=freeze<=neural_end<=end['at_unix'],'grant/freeze/prediction/end chronology')
                p.need(neural_end-start<=lock['campaign']['neural_seconds'],'original cumulative10h neural stage')
                for detail, in db.execute("select detail from artifacts where kind='probability'"):
                    a=json.loads(detail)
                    if a['side']=='target':p.need(a['at_unix']>=freeze,'every new target prediction after all360 source models freeze')
                p.need(meta['SOURCE_FREEZE']['new_models']==360 and meta['SOURCE_FREEZE']['reused_models']==45,'reuse not counted as fresh training')
                events=meta.get('ARCHIVE_READS',[])
                p.need(len(events)==90 and {v['path'] for v in events}==set(lock['archive_predictions']),'actual90 archived-read chronology records')
                p.need(all(freeze<=v['at_unix']<=meta['DONE']['at_unix'] and v['parent_fingerprint']==lock['archive_predictions'][v['path']]['parent_fingerprint'] for v in events),'all archive reads after new-source freeze with original identities')
        finally:db.close()
    p.need(closed['at_unix']<datetime.fromisoformat(lock['campaign']['deadline']).timestamp(),'preparation-inclusive original16h total clock')
    value=dict(status='PASS_ALL810_ROLES_REUSE_AND_CLOSED_TIMING',scientific=r['status'],result_sha256=p.sha(out/'SCREEN_RESULT_V1.json'),at_unix=time.time(),new_neural_actions=0,new_scientific_payload_reads=0)
    p.save(out/'CLOSED_POSTCHECK_V1.json',value);print('FULL_EVENT_INDEPENDENT_POSTCHECK_PASS',flush=True)


if __name__=='__main__':run()
