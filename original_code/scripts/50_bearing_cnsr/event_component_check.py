"""Independent stdlib closed artifact, scope, confusion, paid work and time audit."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import sqlite3
import time

import event_component_execution as e
from check_rate_impulse_screen import stats,close,mean,FIELDS


def aggregate(rows,policy):
    from event_component_decision import ALL_ARMS
    out={}
    for arm in ALL_ARMS:
        by=defaultdict(list)
        for r in rows:
            if r['arm']==arm and r['role']==policy:by[r['protocol'],r['fold']].append(stats(r['metrics']['window_confusion'],r['metrics']['recording_confusion']))
        tasks={k:{f:mean(v[f] for v in values) for f in FIELDS} for k,values in by.items()}
        per={p:{f:mean(v[f] for (ds,_),v in tasks.items() if ds==p) for f in FIELDS} for p in sorted({p for p,_ in tasks})}
        out[arm]=dict(per_protocol=per,dataset_equal={f:mean(v[f] for v in per.values()) for f in FIELDS},task_equal={f:mean(v[f] for v in tasks.values()) for f in FIELDS})
    return out


def run():
    from event_component_decision import decide
    r=e.read(e.OUT/'SCREEN_RESULT_V1.json');receipt=e.read(e.OUT/'CLOSED_RECEIPT_V1.json')
    lock=e.load(e.OUT/e.NN_NAME);source=e.read(e.OUT/'SOURCE_LOCK_V1.json');banks=e.read(e.OUT/'SOURCE_BANKS_V1.json')
    for filename,h in receipt['files_sha256'].items():e.need(e.sha(e.OUT/filename)==h,'complete closed artifact '+filename)
    for spec in (lock,source):
        e.need(e.canonical({k:v for k,v in spec.items() if k!='fingerprint'})==spec['fingerprint'],'same complete lock')
        for path,h in {**spec['input_sha256'],**spec['code_sha256']}.items():e.need(e.sha(path)==h,'immutable code/input '+path)
    e.need(r['execution']=='COMPLETE' and e.read(e.OUT/'AUDIT_V1.json')['result_sha256']==e.sha(e.OUT/'SCREEN_RESULT_V1.json'),'actual saved probability audit')
    for k,v in decide(r['rows'],list(lock['cases'].values()),aggregate).items():close(v,r[k],k)
    e.need(sum(x['origin']=='NEW' for x in r['rows'])==324 and sum(x['origin']=='CLOSED_405_MODEL_METRICS' for x in r['rows'])==810,'162 new and405 preserved archived identities')
    parent=e.read(lock['parent_result_path'])
    e.need(e.sha(lock['parent_result_path'])==lock['parent_result_sha256'] and parent['status']==r['original_scientific_status']=='FULL_SCOPE_EVENT_OPERATOR_SUPPORTED','unchanged original positive performance result')
    old_index={(v['protocol'],v['fold'],v['seed'],v['arm'],v['role']):v for v in parent['rows']}
    for row in r['rows']:
        values=stats(row['metrics']['window_confusion'],row['metrics']['recording_confusion'])
        for field in FIELDS:e.need(abs(values[field]-row['metrics'][field])<1e-12,'all1134 independent confusion metrics')
        if row['origin']!='NEW':
            before=old_index[row['protocol'],row['fold'],row['seed'],row['arm'],row['role']]
            e.need(all(row[k]==v for k,v in before.items() if k!='origin') and row['original_origin']==before['origin'] and row['closed_comparison_fingerprint']==parent['fingerprint'],'each archived metric role and nested ancestry unchanged')
    e.need(len(banks['banks'])==81 and len(banks['groups'])==banks['canonical_Q2_exact_groups']==3372,'every canonical Q2 anchor and every new source bank')
    for label,key,spec,result in (('SOURCE','source_database',source,banks),('NEURAL','database',lock,r)):
        exit_record=e.read(e.OUT/(label+'_PRODUCER_EXIT_V1.json'))
        e.need(exit_record['exit_code']==0 and exit_record['wait_returned'],'actual '+label+' native exit0')
        binding=receipt[key];e.need(e.sha(binding['path'])==binding['sha256'],'closed '+label+' ledger bytes')
        db=sqlite3.connect(Path(binding['path']).resolve().as_uri()+'?mode=ro',uri=True)
        try:
            counters=dict(db.execute('select k,n from counters'));meta={k:json.loads(v) for k,v in db.execute('select k,v from meta')}
            e.need('DONE' in meta and 'STOPPED' not in meta and counters==result['counts'],'complete '+label+' metadata/counters')
            for k,n in counters.items():e.need(k in spec['limits'] and n<=spec['limits'][k],'resource within bound '+k)
            for k,n in spec['limits'].items():e.need(counters.get(k,0)==n,'exact complete resource '+label+'/'+k)
            if label=='SOURCE':
                e.need(db.execute('select state,count(*) from units group by state').fetchall()==[('DONE',3453)],'all3372 groups/81 banks saved')
                e.need(banks['source_end']-banks['source_start']<=spec['campaign']['source_seconds'],'same-start2h source phase')
            else:
                e.need(db.execute('select count(*),sum(done) from steps').fetchone()==(324060,324060),'every actual paid/done CUDA attempt')
                e.need(db.execute('select count(*) from finished').fetchone()[0]==162 and db.execute('select count(*) from artifacts').fetchone()[0]==2310,'all162 models/1476 states/834 probability files')
                e.need(all(db.execute('select count(*),sum(done) from steps where chain=?',(name,)).fetchone()==(v['steps'],v['steps']) for name,v in lock['chains'].items()),'each174 registered chain complete')
                start,freeze,end=meta['neural_start']['at_unix'],meta['SOURCE_FREEZE']['at_unix'],meta['NEURAL_END']['at_unix']
                grant=e.read(e.OUT/'NEURAL_ACTUAL_GRANT_V1.json')
                e.need(grant['at_unix']<=start<=freeze<=end<=exit_record['at_unix'],'actual grant/freeze/end chronology')
                e.need(end-start<=lock['campaign']['neural_seconds'],'same-start5h neural allocation')
                e.need(meta['SOURCE_FREEZE']['models']==162 and meta['SOURCE_FREEZE']['reused_closed_models']==405,'reuse not counted as new source training')
                for detail, in db.execute("select detail from artifacts where kind='probability'"):
                    item=json.loads(detail)
                    if item['side']=='target':e.need(item['at_unix']>=freeze,'every actual new target after common162-model freeze')
        finally:db.close()
    e.need(receipt['at_unix']<datetime.fromisoformat(lock['campaign']['deadline']).timestamp(),'preparation-inclusive original12h deadline')
    value=dict(status='PASS_ALL1134_ROLES_FIXED_COMPONENT_SCOPE_AND_TIMING',scientific=r['status'],
        original_scientific_status=r['original_scientific_status'],result_sha256=e.sha(e.OUT/'SCREEN_RESULT_V1.json'),
        new_neural_actions=0,new_scientific_payload_reads=0,at_unix=time.time())
    e.save(e.OUT/'CLOSED_POSTCHECK_V1.json',value);print('COMPONENT_INDEPENDENT_POSTCHECK_PASS',flush=True)


if __name__=='__main__':run()
