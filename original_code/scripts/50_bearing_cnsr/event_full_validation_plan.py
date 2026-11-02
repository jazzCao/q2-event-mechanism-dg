"""Prospective whole27-task event-operator validation; metadata only at preflight."""
from collections import Counter
from datetime import datetime, timedelta
import math
from pathlib import Path
import sqlite3
import sys
import protected_response_run_spec as parent
from natural_mapping_probe_plan import source_case

ROOT=parent.ROOT
OUT=ROOT/'results/wfdg/ress_event_full_validation'
read,sha,canonical=parent.read,parent.sha,parent.canonical
ARMS=('A0_STRONG_IN_MIXUP','F0_UNEDITED_BANK','A1_MATCHED_E0','Q2_FIXED_IMPULSE','Q2')
CANDIDATE='Q2'
P0_CASES=('pu_clock4/ABCtoD/s0','phm09_w3/adj_AC_B/s0','jnu_speed/A/s0')
Q=(2/3,1.5,math.exp(-math.log(1.5)/math.sqrt(3)),math.exp(math.log(1.5)/math.sqrt(3)))


def need(value,message):
    if not value:raise ValueError(message)


def save(path,value):
    import json
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def start():
    OUT.mkdir(exist_ok=True)
    name=OUT/'PREPARATION_SCOPE_V1.json'
    if name.exists():return read(name)
    now=datetime.now().astimezone()
    scope=dict(kind='FINITE_EVENT_OPERATOR_FULL_VALIDATION_PREPARATION',authorization='standing_authorization',
        human_message_id='01a1715e-74fe-7eb2-8a68-306e29cb5cf3',actual_start=now.isoformat(),
        deadline=(now+timedelta(hours=16)).isoformat(),source_seconds=7200,neural_seconds=36000,
        task_count=27,seeds=[0,1,2],arms=list(ARMS),candidate=CANDIDATE,
        logical_models=405,planned_new_models=360,planned_reused_models=45,
        formal_updates=720000,qualification_updates=150,optimizer_attempt_limit=720150,
        parameter_search=False,new_datasets=False,new_raw_reads=0,new_inverse_fits=0,reserved_signal_reads=0,
        payload_execution_ready=False,reason='Q4 failed; separately validate development-selected existing Q2, without changing old verdict')
    save(name,scope);return scope


def metadata():
    import json
    scope=start();need(datetime.now().astimezone()<datetime.fromisoformat(scope['deadline']),'preparation-inclusive deadline')
    data=parent.load_plan();need(len(data['cases'])==81,'whole original27 tasks x3 seeds')
    final_path=parent.OUT/'SOURCE_QUALIFICATION_FINAL_V1.json';final=read(final_path)
    db_binding=final['database'];need(sha(db_binding['path'])==db_binding['sha256'],'closed source qualification bytes')
    db=sqlite3.connect(Path(db_binding['path']).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        need(db.execute("select value from meta where key='DONE'").fetchone(),'original source qualification completed')
        receipts={k:json.loads(v) for k,v in db.execute("select key,detail from units where status='DONE' and key like 'I/%'")}
    finally:db.close()
    prior=ROOT/'results/wfdg/ress_interval_coverage'
    closed=read(prior/'CLOSED_RECEIPT_V1.json')
    for name in ('SOURCE_BANKS_V1.json','SCREEN_RESULT_V1.json','NEURAL_LOCK_V1.json','FROZEN_SOURCE_V1.json'):
        need(sha(prior/name)==closed['files_sha256'][name],'closed development parent '+name)
    old=read(prior/'NEURAL_LOCK_V1.json');old_source=read(prior/'SOURCE_BANKS_V1.json')
    groups,assets,cases={}, {}, []
    for key,case in sorted(data['cases'].items()):
        task=case['task'];bank=[];n=32768;m=(n-1)/2;length=math.ceil(.004*data['records'][case['bank'][0]['record']]['fs'])
        from design_e0_natural_source_probe_v2 import centered_three
        starts=centered_three(n,task['window'])
        for item in case['bank']:
            origin=item['record'];pair_key=item['pair'];pair=case['pairs'][pair_key]
            for record in pair:source_case(case,data['records'],record,context=True)
            identity=data['case_inverses'][key][pair_key];inverse=data['inverses'][identity]['binding']
            need(inverse['pair']==pair,'one exact original inverse and source pair')
            need('I/'+identity in receipts,'qualified context for every full-source inverse')
            asset=dict(inverse=inverse,qualified_context=receipts['I/'+identity]['packet'])
            if identity in assets:need(assets[identity]==asset,'shared source inverse binding')
            assets[identity]=asset;side=pair.index(origin);gkey=canonical(dict(inverse_identity=identity,source=side))
            g=dict(key=gkey,inverse_identity=identity,source=side,origin=origin,pair=pair,
                fs=data['records'][origin]['fs'],kernel_length=length,window=task['window'],starts=starts)
            if gkey in groups:need(groups[gkey]==g,'shared source geometry unchanged')
            groups[gkey]=g
            for q in (1.,*Q):
                lo=m+q*(min(starts)-m-(length-1));hi=m+q*(max(starts)+task['window']-1-m)
                need(lo>=0 and hi<=n-1,'all real-ratio queries have full event/kernel support')
            bank.append(dict(record=origin,domain=data['records'][origin]['domain'],label=data['records'][origin]['label'],group=gkey))
        cases.append(dict(key=key,task=task,seed=case['seed'],original_case_sha256=canonical(case),
            original_roles_sha256=canonical(case['roles']),source_training_windows=case['training_windows'],
            source_validation_windows=case['validation_windows'],bank_windows=3*len(bank),bank=bank))
    need(len(cases)==81 and Counter(c['task']['protocol'] for c in cases)=={'pu_clock4':12,'phm09_w3':60,'jnu_speed':9},'whole dataset/task panel')
    current={c['key']:c for c in cases};old_cases=old['cases']
    need(len(old_cases)==15 and all(current[k]['original_case_sha256']==canonical(old['original_cases'][k]) for k in old_cases),'same15 original cases for legal reuse')
    for identity,asset in assets.items():
        for b in asset.values():need(Path(b['path']).drive.upper()=='D:' and Path(b['path']).is_file(),'existing D source payload; no decode')
    compatibility={k:old_source['banks'][k] for k in old_cases}
    bindings={str(Path(b['path']).resolve()):b for a in assets.values() for b in a.values()}
    bindings.update({str(Path(b['path']).resolve()):b for b in compatibility.values()})
    groups=sorted(groups.values(),key=lambda g:(g['inverse_identity'],g['key']));count=len(groups)
    forecast=dict(source_groups=count,source_cases=81,source_inverse_pairs=len(assets),source_payload_decodes=len(bindings),
        source_payload_bytes=sum(Path(b['path']).stat().st_size for b in bindings.values()),
        renderer_calls=5*count,mode_sample_updates=5*3*48*32768*count,filter_calls=5*3*48*count,
        event_grid_scans=5*2*48*32768*count,normalization_samples=sum(5*48*g['kernel_length'] for g in groups),
        source_fft_1d_transforms=27*count,source_fft_points=27*sum(g['window'] for g in groups),
        owned_writes=count+81,owned_reads=count+81,owned_hashes=2*(count+81),compatibility_bank_reads=15,
        logical_models=405,reused_models=45,new_models=360,formal_updates=720000,qualification_updates=150)
    inputs={**data['input_sha256'],str(final_path):sha(final_path),str(prior/'CLOSED_RECEIPT_V1.json'):sha(prior/'CLOSED_RECEIPT_V1.json')}
    inputs.update({str(prior/n):sha(prior/n) for n in ('SOURCE_BANKS_V1.json','SCREEN_RESULT_V1.json','NEURAL_LOCK_V1.json','FROZEN_SOURCE_V1.json')})
    output=dict(kind='FULL27_EVENT_METADATA_PREFLIGHT',status='SOURCE_METADATA_COMPLETE_PAYLOAD_ENTRY_NOT_READY',campaign=scope,
        cases=cases,groups=groups,assets=assets,compatibility_banks=compatibility,
        original_case_references={v['key']:v['original_case_reference'] for v in data['graph']['cases']},
        original_records_metadata_sha256=canonical(data['records']),ratios=list(Q),forecast=forecast,
        reuse=dict(parent=str(prior),case_keys=sorted(old_cases),arms=['A0_STRONG_IN_MIXUP','A1_MATCHED_E0','Q2'],
            must_prove_actual_bank_byte_equivalence_before_any_archive_reuse=True),
        input_sha256=inputs,new_scientific_payload_reads=0,new_neural_actions=0,
        preparation_code_sha256={str(Path(__file__).resolve()):sha(__file__)})
    output['fingerprint']=canonical(output);save(OUT/'SOURCE_METADATA_V1.json',output)
    print(json.dumps(dict(status=output['status'],fingerprint=output['fingerprint'],forecast=forecast),ensure_ascii=False))
    return output


if __name__=='__main__':metadata()
