"""Complete source/neural lock compiler with exact archive identities."""
from datetime import datetime
import copy
import gzip
import json
import os
from pathlib import Path
import sqlite3
import event_full_validation_plan as p
from event_full_validation_schedule import forecast
from source_rate_screen_plan import expected,paths
from e0_selection_entry_spec import local_closure

ENTRY=tuple('event_full_validation_'+n+'.py' for n in
    ('plan','schedule','source','decision','execution','neural','run','check','continue','test'))
STEPS=(0,1000,2000)
NN_NAME='NEURAL_LOCK_V1.json.gz'
TEST_NAME='CPU_TESTS_V2.json'


def clock(scope):
    p.need(datetime.now().astimezone()<datetime.fromisoformat(scope['deadline']),'original16h total deadline includes preparation')


def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)
    return p.read(path)


def compressed_save(path,value):
    with Path(path).open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as z:
            z.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8'))
        f.flush();os.fsync(f.fileno())


def verify(lock):
    clock(lock['campaign'])
    p.need(p.canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'complete scientific fingerprint')
    for path,h in {**lock['input_sha256'],**lock['code_sha256']}.items():p.need(p.sha(path)==h,'bound code/input '+path)


def prepare_source():
    meta=p.read(p.OUT/'SOURCE_METADATA_V1.json');clock(meta['campaign'])
    p.need(p.canonical({k:v for k,v in meta.items() if k!='fingerprint'})==meta['fingerprint'],'original whole-source metadata preflight')
    tests=p.read(p.OUT/TEST_NAME);code=local_closure([Path(__file__).with_name(n) for n in ENTRY])
    p.need(tests['status']=='PASS' and not tests['torch_imported'] and tests['code_sha256']==code,'tested entire source/neural/audit entry before any payload')
    campaign=dict(meta['campaign'],store=str(Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f')/'ress_event_full_validation_20261027_1723'),payload_execution_ready=True)
    counters=('source_payload_decodes','source_payload_bytes','renderer_calls','mode_sample_updates','filter_calls','event_grid_scans',
        'normalization_samples','source_fft_1d_transforms','source_fft_points','owned_writes','owned_reads','owned_hashes','compatibility_bank_reads')
    limits={k:meta['forecast'][k] for k in counters}
    bindings={str(Path(b['path']).resolve()):b for a in meta['assets'].values() for b in a.values()}
    bindings.update({str(Path(b['path']).resolve()):b for b in meta['compatibility_banks'].values()})
    sizes={k:Path(k).stat().st_size for k in bindings}
    p.need(len(sizes)==limits['source_payload_decodes'] and sum(sizes.values())==limits['source_payload_bytes'],'complete actual source read forecast')
    inputs=dict(meta['input_sha256'])
    for name in ('PREPARATION_SCOPE_V1.json','SOURCE_METADATA_V1.json','NEURAL_FORECAST_V1.json','RESEARCH_CARD_V1.md',TEST_NAME,'PRELAUNCH_ENGINEERING_NOTE_V1.md'):
        inputs[str(p.OUT/name)]=p.sha(p.OUT/name)
    auth=p.ROOT/'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md';inputs[str(auth)]=p.sha(auth)
    numerical=p.ROOT/'results/wfdg/ress_continuous_event_operator/repair_v2/CLOSED_RECEIPT_V1.json';inputs[str(numerical)]=p.sha(numerical)
    lock=dict(kind='FULL27_FIXED_Q2_SOURCE',campaign=campaign,cases=meta['cases'],groups=meta['groups'],assets=meta['assets'],
        compatibility_banks=meta['compatibility_banks'],ratios=meta['ratios'],payload_sizes=sizes,forecast=meta['forecast'],
        limits=limits,input_sha256=inputs,code_sha256=code)
    lock['fingerprint']=p.canonical(lock);p.save(p.OUT/'SOURCE_LOCK_V1.json',lock);verify(lock);return lock


def schedule(meta,original,store):
    cases={c['key']:copy.deepcopy(c) for c in meta['cases']};old=set(meta['reuse']['case_keys']);anchors=set(meta['reuse']['arms'])
    chains={};states={};predictions={};vectors={}
    for key,c in cases.items():c['real_arms']=[a for a in p.ARMS if key not in old or a not in anchors]
    def add(key,arm,qualification=False,branch=None):
        name=('P0/' if qualification else 'MAIN/')+key+'/'+arm+('/'+branch if qualification else '')
        chains[name]=dict(case_key=key,arm=arm,qualification=qualification,branch=branch,steps=5 if qualification else 2000)
        for step in ((5,) if branch=='continuous' else (2,5) if qualification else range(0,2001,250)):
            states[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        vkey=('qualification_source/' if qualification else 'source/')+key
        v=expected(original['records'],original['cases'][key],'va');vectors[vkey]={k:x[:32] for k,x in v.items()} if qualification else v
        for step in ((0,5) if qualification else STEPS):
            predictions[paths(store,name,'source',step)]=dict(chain=name,case_key=key,side='source',step=step,qualification=qualification,expected_key=vkey)
        if not qualification:
            vkey='target/'+key;vectors[vkey]=expected(original['records'],original['cases'][key],'target')
            for policy in ('last','selected'):
                predictions[str(Path(store)/'neural'/name/f'target_{policy}.npz')]=dict(chain=name,case_key=key,side='target',policy=policy,qualification=False,expected_key=vkey)
    for key in p.P0_CASES:
        for arm in p.ARMS:
            for branch in ('continuous','resume'):add(key,arm,True,branch)
    for key,c in cases.items():
        for arm in c['real_arms']:add(key,arm)
        vectors.setdefault('target/'+key,expected(original['records'],original['cases'][key],'target'))
    f=forecast(meta)
    p.need(len(chains)==390 and len(states)==f['checkpoint_writes'] and len(predictions)==f['probability_writes'],'exact360 new plus30 P0 chains')
    p.need(sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='source')==f['source_prediction_windows'],'all source windows')
    p.need(sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='target')==f['target_prediction_windows'],'all new target windows')
    return dict(cases=cases,chains=chains,checkpoints=states,predictions=predictions,vectors=vectors)


def archive(meta,vectors):
    folder=Path(meta['reuse']['parent']);closed=p.read(folder/'CLOSED_RECEIPT_V1.json');old=p.read(folder/'NEURAL_LOCK_V1.json')
    p.need(p.sha(closed['database']['path'])==closed['database']['sha256'],'closed parent model/artifact ledger')
    for path,h in old['code_sha256'].items():p.need(p.sha(path)==h,'archived training implementation still bound')
    frozen=p.read(folder/'FROZEN_SOURCE_V1.json')
    db=sqlite3.connect(Path(closed['database']['path']).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        finished={k:json.loads(v) for k,v in db.execute('select chain,detail from finished')}
        artifacts={k:json.loads(v) for k,v in db.execute('select path,detail from artifacts')}
    finally:db.close()
    models={};probabilities={};critical={}
    for key in meta['reuse']['case_keys']:
        for arm in meta['reuse']['arms']:
            name='MAIN/'+key+'/'+arm;info=finished[name];states={}
            for step in STEPS:
                path=paths(old['campaign']['store'],name,'checkpoint',step);h=artifacts[path]['sha256']
                p.need(frozen['states'][path]==h,'old critical state froze before parent targets');states[path]=h;critical[path]=h
            models[name]=dict(case_key=key,arm=arm,detail=info,states=states,parent_fingerprint=old['fingerprint'])
            for policy in ('last','selected'):
                path=str(Path(old['campaign']['store'])/'neural'/name/f'target_{policy}.npz');a=artifacts[path]
                p.need(a['at_unix']>=frozen['at_unix'],'actual parent target chronology')
                probabilities[path]=dict(path=path,sha256=a['sha256'],chain=name,case_key=key,arm=arm,policy=policy,
                    expected_key='target/'+key,n=len(vectors['target/'+key]['y']),parent_fingerprint=old['fingerprint'],
                    checkpoint=info[policy],checkpoint_sha256=artifacts[info[policy]]['sha256'],
                    step=2000 if policy=='last' else info['selected_step'],original_at_unix=a['at_unix'])
    p.need(len(models)==45 and len(probabilities)==90 and len(critical)==135,'exact45 original models/90 target archives/135 state hashes')
    return dict(archive_models=models,archive_predictions=probabilities,archive_critical_states=critical)


def compile_neural():
    source=p.read(p.OUT/'SOURCE_LOCK_V1.json');verify(source);banks=p.read(p.OUT/'SOURCE_BANKS_V1.json')
    p.need(banks['status']=='FULL81_SOURCE_BANKS_COMPLETE' and banks['fingerprint']==source['fingerprint'] and len(banks['archive_input_proofs'])==15,'actual all81 banks and45 reusable model inputs')
    db=sqlite3.connect((p.OUT/'SOURCE_CONTROL.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        p.need(db.execute("select count(*) from units where state='DONE'").fetchone()[0]==3453 and db.execute("select count(*) from units where state!='DONE'").fetchone()[0]==0,'all3372 groups/81 banks actually done')
        p.need(db.execute("select v from meta where k='DONE'").fetchone() and not db.execute("select v from meta where k='STOPPED'").fetchone(),'closed source producer')
    finally:db.close()
    meta=p.read(p.OUT/'SOURCE_METADATA_V1.json');original=p.parent.load_plan();s=schedule(meta,original,source['campaign']['store']);a=archive(meta,s['vectors'])
    registry_path=p.ROOT/'results/wfdg/ress_e0_source_selection/INPUT_REGISTRY_V1.json';anchor_path=p.ROOT/'results/wfdg/ress_e0_source_selection/ANCHOR_METADATA_V1.json'
    registry=p.read(registry_path);anchors=p.read(anchor_path);features=[v['binding'] for v in registry['caches'] if v['kind']=='full_feature']
    initial={v['case_key']:v['initial_encoder_sha256'] for v in anchors['cases'] if v['case_key'] in s['cases']}
    p.need(len(features)==17 and len(initial)==81,'all physical domains and old initial encoders')
    old=p.read(Path(meta['reuse']['parent'])/'NEURAL_LOCK_V1.json')
    p.need(all(initial[k]==h for k,h in old['initial_encoder_sha256'].items()),'same archived encoder initializations')
    f=forecast(meta);p.need(f==p.read(p.OUT/'NEURAL_FORECAST_V1.json')['forecast'],'whole exact budget unchanged')
    limits={k:v for k,v in f.items() if k not in ('real_new_models','reused_models','logical_models','new_models_by_protocol','critical_CPU_new_states')}
    inputs=dict(source['input_sha256'])
    for path in (p.OUT/'SOURCE_LOCK_V1.json',p.OUT/'SOURCE_BANKS_V1.json',p.OUT/'SOURCE_CONTROL.sqlite',p.OUT/'SOURCE_ACTUAL_GRANT_V1.json',p.OUT/'SOURCE_PRODUCER_EXIT_V1.json',registry_path,anchor_path,
                 Path(meta['reuse']['parent'])/'NEURAL_CONTROL.sqlite',Path(meta['reuse']['parent'])/'CLOSED_POSTCHECK_V1.json'):
        inputs[str(path)]=p.sha(path)
    lock=dict(kind='FULL27_FIXED_Q2_CLASSIFICATION',campaign=source['campaign'],**s,**a,
        original_cases={k:original['cases'][k] for k in s['cases']},records=original['records'],banks=banks['banks'],
        feature_bindings=features,initial_encoder_sha256=initial,expected_FFT_code_sha256=old['expected_FFT_code_sha256'],
        control_filename=str(p.OUT/'NEURAL_CONTROL.sqlite'),forecast=f,limits=limits,input_sha256=inputs,code_sha256=source['code_sha256'])
    p.need(sum(v['n'] for v in a['archive_predictions'].values())==f['archive_prediction_windows_read'],'exact archive window read bound')
    lock['fingerprint']=p.canonical(lock);compressed_save(p.OUT/NN_NAME,lock);verify(lock);return lock
