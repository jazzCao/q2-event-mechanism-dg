"""Source and neural schedules; metadata only until a locked runner starts."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import sqlite3
import compatible_set_plan as old
from source_rate_screen_plan import paths, expected
from e0_selection_entry_spec import local_closure

ROOT=old.ROOT
OUT=ROOT/'results/wfdg/ress_rate_impulse_screen'
PARENT=ROOT/'results/wfdg/ress_compatible_set'
read,save,sha,canonical,require=old.read,old.save,old.sha,old.canonical,old.require
ARMS=('C0_RATE_IMPULSE','G0_GLOBAL_GAIN','G1_EVENT_GAIN','N0_INVERSE_AMPLITUDE','F0_UNEDITED_BANK')
ANCHORS=('A0_STRONG_IN_MIXUP','A1_MATCHED_E0')
P0_CASES=('pu_clock4/ABCtoD/s0','phm09_w3/adj_AC_B/s0','jnu_speed/A/s0')
STEPS=(0,1000,2000)
LOCK=OUT/'NEURAL_LOCK_V1.json'
TESTS=OUT/'CPU_TESTS_V1.json'
ENTRY=('rate_impulse_screen_plan.py','rate_impulse_screen_source.py','rate_impulse_screen_neural.py',
       'run_rate_impulse_screen.py','test_rate_impulse_screen.py','continue_rate_impulse_screen.py','check_rate_impulse_screen.py')


def clock(scope):
    require(datetime.now().astimezone()<datetime.fromisoformat(scope['deadline']),'original rate-impulse campaign deadline')


def verify(lock):
    clock(lock['campaign'])
    require(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'immutable current lock')
    for path,digest in {**lock['input_sha256'],**lock['code_sha256']}.items():
        require(sha(path)==digest,'changed locked input/code '+path)


def prepare_source():
    scope=read(OUT/'CAMPAIGN_SCOPE_V1.json');clock(scope)
    source=read(PARENT/'SOURCE_LOCK_V2.json');parent=read(ROOT/'results/wfdg/ress_source_rate_screen/NEURAL_LOCK_V2.json')
    from natural_mapping_probe_plan import source_case
    require(len(source['cases'])==10 and len(source['groups'])==2150 and len(source['assets'])==1428,'same entire five-task source')
    closed=read(PARENT/'CLOSED_RECEIPT_V1.json')
    require(sha(PARENT/'NEURAL_LOCK_V1.json')==closed['files_sha256']['NEURAL_LOCK_V1.json'],'closed matched anchor implementation')
    qualified=ROOT/'results/wfdg/ress_rate_impulse_qualification/CLOSED_RECEIPT_V1.json'
    require(read(qualified)['scientific']=='SOURCE_QUALIFIED_CLASSIFICATION_UNTESTED','actual new source qualification passed')
    for case in source['cases']:
        original=source['original_cases'][case['key']]
        require(canonical(original)==case['original_case_sha256'],'same original source roles')
        by_group={g['key']:g for g in source['groups']}
        for item in case['bank']:
            group=by_group[item['group']]
            for record in group['pair']:source_case(original,source['records'],record,context=True)
            require(item['record']==group['origin'],'source bank identity')
    bindings=[v for asset in source['assets'].values() for v in asset.values()]+list(source['anchors'].values())
    sizes={str(Path(b['path']).resolve()):Path(b['path']).stat().st_size for b in bindings}
    require(len(sizes)==5006 and sum(sizes.values())<=scope['limits']['source_payload_bytes'],'all finite existing source files')
    groups=sorted(source['groups'],key=lambda g:(g['inverse_identity'],g['key']))
    forecast=dict(cases=10,groups=2150,inverse_pairs=1428,source_payload_decodes=5006,source_payload_bytes=sum(sizes.values()),
        source_fft_1d_transforms=515*2150,source_fft_points=482*65536*2150+sum(33*g['window'] for g in groups),
        owned_writes=2160,owned_reads=2160,owned_hashes=4320)
    for name in ('source_payload_decodes','source_fft_1d_transforms','source_fft_points'):
        require(forecast[name]<=scope['limits'][name],'compiled source quota '+name)
    tests=read(OUT/'CPU_TESTS_V1.json');require(tests['status']=='PASS' and tests['torch_imported'] is False,'CPU qualification before source')
    inputs={str(v):sha(v) for v in (OUT/'CAMPAIGN_SCOPE_V1.json',OUT/'EXECUTION_SPEC_V1.md',OUT/'CPU_TESTS_V1.json',
        PARENT/'SOURCE_LOCK_V2.json',PARENT/'CLOSED_RECEIPT_V1.json',qualified,
        ROOT/'results/wfdg/ress_rate_impulse_qualification/NEXT_CLASSIFICATION_RESEARCH_CARD.md')}
    code=local_closure([Path(__file__).with_name(n) for n in ENTRY])
    require(all(sha(k)==v for k,v in tests['code_sha256'].items()),'current tested code')
    lock=dict(kind='RATE_IMPULSE_FULL_SOURCE_V1',campaign=scope,cases=source['cases'],original_cases=source['original_cases'],
        records=source['records'],groups=groups,assets=source['assets'],anchors=source['anchors'],payload_sizes=sizes,
        forecast=forecast,limits=scope['limits'],input_sha256=inputs,code_sha256=code,source_only=True)
    lock['fingerprint']=canonical(lock);save(OUT/'SOURCE_LOCK_V1.json',lock);return lock


def schedule(source,parent):
    cases={c['key']:dict(c,real_arms=list(ARMS),aliases={}) for c in source['cases']}
    store=source['campaign']['store'];chains={};checkpoints={};predictions={};vectors={}
    def add(name,key,arm,qualification=False,branch=None):
        chains[name]=dict(case_key=key,arm=arm,qualification=qualification,branch=branch,steps=5 if qualification else 2000)
        saved=(5,) if branch=='continuous' else (2,5) if qualification else tuple(range(0,2001,250))
        for step in saved:checkpoints[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        vkey=('qualification_source/' if qualification else 'source/')+key
        v=expected(parent['records'],parent['original_cases'][key],'va')
        if qualification:v={k:value[:32] for k,value in v.items()}
        vectors[vkey]=v
        for step in ((0,5) if qualification else STEPS):
            predictions[paths(store,name,'source',step)]=dict(chain=name,case_key=key,side='source',step=step,qualification=qualification,expected_key=vkey)
        if not qualification:
            vkey='target/'+key;vectors[vkey]=expected(parent['records'],parent['original_cases'][key],'target')
            for policy in ('last','selected'):
                path=str(Path(store)/'neural'/name/f'target_{policy}.npz')
                predictions[path]=dict(chain=name,case_key=key,side='target',policy=policy,qualification=False,expected_key=vkey)
    for key in P0_CASES:
        for arm in ARMS:
            for branch in ('continuous','resume'):add('P0/'+key+'/'+arm+'/'+branch,key,arm,True,branch)
    for key in sorted(cases):
        for arm in ARMS:add('MAIN/'+key+'/'+arm,key,arm)
    forward=sum(v['steps']*6*(120//(len(cases[v['case_key']]['task']['sources'])*cases[v['case_key']]['task']['classes']))*
        len(cases[v['case_key']]['task']['sources'])*cases[v['case_key']]['task']['classes'] for v in chains.values())
    forecast=dict(real_models=50,archived_models=20,formal_updates=100000,qualification_updates=150,training_forward_windows=forward,
        source_prediction_windows=sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='source'),
        target_prediction_windows=sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='target'),
        checkpoint_writes=len(checkpoints),probability_writes=len(predictions),neural_model_builds=245,checkpoint_reads=165,
        archive_probability_reads=40,repair_updates_not_automatically_enabled=50)
    require(len(chains)==80 and len(checkpoints)==495 and len(predictions)==310 and forward==69464700,'exact six-view50model schedule')
    return dict(cases=cases,chains=chains,checkpoints=checkpoints,predictions=predictions,vectors=vectors,forecast=forecast)


def compile_neural():
    source=read(OUT/'SOURCE_LOCK_V1.json');verify(source);banks=read(OUT/'SOURCE_BANKS_V1.json')
    require(banks['fingerprint']==source['fingerprint'] and banks['status']=='RATE_IMPULSE_SOURCE_BANKS_COMPLETE','all source generation first')
    source_db=OUT/'SOURCE_CONTROL.sqlite';db=sqlite3.connect(source_db.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        require(db.execute("SELECT v FROM meta WHERE k='DONE'").fetchone() and
            db.execute("SELECT count(*) FROM units WHERE state='DONE'").fetchone()[0]==2160 and
            db.execute("SELECT count(*) FROM units WHERE state!='DONE'").fetchone()[0]==0,'complete actual source journal')
    finally:db.close()
    parent_path=PARENT/'NEURAL_LOCK_V1.json';parent=read(parent_path);closed=read(PARENT/'CLOSED_RECEIPT_V1.json')
    require(closed['files_sha256']['NEURAL_LOCK_V1.json']==sha(parent_path),'unchanged closed parent lock')
    archive_db=PARENT/'NEURAL_CONTROL.sqlite';db=sqlite3.connect(archive_db.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        require(db.execute("SELECT v FROM meta WHERE k='DONE'").fetchone(),'closed archived NN producer')
        artifacts={path:read_detail for path,read_detail in db.execute('SELECT path,detail FROM artifacts')}
        import json
        artifacts={k:json.loads(v) for k,v in artifacts.items()}
        draws=json.loads(db.execute("SELECT v FROM meta WHERE k='P0_DRAWS'").fetchone()[0])
        finished={k:json.loads(v) for k,v in db.execute('SELECT chain,detail FROM finished')}
    finally:db.close()
    plan=schedule(source,parent);archive=[]
    for path,spec in parent['predictions'].items():
        arm=parent['chains'][spec['chain']]['arm']
        if spec['side']=='target' and arm in ANCHORS:
            archive.append(dict(path=path,sha256=artifacts[path]['sha256'],spec=spec,arm=arm,
                checkpoint_sha256=artifacts[finished[spec['chain']][spec['policy']]]['sha256'],
                parent_fingerprint=parent['fingerprint'],expected=parent['vectors'][spec['expected_key']]))
    require(len(archive)==40,'exact20 matched archived models, two policies')
    domains={(c['task']['protocol'],d) for c in plan['cases'].values() for d in [*c['task']['sources'],c['task']['target']]}
    features=[v for v in parent['feature_bindings'] if (v['protocol'],v['domain']) in domains]
    limits=dict(source['campaign']['limits'])
    limits.update(feature_decodes=len(features),feature_hashes=len(features),bank_decodes=10,bank_hashes=10,
        checkpoint_writes=495,checkpoint_reads=165,checkpoint_hashes=1100,probability_writes=310,
        probability_reads=310,probability_hashes=700,neural_model_builds=245,archive_probability_reads=40,archive_probability_hashes=40)
    for field in ('training_forward_windows','source_prediction_windows','target_prediction_windows'):
        require(plan['forecast'][field]<=limits[field],'exact full NN quota '+field)
    inputs=dict(source['input_sha256'],**{str(v):sha(v) for v in
        (OUT/'SOURCE_LOCK_V1.json',OUT/'SOURCE_BANKS_V1.json',source_db,parent_path,archive_db,
         PARENT/'CLOSED_RECEIPT_V1.json',PARENT/'SCREEN_RESULT_V1.json',OUT/'CPU_TESTS_V1.json')})
    lock=dict(kind='RATE_IMPULSE_SEVEN_ROW_CLASSIFICATION_V1',campaign=source['campaign'],**plan,
        original_cases=source['original_cases'],records=source['records'],banks=banks['banks'],feature_bindings=features,
        expected_FFT_code_sha256=parent['expected_FFT_code_sha256'],initial_encoder_sha256=parent['initial_encoder_sha256'],
        parent_p0_draws={k:draws['P0/'+k+'/A1_MATCHED_E0/continuous'] for k in P0_CASES},archive=archive,
        source_fingerprint=source['fingerprint'],limits=limits,control_filename=str(OUT/'NEURAL_CONTROL.sqlite'),
        input_sha256=inputs,code_sha256=source['code_sha256'],execution_ready=True,
        gate=dict(primary='last',acc=.003,specificity=.001,protocol_floor=-.005,epsilon=1e-12))
    lock['fingerprint']=canonical(lock);save(LOCK,lock);return lock
