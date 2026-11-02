"""Metadata-only fixed third-seed replication; original failures remain immutable."""
from datetime import datetime
from pathlib import Path
import math
import sqlite3
import json
import rate_impulse_screen_plan as old
from e0_selection_entry_spec import local_closure
from source_rate_screen_plan import expected,paths
ROOT=old.ROOT
OUT=ROOT/'results/wfdg/ress_rate_impulse_repeat'
read,save,sha,canonical,require=old.read,old.save,old.sha,old.canonical,old.require
ARMS=('A0_STRONG_IN_MIXUP','A1_MATCHED_E0','C0_RATE_IMPULSE')
TASKS=(('pu_clock4','ABCtoD'),('pu_clock4','BCDtoA'),('phm09_w3','adj_AC_B'),('phm09_w3','ext_AB_F'),('jnu_speed','A'))
P0_CASES=('pu_clock4/ABCtoD/s2','phm09_w3/adj_AC_B/s2','jnu_speed/A/s2')
STEPS=(0,1000,2000)
ENTRY=('rate_impulse_repeat_plan.py','rate_impulse_repeat_source.py','rate_impulse_repeat_neural.py','run_rate_impulse_repeat.py','test_rate_impulse_repeat.py','continue_rate_impulse_repeat.py','check_rate_impulse_repeat.py')


def clock(scope):require(datetime.now().astimezone()<datetime.fromisoformat(scope['deadline']),'original replication total deadline')


def verify(lock):
    clock(lock['campaign']);require(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'replication lock identity')
    for path,digest in {**lock['input_sha256'],**lock['code_sha256']}.items():require(sha(path)==digest,'unchanged replication code/input '+path)


def source_metadata(scope):
    import protected_response_run_spec as parent
    from natural_mapping_probe_plan import source_case
    from design_e0_natural_source_probe_v2 import centered_three,valid_support
    from fractions import Fraction
    data=parent.load_plan();original={k:v for k,v in data['cases'].items() if v['seed']==2 and (v['task']['protocol'],v['task']['fold']) in TASKS}
    require(len(original)==5,'exact five original seed2 splits')
    final=read(parent.OUT/'SOURCE_QUALIFICATION_FINAL_V1.json');dbpath=Path(final['database']['path'])
    require(sha(dbpath)==final['database']['sha256'],'closed all-source context journal')
    db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        require(db.execute("SELECT value FROM meta WHERE key='DONE'").fetchone(),'all original numerical context qualification complete')
        receipts={k:json.loads(v) for k,v in db.execute("SELECT key,detail FROM units WHERE status='DONE'")}
    finally:db.close()
    cases=[];groups={};assets={}
    for key,c in sorted(original.items()):
        bank=[];task=c['task'];starts=centered_three(32768,task['window'])
        for b in c['bank']:
            pair=c['pairs'][b['pair']]
            for record in pair:source_case(c,data['records'],record,context=True)
            identity=data['case_inverses'][key][b['pair']];inverse=data['inverses'][identity]['binding']
            require(inverse['pair']==pair,'current exact original inverse pair')
            assets[identity]=dict(inverse=inverse,qualified_context=receipts['I/'+identity]['packet'])
            side=pair.index(b['record']);gkey=canonical(dict(inverse_identity=identity,source=side));r=data['records'][b['record']]
            g=dict(key=gkey,inverse_identity=identity,source=side,origin=b['record'],pair=pair,witness_case=key,
                fs=r['fs'],kernel_length=math.ceil(.004*r['fs']),window=task['window'],starts=starts)
            require(all(valid_support(32768,g['kernel_length'],Fraction(q),starts,g['window']) for q in ('2/3','3/2')),'entire current source support')
            if gkey in groups:require(groups[gkey]['origin']==g['origin'] and groups[gkey]['starts']==g['starts'],'same reused source geometry')
            groups[gkey]=g;bank.append(dict(record=b['record'],domain=r['domain'],label=r['label'],group=gkey))
        cases.append(dict(key=key,task=task,seed=2,source_training_windows=c['training_windows'],source_validation_windows=c['validation_windows'],
            source_speed_diversity=len({data['records'][r]['shaft_hz'] for r,v in c['roles'].items() if v['tr']})>1,
            bank=bank,bank_windows=3*len(bank),real_arms=list(ARMS),aliases={},
            original_case_sha256=canonical(c),original_roles_sha256=canonical(c['roles'])))
    require(len(assets)==1187,'fixed original1187 inverse subset')
    groups=sorted(groups.values(),key=lambda g:(g['inverse_identity'],g['key']))
    sizes={str(Path(b['path']).resolve()):Path(b['path']).stat().st_size for a in assets.values() for b in a.values()}
    n=len(groups);forecast=dict(cases=5,groups=n,inverse_pairs=len(assets),source_payload_decodes=2*len(assets),source_payload_bytes=sum(sizes.values()),
        source_fft_1d_transforms=515*n,source_fft_points=482*65536*n+sum(33*g['window'] for g in groups),owned_writes=n+5,owned_reads=n+5,owned_hashes=2*(n+5))
    inputs=dict(data['input_sha256']);inputs.update({str(v):sha(v) for v in (parent.OUT/'SOURCE_RUN_LOCK_V1.json',parent.OUT/'SOURCE_QUALIFICATION_FINAL_V1.json',dbpath)})
    return dict(cases=cases,original_cases=original,records=data['records'],groups=groups,assets=assets,payload_sizes=sizes,forecast=forecast,input_sha256=inputs)


def prepare_source():
    scope=read(OUT/'CAMPAIGN_SCOPE_V1.json');clock(scope);tests=read(OUT/'CPU_TESTS_V3.json')
    require(tests['status']=='PASS' and not tests['torch_imported'],'actual no-Torch CPU tests')
    data=source_metadata(scope);code=local_closure([Path(__file__).with_name(n) for n in ENTRY]);require(code==tests['code_sha256'],'exact tested full entry')
    for k,n in data['forecast'].items():
        if k in scope['source_limits']:require(n<=scope['source_limits'][k],'entire source fits '+k)
    inputs=data.pop('input_sha256');inputs.update({str(v):sha(v) for v in (OUT/'CAMPAIGN_SCOPE_V1.json',OUT/'CPU_TESTS_V3.json',OUT/'RESEARCH_SPEC_V1.md',
        ROOT/'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md')})
    lock=dict(kind='FIXED_RATE_IMPULSE_SEED2_SOURCE',campaign=scope,**data,limits=scope['source_limits'],input_sha256=inputs,code_sha256=code)
    lock['fingerprint']=canonical(lock);save(OUT/'SOURCE_LOCK_V1.json',lock);return lock


def schedule(source):
    chains={};checkpoints={};predictions={};vectors={};store=source['campaign']['store'];cases={c['key']:c for c in source['cases']}
    def add(name,key,arm,qualification=False,branch=None):
        chains[name]=dict(case_key=key,arm=arm,qualification=qualification,branch=branch,steps=5 if qualification else 2000)
        saved=(5,) if branch=='continuous' else (2,5) if qualification else tuple(range(0,2001,250))
        for step in saved:checkpoints[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        vkey=('qualification_source/' if qualification else 'source/')+key
        value=expected(source['records'],source['original_cases'][key],'va')
        if qualification:value={k:v[:32] for k,v in value.items()}
        vectors[vkey]=value
        for step in ((0,5) if qualification else STEPS):predictions[paths(store,name,'source',step)]=dict(chain=name,case_key=key,side='source',step=step,qualification=qualification,expected_key=vkey)
        if not qualification:
            vkey='target/'+key;vectors[vkey]=expected(source['records'],source['original_cases'][key],'target')
            for policy in ('last','selected'):predictions[str(Path(store)/'neural'/name/f'target_{policy}.npz')]=dict(chain=name,case_key=key,side='target',policy=policy,qualification=False,expected_key=vkey)
    for key in P0_CASES:
        for arm in ARMS:
            for branch in ('continuous','resume'):add('P0/'+key+'/'+arm+'/'+branch,key,arm,True,branch)
    for key in sorted(cases):
        for arm in ARMS:add('MAIN/'+key+'/'+arm,key,arm)
    forward=sum(c['steps']*6*(120//(len(cases[c['case_key']]['task']['sources'])*cases[c['case_key']]['task']['classes']))*len(cases[c['case_key']]['task']['sources'])*cases[c['case_key']]['task']['classes'] for c in chains.values())
    forecast=dict(real_models=15,formal_updates=30000,qualification_updates=90,optimizer_attempts=30090,training_forward_windows=forward,
        source_prediction_windows=sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='source'),
        target_prediction_windows=sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='target'),
        checkpoint_writes=len(checkpoints),probability_writes=len(predictions),neural_model_builds=87,checkpoint_reads=99,probability_reads=len(predictions),batch_reference_rows=8376)
    require(len(chains)==33 and len(checkpoints)==162 and len(predictions)==111 and forward==20870820,'exact new-seed-only schedule')
    return dict(cases=cases,chains=chains,checkpoints=checkpoints,predictions=predictions,vectors=vectors,forecast=forecast)


def compile_neural():
    source=read(OUT/'SOURCE_LOCK_V1.json');verify(source);banks=read(OUT/'SOURCE_BANKS_V1.json')
    require(banks['fingerprint']==source['fingerprint'] and banks['status']=='FIXED_SEED2_SOURCE_COMPLETE','whole current source complete')
    parent=ROOT/'results/wfdg/ress_rate_impulse_screen';close=read(parent/'CLOSED_RECEIPT_V1.json')
    require(close['files_sha256']['SCREEN_RESULT_V1.json']==sha(parent/'SCREEN_RESULT_V1.json'),'unchanged failed original pilot')
    registry=ROOT/'results/wfdg/ress_e0_source_selection/INPUT_REGISTRY_V1.json';anchor=ROOT/'results/wfdg/ress_e0_source_selection/ANCHOR_METADATA_V1.json'
    r,a=read(registry),read(anchor);plan=schedule(source)
    domains={(c['task']['protocol'],d) for c in plan['cases'].values() for d in [*c['task']['sources'],c['task']['target']]}
    features=[v['binding'] for v in r['caches'] if v['kind']=='full_feature' and (v['binding']['protocol'],v['binding']['domain']) in domains]
    require(len(features)==11,'exact same eleven feature domains')
    oldlock=read(parent/'NEURAL_LOCK_V1.json')
    inputs=dict(source['input_sha256']);inputs.update({str(v):sha(v) for v in (OUT/'SOURCE_LOCK_V1.json',OUT/'SOURCE_BANKS_V1.json',OUT/'SOURCE_CONTROL.sqlite',registry,anchor,parent/'SCREEN_RESULT_V1.json',parent/'CLOSED_RECEIPT_V1.json',parent/'NEURAL_LOCK_V1.json')})
    limits=source['campaign']['neural_limits']
    for k,n in plan['forecast'].items():
        if k in limits:require(n<=limits[k],'entire neural schedule fits '+k)
    lock=dict(kind='FIXED_RATE_IMPULSE_SEED2_CLASSIFICATION',campaign=source['campaign'],**plan,original_cases=source['original_cases'],records=source['records'],
        banks=banks['banks'],feature_bindings=features,expected_FFT_code_sha256=oldlock['expected_FFT_code_sha256'],
        initial_encoder_sha256={v['case_key']:v['initial_encoder_sha256'] for v in a['cases'] if v['case_key'] in plan['cases']},
        original_rows=[v for v in read(parent/'SCREEN_RESULT_V1.json')['rows'] if v['arm'] in ARMS],
        original_pilot_status='CANDIDATE_SCREEN_FAILED',limits=limits,control_filename=str(OUT/'NEURAL_CONTROL.sqlite'),
        input_sha256=inputs,code_sha256=source['code_sha256'])
    require(len(lock['original_rows'])==60,'all original two-seed three-arm roles retained')
    lock['fingerprint']=canonical(lock);save(OUT/'NEURAL_LOCK_V1.json',lock);return lock
