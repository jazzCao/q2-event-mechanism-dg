"""Whole fixed interval-coverage comparison; metadata-only compilation."""
from datetime import datetime
from pathlib import Path
import math
import copy
import source_rate_plan as base
from source_rate_screen_plan import expected,paths
from e0_selection_entry_spec import local_closure

ROOT=base.ROOT
OUT=ROOT/'results/wfdg/ress_interval_coverage'
read,save,sha,canonical,require=base.read,base.save,base.sha,base.canonical,base.require
ARMS=('A0_STRONG_IN_MIXUP','A1_MATCHED_E0','A2_ENDPOINT','Q2','Q4','NTIME','NFRONT')
TASKS=(('pu_clock4','ABCtoD'),('pu_clock4','BCDtoA'),('phm09_w3','adj_AC_B'),('phm09_w3','ext_AB_F'),('jnu_speed','A'))
P0_CASES=('pu_clock4/ABCtoD/s0','phm09_w3/adj_AC_B/s0','jnu_speed/A/s0')
STEPS=(0,1000,2000)
SOURCE_LOCK_NAME='SOURCE_LOCK_V2.json'
TEST_NAME='CPU_TESTS_V2.json'
ENTRY=('interval_coverage_plan.py','interval_coverage_core.py','interval_coverage_source.py','interval_coverage_neural.py',
       'run_interval_coverage.py','interval_coverage_decision.py','continue_interval_coverage.py','check_interval_coverage.py','test_interval_coverage.py')


def clock(scope):require(datetime.now().astimezone()<datetime.fromisoformat(scope['deadline']),'original fixed interval total deadline')


def verify(lock):
    clock(lock['campaign']);require(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'immutable interval lock')
    for k,v in {**lock['input_sha256'],**lock['code_sha256']}.items():require(sha(k)==v,'changed interval code/input '+k)


def nodes():
    a=math.log(1.5);hi=math.sqrt((3+2*math.sqrt(6/5))/7);lo=math.sqrt((3-2*math.sqrt(6/5))/7)
    return [2/3,1.5,math.exp(-a/math.sqrt(3)),math.exp(a/math.sqrt(3)),math.exp(-a*hi),math.exp(-a*lo),math.exp(a*lo),math.exp(a*hi)]


def source_metadata():
    import protected_response_run_spec as parent
    from natural_mapping_probe_plan import source_case
    data=parent.load_plan();cases=[];groups={};assets={};anchors={};original={};inputs={}
    for dirname in ('ress_rate_impulse_screen','ress_rate_impulse_repeat'):
        folder=ROOT/'results/wfdg'/dirname;old=read(folder/'SOURCE_LOCK_V1.json');closed=read(folder/'CLOSED_RECEIPT_V1.json')
        require(sha(folder/'SOURCE_BANKS_V1.json')==closed['files_sha256']['SOURCE_BANKS_V1.json'],'closed source parent '+dirname)
        for name in ('SOURCE_LOCK_V1.json','SOURCE_BANKS_V1.json','CLOSED_RECEIPT_V1.json'):inputs[str(folder/name)]=sha(folder/name)
        bygroup={g['key']:g for g in old['groups']};oldbanks=read(folder/'SOURCE_BANKS_V1.json')
        for c0 in old['cases']:
            if (c0['task']['protocol'],c0['task']['fold']) not in TASKS:continue
            key=c0['key'];c=copy.deepcopy(c0);o=data['cases'][key]
            require(canonical(o)==c['original_case_sha256']==canonical(old['original_cases'][key]),'current exact case roles '+key)
            require(key not in original,'one original split per seed/task');original[key]=o
            for item in c['bank']:
                g0=bygroup[item['group']]
                for record in g0['pair']:source_case(o,data['records'],record,context=True)
                require(g0['origin']==item['record'] and item['domain'] in c['task']['sources'],'every bank current source')
                require(data['records'][item['record']].get('bearing') not in ('K006','KA08','KA09'),'no reserved bearing')
                g={k:g0[k] for k in ('key','inverse_identity','source','origin','pair','fs','kernel_length','window','starts')}
                if g['key'] in groups:require(groups[g['key']]==g,'shared group exact geometry')
                groups[g['key']]=g;a=old['assets'][g['inverse_identity']]
                anchors[g['key']]=oldbanks['groups'][g['key']]
                if g['inverse_identity'] in assets:require(assets[g['inverse_identity']]==a,'same shared inverse/context binding')
                assets[g['inverse_identity']]=a
            c.update(real_arms=list(ARMS),aliases={});cases.append(c)
    require(len(cases)==15 and {(c['task']['protocol'],c['task']['fold'],c['seed']) for c in cases}=={(*t,s) for t in TASKS for s in (0,1,2)},'complete five task/three seed source panel')
    groups=sorted(groups.values(),key=lambda g:(g['inverse_identity'],g['key']))
    sizes={str(Path(b['path']).resolve()):Path(b['path']).stat().st_size for a in assets.values() for b in a.values()}
    sizes.update({str(Path(b['path']).resolve()):Path(b['path']).stat().st_size for b in anchors.values()})
    bygroup={g['key']:g for g in groups};n=len(groups);items=sum(len(c['bank']) for c in cases)
    forecast=dict(groups=n,cases=15,inverse_pairs=len(assets),source_payload_decodes=2*len(assets)+n,source_payload_bytes=sum(sizes.values()),
        renderer_calls=9*n,mode_sample_updates=9*3*48*32768*n,filter_calls=9*3*48*n,event_grid_scans=9*2*48*32768*n,
        normalization_samples=sum(9*48*g['kernel_length'] for g in groups),
        source_fft_1d_transforms=33*n+6*items,source_fft_points=33*sum(g['window'] for g in groups)+6*sum(bygroup[b['group']]['window'] for c in cases for b in c['bank']),
        owned_writes=n+15,owned_reads=n+15,owned_hashes=2*(n+15))
    return dict(cases=sorted(cases,key=lambda c:c['key']),original_cases=original,records=data['records'],groups=groups,assets=assets,anchors=anchors,
        payload_sizes=sizes,forecast=forecast,input_sha256=inputs)


def schedule(source):
    cases={c['key']:c for c in source['cases']};chains={};checkpoints={};predictions={};vectors={};store=source['campaign']['store']
    def add(key,arm,qualification=False,branch=None):
        name=('P0/' if qualification else 'MAIN/')+key+'/'+arm+('/'+branch if qualification else '')
        chains[name]=dict(case_key=key,arm=arm,qualification=qualification,branch=branch,steps=5 if qualification else 2000)
        for step in ((5,) if branch=='continuous' else (2,5) if qualification else range(0,2001,250)):
            checkpoints[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        vkey=('qualification_source/' if qualification else 'source/')+key
        value=expected(source['records'],source['original_cases'][key],'va')
        vectors[vkey]={k:v[:32] for k,v in value.items()} if qualification else value
        for step in ((0,5) if qualification else STEPS):
            predictions[paths(store,name,'source',step)]=dict(chain=name,case_key=key,side='source',step=step,qualification=qualification,expected_key=vkey)
        if not qualification:
            vkey='target/'+key;vectors[vkey]=expected(source['records'],source['original_cases'][key],'target')
            for policy in ('last','selected'):predictions[str(Path(store)/'neural'/name/f'target_{policy}.npz')]=dict(chain=name,case_key=key,side='target',policy=policy,qualification=False,expected_key=vkey)
    for key in P0_CASES:
        for arm in ARMS:
            for branch in ('continuous','resume'):add(key,arm,True,branch)
    for key in cases:
        for arm in ARMS:add(key,arm)
    forward=sum(v['steps']*6*(120//(len(cases[v['case_key']]['task']['sources'])*cases[v['case_key']]['task']['classes']))*len(cases[v['case_key']]['task']['sources'])*cases[v['case_key']]['task']['classes'] for v in chains.values())
    forecast=dict(real_models=105,formal_updates=210000,qualification_updates=210,optimizer_attempts=210210,training_forward_windows=forward,
        source_prediction_windows=sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='source'),
        target_prediction_windows=sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='target'),
        checkpoint_writes=len(checkpoints),checkpoint_reads=651,neural_model_builds=483,probability_writes=len(predictions),probability_reads=len(predictions),
        batch_reference_rows=2*6*sum(120//(len(cases[k]['task']['sources'])*cases[k]['task']['classes'])*len(cases[k]['task']['sources'])*cases[k]['task']['classes'] for k in P0_CASES)*3)
    require(len(chains)==147 and len(checkpoints)==1008 and len(predictions)==609 and forward==145802580,'complete fixed105 model schedule')
    return dict(cases=cases,chains=chains,checkpoints=checkpoints,predictions=predictions,vectors=vectors,forecast=forecast)


def prepare_source():
    scope=read(OUT/'CAMPAIGN_SCOPE_V1.json');clock(scope);tests=read(OUT/TEST_NAME);code=local_closure([Path(__file__).with_name(n) for n in ENTRY])
    require(tests['status']=='PASS' and not tests['torch_imported'] and code==tests['code_sha256'],'actual tested complete interval entry')
    data=source_metadata();inputs=data.pop('input_sha256')
    for n in ('CAMPAIGN_SCOPE_V1.json',TEST_NAME,'IMPLEMENTATION_SPEC_V1.md','PRELAUNCH_AMENDMENT_V2.md'):inputs[str(OUT/n)]=sha(OUT/n)
    for path in (ROOT/'results/wfdg/ress_continuous_event_operator/NEXT_CLASSIFICATION_DESIGN_V1.md',ROOT/'results/wfdg/ress_continuous_event_operator/repair_v2/CLOSED_RECEIPT_V1.json',ROOT/'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md'):
        inputs[str(path)]=sha(path)
    for k,v in data['forecast'].items():
        if k in scope['source_limits']:require(v<=scope['source_limits'][k],'whole source scope fits '+k)
    lock=dict(kind='FIXED_INTERVAL_COVERAGE_SOURCE',campaign=scope,**data,ratios=nodes(),limits=scope['source_limits'],input_sha256=inputs,code_sha256=code)
    lock['fingerprint']=canonical(lock);save(OUT/SOURCE_LOCK_NAME,lock);return lock


def compile_neural():
    source=read(OUT/SOURCE_LOCK_NAME);verify(source);banks=read(OUT/'SOURCE_BANKS_V1.json')
    require(banks['status']=='INTERVAL_SOURCE_CALIBRATED' and banks['fingerprint']==source['fingerprint'],'all15 calibrated source banks')
    registry=ROOT/'results/wfdg/ress_e0_source_selection/INPUT_REGISTRY_V1.json';anchor=ROOT/'results/wfdg/ress_e0_source_selection/ANCHOR_METADATA_V1.json'
    r,a=read(registry),read(anchor);s=schedule(source);ds={(c['task']['protocol'],d) for c in s['cases'].values() for d in (*c['task']['sources'],c['task']['target'])}
    features=[v['binding'] for v in r['caches'] if v['kind']=='full_feature' and (v['binding']['protocol'],v['binding']['domain']) in ds]
    require(len(features)==11,'same eleven full feature domains')
    old=read(ROOT/'results/wfdg/ress_rate_impulse_repeat/NEURAL_LOCK_V1.json')
    inputs=dict(source['input_sha256']);inputs.update({str(path):sha(path) for path in (OUT/SOURCE_LOCK_NAME,OUT/'SOURCE_BANKS_V1.json',OUT/'SOURCE_CONTROL.sqlite',registry,anchor)})
    limits=source['campaign']['neural_limits']
    for k,v in s['forecast'].items():
        if k in limits:require(v<=limits[k],'entire fixed neural comparison fits '+k)
    lock=dict(kind='FIXED_INTERVAL_COVERAGE_CLASSIFICATION',campaign=source['campaign'],**s,original_cases=source['original_cases'],records=source['records'],banks=banks['banks'],
        feature_bindings=features,expected_FFT_code_sha256=old['expected_FFT_code_sha256'],
        initial_encoder_sha256={v['case_key']:v['initial_encoder_sha256'] for v in a['cases'] if v['case_key'] in s['cases']},
        control_filename=str(OUT/'NEURAL_CONTROL.sqlite'),limits=limits,input_sha256=inputs,code_sha256=source['code_sha256'])
    require(len(lock['initial_encoder_sha256'])==15,'all15 original encoder initializations')
    lock['fingerprint']=canonical(lock);save(OUT/'NEURAL_LOCK_V1.json',lock);return lock
