"""Fixed external-baseline scope and exact bills. Metadata only, no approval."""
import copy
from pathlib import Path
import sqlite3
from pcr_campaign_core import canonical,read,sha,write_new
from event_native_main_sources import inventory

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/wfdg'
OUT=BASE/'ress_event_native_main'
STORE=Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_event_native_main')
CONFIGS=('A0_AUTHOR_CE','A2_AUTHOR_IN_MIXUP','C0_CCDG_CODE')
QUAL_CASES={('phm09_w3','adj_BC_A',0),('jnu_speed','A',0)}


def case_key(case):return case['task']['protocol']+'/'+case['task']['fold']+'/s'+str(case['seed'])
def model_key(case,config):
    if config not in CONFIGS:raise ValueError('only locked three native methods')
    return 'formal/'+case_key(case)+'/'+config


def author_math_bindings():
    path=BASE/'ress_event_native_repeat/PROPOSED_LOCK.json';old=read(path)
    filenames=['scripts/50_bearing_cnsr/event_author_native_engine.py',
        'scripts/50_bearing_cnsr/wfdg/ress_ccdg_matched/author.py',
        'related projects/CCDG-main/domainbed/TSnetworks.py','related projects/CCDG-main/domainbed/lib/misc.py',
        'scripts/50_bearing_cnsr/mocd_audit_cpu_view_v2.py']
    bindings={}
    for rel in filenames:
        p=(ROOT/rel).resolve();expected=old['code_sha256'][str(p)]
        if sha(p)!=expected:raise PermissionError('historically qualified native mathematics changed '+str(p))
        bindings[str(p)]=expected
    return dict(status='HISTORICAL_MATH_SHA_MATCH_NOT_NEW_CUDA_PASS',bindings=bindings,
        legacy_lock_binding={str(path):sha(path)},formal_steps=3001,source_steps=[0,1000,2000,3001],
        optimizer=dict(name='Adam',lr=.001,weight_decay=.00005,betas=[.9,.999],eps=1e-8),
        scheduler=dict(name='StepLR',step_size=2400,gamma=.1),batch_per_source=32,
        losses=dict(A0_AUTHOR_CE='CE',A2_AUTHOR_IN_MIXUP='0.5 factual CE + 0.5 Mixup CE; Beta(0.2,0.2)',
            C0_CCDG_CODE='0.9 CE + 0.1 author FEATURE contrast; temperature0.7'),
        actual_new_neural_actions=0)


def snapshot():
    scope_path=BASE/'ress_event_paper_inventory/PHM_SOURCE_SCOPE.json'
    cross_path=BASE/'ress_cf_cross_range/DATA_MANIFEST.json'
    features_path=BASE/'ress_cf_cross_range/EFFECTIVE_CACHE_BINDINGS.json'
    phm_path=BASE/'ress_event_phm_main';phm_lock=read(phm_path/'PROPOSED_LOCK.json');audit=read(phm_path/'FINAL_AUDIT.json')
    if audit['status']!='PASS' or audit['fingerprint']!=phm_lock['fingerprint'] or audit['metric_sha256']!=sha(phm_path/'METRICS.json'):raise PermissionError('completed fixed PHM source cache first')
    control=phm_path/'control/CONTROL.sqlite'
    with sqlite3.connect(control.resolve().as_uri()+'?mode=ro',uri=True) as db:
        if db.execute('SELECT status FROM producer').fetchall()!=[('STOPPED',)]:raise PermissionError('old PHM producer must be terminal')
        row=db.execute('SELECT value FROM journal WHERE name=?',('EFFECTIVE_CACHE',)).fetchone()
        import json
        effective=json.loads(row[0])
    phm=read(scope_path);cross=read(cross_path);oldfeatures=read(features_path)
    records=copy.deepcopy(phm['records']);records.update({k:r for k,r in cross['records'].items() if r['protocol']=='jnu_speed'})
    raw=dict(effective['raw_bindings']);raw.update({k:b for k,b in oldfeatures['raw_bindings'].items() if k in records and records[k]['protocol']=='jnu_speed'})
    if set(raw)!=set(records) or len(records)!=172:raise AssertionError('160PHM+12JNU existing records')
    features=list(effective['feature_bindings'])+[b for b in oldfeatures['all_features'] if b['protocol']=='jnu_speed']
    if len(features)!=13 or len({(b['protocol'],b['domain']) for b in features})!=13:raise AssertionError('all13 existing common FFT caches')
    cases=[]
    for original in list(phm['cases'])+[x for x in cross['cases'] if x['task']['protocol']=='jnu_speed']:
        case={k:copy.deepcopy(original[k]) for k in ('task','seed','roles')}
        case['source_inventory']=inventory(records,case)
        active=[b for b in features if b['protocol']==case['task']['protocol'] and b['domain'] in case['task']['sources']]
        if len(active)!=2:raise AssertionError('source feature identities, not target fitted context')
        case['source_fingerprint']=canonical(dict(task=case['task'],seed=case['seed'],roles=case['roles'],source_cache=active))
        cases.append(case)
    if len(cases)!=69 or len({model_key(x,a) for x in cases for a in CONFIGS})!=207:raise AssertionError('full69cases/207native models')
    by_task={}
    for case in cases:
        by_task.setdefault(case['task']['protocol'],set()).add(case['task']['fold'])
    if {k:len(v) for k,v in by_task.items()}!={'phm09_w3':20,'jnu_speed':3}:raise AssertionError('no PU/reduced task subset')
    # Existing qualified mathematical engine is not edited or reimplemented.
    legacy=read(BASE/'ress_event_native_repeat/PROPOSED_LOCK.json')
    engine=Path(__file__).with_name('event_author_native_engine.py').resolve()
    if legacy['code_sha256'][str(engine)]!=sha(engine):raise PermissionError('native mathematical recipe must remain unchanged')
    formal_source=sum(sum(v['va']['windows'] for v in x['source_inventory'].values()) for x in cases)*len(CONFIGS)*4
    formal_target=sum(sum(len(r['offsets']) for r in records.values() if r['protocol']==x['task']['protocol'] and r['domain']==x['task']['target']) for x in cases)*len(CONFIGS)*2
    bills=dict(formal_models=207,formal_updates=207*3001,qualification_planned_updates=120,
        optimizer_attempt_cap=207*3001+180,formal_training_forward_windows=69*3001*256,
        qualification_training_forward_windows=2*20*256,training_forward_window_cap=53030000,
        formal_source_prediction_windows=formal_source,qualification_source_prediction_windows=3072,
        source_prediction_window_cap=500000,target_prediction_windows=formal_target,
        raw_signal_loads=0,new_inverse_fits=0,existing_feature_caches=13)
    if (formal_source,formal_target)!=(490752,651024):raise AssertionError('actual source/target window accounting')
    prior=read(BASE/'ress_event_mechanism_factorial/RUN_LOCK.json')
    refs=[]
    for r in read(phm_path/'RESULT.json')['rows']:
        refs.append(dict(protocol='phm09_w3',fold=r['fold'],seed=r['seed'],arm=r['arm'],role=r['role'],probability=r['probability'],archive='ress_event_phm_main'))
    proposal=read(phm_lock['proposal_path'])
    for r in proposal['reused_probability_roles']:
        refs.append(dict(protocol='phm09_w3',fold=r['fold'],seed=r['seed'],arm=r['method'],role=r['policy'],probability=r['reference']['probability'],archive=r['reference']['archive']))
    refs.extend({k:v for k,v in r.items() if k not in ('current_file_exists','fresh_array_SHA_verified')} for r in prior['reused_probability_references'] if r['protocol']=='jnu_speed')
    if len(refs)!=690 or len({(r['protocol'],r['fold'],r['seed'],r['arm'],r['role']) for r in refs})!=690:raise AssertionError('all23task five-core archived690roles, no old forwards')
    paths=[scope_path,cross_path,features_path,phm_path/'PROPOSED_LOCK.json',phm_path/'FINAL_AUDIT.json',
        phm_path/'RESULT.json',Path(phm_lock['proposal_path']),BASE/'ress_event_mechanism_factorial/RUN_LOCK.json',
        BASE/'ress_event_native_repeat/PROPOSED_LOCK.json',control]
    for case in cases:
        t=case['task'];folder=BASE/'splits'
        split=folder/('phm09_w3_grouped_class_guard_'+t['fold']+'_seed'+str(case['seed'])+'.npz') if t['protocol']=='phm09_w3' else folder/'jnu_speed'/('jnu_speed_temporal_block_'+t['fold']+'_seed'+str(case['seed'])+'.npz')
        paths.append(split)
    paths.extend(ROOT/'data/processed'/p/'provenance.csv' for p in ('phm09_w3','jnu_speed'))
    return dict(version='NATIVE_PHM_JNU_FULL_SCOPE_METADATA_V1',records=records,cases=cases,raw_bindings=raw,
        feature_bindings=features,FFT_code_sha256=proposal['FFT_code_sha256'],archived_core_probability_roles=refs,
        bills=bills,source_control_effective_cache_sha256=canonical(effective),
        input_sha256={str(p):sha(p) for p in paths},native_engine_binding={str(engine):sha(engine)},
        author_model_input=dict(network_input=2048,flatten_dimension=464,PHM_raw_window=5333,JNU_raw_window=4000,PHM_classes=8,JNU_classes=4),
        source_semantics=dict(PHM='unchanged grouped recording-disjoint; validation class completeness not forced',
            JNU='unchanged temporal blocks; full sampled intervals disjoint; recordings/entities not held out'),
        new_signal_feature_inverse_weight_probability_payload_reads=0,new_neural_actions=0,
        runtime_status='PREPARATION_PART1_METADATA_SOURCE_ADAPTER_ONLY_NO_EXECUTABLE_TRAINING_LOCK')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--math-proof',action='store_true');args=parser.parse_args()
    if args.math_proof:
        value=author_math_bindings();value['fingerprint']=canonical(value);write_new(OUT/'AUTHOR_MATH_BINDINGS.json',value)
        print('HISTORICAL_NATIVE_MATH_BINDINGS_VERIFIED_NOT_NEW_CUDA',len(value['bindings']),flush=True)
    else:
        value=snapshot();value['fingerprint']=canonical(value);write_new(OUT/'SOURCE_SCOPE.json',value)
        print('NATIVE_FULL_SCOPE_METADATA_READY_NOT_EXECUTABLE_NOT_APPROVED',len(value['cases']),value['bills'],flush=True)
