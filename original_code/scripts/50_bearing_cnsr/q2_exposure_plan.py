"""Finite 44-model Q2 exposure calibration. Metadata compilation only."""
from datetime import datetime,timedelta
from pathlib import Path
import copy
import sys
import time
import event_full_validation_plan as old
import event_full_validation_execution as oe
from source_rate_screen_plan import paths
from e0_selection_entry_spec import local_closure

ROOT=old.ROOT;OUT=ROOT/'results/wfdg/ress_q2_exposure'
read,save,sha,canonical,need=old.read,old.save,old.sha,old.canonical,old.need
ARMS=('T0_HALF_EXPOSURE','M0_EXPECTED_ENERGY_SHRINK')
REFS=('A0_STRONG_IN_MIXUP','F0_UNEDITED_BANK','Q2')
P0_CASES=old.P0_CASES;STEPS=(0,1000,2000)
ENTRY=tuple('q2_exposure_'+v+'.py' for v in ('core','plan','neural','run','test','close'))
NN_NAME='NEURAL_LOCK_V1.json.gz'
PHM=('adj_AC_B','adj_HJ_I','ext_AB_F','ext_IJ_E')

def clock(campaign):need(time.time()<datetime.fromisoformat(campaign['deadline']).timestamp(),'preparation-inclusive4h exposure deadline')

def verify(lock):
    clock(lock['campaign'])
    need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'immutable exposure lock')
    for path,h in {**lock['input_sha256'],**lock['code_sha256']}.items():need(sha(path)==h,'bound code/input '+path)

def schedule(cases,parent,store):
    chains={};states={};predictions={};vectors={}
    def add(key,arm,qualification=False,branch=None):
        name=('P0/' if qualification else 'MAIN/')+key+'/'+arm+('/'+branch if qualification else '')
        chains[name]=dict(case_key=key,arm=arm,qualification=qualification,branch=branch,steps=12 if qualification else 2000)
        for step in ((12,) if branch=='continuous' else (6,12) if qualification else range(0,2001,250)):
            states[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        keyv=('qualification_source/' if qualification else 'source/')+key
        vec=parent['vectors']['source/'+key]
        vectors[keyv]={k:v[:32] for k,v in vec.items()} if qualification else vec
        for step in ((0,12) if qualification else STEPS):
            predictions[paths(store,name,'source',step)]=dict(chain=name,case_key=key,side='source',step=step,qualification=qualification,expected_key=keyv)
        if not qualification:
            keyv='target/'+key;vectors[keyv]=parent['vectors'][keyv]
            for policy in ('last','selected'):
                predictions[str(Path(store)/'neural'/name/f'target_{policy}.npz')]=dict(chain=name,case_key=key,side='target',policy=policy,qualification=False,expected_key=keyv)
    for key in P0_CASES:
        for arm in ARMS:
            for branch in ('continuous','resume'):add(key,arm,True,branch)
    for key in cases:
        for arm in ARMS:add(key,arm)
    return dict(chains=chains,checkpoints=states,predictions=predictions,vectors=vectors)

def prepare():
    campaign=read(OUT/'SCOPE_V1.json');clock(campaign)
    tests=read(OUT/'CPU_TESTS_V1.json');code=local_closure([Path(__file__).with_name(n) for n in ENTRY])
    need(tests['status']=='PASS' and tests['code_sha256']==code and not tests['torch_imported'],'actual current CPU tests')
    parent_path=old.OUT/oe.NN_NAME;parent=oe.read(parent_path)
    need(canonical({k:v for k,v in parent.items() if k!='fingerprint'})==parent['fingerprint'],'canonical closed full27 parent')
    result_path=old.OUT/'SCREEN_RESULT_V1.json';result=read(result_path)
    need(sha(result_path)=='e4d7be98b7eb6508b1b9c3deac5911992c1ecb3660040defa6a95cbc2aba311e','previously closed full27 metrics')
    cases={k:copy.deepcopy(c) for k,c in parent['cases'].items() if c['seed'] in (0,1) and
        (c['task']['protocol']!='phm09_w3' or c['task']['fold'] in PHM)}
    need(len(cases)==22 and all(k in cases for k in P0_CASES),'eleven fixed tasks and two existing seeds')
    for c in cases.values():c['real_arms']=list(ARMS)
    store=campaign['store'];s=schedule(cases,parent,store);new=44;proofs=6
    batch=lambda c:120//(len(c['task']['sources'])*c['task']['classes'])*len(c['task']['sources'])*c['task']['classes']
    domains={(c['task']['protocol'],d) for c in cases.values() for d in c['task']['sources']+[c['task']['target']]}
    features=[b for b in parent['feature_bindings'] if (b['protocol'],b['domain']) in domains]
    forecasts=dict(optimizer_attempts=88144,formal_updates=88000,qualification_updates=144,
        training_forward_windows=sum(2*2000*6*batch(c) for c in cases.values())+sum(2*24*6*batch(cases[k]) for k in P0_CASES),
        source_prediction_windows=sum(len(s['vectors'][v['expected_key']]['y']) for v in s['predictions'].values() if v['side']=='source'),
        target_prediction_windows=sum(len(s['vectors'][v['expected_key']]['y']) for v in s['predictions'].values() if v['side']=='target'),
        feature_hashes=len(features),feature_decodes=len(features),bank_hashes=22,bank_decodes=22,
        neural_model_builds=4*new+3*proofs,checkpoint_writes=9*new+3*proofs,checkpoint_reads=6*new+proofs,
        checkpoint_hashes=21*new+4*proofs,probability_writes=5*new+4*proofs,
        probability_reads=5*new+4*proofs,probability_hashes=2*(5*new+4*proofs),
        batch_reference_rows=sum(12*batch(cases[k]) for k in P0_CASES))
    need(forecasts['training_forward_windows']==61348512 and forecasts['source_prediction_windows']==672384 and
        forecasts['target_prediction_windows']==777792 and len(features)==15,'exact prospective neural/window schedule')
    need(len(s['chains'])==56 and len(s['checkpoints'])==414 and len(s['predictions'])==244,'actual finite artifact schedule')
    identities={(c['task']['protocol'],c['task']['fold'],c['seed']) for c in cases.values()}
    refs=[dict(r,origin='CLOSED_FULL27_METRICS') for r in result['rows'] if
          (r['protocol'],r['fold'],r['seed']) in identities and r['arm'] in REFS]
    need(len(refs)==132 and len({(r['protocol'],r['fold'],r['seed'],r['arm'],r['role']) for r in refs})==132,'all matched reference rows/policies, no mean substitution')
    input_paths=[parent_path,result_path,old.OUT/'CLOSED_RECEIPT_V1.json',old.OUT/'CLOSED_POSTCHECK_V1.json',
        OUT/'SCOPE_V1.json',OUT/'RESEARCH_CARD_V1.md',OUT/'CPU_TESTS_V1.json',
        ROOT/'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md']
    lock=dict(kind='FIXED_CURRENT_Q2_EXPOSURE_CALIBRATION',campaign=campaign,cases=cases,**s,
        original_cases={k:parent['original_cases'][k] for k in cases},records=parent['records'],
        banks={k:parent['banks'][k] for k in cases},feature_bindings=features,
        initial_encoder_sha256={k:parent['initial_encoder_sha256'][k] for k in cases},
        expected_FFT_code_sha256=parent['expected_FFT_code_sha256'],control_filename=str(OUT/'NEURAL_CONTROL.sqlite'),
        reference_rows=refs,parent_fingerprint=parent['fingerprint'],forecast=forecasts,limits=forecasts,
        input_sha256={str(path):sha(path) for path in input_paths},code_sha256=code,
        effect_gate=dict(candidate=ARMS[0],primary_policy='last',mean_acc_gain=.003,
            nonnegative_metrics=['macro_f1','recording_ba'],both_seed_acc_positive=True,protocol_floor=-.005,
            comparators=list(REFS),secondary_cannot_rescue=True,control_not_promoted=True))
    lock['fingerprint']=canonical(lock);oe.compressed_save(OUT/NN_NAME,lock);verify(lock)
    save(OUT/'NEURAL_ACTUAL_GRANT_V1.json',dict(authorization='standing_authorization',human_message_id=campaign['human_message_id'],
        fingerprint=lock['fingerprint'],limits=forecasts,neural_seconds=7200,total_deadline=campaign['deadline'],at_unix=time.time()))
    print('EXPOSURE_LOCKED',lock['fingerprint'],forecasts,flush=True)

if __name__=='__main__':prepare()
