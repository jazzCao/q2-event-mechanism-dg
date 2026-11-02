"""Complete immutable source/neural schedule for two controls; preparation is stdlib."""
from datetime import datetime
from pathlib import Path
import copy

import event_component_metadata as meta
import event_full_validation_plan as old
import event_full_validation_execution as old_execution
from e0_selection_entry_spec import local_closure
from source_rate_screen_plan import paths

ROOT,OUT,ARMS=meta.ROOT,meta.OUT,meta.ARMS
read,save,sha,canonical,need=old.read,old.save,old.sha,old.canonical,old.need
STEPS=(0,1000,2000)
NN_NAME='NEURAL_LOCK_V1.json.gz'
TEST_NAME='CPU_ENTRY_TESTS_V2.json'
ENTRY=tuple('event_component_'+n+'.py' for n in ('core','core_test','metadata','execution','source','neural','decision','run','check','continue','test'))


def clock(campaign):need(datetime.now().astimezone()<datetime.fromisoformat(campaign['deadline']),'component preparation-inclusive12h deadline')


def load(path):return old_execution.read(path)


def verify(lock):
    clock(lock['campaign'])
    need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'complete component lock fingerprint')
    for path,h in {**lock['input_sha256'],**lock['code_sha256']}.items():need(sha(path)==h,'bound complete component code/input '+path)


def schedule(cases,parent,store):
    cases=copy.deepcopy(cases);chains={};checkpoints={};predictions={};vectors={}
    def add(key,arm,qualification=False,branch=None):
        name=('P0/' if qualification else 'MAIN/')+key+'/'+arm+('/'+branch if qualification else '')
        chains[name]=dict(case_key=key,arm=arm,qualification=qualification,branch=branch,steps=5 if qualification else 2000)
        for step in ((5,) if branch=='continuous' else (2,5) if qualification else range(0,2001,250)):
            checkpoints[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        vector_key=('qualification_source/' if qualification else 'source/')+key
        vector=parent['vectors']['source/'+key]
        vectors[vector_key]={k:v[:32] for k,v in vector.items()} if qualification else vector
        for step in ((0,5) if qualification else STEPS):
            predictions[paths(store,name,'source',step)]=dict(chain=name,case_key=key,side='source',step=step,qualification=qualification,expected_key=vector_key)
        if not qualification:
            vector_key='target/'+key;vectors[vector_key]=parent['vectors'][vector_key]
            for policy in ('last','selected'):
                predictions[str(Path(store)/'neural'/name/f'target_{policy}.npz')]=dict(chain=name,case_key=key,side='target',policy=policy,qualification=False,expected_key=vector_key)
    for key in old.P0_CASES:
        for arm in ARMS:
            for branch in ('continuous','resume'):add(key,arm,True,branch)
    for key,case in cases.items():
        case['real_arms']=list(ARMS)
        for arm in ARMS:add(key,arm)
    f=meta.forecast(cases,parent)
    need(len(chains)==174 and len(checkpoints)==f['checkpoint_writes'] and len(predictions)==f['probability_writes'],'exact162 new and12 P0 chains')
    need(sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='source')==f['source_prediction_windows'],'exact source prediction schedule')
    need(sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='target')==f['target_prediction_windows'],'exact target prediction schedule')
    return dict(cases=cases,chains=chains,checkpoints=checkpoints,predictions=predictions,vectors=vectors)


def prepare_source():
    m=read(OUT/'METADATA_PREFLIGHT_V1.json');clock(m['campaign'])
    # The metadata preflight uses its documented ASCII-safe fingerprint; original locks use UTF8.
    need(meta.closed.canonical({k:v for k,v in m.items() if k!='fingerprint'})==m['fingerprint'],'unchanged complete preflight')
    tests=read(OUT/TEST_NAME);code=local_closure([Path(__file__).with_name(n) for n in ENTRY])
    need(tests['status']=='PASS' and not tests['torch_imported'] and tests['code_sha256']==code,'complete actual entry tests before source payload')
    campaign=dict(m['campaign'],payload_execution_ready=True,neural_execution_ready=False,
        store='D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_event_components_20261028_1039')
    fields=('source_payload_decodes','source_payload_bytes','canonical_anchor_group_reads','renderer_calls','mode_sample_updates',
        'filter_calls','event_grid_scans','normalization_samples','source_fft_1d_transforms','source_fft_points','owned_writes','owned_reads','owned_hashes')
    inputs=dict(m['input_sha256'])
    for name in ('METADATA_PREFLIGHT_V1.json',TEST_NAME):inputs[str(OUT/name)]=sha(OUT/name)
    inputs[str(ROOT/'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md')]=sha(ROOT/'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md')
    lock=dict(kind='FIXED_Q2_TWO_CONTROL_SOURCE',campaign=campaign,cases=m['cases'],groups=m['groups'],assets=m['assets'],
        canonical_group_anchors=m['canonical_group_anchors'],ratios=m['ratios'],payload_sizes=m['source_payload_sizes'],
        forecast=m['source_forecast'],limits={k:m['source_forecast'][k] for k in fields},input_sha256=inputs,code_sha256=code)
    lock['fingerprint']=canonical(lock);save(OUT/'SOURCE_LOCK_V1.json',lock);verify(lock);return lock


def compile_neural():
    import sqlite3
    source=read(OUT/'SOURCE_LOCK_V1.json');verify(source);banks=read(OUT/'SOURCE_BANKS_V1.json')
    need(banks['status']=='ALL81_COMPONENT_BANKS_WITH_EXACT_Q2_ANCHORS' and banks['fingerprint']==source['fingerprint'],'actual complete source banks')
    db=sqlite3.connect((OUT/'SOURCE_CONTROL.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        need(db.execute("select state,count(*) from units group by state").fetchall()==[('DONE',3453)],'all3372 groups and81 banks')
        need(db.execute("select v from meta where k='DONE'").fetchone() and not db.execute("select v from meta where k='STOPPED'").fetchone(),'source actually completed')
    finally:db.close()
    parent=load(old.OUT/old_execution.NN_NAME);m=read(OUT/'METADATA_PREFLIGHT_V1.json');f=m['neural_forecast']
    s=schedule(source['cases'],parent,source['campaign']['store'])
    need(f==meta.forecast(s['cases'],parent),'exact complete neural forecast')
    limits={k:v for k,v in f.items() if k not in ('real_new_models','reused_closed_models','logical_models','critical_CPU_new_states')}
    inputs=dict(source['input_sha256'])
    for name in ('SOURCE_LOCK_V1.json','SOURCE_BANKS_V1.json','SOURCE_CONTROL.sqlite','SOURCE_ACTUAL_GRANT_V1.json','SOURCE_PRODUCER_EXIT_V1.json'):
        inputs[str(OUT/name)]=sha(OUT/name)
    lock=dict(kind='FIXED_Q2_TWO_CONTROL_CLASSIFICATION',campaign=dict(source['campaign'],neural_execution_ready=True),**s,
        original_cases=parent['original_cases'],records=parent['records'],banks=banks['banks'],
        feature_bindings=parent['feature_bindings'],initial_encoder_sha256=parent['initial_encoder_sha256'],
        expected_FFT_code_sha256=parent['expected_FFT_code_sha256'],control_filename=str(OUT/'NEURAL_CONTROL.sqlite'),
        parent_result_path=str(old.OUT/'SCREEN_RESULT_V1.json'),parent_result_sha256=sha(old.OUT/'SCREEN_RESULT_V1.json'),
        forecast=f,limits=limits,input_sha256=inputs,code_sha256=source['code_sha256'])
    lock['fingerprint']=canonical(lock);old_execution.compressed_save(OUT/NN_NAME,lock);verify(lock);return lock


def test():
    import unittest
    import sys
    import time
    import event_component_core_test as core_tests
    import event_component_test as entry_tests
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(core_tests),unittest.defaultTestLoader.loadTestsFromModule(entry_tests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    need(result.wasSuccessful() and 'torch' not in sys.modules,'actual complete CPU entry tests without neural imports')
    code=local_closure([Path(__file__).with_name(n) for n in ENTRY])
    save(OUT/TEST_NAME,dict(status='PASS',tests=result.testsRun,code_sha256=code,
        torch_imported=False,new_scientific_payload_reads=0,new_actual_FFTs=0,new_neural_actions=0,at_unix=time.time()))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare-source'));args=parser.parse_args()
    if args.mode=='test':test()
    else:print('COMPONENT_COMPLETE_SOURCE_LOCK',prepare_source()['fingerprint'],flush=True)
