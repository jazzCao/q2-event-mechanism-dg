"""Metadata-only exact workload for two controls of closed Q2; no payload entry."""
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time

import event_full_closed_comparison as closed
import event_full_validation_execution as original_execution
import event_full_validation_plan as original
from natural_mapping_probe_plan import source_case

ROOT = original.ROOT
OUT = ROOT/'results/wfdg/ress_event_component_attribution'
ARMS = ('A0_IMPULSE_ONLY','M0_GLOBAL_AFFINE_MOMENTS')
ENTRY = ('event_component_core.py','event_component_core_test.py','event_component_metadata.py')
TEST_NAME = 'SYNTHETIC_TESTS_V2.json'


def code_hashes(): return {str(Path(__file__).with_name(name).resolve()):closed.sha(Path(__file__).with_name(name)) for name in ENTRY}


def scope():
    path = OUT/'PREPARATION_SCOPE_V1.json'
    if path.exists(): return closed.read(path)
    card = OUT/'RESEARCH_CARD_V1.md';stat = card.stat()
    started = getattr(stat,'st_birthtime',stat.st_ctime)
    start = datetime.fromtimestamp(started).astimezone()
    result = dict(kind='SAME_FIXED_Q2_TWO_CONTROL_PREPARATION',authorization='standing_authorization',human_message_id=closed.AUTH,
        actual_start=start.isoformat(),start_basis='new research card creation time in current Windows filesystem',
        deadline=(start+timedelta(hours=12)).isoformat(),source_seconds=7200,neural_seconds=18000,
        task_count=27,seeds=[0,1,2],arms=ARMS,new_models=162,reused_closed_models=405,
        formal_updates=324000,qualification_updates=60,optimizer_attempt_limit=324060,
        parameter_search=False,new_datasets=False,new_raw_reads=0,new_inverse_fits=0,reserved_signal_reads=0,
        payload_execution_ready=False,neural_execution_ready=False,
        reason='Test timing and selective event editing against two fixed amplitude explanations; do not change Q2 or its closed result')
    closed.save(path,result);return result


def forecast(cases, nn):
    target_counts={k:len(nn['vectors']['target/'+k]['y']) for k in cases}
    new=len(cases)*len(ARMS);proofs=len(original.P0_CASES)*len(ARMS)
    batch=lambda c:120//(len(c['task']['sources'])*c['task']['classes'])*len(c['task']['sources'])*c['task']['classes']
    training=sum(len(ARMS)*2000*6*batch(v) for v in cases.values())
    training+=sum(len(ARMS)*10*6*batch(cases[k]) for k in original.P0_CASES)
    return dict(real_new_models=new,reused_closed_models=405,logical_models=new+405,
        formal_updates=2000*new,qualification_updates=10*proofs,optimizer_attempts=2000*new+10*proofs,
        training_forward_windows=training,
        source_prediction_windows=sum(3*len(ARMS)*v['source_validation_windows'] for v in cases.values())+proofs*4*32,
        target_prediction_windows=sum(2*len(ARMS)*n for n in target_counts.values()),
        feature_hashes=17,feature_decodes=17,bank_hashes=len(cases),bank_decodes=len(cases),
        neural_model_builds=4*new+3*proofs,checkpoint_writes=9*new+3*proofs,
        checkpoint_reads=6*new+proofs,checkpoint_hashes=21*new+4*proofs,
        probability_writes=5*new+4*proofs,probability_reads=5*new+4*proofs,probability_hashes=2*(5*new+4*proofs),
        critical_CPU_new_states=3*new,batch_reference_rows=sum(12*batch(cases[k]) for k in original.P0_CASES),
        archived_probability_reads=0,archived_model_reads=0,new_raw_reads=0,new_inverse_fits=0,protected_reads=0)


def prepare():
    campaign=scope();original.need(time.time()<datetime.fromisoformat(campaign['deadline']).timestamp(),'current preparation-inclusive deadline')
    post,receipt=closed.require_closed();parent=original.OUT
    tests=closed.read(OUT/TEST_NAME)
    original.need(tests['status']=='PASS' and tests['tests']==8 and tests['code_sha256']==code_hashes() and not tests['torch_imported'],'actual same pure-control tests')
    nn=original_execution.read(parent/original_execution.NN_NAME)
    source=closed.read(parent/'SOURCE_LOCK_V1.json');banks=closed.read(parent/'SOURCE_BANKS_V1.json')
    for value in (source,nn):original.need(original.canonical({k:v for k,v in value.items() if k!='fingerprint'})==value['fingerprint'],'unchanged canonical closed source/neural lock')
    original.need(post['scientific']==receipt['scientific']=='FULL_SCOPE_EVENT_OPERATOR_SUPPORTED','actual closed Q2 increment before component work')
    original.need(banks['status']=='FULL81_SOURCE_BANKS_COMPLETE' and banks['fingerprint']==source['fingerprint'],'actual complete canonical source banks')
    cases={v['key']:v for v in source['cases']};groups=source['groups'];assets=source['assets'];proofs={}
    original.need(len(cases)==81 and len(groups)==3372 and len(assets)==2124,'whole original source scope')
    original.need(tuple(source['ratios'][2:])==tuple(original.Q[2:]),'exact closed internal Q2 factors')
    original.need(len(nn['feature_bindings'])==17,'all existing physical-domain feature bindings')
    for key,case in cases.items():
        old=nn['original_cases'][key]
        original.need(original.canonical(old)==case['original_case_sha256'],'original full source split identity')
        for item in case['bank']:
            old_item=next(v for v in old['bank'] if v['record']==item['record'])
            pair=old['pairs'][old_item['pair']]
            for record in pair:source_case(old,nn['records'],record,context=True)
        proofs[key]=dict(task=case['task'],seed=case['seed'],roles_sha256=original.canonical(old['roles']),
            bank_windows=case['bank_windows'],source_validation_windows=case['source_validation_windows'])
    group_registry={g['key']:g for g in groups}
    original.need(set(group_registry)==set(banks['groups']),'all canonical group anchors available')
    bindings={str(Path(b['path']).resolve()):b for a in assets.values() for b in a.values()}
    bindings.update({str(Path(b['path']).resolve()):b for b in banks['groups'].values()})
    original.need(len(bindings)==7620,'4248 inverse/context plus3372 previous group anchors')
    sizes={path:Path(path).stat().st_size for path in bindings}
    original.need(all(Path(path).drive.upper()=='D:' and Path(path).suffix=='.npz' for path in bindings),'only existing D source packets; no payload decoded')
    count=len(groups)
    source_forecast=dict(source_groups=count,source_cases=81,source_inverse_pairs=2124,
        source_payload_decodes=len(bindings),source_payload_bytes=sum(sizes.values()),
        canonical_anchor_group_reads=count,renderer_calls=3*count,
        mode_sample_updates=3*3*48*32768*count,filter_calls=3*3*48*count,
        event_grid_scans=3*2*48*32768*count,normalization_samples=sum(3*48*g['kernel_length'] for g in groups),
        source_fft_1d_transforms=21*count,source_fft_points=21*sum(g['window'] for g in groups),
        owned_writes=count+81,owned_reads=count+81,owned_hashes=2*(count+81),
        new_inverse_fits=0,new_raw_reads=0,protected_reads=0)
    nn_forecast=forecast(cases,nn)
    original.need(nn_forecast['real_new_models']==162 and nn_forecast['optimizer_attempts']==324060,'exact two controls and real qualification')
    inputs={str(parent/name):closed.sha(parent/name) for name in ('SOURCE_LOCK_V1.json','SOURCE_BANKS_V1.json',original_execution.NN_NAME,
        'SCREEN_RESULT_V1.json','AUDIT_V1.json','CLOSED_RECEIPT_V1.json','CLOSED_POSTCHECK_V1.json','POSTCLOSURE_AUDIT_V1.json')}
    inputs.update({str(OUT/name):closed.sha(OUT/name) for name in ('RESEARCH_CARD_V1.md','PREPARATION_SCOPE_V1.json',TEST_NAME)})
    output=dict(kind='FIXED_Q2_TWO_CONTROL_METADATA_PREFLIGHT',status='METADATA_COMPLETE_EXECUTABLE_ENTRY_NOT_READY',campaign=campaign,
        parent_scientific_fingerprint=nn['fingerprint'],source_parent_fingerprint=source['fingerprint'],
        cases=cases,source_role_proofs=proofs,groups=groups,assets=assets,canonical_group_anchors=banks['groups'],
        source_payload_sizes=sizes,source_forecast=source_forecast,neural_forecast=nn_forecast,
        ratios=source['ratios'][2:],P0_cases=original.P0_CASES,
        input_sha256=inputs,preparation_code_sha256=code_hashes(),
        new_scientific_payload_reads=0,new_neural_actions=0,torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
        waveform_controls_tested=True,complete_source_trainer_evaluator_and_auditor_tested=False,
        CUDA_qualification_passed=False,formal_models_started=0,
        interpretation='Metadata eligibility and exact prospective costs only; no scientific payload or CUDA grant and no method result.')
    original.need(not output['torch_imported'] and not output['numpy_imported'],'stdlib-only actual metadata preflight')
    output['fingerprint']=closed.canonical(output);closed.save(OUT/'METADATA_PREFLIGHT_V1.json',output)
    print(output['status'],output['fingerprint'],flush=True)
    print('SOURCE',source_forecast,flush=True);print('NEURAL',nn_forecast,flush=True)


def test():
    import unittest
    import event_component_core_test as tests
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    original.need(result.wasSuccessful() and 'torch' not in sys.modules,'actual synthetic tests without neural imports')
    closed.save(OUT/TEST_NAME,dict(status='PASS',tests=result.testsRun,code_sha256=code_hashes(),
        torch_imported=False,new_source_payload_reads=0,new_classifier_FFTs=0,new_neural_actions=0,at_unix=time.time()))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare'));args=parser.parse_args()
    if args.mode=='test':test()
    else:prepare()
