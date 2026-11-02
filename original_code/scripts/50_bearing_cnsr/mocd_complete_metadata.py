"""Unified complete MOCD inputs and finite accounting forecast, JSON only.

This is preparation, not an execution lock/approval. No array payload, neural
module or prior model is loaded. Original source roles and archives are kept.
"""
from pathlib import Path
import time
from pcr_campaign_core import ROOT,canonical,read,sha,write_new
import mocd_complete_source_adapter as source

BASE=ROOT/'results/wfdg'
OUT=BASE/'ress_mocd_complete'
PREP=BASE/'ress_mocd_complete_scope'
PROFILES=('EQ_LITERAL','PROSE_STOPGRAD_SENSITIVITY')
QUAL_FOLDS={'pu_clock4':'BCDtoA','phm09_w3':'adj_BC_A','jnu_speed':'A'}


def case_key(case):return case['task']['protocol']+'/'+case['task']['fold']+'/s'+str(case['seed'])
def profile_streams(seed):return {k:v for k,v in source.random_stream_spec(seed).items() if k!='status'}


def qualification_bill(cases):
    selected=[c for c in cases if c['task']['fold']==QUAL_FOLDS[c['task']['protocol']]]
    if len(selected)!=9 or {(c['task']['protocol'],c['seed']) for c in selected}!={(p,s) for p in QUAL_FOLDS for s in range(3)}:raise ValueError('three protocol shapes, all three seeds')
    rows=[]
    for case in selected:
        task=case['task'];d=len(task['sources']);k=task['classes'];inv=case['source_inventory']
        tr=sum(inv[x]['tr']['windows'] for x in task['sources']);va=sum(inv[x]['va']['windows'] for x in task['sources'])
        # Per profile: continuous10 plus disk5+5, mini A4/B6, two refreshes/trajectory.
        rows.append(dict(case_key=case_key(case),profiles=list(PROFILES),mini_schedule=dict(A_outer=1,A_inner=4,B_outer=1,B_inner=6),
            trajectories=['continuous10','disk5+5'],optimizer_attempts=40,
            training_encoder_windows=2*(128*d*k+3072),matching_encoder_windows=16*tr,
            source_prediction_windows=4*va,model_builds=6,complete_packets=6,source_probability_files=4))
    return dict(cases=rows,combinations=18,optimizer_attempts=sum(v['optimizer_attempts'] for v in rows),
        training_encoder_windows=sum(v['training_encoder_windows'] for v in rows),matching_encoder_windows=sum(v['matching_encoder_windows'] for v in rows),
        source_prediction_windows=sum(v['source_prediction_windows'] for v in rows),model_builds=54,complete_packets=54,source_probability_files=36,
        repair_reserve_not_yet_authorized=120,real_CUDA_qualification_status='NOT_RUN_NOT_AUTHORIZED')


def snapshot():
    original=read(PREP/'SOURCE_SCHEDULE_SCOPE_V2.json');target_bill=read(PREP/'TARGET_BILL_V1.json')
    for value in (original,target_bill):
        if canonical({k:v for k,v in value.items() if k!='fingerprint'})!=value['fingerprint']:raise PermissionError('original finite metadata fingerprints')
    reference_paths={c['source_case_reference']['path'] for c in original['cases']};documents={}
    for path in sorted(reference_paths):
        if sha(path)!=original['input_sha256'][path]:raise PermissionError('original source JSON changed '+path)
        documents[path]=read(path)
    records={};features={};raw={};fft=set()
    for document in documents.values():
        for key,record in document['records'].items():
            if key in records and records[key]!=record:raise PermissionError('conflicting recording metadata')
            records[key]=record
        for binding in document['feature_bindings']:
            key=binding['protocol'],binding['domain']
            if key in features and features[key]!=binding:raise PermissionError('conflicting source feature binding')
            features[key]=binding;fft.add(binding['FFT_code_sha256'])
            for row in binding['source_input_records']:
                value={k:v for k,v in row.items() if k!='record'}
                if row['record'] in raw and raw[row['record']]!=value:raise PermissionError('conflicting raw SHA provenance')
                raw[row['record']]=value
    if len(records)!=1612 or len(features)!=17 or set(raw)!=set(records) or len(fft)!=1:raise ValueError('all1612 records and17 existing FFT caches, no new data')
    for key,record in records.items():
        binding=raw[key]
        if binding['fs']!=record['fs'] or ('path' in record and binding['path']!=record['path']) or Path(binding['path']).name!=key.split('/')[-1] and record['protocol']!='pu_clock4':raise ValueError('raw provenance path/fs binding')
        if binding['samples']<max(record['offsets'])+record['window'] or (record.get('expected_samples') is not None and binding['samples']!=record['expected_samples']) or record.get('bearing') in source.pu.PROTECTED:raise ValueError('unchanged complete source/target metadata support')
    cases=[];models=[]
    for item in original['cases']:
        ref=item['source_case_reference'];entry=documents[ref['path']]['cases'][ref['index']];case=entry['case'] if ref['field']=='cases[].case' else entry
        if canonical(case)!=item['source_case_sha256'] or source.source_metadata(records,case)!=item['source_inventory']:raise PermissionError('same actual source case/indices')
        case=dict(case);case['source_inventory']=item['source_inventory'];case['original_source_case_sha256']=item['source_case_sha256']
        case['new_source_fingerprint']=canonical(dict(task=case['task'],seed=case['seed'],roles=case['roles'],FFT_code_sha256=next(iter(fft)),
            source_feature_bytes={d:features[case['task']['protocol'],d]['sha256'] for d in case['task']['sources']}))
        case['source_validation_rule']='TEMPORAL_BLOCK_FULL_INTERVAL_DISJOINT' if case['task']['protocol']=='jnu_speed' else 'GROUPED_RECORDING_DISJOINT'
        cases.append(case)
        for old in item['models']:models.append(dict(case_key=case_key(case),profile=old['profile'],action=old['action'],archive=old['reference']))
    if len(cases)!=81 or len({case_key(c) for c in cases})!=81 or sum(m['action']=='MISSING_NOT_AUTHORIZED' for m in models)!=154:raise ValueError('all27tasks three seeds two profiles')
    archive_path=BASE/'ress_event_author_comparison/formal_batched/RESULT.json';archived=read(archive_path)
    old_lock_path=archive_path.parent/'PROPOSED_LOCK.json';old_lock=read(old_lock_path)
    profiles_path=BASE/'ress_mocd_state_qualification/INTERPRETATION_PROFILES.json';profiles=read(profiles_path)
    if profiles!=old_lock['profiles'] or profiles['primary']!='EQ_LITERAL' or tuple(profiles['profiles'])!=PROFILES:raise PermissionError('both original declared interpretations, no target-selected replacement')
    old_rows=[row for row in archived['rows'] if row['config'] in PROFILES]
    if len(old_rows)!=16:raise ValueError('all16 archived target roles, no new inference or old rebranding')
    for row in old_rows:
        matching=[m for m in models if m['case_key']=='pu_clock4/'+row['fold']+'/s0' and m['profile']==row['config'] and m['archive']]
        if len(matching)!=1 or row['engine_fingerprint']!=matching[0]['archive']['historical_engine_fingerprint'] or row['weights_sha256_before']!=row['weights_sha256_after']:raise PermissionError('original source/target archive identity')
    math_names=('mocd_full_engine.py','mocd_full_source_engine.py','mocd_full_matching.py','mocd_matching_state.py','mocd_schedule_control.py','mocd_paper_components.py','qualify_ress_literature_components.py')
    math_bindings={}
    for name in math_names:
        path=Path(__file__).with_name(name).resolve();h=sha(path)
        if old_lock['code_sha256'].get(str(path))!=h:raise PermissionError('original qualified mathematics changed '+name)
        math_bindings[str(path)]=h
    qualification=qualification_bill(cases)
    formal=dict(optimizer_attempts=123200,training_encoder_windows=original['future_training_encoder_windows'],matching_encoder_windows=original['future_matching_encoder_windows'],source_prediction_windows=original['future_source_validation_windows'],target_prediction_windows=target_bill['total']['new_target_prediction_windows'],model_builds=154*4)
    normal={k:formal[k]+qualification.get(k,0) for k in formal}
    paths=[PREP/'SOURCE_SCHEDULE_SCOPE_V2.json',PREP/'TARGET_BILL_V1.json',profiles_path,archive_path,old_lock_path,
        archive_path.parent/'FROZEN_SOURCE.json',archive_path.parent/'INDEPENDENT_AUDIT.json',archive_path.parent/'FINAL_EXECUTION_RECEIPT.json',
        BASE/'ress_event_native_main/recovery135/METRICS.json',BASE/'ress_event_native_main/recovery135/FINAL_AUDIT.json',BASE/'ress_event_native_main/recovery135/FINAL_EXECUTION_AND_DELIVERY.json',
        Path(__file__),Path(source.__file__)]+[Path(p) for p in sorted(reference_paths)]
    value=dict(status='COMPLETE_SOURCE_MANIFEST_PART1_NOT_RUNTIME_OR_NEURAL_AUTHORITY',records=records,cases=cases,models=models,profiles=profiles,
        feature_bindings=[features[k] for k in sorted(features)],raw_bindings=raw,FFT_code_sha256=next(iter(fft)),archived_target_roles=old_rows,
        formal_forecast=formal,qualification_forecast=qualification,normal_total_forecast=normal,
        complete_models=162,new_models=154,archive_models=8,source_scientific_steps=[0,400,421,800],source_score_steps=list(range(420,801,20)),
        primary='fixed_last800',secondary='earliest_strict_source_best',common_freeze='all154 new complete plus8 archive bindings before any new target',
        seed_stream_specs={str(s):profile_streams(s) for s in range(3)},
        original_math_sha256=math_bindings,input_sha256={str(p):sha(p) for p in paths},
        no_new_neural_actions=True,no_real_array_payload_reads=True,no_new_dataset_split_seed_or_operator=True,
        cache_payloads_old_states_and_probabilities='NOT_LOADED_HERE_REQUIRE_SHA_AND_ROLE_CHECKS_AFTER_NEW_AUTHORITY',
        complete_runtime_auditor_time_and_repair_budget='NOT_IMPLEMENTED_OR_APPROVED_BY_THIS_PART1',at_unix=time.time())
    value['fingerprint']=canonical(value);return value


if __name__=='__main__':
    import sys
    def guard(event,args):
        if event=='open' and isinstance(args[0],str) and Path(args[0]).suffix.lower() in ('.pt','.pth','.npz','.npy','.mat'):raise PermissionError('metadata builder does not open real payloads')
    sys.addaudithook(guard);value=snapshot()
    if 'torch' in sys.modules:raise AssertionError('metadata builder imported a neural module')
    write_new(OUT/'PART1_SOURCE_MANIFEST_V2.json',value)
    print('COMPLETE_MOCD_PART1_SOURCE_MANIFEST_READY_NOT_AUTHORIZED',len(value['cases']),value['new_models'],value['normal_total_forecast'],flush=True)
