"""Exact fifty-model schedule, source/target vectors and bounded artifacts."""
from pathlib import Path
import sqlite3
import compatible_set_plan as p
from source_rate_screen_plan import paths, expected
from e0_selection_entry_spec import local_closure

LOCK=p.OUT/'NEURAL_LOCK_V1.json'
TESTS=p.OUT/'NEURAL_CPU_TESTS_V6.json'
ARMS=(*p.ARMS[:4],'N1_GLOBAL_ORTHOGONAL_WORST')
P0_CASES=('pu_clock4/ABCtoD/s0','phm09_w3/adj_AC_B/s0','jnu_speed/A/s0')
STEPS=(0,1000,2000)
ENTRY=('compatible_set_screen_plan.py','compatible_set_neural.py','run_compatible_set.py','test_compatible_set_screen.py','continue_compatible_set.py')


def schedule(source,parent):
    cases={c['key']:dict(c,real_arms=list(ARMS),aliases={}) for c in source['cases']};store=source['campaign']['store']
    chains={};checkpoints={};predictions={};vectors={}
    def add(name,key,arm,qualification=False,branch=None):
        chains[name]=dict(case_key=key,arm=arm,qualification=qualification,branch=branch,steps=5 if qualification else 2000)
        saved=(5,) if branch=='continuous' else (2,5) if qualification else tuple(range(0,2001,250))
        for step in saved:checkpoints[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        vkey=('qualification_source/' if qualification else 'source/')+key
        values=expected(parent['records'],parent['original_cases'][key],'va')
        if qualification:values={k:v[:32] for k,v in values.items()}
        vectors[vkey]=values
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
    forecast=dict(real_models=50,formal_updates=100000,qualification_updates=150,training_forward_windows=forward,
        source_prediction_windows=sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='source'),
        target_prediction_windows=sum(len(vectors[v['expected_key']]['y']) for v in predictions.values() if v['side']=='target'),
        checkpoints=len(checkpoints),predictions=len(predictions),planned_model_builds=245,planned_checkpoint_reads=165,
        synthetic_loss_gradient_fixtures=10,repair_updates_not_automatically_enabled=90,
        orthogonal_feature_rows=4*sum(c.get('bank_windows',0) for c in cases.values()),control_mixed_pairs=640*len(cases))
    p.require(len(chains)==80 and len(checkpoints)==495 and len(predictions)==310 and forward==69464700,'exact fifty-model six-view accounting')
    return dict(cases=cases,chains=chains,checkpoints=checkpoints,predictions=predictions,vectors=vectors,forecast=forecast)


def compile_neural():
    source=p.read(p.OUT/'SOURCE_LOCK_V2.json');p.verify(source)
    banks=p.read(p.OUT/'SOURCE_BANKS_V1.json')
    p.require(banks['status']=='SOURCE_BANKS_COMPLETE_CLASSIFICATION_UNTESTED' and banks['fingerprint']==source['fingerprint'],'complete current source producer first')
    dbpath=p.OUT/'SOURCE_CONTROL.sqlite';db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        p.require(db.execute("SELECT v FROM meta WHERE k='DONE'").fetchone() and
            db.execute("SELECT count(*) FROM units WHERE state='DONE'").fetchone()[0]==2160 and
            db.execute("SELECT count(*) FROM units WHERE state!='DONE'").fetchone()[0]==0,'all2150 groups and10case banks actually completed')
    finally:db.close()
    parent_path=p.ROOT/'results/wfdg/ress_source_rate_screen/NEURAL_LOCK_V2.json';parent=p.read(parent_path)
    campaign=p.read(p.OUT/'CAMPAIGN_SCOPE_V2.json')
    p.require(campaign['actual_start']==source['campaign']['actual_start'] and campaign['deadline']==source['campaign']['deadline'] and
        campaign['neural_seconds']==source['campaign']['neural_seconds'] and campaign['arms']==list(ARMS),'control-only amendment, original clocks/counts')
    plan=schedule(source,parent);forecast=plan['forecast'];limits=dict(campaign['limits'])
    for field in ('source_prediction_windows','target_prediction_windows','training_forward_windows'):
        p.require(forecast[field]<=limits[field],'exact neural forecast fits finite envelope: '+field)
    domains={(c['task']['protocol'],d) for c in plan['cases'].values() for d in [*c['task']['sources'],c['task']['target']]}
    features=[b for b in parent['feature_bindings'] if (b['protocol'],b['domain']) in domains]
    p.require(len(features)==len(domains),'complete fullfeature domains, no new inputs')
    limits.update(feature_decodes=len(features),feature_hashes=len(features),bank_decodes=10,bank_hashes=10,
        checkpoint_writes=495,checkpoint_reads=165,checkpoint_hashes=1100,
        probability_writes=310,probability_reads=310,probability_hashes=1000,synthetic_loss_gradient_fixtures=10,
        orthogonal_feature_rows=forecast['orthogonal_feature_rows'],control_mixed_pairs=forecast['control_mixed_pairs'])
    tests_path=TESTS;tests=p.read(tests_path)
    p.require(tests['status']=='PASS' and tests['torch_imported'] is False,'actual synthetic-only neural CPU tests first')
    code=local_closure([Path(__file__).with_name(n) for n in ENTRY])
    for path,digest in tests['code_sha256'].items():p.require(p.sha(path)==digest,'exact tested bytes')
    inputs=dict(source['input_sha256'],**{str(v):p.sha(v) for v in
        (p.OUT/'SOURCE_LOCK_V2.json',p.OUT/'SOURCE_BANKS_V1.json',dbpath,tests_path,
         p.OUT/'NEURAL_EXECUTION_SPEC_V2.md',p.OUT/'CAMPAIGN_SCOPE_V2.json',p.OUT/'PRETRAINING_AMENDMENT_GLOBAL_CONTROL_V2.md',
         p.ROOT/'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md')})
    lock=dict(kind='COMPATIBLE_SET_FIVE_ARM_NEURAL_V2_GLOBAL_CONTROL',campaign=campaign,source_fingerprint=source['fingerprint'],
        **plan,original_cases={k:parent['original_cases'][k] for k in plan['cases']},records=parent['records'],
        banks=banks['banks'],feature_bindings=features,expected_FFT_code_sha256=parent['expected_FFT_code_sha256'],
        initial_encoder_sha256={k:parent['initial_encoder_sha256'][k] for k in plan['cases']},
        limits=limits,input_sha256=inputs,code_sha256=code,
        gate=dict(primary='last',secondary='selected_no_rescue',minimum_acc=.003,both_seeds_strict_positive=True,
            macro_f1_floor=0.,protocol_acc_floor=-.005,specificity_acc=.001,epsilon=1e-12),
        source_selection='source-domain equal accuracy;0/1000/2000;earliest strict best',
        neural_policy=dict(lr=1e-4,Adam_betas=[.9,.999],Adam_eps=1e-8,weight_decay=0,foreach=False,
            AMP=False,scheduler=None,common_forward='F,F,F,M0,Mplus,Mminus',checkpoint_every=250,disk_restore=1000),
        execution_ready=True,new_raw_reads=0,new_inverse_fits=0,protected_reads=0)
    lock['fingerprint']=p.canonical(lock);p.save(LOCK,lock);return lock


if __name__=='__main__':
    import json
    result=compile_neural();print(json.dumps(dict(fingerprint=result['fingerprint'],forecast=result['forecast'],limits=result['limits']),indent=2))
