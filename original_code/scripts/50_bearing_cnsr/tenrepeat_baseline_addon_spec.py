"""Freeze three standard source-only baselines after the existing campaign."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import gzip
import json
import time
from tenrepeat_source_metadata import ROOT, OUT as PARENT, read, write, sha, canonical, need

OUT = ROOT/'results/wfdg/ress_tenseed_baseline_addon'
STORE = Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_tenseed_baseline_addon')
ARMS = ('ERM_TABLE3_IN_CE', 'DANN_SOURCE_ONLY', 'DCORAL_SOURCE_ONLY')


def metadata():
    from protected_response_run_spec import load_plan
    old=load_plan()
    new=json.loads(gzip.decompress((PARENT/'MDG_SOURCE_MANIFEST_V1.json.gz').read_bytes()))
    cases={}
    for k,c in {**old['cases'],**new['cases']}.items():
        v=copy.deepcopy(c);v['key']=k;v['source_training_windows']=v['training_windows'];v['source_validation_windows']=v['validation_windows']
        need(v['seed'] in range(10) and len(v['task']['sources'])>1,'same MDG panel and fixed seeds')
        need(v['task']['target'] not in v['task']['sources'],'held-out condition excluded from all source roles')
        for r,roles in v['roles'].items():
            record=new['records'][r]
            need(record['domain'] in v['task']['sources'] and record.get('bearing') not in ('K006','KA08','KA09'),'legal existing source recording')
            need(not set(roles['tr'])&set(roles['va']),'source training/validation windows disjoint')
        cases[k]=v
    need(len(cases)==270 and len({(v['task']['protocol'],v['task']['fold']) for v in cases.values()})==27,'all270 original task/seed cases')
    with gzip.open(ROOT/'results/wfdg/ress_event_full_validation/NEURAL_LOCK_V1.json.gz','rt',encoding='utf8') as f:base=json.load(f)
    initials={**base['initial_encoder_sha256'],**read(PARENT/'INITIAL_ENCODERS_V1.json')['new189']}
    need(all(k in initials for k in cases),'all270 previously fixed encoder initializations')
    return cases,new,initials


def prepare():
    from e0_selection_entry_spec import local_closure
    OUT.mkdir(exist_ok=True)
    checks=[OUT/'CPU_TESTS_V2.json',OUT/'INTEGRATION_CPU_TESTS_V2.json']
    for p in checks:
        value=read(p);need(value['status'].startswith('PASS'),'actual source/engine CPU readiness')
        for name,h in value['code_sha256'].items():need(sha(name)==h,'tested extension source unchanged '+name)
    cases,m,initials=metadata()
    units=[dict(name='BASELINE/'+k+'/'+arm,case_key=k,arm=arm,steps=2000) for k in sorted(cases) for arm in ARMS]
    batch=lambda c:120//(len(c['task']['sources'])*c['task']['classes'])*len(c['task']['sources'])*c['task']['classes']
    src=sum(3*c['source_validation_windows']*len(ARMS) for c in cases.values())
    target=sum(2*len(ARMS)*sum(len(r['offsets']) for r in m['records'].values() if (r['protocol'],r['domain'])==(c['task']['protocol'],c['task']['target'])) for c in cases.values())
    inputs=[PARENT/'MDG_SOURCE_MANIFEST_V1.json.gz',PARENT/'INITIAL_ENCODERS_V1.json',PARENT/'HUMAN_SCOPE_V1.md',
        ROOT/'results/wfdg/ress_event_full_validation/NEURAL_LOCK_V1.json.gz']+checks
    qkeys=[next(k for k,v in sorted(cases.items()) if v['task']['protocol']==p and v['seed']==0) for p in ('jnu_speed','phm09_w3','pu_clock4')]
    spec=dict(kind='AUTHORIZED_THREE_BASELINE_TEN_SEED_EXTENSION',at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        authorization='User: append missing baselines after existing experiments; fully use GPU memory and compute',
        parent_campaign=str(PARENT),start_after=dict(path=str(PARENT/'FINAL_AUDIT_V1.json'),status='PASS',models=5460),
        queued_wait_deadline=(datetime.now(timezone(timedelta(hours=8)))+timedelta(days=14)).isoformat(),
        cases=cases,original_cases=cases,records=m['records'],feature_bindings=m['features'],expected_FFT_code_sha256=m['FFT_code_sha256'],
        initial_encoder_sha256={k:initials[k] for k in cases},units=units,arms=list(ARMS),models=810,seeds=list(range(10)),
        model=dict(backbone='unchanged Table-3 enc+classifier',feature_IN='same fixed terminal feature IN as current Q2',feature_dim=256,
            extra_DANN_head=[256,128,'source-domain-count'],domain_head_activation='ReLU',dropout=False),
        optimization=dict(steps=2000,Adam_lr=1e-4,Adam_betas=[.9,.999],Adam_eps=1e-8,weight_decay=0,
            AMP=False,TF32=False,sampler='same balanced source-domain/class full pools; case-specific original seed+11 RandomState',
            factual_batch='D*K*floor(120/(D*K))',forward='one factual B; no Mixup, bank, event edits or target samples'),
        losses=dict(ERM='CE(yhat,y)',DANN='CE(yhat,y)+CE(domain_head(GRL(z,lambda)),source_domain)',
            DANN_lambda='2/(1+exp(-10*(step-1)/1999))-1; GRL applied once, no second lambda on domain CE',
            DCORAL='CE(yhat,y)+mean_over_unordered_source_pairs(||C_i-C_j||_F^2/(4*256^2))',
            covariance='unbiased centered covariance with denominator n-1',DCORAL_weight=1.0,
            alignment_level='256-dimensional encoder output after fixed feature IN; before ClfHead'),
        selection=dict(primary='last2000',secondary='earliest strict domain-equal source best among0/1000/2000'),
        qualification=dict(cases=qkeys,arms=list(ARMS),steps=30,restore_at=15,
            modes=['reference_serial','fast_serial','fast_parallel2','fast_parallel3','fast_disk_resume'],
            actual_updates=1350,source_probe_windows=2880,extra_target_predictions=0,
            require='exact model/Adam/BN/all RNG/sampled-batch trace/loss/source-probability state across all five modes'),
        limits=dict(formal_updates=1620000,qualification_updates=1350,optimizer_attempts=1621350,
            training_forward_windows=sum(len(ARMS)*2000*batch(c) for c in cases.values())+sum(5*3*30*batch(cases[k]) for k in qkeys),
            source_prediction_windows=src+2880,target_prediction_windows=target,
            checkpoint_writes=810*9+90,checkpoint_reads=810*6+200,probability_writes=810*3+1620,
            probability_reads=810*3+2*1620,feature_hashes=17*16,feature_decodes=17*16,neural_model_builds=810*3+120),
        execution=dict(max_concurrency=3,min_processes=1,ledger='C small SQLite WAL synchronous FULL, prepay+complete two transactions',
            store=str(STORE),neural_hours=48,total_hours=96,clocks_start='after original5460-scope final audit, not while queued',
            choose_concurrency='highest aggregate active-update throughput270/max(worker update seconds) among1/2/3; retain actual wall time separately; exact states first; peak GPU memory below21GiB',
            all810_source_models_before_any_new_target_evaluation=True,failures='preserve paid prefix and saved state; no silent full-model retry'),
        original_scope_unchanged=True,extended_models=6270,extended_metric_roles=12720,
        prior_results='old11100 roles preserved; append1620 new roles; common task-equal mean/sample-SD per10 seeds',
        literature=dict(DANN='https://jmlr.org/papers/v17/15-239.html',DCORAL='https://arxiv.org/abs/1607.01719',
            adaptation='multi-source DG: discriminate/align only observed sources; not original source-target DA experiments'),
        historical_nonreuse='old dann_t3fft uses2500 steps, different split/sampler/normalization and lambda semantics; never reuse its result as this fixed2000-step recipe',
        input_sha256={str(p):sha(p) for p in inputs},code_sha256=local_closure([Path(__file__),Path(__file__).with_name('tenrepeat_baseline_addon.py')]),
        new_training_so_far=0)
    need(len(units)==810 and 3*3*30*5==1350,'finite model and qualification counts')
    spec['fingerprint']=canonical(spec);write(OUT/'SPEC_V1.json.gz',spec)
    write(OUT/'STATIC_PLAN_V1.json',dict(status='PASS',models=810,formal_updates=1620000,qualification_updates=1350,
        source_windows=src,target_windows=target,source_only=True,existing_workers_unchanged=True,new_neural_actions=0))
    print('BASELINE_EXTENSION_SPEC_READY_810_AFTER_EXISTING',flush=True)


def specification():
    v=json.loads(gzip.decompress((OUT/'SPEC_V1.json.gz').read_bytes()))
    need(canonical({k:x for k,x in v.items() if k!='fingerprint'})==v['fingerprint'],'immutable810 model extension specification')
    for p,h in {**v['input_sha256'],**v['code_sha256']}.items():need(sha(p)==h,'bound baseline extension code/input '+p)
    return v


def activate():
    spec=specification();old=read(PARENT/'FINAL_AUDIT_V1.json')
    need(old['status']=='PASS' and old['models']==5460,'finish original scope first')
    need(sha(PARENT/'FULL_TEN_RUN_RESULTS_V1.json')==old['results_sha256'],'actual original completed results binding')
    now=time.time();lock=copy.deepcopy(spec)
    lock.update(parent_final_audit_sha256=sha(PARENT/'FINAL_AUDIT_V1.json'),parent_results_sha256=old['results_sha256'],
        campaign=dict(start=now,deadline=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).isoformat(),store=str(STORE)))
    lock['campaign'].update(neural_start=now,neural_seconds=48*3600,
        deadline=datetime.fromtimestamp(now+96*3600,timezone(timedelta(hours=8))).isoformat())
    lock['specification_fingerprint']=lock.pop('fingerprint');lock['fingerprint']=canonical(lock)
    write(OUT/'LOCK_V1.json.gz',lock)
    return lock


if __name__=='__main__':prepare()
