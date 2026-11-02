"""168 PCR source repeats using the unchanged two physical-width frontends.

Reuse all 1440 archived factual record features. Counterfactual features missing
for a newly eligible source TRAIN record are generated from its frozen spectrum.
No new raw signal, angular frontend fit or target observation is required.
"""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
import ast
import gzip
import hashlib
import io
import json
import os
import sqlite3
import sys
import time
from tenrepeat_source_metadata import ROOT,OUT,read,sha,canonical,need,write

HERE=OUT/'pcr_source';FRONTENDS=('NATIVE40','SHORT_PHYSICAL_WIDTH');PROFILES=('AUTHOR_CE','CE_PLUS_REAL_HEALTHY','PCR_FAULT_TO_HEALTHY')

def sampler_adapter():
    import pcr_training_design as old
    ns=dict(old.__dict__);tree=ast.parse(Path(old.__file__).read_text('utf-8'))
    nodes=[v for v in tree.body if isinstance(v,(ast.FunctionDef,ast.ClassDef)) and v.name in ('model_initialization_seed','source_stream_seed','SourceSampler')];changes=0
    for n in nodes:
        for v in ast.walk(n):
            if isinstance(v,ast.Tuple) and [q.value if isinstance(q,ast.Constant) else None for q in v.elts]==[0,1,2]:v.elts=[ast.Constant(value=i) for i in range(10)];changes+=1
    need(changes==2,'only two PCR seed guards expanded')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::PCR_seed_guard_extension','exec'),ns);return ns

def prepare(version='V1'):
    from e0_selection_entry_spec import local_closure
    from prepare_pcr_comparison import case_metadata
    import prepare_pcr_comparison as old
    from pcr_physics_adapter import native_layout,RECORD_CAP
    HERE.mkdir(parents=True,exist_ok=True);m=json.loads(gzip.decompress((OUT/'MDG_SOURCE_MANIFEST_V1.json.gz').read_bytes()));records={k:r for k,r in m['records'].items() if r['protocol']=='pu_clock4'}
    # Bind the same case compiler with the sole seed guard widened.
    ns=dict(old.__dict__);ns['SEEDS']=tuple(range(10));tree=ast.parse(Path(old.__file__).read_text('utf-8'));node=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='case_metadata');exec(compile(ast.Module(body=[node],type_ignores=[]),__file__+'::PCR_case_seed_extension','exec'),ns)
    cases={}
    for k,c in m['cases'].items():
        if c['task']['protocol']!='pu_clock4':continue
        cc=dict(c,inventory=dict(source_fingerprint=canonical(dict(task=c['task'],seed=c['seed'],roles=c['roles']))));cases[k]=dict(ns['case_metadata'](cc,records),key=k)
    parent=ROOT/'results/wfdg/ress_pcr_comparison';dbpath=parent/'control/CONTROL.sqlite'
    with sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True) as db:generation=json.loads(db.execute("SELECT value FROM journal WHERE name='SOURCE_GENERATION'").fetchone()[0])
    need(len(generation['factual_bindings'])==2880,'all1440 two-frontend archived factual record caches')
    needed={r+'|'+f for c in cases.values() for r in c['roles']['source_train'] if records[r]['label']!=0 for f in FRONTENDS}
    missing=sorted(needed-set(generation['cf_bindings']))
    # The whole source split changes with each run exactly as before.
    source_count=sum(3*3*sum(60 if f=='SHORT_PHYSICAL_WIDTH' else native_layout(RECORD_CAP,records[r]['fs'],records[r]['shaft_hz'])['windows'] for r in c['roles']['source_validation']) for c in cases.values() for f in FRONTENDS)
    qkeys=[k for k,c in cases.items() if c['task']['fold'] in ('BCDtoA','ABCtoD')]
    need(len(cases)==28 and len(qkeys)==14,'four original folds and seven seeds, both speed layouts qualified')
    qualification=len(qkeys)*6*10;models=168;limits=dict(optimizer_attempts=models*2000+qualification,actual_optimizer_updates=models*2000+qualification,
        training_forward_windows=28*2*2000*(200+400+400)+len(qkeys)*2*10*(200+400+400),source_prediction_windows=source_count+len(qkeys)*6*4*64,target_prediction_windows=0,
        neural_model_constructions=models*3+len(qkeys)*18,checkpoint_writes=models*9+len(qkeys)*18,checkpoint_reads=models*4+len(qkeys)*6,
        probability_writes=models*3+len(missing),probability_reads=models*3,source_payload_reads=100000,source_cf_generations=len(missing))
    now=datetime.now(timezone(timedelta(hours=8)))
    oldlock=read(parent/'PROPOSED_RUN_LOCK.json')
    for p,h in oldlock['code_sha256'].items():need(sha(p)==h,'PCR original implementation still qualified '+p)
    lock=dict(kind='TEN_REPEAT_PCR_SIX_FIXED_SOURCE_RECIPES',campaign=dict(start=now.isoformat(),deadline=(now+timedelta(hours=72)).isoformat(),neural_seconds=40*3600,store='D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_tenseed_all/pcr_source'),
        cases=cases,records=records,factual=generation['factual_bindings'],counterfactual=generation['cf_bindings'],missing_source_cf=missing,
        models=models,P0_cases=qkeys,limits=limits,torch_version=oldlock['torch_version'],numpy_version=oldlock['numpy_version'],
        input_sha256={str(parent/'PROPOSED_RUN_LOCK.json'):sha(parent/'PROPOSED_RUN_LOCK.json'),str(dbpath):sha(dbpath),str(OUT/'MDG_SOURCE_MANIFEST_V1.json.gz'):sha(OUT/'MDG_SOURCE_MANIFEST_V1.json.gz')},
        code_sha256=local_closure([Path(__file__)]),author_code_sha256=oldlock['code_sha256'],source_only=True,seed_guard_changes=2,new_raw_reads=0,new_frontend_fft=0)
    lock['fingerprint']=canonical(lock);write(HERE/f'LOCK_{version}.json.gz',lock);print('TEN_REPEAT_PCR_SOURCE_READY_168',len(missing),'new source CF spectra',flush=True)

def run(version='V1'):
    lock=json.loads(gzip.decompress((HERE/f'LOCK_{version}.json.gz').read_bytes()))
    need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'PCR repeat lock')
    for p,h in {**lock['input_sha256'],**lock['code_sha256'],**lock['author_code_sha256']}.items():need(sha(p)==h,'PCR bound source/code '+p)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from tenrepeat_durable import Runtime as BaseRuntime
    class Runtime(BaseRuntime):
        def __init__(self,*a):
            super().__init__(*a);self.proposal=dict(records=lock['records']);self.ledger=SimpleNamespace(completed=self.completed,cursor=lambda name:(self.cursor.get(self.active,0),)*3);self.expected_engine_name=None
        neural=BaseRuntime.before_neural
        def attempt(self,name,n,profile):need(name==self.expected_engine_name and profile in PROFILES,'PCR exact active engine');BaseRuntime.attempt(self,n);self.claim('training_forward_windows',200 if profile=='AUTHOR_CE' else 400)
        def completed(self,name,n):need(self.pending==(self.active,n) and name==self.expected_engine_name,'PCR actual paid cursor');self.claim('actual_optimizer_updates')
    rt=Runtime(lock,HERE);sys.addaudithook(rt.guard)
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import pcr_campaign_neural as original
    adapter=sampler_adapter();ns=dict(original.__dict__);ns.update(Runtime=Runtime,SourceSampler=adapter['SourceSampler'],model_initialization_seed=adapter['model_initialization_seed'])
    tree=ast.parse(Path(original.__file__).read_text('utf-8'));names={'Chain','build_model'};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),__file__+'::unchanged_PCR_neural_math','exec'),ns)
    Chain=ns['Chain'];cache={};cf=dict(lock['counterfactual'])
    def payload(b):
        rt.claim('source_payload_reads');rt.permit(b['path'])
        try:
            raw=Path(b['path']).read_bytes();need(hashlib.sha256(raw).hexdigest()==b['sha256'],'PCR exact feature packet SHA')
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
        finally:rt.revoke(b['path'])
    def data(c,f):
        tr={};va={};aux={};bindings={}
        for role,dest in (('source_train',tr),('source_validation',va)):
            for r in c['roles'][role]:
                need(lock['records'][r]['domain'] in c['task']['sources'] and lock['records'][r]['bearing'] not in ('K006','KA08','KA09'),'PCR source role on every cached use')
                key=r+'|'+f
                if key not in cache:cache[key]=payload(lock['factual'][key]['array'])
                dest[r]=cache[key]['x'];bindings[role+'/'+key]=lock['factual'][key]['identity']
                if role=='source_train' and lock['records'][r]['label']!=0:
                    if key not in cf:
                        need(key in lock['missing_source_cf'],'only registered missing source spectra')
                        from pcr_campaign_data import source_cf
                        rt.claim('source_cf_generations');values,width=source_cf(cache[key],lock['records'][r]['label'],f)
                        b=rt.save(Path('source_cf')/(hashlib.sha256(key.encode()).hexdigest()+'.npz'),dict(x=values));cf[key]=dict(array=b,record=r,frontend=f,width=width)
                    ck='cf/'+key
                    if ck not in cache:cache[ck]=payload(cf[key]['array'])
                    aux[r]=cache[ck]['x']
        return dict(train=tr,validation=va,cf=aux,source_fingerprint=canonical(dict(roles=c['role_fingerprint'],frontend=f,bindings=bindings)))
    def probe(e,name,mini=False):
        values=original.predict(e.model,lock['records'],e.data['validation'],rt,'source',limit=64 if mini else None)
        if mini:return values['prob']
        from pcr_training_design import source_selection_score
        score=source_selection_score(values['prob'],values['y'],values['domain'],e.case['task']['sources']);b=rt.save(Path(name)/f'source{e.step_n}.npz',values)
        row=dict(step=e.step_n,score=score,probability=b,target_used_for_selection=False);e.curve.append(row)
        if e.best is None or score>e.best:e.best=score;e.selected_step=e.step_n
        return values['prob']
    def engine(c,f,profile,d,name,packet=None):rt.expected_engine_name=name;return Chain(rt,c,f,profile,d,name,packet=packet)
    try:
        rt.before_neural();proofs=[]
        for k in lock['P0_cases']:
            c=lock['cases'][k]
            for f in FRONTENDS:
                d=data(c,f)
                for profile in PROFILES:
                    outputs=[];common='P0/'+k+'/'+f+'/'+profile
                    for branch in ('continuous','resume'):
                        rt.active=common+'/'+branch;e=engine(c,f,profile,d,common);first=probe(e,rt.active,True)
                        while e.step_n<5:
                            e.step(inspect=e.step_n==0)
                            if branch=='resume' and e.step_n==2:
                                p=e.packet();b=rt.save(Path(rt.active)/'state2.pt',p,packet=True);del e;e=engine(c,f,profile,d,common,rt.load(b));original.exact(p,e.packet(),'PCR disk2 complete new-seed restore')
                        outputs.append((e.packet(),first,probe(e,rt.active,True)));rt.save(Path(rt.active)/'state5.pt',outputs[-1][0],packet=True);del e
                    original.exact(outputs[0],outputs[1],'PCR continuous5/disk2+3 state/AdamW/scheduler/sampler/BN-hook/probabilities');proofs.append(dict(case=k,frontend=f,profile=profile,complete_state_exact=True))
        need(rt.counts['optimizer_attempts']==840,'PCR actual840 new-seed qualification attempts')
        write(HERE/'CUDA_P0_V1.json',dict(status='PASS',proofs=proofs,attempts=840,target_predictions=0));rt.set('P0_PASS',True)
        for k,c in sorted(lock['cases'].items()):
            for f in FRONTENDS:
                d=data(c,f)
                for profile in PROFILES:
                    name='MAIN/'+k+'/'+f+'/'+profile;rt.active=name;e=engine(c,f,profile,d,name);probe(e,name);states={0:rt.save(Path(name)/'state0.pt',e.packet(),packet=True)}
                    while e.step_n<2000:
                        e.step(inspect=e.step_n==0)
                        if e.step_n in (1000,2000):probe(e,name)
                        if e.step_n%250==0:
                            states[e.step_n]=rt.save(Path(name)/f'state{e.step_n}.pt',e.packet(),packet=True)
                            if e.step_n==1000:
                                p=e.packet();del e;e=engine(c,f,profile,d,name,rt.load(states[1000]));original.exact(p,e.packet(),'PCR formal disk1000 exact continuation')
                    for row in e.curve:
                        a=rt.load(row['probability']);from pcr_training_design import source_selection_score
                        need(source_selection_score(a['prob'],a['y'],a['domain'],c['task']['sources'])==row['score'],'PCR independent saved source score')
                    selected=min(e.curve,key=lambda v:(-v['score'],v['step']));need(selected['step']==e.selected_step,'PCR strict earliest source selection')
                    rt.finish(name,dict(case_key=k,frontend=f,profile=profile,last=states[2000],selected=states[e.selected_step],selected_step=e.selected_step,curve=e.curve,states={str(n):b for n,b in states.items()},actual_disk1000=True))
                    del e;print('TEN_REPEAT_PCR_SOURCE_MODEL',len(rt.models()),'/168',name,flush=True)
        need(len(rt.models())==168 and rt.counts['optimizer_attempts']==336840,'all168 frozen PCR source repeats')
        write(HERE/'SOURCE_STAGE_COMPLETE_V1.json',dict(status='SOURCE_COMPLETE_TARGET_PENDING_MASTER',models=rt.models(),counts=dict(rt.counts),counterfactual=cf,fingerprint=lock['fingerprint'],at=time.time(),target_predictions=0));rt.set('DONE',True)
    except BaseException as e:rt.set('STOPPED',dict(type=type(e).__name__,message=str(e),counts=dict(rt.counts)));raise
    finally:rt.db.close()

if __name__=='__main__':{'prepare':prepare,'run':run,'prepare-v2':lambda:prepare('V2'),'run-v2':lambda:run('V2')}[sys.argv[1]]()
