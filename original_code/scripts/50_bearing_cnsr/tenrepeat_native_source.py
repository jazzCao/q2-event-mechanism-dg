"""567 native author-family source repeats; original 3001-step engine intact."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import os
import sys
import time
from tenrepeat_source_metadata import ROOT,OUT,read,sha,canonical,need,write

ARMS=('A0_AUTHOR_CE','A2_AUTHOR_IN_MIXUP','C0_CCDG_CODE')
HERE=OUT/'native_source'

def prepare(version='V1'):
    from e0_selection_entry_spec import local_closure
    m=json.loads(gzip.decompress((OUT/'MDG_SOURCE_MANIFEST_V1.json.gz').read_bytes()))
    HERE.mkdir(parents=True,exist_ok=True);n=189*3
    windows=sum(4*3*c['validation_windows'] for c in m['cases'].values())+3*4*64*(3+2+2)
    train_windows=sum(3001*4*32*len(c['task']['sources']) for c in m['cases'].values())+3*20*4*32*(3+2+2)
    limits=dict(optimizer_attempts=n*3001+180,native_optimizer_attempts=n*3001+180,actual_optimizer_updates=n*3001+180,
        native_training_forward_windows=train_windows,source_prediction_windows=windows,target_prediction_windows=0,
        feature_hashes=17,feature_decodes=17,checkpoint_writes=10*n+27,checkpoint_reads=5*n+9,
        probability_writes=4*n+36,probability_reads=4*n+36)
    cases={k:dict(key=k,task=c['task'],seed=c['seed'],source_training_windows=c['training_windows'],source_validation_windows=c['validation_windows']) for k,c in m['cases'].items()}
    now=datetime.now(timezone(timedelta(hours=8)))
    files=[OUT/'MDG_SOURCE_MANIFEST_V1.json.gz',OUT/'HUMAN_SCOPE_V1.md',ROOT/'results/wfdg/ress_event_native_main/AUTHOR_MATH_BINDINGS.json']
    bindings=read(files[-1])
    for p,h in bindings['bindings'].items():need(sha(p)==h,'qualified author implementation '+p)
    lock=dict(kind='TEN_REPEAT_NATIVE_AUTHOR_SOURCE_ONLY',campaign=dict(start=now.isoformat(),deadline=(now+timedelta(hours=72)).isoformat(),neural_seconds=36*3600,store='D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_tenseed_all/native_source'),
        cases=cases,original_cases=m['cases'],records=m['records'],feature_bindings=m['features'],expected_FFT_code_sha256=m['FFT_code_sha256'],
        models=n,arms=list(ARMS),P0_cases=['pu_clock4/ABCtoD/s3','phm09_w3/adj_AC_B/s3','jnu_speed/A/s3'],limits=limits,
        input_sha256={str(p):sha(p) for p in files},code_sha256=local_closure([Path(__file__)]),qualified_author_bindings=bindings['bindings'],
        schedule=dict(steps=3001,source_steps=[0,1000,2000,3001],optimizer='unchanged native Adam',scheduler='unchanged native StepLR'),source_only=True)
    lock['fingerprint']=canonical(lock);write(HERE/f'LOCK_{version}.json.gz',lock)
    print('TEN_REPEAT_NATIVE_READY_567',limits['optimizer_attempts'],flush=True)

def sources(rt,files,c):
    import numpy as np
    import torch
    case=rt.lock['original_cases'][c['key']];a=files.source(c);arrays={k:{q:v for q,v in val.items() if q!='offset'} for k,val in a.items()}
    fp=canonical(dict(case=c,roles=case['roles'],features={d:next(b['sha256'] for b in rt.lock['feature_bindings'] if b['protocol']==c['task']['protocol'] and b['domain']==d) for d in c['task']['sources']}))
    if c['task']['protocol']=='pu_clock4':
        from event_author_sources import bind_sources
        return bind_sources(rt.lock['records'],case,arrays,fp)
    from event_native_main_sources import bind_sources
    return bind_sources(rt.lock['records'],case,arrays,fp,tensor_factory=torch.from_numpy)

def probe(rt,engine,source,name,qualification=False):
    import numpy as np
    import torch
    from event_author_native_engine import NativeAuthorEngine
    from source_rate_neural import tensor_sha
    parts=[source.get(d,'val') for d in source.allowed];n=64 if qualification else None
    x=torch.cat([v.x[:n] for v in parts]);y=np.concatenate([v.y[:n] for v in parts]);g=np.concatenate([v.groups[:n] for v in parts]);domains=np.concatenate([np.full(len(v.y[:n]),d) for d,v in zip(source.allowed,parts)])
    rt.claim('source_prediction_windows',len(y));engine.model.eval();before=tensor_sha(engine.model.state_dict());prob=[]
    with torch.no_grad():
        for start in range(0,len(y),512):
            rt.clock();logits=engine.model(x[start:start+512].cuda())['logits'].detach().cpu().numpy().astype(np.float64);logits-=logits.max(1,keepdims=True);q=np.exp(logits);prob.append(q/q.sum(1,keepdims=True))
    need(tensor_sha(engine.model.state_dict())==before,'frozen native source prediction parameters and BN')
    prob=np.concatenate(prob);score=float(np.mean([(prob[domains==d].argmax(1)==y[domains==d]).mean() for d in source.allowed]))
    b=rt.save(Path(name)/f'source_{engine.step_n}.npz',dict(prob=prob,y=y,g=g,domain=domains,model_tensor_sha256=np.array(before)))
    row=dict(step=engine.step_n,score=score,target_used_for_selection=False)
    if not qualification:row['probability']=b
    engine.consider(row);return prob

def engine_for(rt,c,source,arm,qualification=False):
    from event_author_native_engine import NativeAuthorEngine
    from tenrepeat_durable import NativePermit
    permit=NativePermit(rt,c['key'],arm,qualification)
    return NativeAuthorEngine(source,c['task'],c['seed'],arm,permit.fingerprint,permit,rt,maximum_steps=10 if qualification else 3001,validation_steps=(0,10) if qualification else (0,1000,2000,3001))

def qualify(rt,files):
    from event_native_main_neural import exact
    import torch
    proofs=[]
    for k in rt.lock['P0_cases']:
        c=rt.lock['cases'][k];s=sources(rt,files,c)
        for arm in ARMS:
            outputs=[]
            for branch in ('continuous','resume'):
                name='P0/'+k+'/'+arm+'/'+branch;rt.active=name;e=engine_for(rt,c,s,arm,True);probe(rt,e,s,name,True);draws=[]
                while e.step_n<10:
                    _,ids=e.step(inspect=e.step_n==0);draws.append(ids)
                    if branch=='resume' and e.step_n==5:
                        p=e.packet();b=rt.save(Path(name)/'state5.pt',p,packet=True);del e;e=engine_for(rt,c,s,arm,True);e.restore(rt.load(b));exact(p,e.packet(),'native new-seed disk5 exact')
                prob=probe(rt,e,s,name,True);outputs.append((e.packet(),prob,draws));rt.save(Path(name)/'state10.pt',e.packet(),packet=True);del e
            exact(outputs[0],outputs[1],'native continuous10 versus actual disk5+5 complete state/probabilities/draws')
            proofs.append(dict(case=k,arm=arm,full_restore_exact=True,gradients=True))
        del s;torch.cuda.empty_cache()
    need(rt.counts['optimizer_attempts']==180,'native new-seed P0 actual180')
    write(HERE/'CUDA_P0_V1.json',dict(status='PASS',proofs=proofs,updates=180,target_predictions=0));rt.set('P0_PASS',True)
    print('TEN_REPEAT_NATIVE_CUDA_PASS',flush=True)

def train(rt,files):
    from event_native_main_neural import exact
    from source_rate_neural import tensor_sha
    import torch
    for k,c in sorted(rt.lock['cases'].items()):
        s=sources(rt,files,c)
        for arm in ARMS:
            name='MAIN/'+k+'/'+arm;rt.active=name;e=engine_for(rt,c,s,arm);probe(rt,e,s,name);states={0:rt.save(Path(name)/'state0.pt',e.packet(),packet=True)}
            while e.step_n<3001:
                e.step(inspect=e.step_n==0)
                if e.step_n in (1000,2000,3001):probe(rt,e,s,name)
                if e.step_n%500==0 or e.step_n==3001:
                    states[e.step_n]=rt.save(Path(name)/f'state{e.step_n}.pt',e.packet(),packet=True)
                    if e.step_n==1500:
                        packet=e.packet();del e;e=engine_for(rt,c,s,arm);e.restore(rt.load(states[1500]));exact(packet,e.packet(),'native actual full disk1500 continuation')
            value=dict(case_key=k,arm=arm,step=3001,states={str(q):v for q,v in states.items()},last=states[3001],selected=states[e.best_step],selected_step=e.best_step,curve=e.curve,logs=e.logs,actual_disk1500_restore=True)
            rt.finish(name,value);del e;print('TEN_REPEAT_NATIVE_MODEL',len(rt.models()),'/567',name,flush=True)
        del s;torch.cuda.empty_cache()
    need(len(rt.models())==567 and rt.counts['optimizer_attempts']==567*3001+180,'all567 native new source repeats')
    checks=[]
    for name,model in rt.models().items():
        chosen=min(model['curve'],key=lambda v:(-v['score'],v['step']));need(chosen['step']==model['selected_step'],'native earliest strict source selection')
        for row in model['curve']:
            p=rt.load(row['probability']);state=rt.load(model['states'][str(row['step'])]);need(tensor_sha(state['model'])==str(p['model_tensor_sha256']),'actual native saved source state/probability binding')
            need(state['scheduler']['last_epoch']==row['step'],'actual native scheduler step')
            import numpy as np
            task=rt.lock['cases'][model['case_key']]['task'];score=float(np.mean([(p['prob'][p['domain']==d].argmax(1)==p['y'][p['domain']==d]).mean() for d in task['sources']]))
            need(score==row['score'],'independent saved probability source score reconstruction')
        checks.append(dict(model=name,selected=model['selected_step'],actual_disk1500=True))
    write(HERE/'SOURCE_AUDIT_V1.json',dict(status='PASS',models=567,checks=checks,target_predictions=0))
    write(HERE/'SOURCE_STAGE_COMPLETE_V1.json',dict(status='SOURCE_COMPLETE_TARGET_PENDING_MASTER',models=rt.models(),counts=dict(rt.counts),fingerprint=rt.lock['fingerprint'],at=time.time()))
    rt.set('DONE',True);print('TEN_REPEAT_NATIVE_SOURCE_COMPLETE_567_NO_TARGET',flush=True)

def run(version='V1'):
    lock=json.loads(gzip.decompress((HERE/f'LOCK_{version}.json.gz').read_bytes()));need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'native source lock')
    for p,h in {**lock['input_sha256'],**lock['code_sha256'],**lock['qualified_author_bindings']}.items():need(sha(p)==h,'native bound code/source '+p)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from tenrepeat_durable import Runtime
    rt=Runtime(lock,HERE);sys.addaudithook(rt.guard)
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from source_rate_screen_data import Files
    try:rt.before_neural();files=Files(rt);qualify(rt,files);train(rt,files)
    except BaseException as e:rt.set('STOPPED',dict(type=type(e).__name__,message=str(e),counts=dict(rt.counts)));raise
    finally:rt.db.close()

if __name__=='__main__':{'prepare':prepare,'run':run,'prepare-v2':lambda:prepare('V2'),'run-v2':lambda:run('V2')}[sys.argv[1]]()
