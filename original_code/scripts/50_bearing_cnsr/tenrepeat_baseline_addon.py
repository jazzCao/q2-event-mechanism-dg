"""Queued810-model ERM/DANN/DCORAL extension with qualified GPU concurrency."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import copy
import gzip
import hashlib
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import time

from tenrepeat_baseline_addon_spec import ROOT, PARENT, OUT, ARMS, specification, activate
from tenrepeat_source_metadata import read, write, sha, need


def configure():
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    need(torch.cuda.is_available(),'real CUDA, no training CPU fallback')


def coral(features, domains):
    """Unbiased source covariance alignment; exact Deep-CORAL normalization."""
    import torch
    values=[]
    for d in torch.unique(domains,sorted=True):
        z=features[domains==d];need(len(z)>1,'at least two source examples per covariance')
        z=z-z.mean(0,keepdim=True);values.append(z.T@z/(len(z)-1))
    pairs=[(a-b).square().sum()/(4*features.shape[1]**2)
           for i,a in enumerate(values) for b in values[i+1:]]
    need(bool(pairs),'at least two observed source domains')
    return torch.stack(pairs).mean()


def grl(features, scale):
    import torch
    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):return x.view_as(x)
        @staticmethod
        def backward(ctx,grad):return -scale*grad
    return Reverse.apply(features)


def runtime_class():
    from tenrepeat_durable import Runtime as Base
    class Runtime(Base):
        def __init__(self,lock,here,reference=False):
            super().__init__(lock,here);self.reference=reference;self.forward_paid=False
            self.gpu_start=lock['campaign']['neural_start']
            self.set('gpu_start',self.gpu_start)
        def begin(self,name,n,windows,qualification):
            self.active=name
            if self.reference:
                Base.attempt(self,n)
                self.claim('qualification_updates' if qualification else 'formal_updates')
                return
            self.before_neural()
            need(self.pending is None and n==self.cursor.get(name,0)+1,'sequential prepaid source update')
            with self.db:
                for k,v in [('optimizer_attempts',1),('qualification_updates' if qualification else 'formal_updates',1),('training_forward_windows',windows)]:
                    need(self.counts[k]+v<=self.lock['limits'][k],'finite prepaid resource '+k)
                    self.counts[k]+=v
                    self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)',(k,self.counts[k]))
                self.db.execute('INSERT INTO steps VALUES (?,?,0)',(name,n))
            self.pending=(name,n);self.forward_paid=(windows,False)
        def forward(self,windows):
            if self.reference:return self.claim('training_forward_windows',windows)
            need(self.pending is not None and self.forward_paid==(windows,False),'exact one prepaid B forward')
            self.forward_paid=(windows,True)
        def complete(self,name,n):
            if not self.reference:need(self.pending==(name,n) and self.forward_paid[1],'actual forward before completion')
            self.claim('actual_optimizer_updates');self.forward_paid=False
    return Runtime


class Data:
    def __init__(self,case,source):
        import numpy as np
        import torch
        self.case=case;self.source=source;task=case['task'];self.per=120//(len(task['sources'])*task['classes'])
        self.x=torch.as_tensor(source['tr']['x'],device='cuda');self.y=torch.as_tensor(source['tr']['y'],device='cuda').long()
        self.pools=[np.flatnonzero((source['tr']['domain']==d)&(source['tr']['y']==y)) for d in task['sources'] for y in range(task['classes'])]
        need(all(len(v)>0 for v in self.pools),'all source training classes represented in every domain')
        self.batch=len(self.pools)*self.per
        self.domains=torch.arange(len(task['sources']),device='cuda').repeat_interleave(task['classes']*self.per)
    def sample(self,sampler):
        import numpy as np
        import torch
        ids=np.concatenate([sampler.choice(p,self.per,replace=True) for p in self.pools])
        idx=torch.as_tensor(ids,device='cuda');return self.x[idx],self.y[idx],self.domains,ids


class Chain:
    def __init__(self,rt,case,arm,name,data,packet=None,qualification=False):
        import random
        import numpy as np
        import torch
        from clock_real_model import Net
        from source_rate_neural import initial_seed,tensor_sha
        rt.claim('neural_model_builds');self.rt=rt;self.case=case;self.arm=arm;self.name=name;self.data=data;self.qualification=qualification
        seed=initial_seed(case['task'],case['seed']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
        self.model=Net(case['task']['classes'],'E0_EVENTS').cuda();self.disc=None
        need(tensor_sha(self.model.enc.state_dict())==rt.lock['initial_encoder_sha256'][case['key']],'original matching encoder initialization')
        if arm=='DANN_SOURCE_ONLY':
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed((seed+701)%2**32)
                self.disc=torch.nn.Sequential(torch.nn.Linear(256,128),torch.nn.ReLU(),torch.nn.Linear(128,len(case['task']['sources']))).cuda()
        params=list(self.model.parameters())+([] if self.disc is None else list(self.disc.parameters()))
        self.opt=torch.optim.Adam(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        self.sampler=np.random.RandomState((seed+11)%2**32);self.n=0;self.logs=[];self.curve=[];self.best=None;self.selected=0;self.trace=hashlib.sha256(b'').hexdigest()
        if packet is not None:self.restore(packet)
    def step(self,inspect=False):
        import torch
        from torch.nn import functional as F
        self.rt.begin(self.name,self.n+1,self.data.batch,self.qualification)
        self.model.train()
        if self.disc is not None:self.disc.train()
        x,y,d,ids=self.data.sample(self.sampler);self.rt.forward(len(x))
        h=self.model.enc.net(x[:,None]);h=(h-h.mean(-1,keepdim=True))/torch.sqrt(h.var(-1,unbiased=False,keepdim=True)+1e-5)
        z=self.model.enc.fc(h);logits=self.model.clf(z);ce=F.cross_entropy(logits,y);regularizer=ce.new_zeros(());scale=0.
        if self.arm=='DANN_SOURCE_ONLY':
            scale=2/(1+math.exp(-10*self.n/1999))-1
            regularizer=F.cross_entropy(self.disc(grl(z,scale)),d)
        elif self.arm=='DCORAL_SOURCE_ONLY':regularizer=coral(z,d)
        loss=ce+regularizer;need(bool(torch.isfinite(loss)),'finite CE and regularizer')
        self.opt.zero_grad(set_to_none=True);loss.backward()
        if inspect:
            params=list(self.model.parameters())+([] if self.disc is None else list(self.disc.parameters()))
            need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),'actual finite gradients, including DANN head')
            need(sum(float(p.grad.square().sum()) for p in params)>1e-24,'actual nonzero classification gradient')
        self.opt.step();self.n+=1;self.rt.complete(self.name,self.n)
        self.trace=hashlib.sha256(bytes.fromhex(self.trace)+ids.tobytes()).hexdigest()
        if self.qualification or self.n%100==0:
            self.logs.append(dict(step=self.n,ce=float(ce.detach()),regularizer=float(regularizer.detach()),total=float(loss.detach()),GRL_scale=scale))
    def packet(self):
        import random,numpy as np,torch
        from source_rate_neural import cpu
        return cpu(dict(fingerprint=self.rt.lock['fingerprint'],case_key=self.case['key'],arm=self.arm,name=self.name,step=self.n,
            model=self.model.state_dict(),disc=None if self.disc is None else self.disc.state_dict(),optimizer=self.opt.state_dict(),
            model_modes={k:v.training for k,v in self.model.named_modules()},disc_modes=None if self.disc is None else {k:v.training for k,v in self.disc.named_modules()},
            sampler=self.sampler.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            logs=self.logs,curve=self.curve,best=self.best,selected=self.selected,trace=self.trace),torch,np)
    def restore(self,p):
        import random,numpy as np,torch
        need((p['fingerprint'],p['case_key'],p['arm'],p['name'])==(self.rt.lock['fingerprint'],self.case['key'],self.arm,self.name),'full saved source state identity')
        self.model.load_state_dict(p['model']);self.opt.load_state_dict(p['optimizer'])
        if self.disc is not None:self.disc.load_state_dict(p['disc'])
        for k,v in self.model.named_modules():v.training=p['model_modes'][k]
        if self.disc is not None:
            for k,v in self.disc.named_modules():v.training=p['disc_modes'][k]
        self.n=p['step'];self.logs=copy.deepcopy(p['logs']);self.curve=copy.deepcopy(p['curve']);self.best=p['best'];self.selected=p['selected'];self.trace=p['trace']
        self.sampler.set_state(p['sampler']);torch.set_rng_state(p['torch_rng']);torch.cuda.set_rng_state_all(p['cuda_rng']);np.random.set_state(p['numpy_rng']);random.setstate(p['python_rng'])


def predict(rt,e,arrays,qualification=False):
    import numpy as np,torch
    from source_rate_neural import tensor_sha
    n=min(32,len(arrays['y'])) if qualification else len(arrays['y']);rt.claim('source_prediction_windows',n)
    e.model.eval();before=tensor_sha(e.model.state_dict());parts=[]
    with torch.no_grad():
        for lo in range(0,n,512):
            rt.before_neural();z=e.model(torch.as_tensor(arrays['x'][lo:lo+min(512,n-lo)],device='cuda'))['logits'].cpu().numpy().astype(float)
            z-=z.max(1,keepdims=True);p=np.exp(z);parts.append(p/p.sum(1,keepdims=True))
    need(before==tensor_sha(e.model.state_dict()),'frozen source probe does not update parameters or BN')
    return dict(prob=np.concatenate(parts),y=arrays['y'][:n],g=arrays['g'][:n],domain=arrays['domain'][:n],model_tensor_sha256=np.array(before))


def source_score(rt,e,source):
    import numpy as np
    a=predict(rt,e,source['va']);score=float(np.mean([(a['prob'][a['domain']==d].argmax(1)==a['y'][a['domain']==d]).mean() for d in e.case['task']['sources']]))
    b=rt.save(Path(e.name)/f'source{e.n}.npz',a);e.curve.append(dict(step=e.n,score=score,probability=b,target_used=False))
    if e.best is None or score>e.best:e.best=score;e.selected=e.n


def load_lock():
    lock=json.loads(gzip.decompress((OUT/'LOCK_V1.json.gz').read_bytes()))
    from tenrepeat_source_metadata import canonical
    need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'immutable activated extension lock')
    for p,h in {**lock['input_sha256'],**lock['code_sha256']}.items():need(sha(p)==h,'unchanged extension code/input '+p)
    return lock


def qworker(mode,index,count):
    configure();import torch
    from source_rate_screen_data import Files
    from source_rate_neural import exact
    lock=load_lock();jobs=[(k,a) for k in lock['qualification']['cases'] for a in ARMS][index::count]
    here=OUT/'qualification'/mode/f'worker{index}';here.mkdir(parents=True,exist_ok=True)
    lock=copy.deepcopy(lock);lock['campaign']['store']=str(Path(lock['campaign']['store'])/'qualification'/mode/f'worker{index}')
    rt=runtime_class()(lock,here,mode=='reference_serial');sys.addaudithook(rt.guard);files=Files(rt);outputs=[];seconds=0.
    try:
        for j,(key,arm) in enumerate(jobs):
            c=lock['cases'][key];source=files.source(c);data=Data(c,source);name='QUAL/'+key+'/'+arm;e=Chain(rt,c,arm,name,data,qualification=True)
            first=predict(rt,e,source['va'],True)
            if j==0:
                write(here/'READY.json',dict(pid=os.getpid(),at=time.time()))
                limit=time.time()+600
                while not (OUT/'qualification'/mode/'GO.json').exists():need(time.time()<limit,'bounded engineering start barrier');time.sleep(.05)
            torch.cuda.synchronize();start=time.perf_counter()
            for step in range(30):
                e.step(inspect=step==0)
                if mode=='fast_disk_resume' and e.n==15:
                    before=e.packet();b=rt.save(Path(name)/'state15.pt',before,packet=True);del e
                    e=Chain(rt,c,arm,name,data,rt.load(b),qualification=True);exact(before,e.packet(),torch,__import__('numpy'),'real disk15 full extension restore')
            torch.cuda.synchronize();seconds+=time.perf_counter()-start
            last=predict(rt,e,source['va'],True);b=rt.save(Path(name)/'state30.pt',dict(packet=e.packet(),first=first,last=last),packet=True)
            outputs.append(dict(case_key=key,arm=arm,packet=b));del e,data;torch.cuda.empty_cache()
        write(here/'RESULT.json',dict(status='PASS',models=outputs,counts=dict(rt.counts),training_seconds=seconds,
            peak_allocated=torch.cuda.max_memory_allocated(),at=time.time()))
    except BaseException as error:rt.set('STOPPED',dict(type=type(error).__name__,message=str(error)));raise
    finally:rt.db.close()


def qualify():
    import numpy as np,torch
    from source_rate_neural import exact
    lock=load_lock();allresults={};timings={};gpu={};total=Counter()
    for mode,n in [('reference_serial',1),('fast_serial',1),('fast_parallel2',2),('fast_parallel3',3),('fast_disk_resume',1)]:
        here=OUT/'qualification'/mode;here.mkdir(parents=True,exist_ok=True);processes=[]
        for i in range(n):
            log=(here/f'worker{i}.log').open('xb');cmd=[sys.executable,'-u','-B','-X','utf8',__file__,'qworker',mode,str(i),str(n)]
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);processes.append((proc,log))
        start=time.time()
        while not all((here/f'worker{i}/READY.json').exists() for i in range(n)):
            need(all(p.poll() is None for p,_ in processes),'qualification worker failed before start; preserve evidence')
            need(time.time()-start<600,'bounded qualification setup');time.sleep(.2)
        write(here/'GO.json',dict(at=time.time()));begin=time.time();peak=0
        while any(p.poll() is None for p,_ in processes):
            rr=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            peak=max(peak,int(rr.stdout.strip().splitlines()[0]));time.sleep(.5)
        for p,log in processes:need(p.returncode==0,'all actual CUDA qualification workers successful');log.close()
        rows=[read(here/f'worker{i}/RESULT.json') for i in range(n)];timings[mode]=dict(wall_seconds=time.time()-begin,
            update_seconds_max_worker=max(v['training_seconds'] for v in rows),throughput=270/max(v['training_seconds'] for v in rows),
            wall_throughput=270/(time.time()-begin));gpu[mode]=peak
        allresults[mode]={}
        for v in rows:
            total.update(v['counts'])
            for row in v['models']:
                b=row['packet'];need(sha(b['path'])==b['sha256'],'actual qualification payload SHA')
                with open(b['path'],'rb') as f:value=torch.load(f,map_location='cpu',weights_only=False)
                total['checkpoint_reads']+=1
                allresults[mode][(row['case_key'],row['arm'])]=value
    reference=allresults['reference_serial']
    need(len(reference)==9,'all three protocols and three baseline mechanisms qualified')
    for mode,values in allresults.items():
        need(values.keys()==reference.keys(),'same qualified cases')
        for key,v in values.items():exact(reference[key],v,torch,np,'reference/fast/concurrent/disk exact state '+mode+'/'+str(key))
    need(total['optimizer_attempts']==1350 and total['source_prediction_windows']==2880,'all real qualification work counted')
    choices=[('fast_serial',1),('fast_parallel2',2),('fast_parallel3',3)]
    mode,n=max((v for v in choices if gpu[v[0]]<21504),key=lambda v:timings[v[0]]['throughput'])
    value=dict(status='PASS',full_state_all_five_modes_exact=True,actual_updates=1350,source_probe_windows=2880,
        counts=dict(total),timings=timings,peak_GPU_MiB=gpu,chosen_concurrency=n,chosen_mode=mode,target_predictions=0,at=time.time())
    write(OUT/'CUDA_QUALIFICATION_V1.json',value);print('ADDED_BASELINES_CUDA_EXACT_CONCURRENCY',n,flush=True)


def worker(index,count):
    configure();import torch,numpy as np
    from source_rate_screen_data import Files
    from source_rate_neural import exact,tensor_sha
    lock=load_lock();keys=sorted(lock['cases'])[index::count];here=OUT/f'worker{index}';here.mkdir(exist_ok=True)
    rt=runtime_class()(lock,here);sys.addaudithook(rt.guard);files=Files(rt)
    try:
        for key in keys:
            c=lock['cases'][key];source=files.source(c);data=Data(c,source)
            for arm in ARMS:
                name='BASELINE/'+key+'/'+arm;e=Chain(rt,c,arm,name,data);source_score(rt,e,source)
                states={0:rt.save(Path(name)/'state0.pt',e.packet(),packet=True)}
                while e.n<2000:
                    e.step(inspect=e.n==0)
                    if e.n in (1000,2000):source_score(rt,e,source)
                    if e.n%250==0:
                        states[e.n]=rt.save(Path(name)/f'state{e.n}.pt',e.packet(),packet=True)
                        if e.n==1000:
                            packet=e.packet();del e;e=Chain(rt,c,arm,name,data,rt.load(states[1000]));exact(packet,e.packet(),torch,np,'formal real disk1000 full state')
                rt.finish(name,dict(case_key=key,arm=arm,last=states[2000],selected=states[e.selected],selected_step=e.selected,
                    states={str(k):v for k,v in states.items()},curve=e.curve,losses=e.logs,actual_disk1000_restore=True))
                del e;print('BASELINE_EXTENSION_MODEL',index,len(rt.models()),'/',len(keys)*3,name,flush=True)
            del data;torch.cuda.empty_cache()
        need(len(rt.models())==len(keys)*3 and rt.counts['formal_updates']==len(keys)*6000,'all assigned fixed baseline units complete')
        for name,v in rt.models().items():
            need(min(v['curve'],key=lambda r:(-r['score'],r['step']))['step']==v['selected_step'],'earliest strict source-only selection')
            for row in v['curve']:
                a=rt.load(dict(row['probability'],packet=False));p=rt.load(v['states'][str(row['step'])])
                need(tensor_sha(p['model'])==str(a['model_tensor_sha256']),'actual saved source probability/checkpoint binding')
                score=float(np.mean([(a['prob'][a['domain']==d].argmax(1)==a['y'][a['domain']==d]).mean() for d in lock['cases'][v['case_key']]['task']['sources']]))
                need(score==row['score'],'independent source probability reconstruction')
        write(here/'SOURCE_COMPLETE_V1.json',dict(status='SOURCE_AUDIT_PASS',models=rt.models(),counts=dict(rt.counts),at=time.time()))
        rt.set('DONE',True)
    except BaseException as error:rt.set('STOPPED',dict(type=type(error).__name__,message=str(error),counts=dict(rt.counts)));raise
    finally:rt.db.close()


def train():
    q=read(OUT/'CUDA_QUALIFICATION_V1.json');need(q['status']=='PASS','actual full-state/throughput qualification first')
    count=q['chosen_concurrency'];processes=[]
    for i in range(count):
        log=(OUT/f'worker{i}.log').open('xb');proc=subprocess.Popen([sys.executable,'-u','-B','-X','utf8',__file__,'worker',str(i),str(count)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        processes.append((proc,log));write(OUT/f'WORKER{i}_NATIVE_V1.json',dict(pid=proc.pid,at=time.time()))
    for p,log in processes:need(p.wait()==0,'fixed baseline worker failure; no duplicate-model retry');log.close()
    models={};counts=Counter(q['counts'])
    for i in range(count):
        v=read(OUT/f'worker{i}/SOURCE_COMPLETE_V1.json');need(set(models).isdisjoint(v['models']),'disjoint independent model assignments');models.update(v['models']);counts.update(v['counts'])
    lock=load_lock();need(set(models)=={v['name'] for v in lock['units']} and len(models)==810,'all810 fixed new source models')
    need(counts['optimizer_attempts']==1621350 and counts['formal_updates']==1620000,'exact formal and qualification updates')
    for k,v in counts.items():need(v<=lock['limits'][k],'total aggregated baseline resource '+k)
    write(OUT/'SOURCE_FREEZE_V1.json',dict(status='ALL810_SOURCE_MODELS_FROZEN',models=models,counts=dict(counts),at=time.time(),new_target_predictions=0))


def evaluate(_child=False):
    if not _child:
        proc=subprocess.run([sys.executable,'-u','-B','-X','utf8',__file__,'evaluate-worker'],cwd=ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        need(proc.returncode==0,'baseline frozen evaluation child successful')
        return
    configure();import numpy as np,torch
    from clock_real_model import Net
    from source_rate_neural import tensor_sha
    from e0_selection_metrics import from_probabilities
    from run_source_rate_screen import independent_metrics
    lock=load_lock();freeze=read(OUT/'SOURCE_FREEZE_V1.json');need(len(freeze['models'])==810,'complete new source barrier')
    here=OUT/'evaluation';here.mkdir(exist_ok=True);lock=copy.deepcopy(lock);lock['campaign']['store']=str(Path(lock['campaign']['store'])/'evaluation')
    rt=runtime_class()(lock,here);rt.phase='TARGET';sys.addaudithook(rt.guard);loaded={};rows=[]
    try:
        for name,v in sorted(freeze['models'].items()):
            c=lock['cases'][v['case_key']];task=c['task'];key=(task['protocol'],task['target'])
            if key not in loaded:
                b=next(b for b in lock['feature_bindings'] if (b['protocol'],b['domain'])==key)
                need(rt.hash(b['path'],'feature_hashes')==b['sha256'],'actual frozen target feature SHA');rt.permit(b['path']);rt.claim('feature_decodes')
                try:
                    with np.load(b['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in ('x','y','g','offset')}
                finally:rt.revoke(b['path'])
                ys=[];gs=[];offset=[]
                for r,record in sorted(lock['records'].items()):
                    if (record['protocol'],record['domain'])==key:
                        ys.extend([record['label']]*len(record['offsets']));gs.extend([r]*len(record['offsets']));offset.extend(record['offsets'])
                need(np.array_equal(a['y'],ys) and np.array_equal(a['g'],gs) and np.array_equal(a['offset'],offset),'complete exact target recording/class/offset support')
                loaded[key]=a
            a=loaded[key];aliases={}
            for policy in ('last','selected'):
                packet=rt.load(v[policy]);net=Net(task['classes'],'E0_EVENTS').cuda();net.load_state_dict(packet['model'],strict=True);net.eval();before=tensor_sha(net.state_dict())
                if before in aliases:
                    b=aliases[before];prob=rt.load(dict(b,packet=False))['prob']
                else:
                    rt.claim('target_prediction_windows',len(a['y']));parts=[]
                    with torch.no_grad():
                        for lo in range(0,len(a['y']),512):
                            rt.before_neural();z=net(torch.as_tensor(a['x'][lo:lo+512],device='cuda'))['logits'].cpu().numpy().astype(float);z-=z.max(1,keepdims=True);p=np.exp(z);parts.append(p/p.sum(1,keepdims=True))
                    prob=np.concatenate(parts);need(before==tensor_sha(net.state_dict()),'target never updates model or BN')
                    b=rt.save(Path('probabilities')/(hashlib.sha256((name+policy).encode()).hexdigest()+'.npz'),dict(prob=prob,y=a['y'],g=a['g'],model_tensor_sha256=np.array(before),source_checkpoint_sha256=np.array(v[policy]['sha256'])));aliases[before]=b
                m=from_probabilities(prob,a['y'],a['g'],task['classes']);other=independent_metrics(prob,a['y'],a['g'],task['classes'])
                for field in ('acc','macro_f1','recording_ba'):need(abs(m[field]-other[field])<1e-12,'independent target metric '+field)
                rows.append(dict(unit='MDG/baseline/'+v['arm']+'/'+v['case_key'],family='MDG',recipe='baseline/'+v['arm'],protocol=task['protocol'],fold=task['fold'],target=task['target'],seed=c['seed'],role=policy,metrics=m,probability=b,checkpoint=v[policy]))
                del net
        need(len(rows)==1620,'all810 new models, two fixed checkpoint policies')
        for row in rows:
            a=rt.load(dict(row['probability'],packet=False));k=lock['cases'][row['unit'].split('baseline/')[1].split('/',1)[1]]['task']['classes']
            m=independent_metrics(a['prob'],a['y'],a['g'],k)
            for field in ('acc','macro_f1','recording_ba'):need(abs(m[field]-row['metrics'][field])<1e-12,'saved probability independent reread '+field)
        write(OUT/'RESULT_V1.json',dict(status='ALL1620_NEW_BASELINE_TARGET_ROLES_COMPLETE',rows=rows,counts=dict(rt.counts),at=time.time()))
        rt.set('DONE',True)
    except BaseException as error:rt.set('STOPPED',dict(type=type(error).__name__,message=str(error)));raise
    finally:rt.db.close()


def summarize():
    import csv,statistics
    from tenrepeat_summarize import summaries
    lock=load_lock();result=read(OUT/'RESULT_V1.json');parent=PARENT/'FULL_TEN_RUN_RESULTS_V1.json'
    need(sha(parent)==lock['parent_results_sha256'],'old ten-run results unchanged')
    old=read(parent);need(len(old['rows'])==11100 and len(result['rows'])==1620,'old/new metric roles exact')
    rows=old['rows']+result['rows'];need(len({r['unit'] for r in rows})==6270,'all6270 distinct semantic models')
    need(len({(r['unit'],r['role'],r['fold']) for r in rows})==12720,'no duplicated policy/target role')
    stats=summaries(rows);tasks=summaries([dict(r,protocol=r['protocol']+'/'+r['fold']) for r in rows])
    need(len(stats)==450 and len(tasks)==3816,'complete extended dataset/task summary counts')
    for v in stats+tasks:
        need(abs(statistics.mean(v['per_seed_percent'])-v['mean_percent'])<1e-10,'independent ten-run mean')
        need(abs(statistics.stdev(v['per_seed_percent'])-v['std_pp'])<1e-10,'independent sample SD')
    write(OUT/'COMBINED_FULL_TEN_RUN_RESULTS_V2.json',dict(status='ORIGINAL_AND_THREE_BASELINES_COMPLETE',models=6270,metric_roles=12720,rows=rows,summary=stats,task_summary=tasks,parent_sha256=sha(parent)))
    with (OUT/'MEAN_PLUS_MINUS_STD_V2.csv').open('x',encoding='utf-8-sig',newline='') as f:
        fields=['recipe','protocol','role','metric','n','tasks','mean_percent','std_pp','formatted'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(stats)
    lines=['% All methods use ten actual fixed seeds. Percent / sample SD.','\\begin{tabular}{llllr}','Recipe & Dataset & Policy & Metric & Mean $\\pm$ SD \\\\','\\hline']
    esc=lambda s:s.replace('_','\\_')
    for r in stats:lines.append(' & '.join([esc(r['recipe']),esc(r['protocol']),r['role'],esc(r['metric']),f'{r["mean_percent"]:.2f} $\\pm$ {r["std_pp"]:.2f}'])+' \\\\')
    lines.append('\\end{tabular}');(OUT/'MEAN_PLUS_MINUS_STD_V2.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(OUT/'FINAL_AUDIT_V1.json',dict(status='PASS',models=6270,added_models=810,metric_roles=12720,seed_coverage=list(range(10)),dataset_summary_rows=450,task_summary_rows=3816,
        parent_results_unchanged=True,all_source_frozen_before_new_target=True,combined_sha256=sha(OUT/'COMBINED_FULL_TEN_RUN_RESULTS_V2.json'),at=time.time()))


def queue():
    spec=specification();write(OUT/'QUEUE_NATIVE_V1.json',dict(pid=os.getpid(),at=time.time(),status='WAITING_FOR_ORIGINAL5460_SCOPE_FINAL_AUDIT'))
    try:
        while not (PARENT/'FINAL_AUDIT_V1.json').exists():
            need(time.time()<datetime.fromisoformat(spec['queued_wait_deadline']).timestamp(),'finite14-day queue wait expired')
            failure=PARENT/'speedup_20261030/interruption_recovery/attempt2/SUPERVISOR_FAILURE_V1.json'
            need(not failure.exists(),'original campaign failed; preserve queue and do not start overlapping models')
            time.sleep(30)
        activate();qualify();train();evaluate();summarize()
        from continue_compatible_set import checked_git_stage
        paths=[p for p in OUT.rglob('*') if p.is_file() and (p.suffix in ('.json','.csv','.tex','.md') or p.name.endswith('.json.gz'))]
        checked_git_stage(paths,'results: complete810 additional ERM DANN DCORAL models and extended ten-run comparison')
        print('BASELINE_EXTENSION_COMPLETE_6270_MODELS_12720_ROLES',flush=True)
    except BaseException as error:
        write(OUT/'QUEUE_FAILURE_V1.json',dict(type=type(error).__name__,message=str(error),at=time.time()));raise


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='queue':queue()
    elif mode=='qworker':qworker(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
    elif mode=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
    elif mode=='evaluate-worker':evaluate(_child=True)
