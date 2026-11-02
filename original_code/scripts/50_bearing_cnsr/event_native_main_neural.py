"""Native PHM/JNU neural wiring. Import ONLY after the new finite approval.

The historical author's mathematical engine is deliberately unchanged.
"""
import copy
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from event_author_native_engine import NativeAuthorEngine,NativeNetwork
from clock_factual_data import CachedStore
from event_native_main_sources import bind_sources
import event_native_main_core as r


def setup(rt):
    rt.clock()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def exact(a,b,location='packet'):
    if type(a) is not type(b):raise AssertionError('type '+location)
    if isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise AssertionError('tensor '+location)
    elif isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):raise AssertionError('array '+location)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('keys '+location)
        for k in a:exact(a[k],b[k],location+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError('length '+location)
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,location+'/'+str(i))
    elif a!=b:raise AssertionError('value '+location)


def tensor_sha(state,include_names=True):
    h=hashlib.sha256()
    for k,v in state.items():
        if include_names:h.update(k.encode())
        a=v.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def parameter_sha(model):return tensor_sha(dict(enumerate(model.parameters())),False)


def checkpoint(folder,packet):
    path=r.fresh_path(folder,'.pt')
    with path.open('xb') as f:torch.save(packet,f);f.flush();os.fsync(f.fileno())
    return dict(path=str(path),sha256=r.sha(path),step=packet['step'])


def load_state(binding):
    path=r.store_path(binding['path'])
    if r.sha(path)!=binding['sha256']:raise PermissionError('own complete-state SHA changed')
    packet=torch.load(path,map_location='cpu',weights_only=False)
    if packet['step']!=binding['step']:raise PermissionError('checkpoint step binding')
    return packet


def save_probability(folder,prob,y,g):
    path=r.fresh_path(folder,'.npz')
    with path.open('xb') as f:
        np.savez_compressed(f,prob=prob,y=y,g=g);f.flush();os.fsync(f.fileno())
    return dict(path=str(path),sha256=r.sha(path),windows=len(y))


class SourcesCache(CachedStore):
    def __init__(self,rt):
        super().__init__(rt);self.accesses=rt.get('CACHE_ACCESSES') or []
    def domain(self,protocol,domain,task,role):
        rt=self.rt
        if rt.active_case is None or task!=rt.active_case['task']:raise PermissionError('exact active case cache role')
        rt.require_case(rt.active_case)
        if role=='target':rt.require_target(rt.active_case)
        elif role!='source' or rt.phase not in ('SOURCE_ADMISSION','QUALIFICATION','SOURCE'):raise PermissionError('source-only cache phase')
        return super().domain(protocol,domain,task,role)
    def flush_accesses(self):self.rt.put('CACHE_ACCESSES',self.accesses)


def actual_sources(rt,cache,case):
    rt.set_case(case)
    arrays=cache.source_case(case);cache.flush_accesses()
    return bind_sources(rt.scope['records'],case,arrays,case['source_fingerprint'],tensor_factory=torch.from_numpy)


def admission(rt,cache):
    old=rt.get('SOURCE_ADMISSION')
    if old is not None:return old
    rt.phase='SOURCE_ADMISSION';rows=[]
    for case in rt.scope['cases']:
        sources=actual_sources(rt,cache,case)
        rows.append(dict(case=r.case_key(case),source_fingerprint=sources.fingerprint,
            train=sum(len(sources.get(d,'train').y) for d in sources.allowed),
            validation=sum(len(sources.get(d,'val').y) for d in sources.allowed)))
    value=dict(status='ALL69_CACHED_SOURCE_SUPPORT_PASS_NO_NETWORK',fingerprint=rt.lock['fingerprint'],
        cases=rows,counters=rt.count(),at_unix=time.time())
    r.write_new(r.OUT/'SOURCE_ADMISSION.json',value);rt.put('SOURCE_ADMISSION',value)
    return value


def make_engine(rt,case,config,sources,trajectory=None):
    rt.set_case(case);rt.gpu();rt.claim('neural_model_builds')
    permit=rt.permit_for(case,config,trajectory)
    return NativeAuthorEngine(sources,case['task'],case['seed'],config,permit.fingerprint,permit,rt,
        maximum_steps=10 if trajectory else 3001,validation_steps=(0,10) if trajectory else r.SOURCE_STEPS)


def prediction(rt,model,x,role):
    if role not in ('source','target'):raise PermissionError('explicit prediction role')
    if role=='source' and rt.phase not in ('QUALIFICATION','SOURCE'):raise PermissionError('source inference phase')
    if role=='target':rt.require_target(rt.active_case)
    rt.claim(role+'_prediction_windows',len(x));model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(x),512):
            rt.clock(True)
            part=x[start:start+512]
            if isinstance(part,np.ndarray):part=torch.from_numpy(np.ascontiguousarray(part))
            values.append(model(part.cuda())['logits'].detach().cpu().numpy())
    logits=np.concatenate(values).astype(np.float64);logits-=logits.max(axis=1,keepdims=True)
    prob=np.exp(logits);prob/=prob.sum(axis=1,keepdims=True)
    if not np.isfinite(prob).all():raise FloatingPointError('actual prediction probabilities')
    return prob


def operation_start(rt,key):
    if rt.get(key) is not None:raise PermissionError('completed operation must be reused, never paid again')
    if rt.get('PENDING/'+key) is not None:raise PermissionError('incomplete paid inference needs audit, no replay')
    rt.put('PENDING/'+key,dict(at_unix=time.time(),phase=rt.phase))


def source_probe(rt,engine,folder,trajectory=None):
    key='PROBE/'+r.chain_key(rt.active_case,engine.config,trajectory)+'/'+str(engine.step_n)
    operation_start(rt,key)
    xs=[];ys=[];gs=[];domains=[]
    for d in engine.sources.allowed:
        part=engine.sources.get(d,'val');n=64 if trajectory else len(part.y)
        if len(part.y)<n:raise ValueError('fixed probe support')
        xs.append(part.x[:n]);ys.append(part.y[:n]);gs.append(part.groups[:n]);domains.extend([d]*n)
    y=np.concatenate(ys);g=np.concatenate(gs);domains=np.asarray(domains)
    prob=prediction(rt,engine.model,torch.cat(xs),'source');pred=prob.argmax(1)
    by_domain={d:float(np.mean(pred[domains==d]==y[domains==d])) for d in engine.sources.allowed}
    binding=save_probability(folder,prob,y,g)
    curve=dict(step=engine.step_n,score=float(np.mean(list(by_domain.values()))),target_used_for_selection=False)
    # File names/timestamps differ across independent P0 trajectories, not mathematics.
    if not trajectory:curve.update(probability=binding,domain_accuracy=by_domain)
    engine.consider(curve)
    receipt=dict(curve=curve,probability=binding,domain_accuracy=by_domain,at_unix=time.time())
    rt.put(key,receipt)
    return prob,receipt


def qualification(rt,cache):
    if rt.get('P0_PASS') is not None:return rt.get('P0_PASS')
    if rt.count().get('qualification_updates',0):raise PermissionError('partial paid qualifier cannot start from scratch')
    rt.phase='QUALIFICATION';rows=[];init_by_case={};batch_by_case={}
    for case in rt.scope['cases']:
        if (case['task']['protocol'],case['task']['fold'],case['seed']) not in r.m.QUAL_CASES:continue
        sources=actual_sources(rt,cache,case);initial=[];streams=[]
        for config in r.CONFIGS:
            packets=[];probs=[];traces=[];states=[];receipts=[];gradients=[]
            for trajectory in ('continuous','resumed'):
                folder=r.STORE/'qualification'/r.case_key(case)/config/trajectory
                engine=make_engine(rt,case,config,sources,trajectory);initial.append(parameter_sha(engine.model))
                source_probe(rt,engine,folder,trajectory);trace=[]
                for stop in ((10,) if trajectory=='continuous' else (5,10)):
                    while engine.step_n<stop:
                        _,indices=engine.step(inspect=True);trace.append(indices)
                        if engine.step_n==1:
                            grads={k:p.grad for k,p in engine.model.named_parameters()}
                            gradients.append(dict(trajectory=trajectory,step=1,finite_nonzero=True,
                                squared_norm=sum(float(g.square().sum()) for g in grads.values()),sha256=tensor_sha(grads)))
                    if stop==5:
                        packet=engine.packet();binding=checkpoint(folder,packet);states.append(binding)
                        del engine;torch.cuda.empty_cache()
                        engine=make_engine(rt,case,config,sources,trajectory);engine.restore(load_state(binding));exact(packet,engine.packet(),'actual disk5 restoration')
                p,receipt=source_probe(rt,engine,folder,trajectory);receipts.append(receipt)
                packets.append(engine.packet());probs.append(p);traces.append(trace);states.append(checkpoint(folder,packets[-1]))
                del engine;torch.cuda.empty_cache()
            exact(packets[0],packets[1],'continuous10 versus disk5+5');exact(probs[0],probs[1]);exact(traces[0],traces[1])
            streams.append(traces[0]);rows.append(dict(case=r.case_key(case),config=config,complete_state_exact=True,
                source_probability_exact=True,states=states,final_probes=receipts,gradients=gradients))
            print('NATIVE_MAIN_P0_EXACT',r.case_key(case),config,flush=True)
        if len(set(initial))!=1 or any(v!=streams[0] for v in streams[1:]):raise AssertionError('same-case native initialization/factual batches')
        init_by_case[r.case_key(case)]=initial[0];batch_by_case[r.case_key(case)]=r.canonical(streams[0])
    counts=rt.count()
    if len(rows)!=6 or counts.get('qualification_updates')!=120 or counts.get('source_prediction_windows')!=3072:raise AssertionError('six fixed native qualifiers and exact bill')
    value=dict(status='ACTUAL_CUDA_DISK_RESUME_PASS',fingerprint=rt.lock['fingerprint'],rows=rows,
        initialization_by_case=init_by_case,factual_trace_by_case=batch_by_case,counters=counts,at_unix=time.time())
    r.write_new(r.OUT/'P0_PASS.json',value);rt.put('P0_PASS',value);return value


def checked_record(rt,case,config):
    name=r.chain_key(case,config);row=rt.get(name);cursor=rt.cursor(name)
    if row is None:
        if cursor!=(0,0,0):raise PermissionError('paid model lacks durable resume state')
    elif cursor!=(row['resume_step'],)*3 or r.sha(row['resume']['path'])!=row['resume']['sha256']:raise PermissionError('durable paid cursor/state mismatch')
    return row


def train_model(rt,cache,case,config):
    rt.phase='SOURCE';rt.set_case(case);name=r.chain_key(case,config);record=checked_record(rt,case,config)
    if record and record['resume_step']==3001:return record
    sources=actual_sources(rt,cache,case);engine=make_engine(rt,case,config,sources);folder=r.STORE/'models'/r.case_key(case)/config
    if record is None:
        initial=parameter_sha(engine.model);source_probe(rt,engine,folder)
        binding=checkpoint(folder,engine.packet());record=dict(name=name,case=r.case_key(case),task=case['task'],seed=case['seed'],config=config,
            engine_fingerprint=engine.fingerprint,source_fingerprint=sources.fingerprint,initial_parameter_sha256=initial,
            states={'0':binding},resume=binding,resume_step=0,curve=copy.deepcopy(engine.curve),selected=engine.best_step,
            restores={},logs=[],completed_unix=None)
        rt.put(name,record)
    else:engine.restore(load_state(record['resume']))
    for boundary in (1000,2000):
        if engine.step_n==boundary and str(boundary) not in record['restores']:
            # The constructor above really restored this same complete disk packet.
            exact(load_state(record['resume']),engine.packet(),'resumed scientific boundary')
            record['restores'][str(boundary)]=dict(binding=record['resume'],actual_disk_restore=True,at_unix=time.time());rt.put(name,record)
    while engine.step_n<3001:
        engine.step(inspect=engine.step_n==0)
        if engine.step_n in r.SOURCE_STEPS:source_probe(rt,engine,folder)
        if engine.step_n%250==0 or engine.step_n==3001:
            packet=engine.packet();binding=checkpoint(folder,packet);states=dict(record['states'])
            if engine.step_n in r.SOURCE_STEPS:states[str(engine.step_n)]=binding
            record=dict(record,resume=binding,resume_step=engine.step_n,states=states,curve=copy.deepcopy(engine.curve),
                selected=engine.best_step,logs=copy.deepcopy(engine.logs),completed_unix=time.time() if engine.step_n==3001 else None)
            rt.put(name,record)
            if engine.step_n in (1000,2000):
                boundary=engine.step_n;del engine;torch.cuda.empty_cache()
                engine=make_engine(rt,case,config,sources);engine.restore(load_state(binding));exact(packet,engine.packet(),'actual formal disk restoration')
                record['restores'][str(boundary)]=dict(binding=binding,actual_disk_restore=True,at_unix=time.time());rt.put(name,record)
    del engine;torch.cuda.empty_cache()
    return record


def target_role(rt,cache,case,record,policy):
    if policy not in ('last','selected'):raise ValueError('two fixed policies')
    rt.set_case(case);rt.require_target(case,record['name']);key='TARGET/'+record['name']+'/'+policy
    prior=rt.get(key)
    if prior is not None:return prior
    operation_start(rt,key)
    binding=record['states'][str(3001 if policy=='last' else record['selected'])]
    packet=load_state(binding)
    identity=dict(fingerprint=record['engine_fingerprint'],source_fingerprint=case['source_fingerprint'],task=case['task'],seed=case['seed'],config=record['config'])
    if any(packet[k]!=v for k,v in identity.items()):raise PermissionError('target frozen own state identity')
    rt.gpu();rt.claim('neural_model_builds')
    model=NativeNetwork(case['task']['classes'],record['config']=='A2_AUTHOR_IN_MIXUP')
    model.load_state_dict(packet['model']);model.cuda().requires_grad_(False).eval();before=tensor_sha(model.state_dict())
    arrays=cache.target(case['task']);cache.flush_accesses();prob=prediction(rt,model,arrays['x'],'target')
    after=tensor_sha(model.state_dict())
    if before!=after:raise AssertionError('target cannot change weights/buffers')
    probability=save_probability(r.STORE/'targets'/r.case_key(case)/record['config'],prob,arrays['y'],arrays['g'])
    row=dict(protocol=case['task']['protocol'],fold=case['task']['fold'],seed=case['seed'],arm=record['config'],role=policy,
        checkpoint=binding,probability=probability,weights_sha256_before=before,weights_sha256_after=after,fresh=True,at_unix=time.time())
    rt.put(key,row);del model,packet;torch.cuda.empty_cache();return row
