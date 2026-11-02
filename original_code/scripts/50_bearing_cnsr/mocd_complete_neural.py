"""Lazy complete-MOCD worker; NOT an approval or a standalone execution entry.

No Torch import at module import. Every actual neural construction first passes
the new Runtime and its actual task/seed/profile/source/schedule permit.
"""
import copy
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import mocd_complete_control as c
import mocd_complete_cache as cache_io
import mocd_complete_factory as factory


def torch_module(rt):
    if not rt.authority_verified:raise PermissionError('real new finite Runtime before neural dependency access')
    rt.clock()
    import torch
    return torch


def setup(rt):
    torch=torch_module(rt);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def exact(a,b,location='packet'):
    """Full value/type equality, comparing saved tensors on CPU (not a forward)."""
    if type(a) is not type(b):raise AssertionError('type '+location)
    if isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):raise AssertionError('array '+location)
    elif type(a).__module__.startswith('torch') and hasattr(a,'detach'):
        if a.dtype!=b.dtype or a.shape!=b.shape or not a.detach().cpu().equal(b.detach().cpu()):raise AssertionError('tensor '+location)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('keys '+location)
        for key in a:exact(a[key],b[key],location+'/'+str(key))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError('length '+location)
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,location+'/'+str(i))
    elif a!=b:raise AssertionError('value '+location)


def tensor_sha(state):
    h=hashlib.sha256()
    for key,value in state.items():
        a=value.detach().cpu().contiguous().numpy()
        h.update(str(key).encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def own_path(binding):
    path=Path(binding['path']).resolve()
    if not path.is_relative_to(c.STORE.resolve()) or path.suffix not in ('.pt','.npz'):raise PermissionError('this complete campaign D-root state/array only')
    return path


def check_bytes(rt,binding):
    rt.claim('cache_hash_checks');path=own_path(binding)
    if c.sha(path)!=binding['sha256']:raise PermissionError('complete campaign state/array bytes changed')
    return path


def checkpoint(rt,folder,packet):
    torch=torch_module(rt);rt.clock(True)
    path=c.fresh_path(folder,'.pt')
    with path.open('xb') as stream:torch.save(packet,stream);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),sha256=c.sha(path),step=packet['progress']['cursor']['completed'])


def load_state(rt,binding):
    torch=torch_module(rt);path=check_bytes(rt,binding)
    packet=torch.load(path,map_location='cpu',weights_only=False)
    if packet['progress']['cursor']['completed']!=binding['step']:raise PermissionError('complete saved step identity')
    return packet


def save_probability(rt,folder,prob,y,g):
    if not rt.authority_verified:raise PermissionError('no synthetic/unauthorized probability artifacts')
    rt.clock();p=np.asarray(prob);y=np.asarray(y);g=np.asarray(g)
    if p.ndim!=2 or p.shape[0]!=len(y) or g.shape!=y.shape or y.dtype.kind not in 'iu' or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=2e-6,rtol=0):raise ValueError('finite probability and full support schema')
    path=c.fresh_path(folder,'.npz')
    with path.open('xb') as stream:np.savez_compressed(stream,prob=p,y=y,g=g);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),sha256=c.sha(path),windows=len(y),support_sha256=c.canonical(dict(y=y.tolist(),g=g.tolist())))


def source_observer(rt,case,folder):
    def observe(step,prob,y,g,row):
        rt.require_case(case)
        if row.get('target_used_for_selection') is not False or float(np.mean(prob.argmax(1)==y))!=row['source_score']:raise PermissionError('actual complete source probabilities determine source score')
        return dict(step=step,source_score=row['source_score'],target_used_for_selection=False,
            probability=save_probability(rt,folder/'source_probabilities',prob,y,g),source_fingerprint=case['new_source_fingerprint'],at_unix=time.time())
    return observe


def make_engine(rt,case,profile,sources,folder,trajectory=None):
    rt.set_case(case);permit=rt.permit_for(case,profile,trajectory)
    # FullSchedule imports Torch indirectly, so it is not imported before the
    # actual new authority check; factory then checks the actual loop fields.
    torch_module(rt)
    from mocd_full_matching import FullSchedule
    return factory.create(sources,profile,rt.scope['profiles'],permit.fingerprint,seed=case['seed'],permit=permit,rt=rt,
        on_source_probe=source_observer(rt,case,folder),schedule=FullSchedule(**c.schedule_spec(trajectory is not None)))


def checked_record(rt,case,profile,trajectory=None):
    name=c.chain_key(case,profile,trajectory);record=rt.get(name);cursor=rt.cursor(name)
    if record is None:
        if cursor!=(0,0,0):raise PermissionError('paid source chain without a complete durable state')
    elif cursor!=(record['resume_step'],)*3 or c.sha(own_path(record['resume']))!=record['resume']['sha256']:raise PermissionError('paid source tail must equal the exact saved resume state; never rewind')
    return record


def trajectory(rt,cache,case,profile,kind=None):
    """One exact paid chain: formal800 or qualification10/5+5.

    run() is advanced ONE step at a time, because complete-controller accounting
    occurs only after optimizer, full refresh and any source observer return.
    """
    rt.phase='SOURCE' if kind is None else 'QUALIFICATION';rt.set_case(case)
    maximum=800 if kind is None else 10;boundary=421 if kind is None else (5 if kind=='resumed' else None)
    science=(0,400,421,800) if kind is None else (0,5,10)
    name=c.chain_key(case,profile,kind);record=checked_record(rt,case,profile,kind)
    if record is not None and record.get('status')=='COMPLETE':return record
    torch=torch_module(rt);sources=cache_io.actual_sources(rt,cache,case,torch.from_numpy)
    folder=c.STORE/name
    engine=make_engine(rt,case,profile,sources,folder,kind)
    if record is None:
        binding=checkpoint(rt,folder,engine.packet())
        record=dict(status='RUNNING',case_key=c.m.case_key(case),profile=profile,seed=case['seed'],source_fingerprint=sources.fingerprint,
            engine_fingerprint=engine.fingerprint,checkpoints={'0':binding},resume=binding,resume_step=0,
            source_curve=[],source_checkpoints={},selected_step=-1,selected_score=-1.,actual_updates=0,amp_skips=0,restored421=None,restored5=None)
        rt.put(name,record)
    else:
        saved=load_state(rt,record['resume']);engine.restore(saved);exact(saved,engine.packet(),'same-clock disk continuation')
        if engine.controller.completed==boundary and record.get('restored'+str(boundary)) is None:
            record['restored'+str(boundary)]=dict(status='EXACT_DISK_RESTORE',checkpoint_sha256=record['resume']['sha256'],at_unix=time.time());rt.put(name,record)
        del saved
    while engine.controller.completed<maximum:
        step=engine.controller.completed+1;engine.run(step);rt.controller_finished(name,step)
        if step%20==0 or step in science:
            packet=engine.packet();binding=checkpoint(rt,folder,packet);states=dict(record['checkpoints']);source_states=dict(record['source_checkpoints'])
            if step in science:states[str(step)]=binding
            if engine.controller.schedule.event(step)['source_val_diagnostic']:source_states[str(step)]=binding
            record=dict(record,resume=binding,resume_step=step,checkpoints=states,source_checkpoints=source_states,
                source_curve=copy.deepcopy(engine.source_scores),selected_step=packet['selection']['best_step'],selected_score=packet['selection']['best_score'],
                actual_updates=engine.controller.actual_updates,amp_skips=step-engine.controller.actual_updates,
                status='COMPLETE' if step==maximum else 'RUNNING',completed_unix=time.time() if step==maximum else None)
            rt.put(name,record)
            if step==boundary and record.get('restored'+str(boundary)) is None:
                del engine;torch.cuda.empty_cache()
                engine=make_engine(rt,case,profile,sources,folder,kind);engine.restore(load_state(rt,binding))
                exact(packet,engine.packet(),'planned actual disk restoration')
                record['restored'+str(boundary)]=dict(status='EXACT_DISK_RESTORE',checkpoint_sha256=binding['sha256'],at_unix=time.time());rt.put(name,record)
            del packet
    if kind is None:rt.validate_new_record(c.m.case_key(case)+'/'+profile,record)
    del engine,sources;torch.cuda.empty_cache();return record


def qualifier_packet(packet):
    cursor=packet['progress']['cursor'];logs=packet['sampler']['logs']
    if cursor['completed']!=10 or cursor['failed_step'] is not None or packet['progress']['stage']!='B' or len(logs)!=10 or packet['sampler']['draws']!=192 or packet['matching']['refreshes']!=4 or packet['selection']['best_step']!=10:raise AssertionError('full-shape A4/B6 qualifier and one source score')
    for stage,names in (('A',('gf','gd','pf','pd')),('B',('gf','gd','hf','hd','dec'))):
        rows=[v for v in logs if v['stage']==stage]
        if not any(v['actual_update'] for v in rows):raise AssertionError('no actual optimizer update in qualification stage '+stage)
        for name in names:
            if not any(v['gradient_norms'][name] is not None and v['gradient_norms'][name]>0 for v in rows):raise AssertionError('missing actual finite nonzero qualified branch gradient '+stage+'/'+name)


def qualification(rt,cache):
    old=rt.get('QUALIFICATION_PASS')
    if old is not None:
        if old['lock_fingerprint']!=rt.lock['fingerprint']:raise PermissionError('same actual qualifier only')
        return old
    rt.phase='QUALIFICATION';rows=[];before=rt.count()
    for q in rt.scope['qualification_forecast']['cases']:
        case=rt.by_case[q['case_key']]
        for profile in c.m.PROFILES:
            continuous=trajectory(rt,cache,case,profile,'continuous');resumed=trajectory(rt,cache,case,profile,'resumed')
            for step in (0,5,10):
                a=load_state(rt,continuous['checkpoints'][str(step)]);b=load_state(rt,resumed['checkpoints'][str(step)])
                exact(a,b,'actual CUDA continuous vs disk5+5 step '+str(step))
                if step==10:qualifier_packet(a);qualifier_packet(b)
            if resumed['restored5'] is None:raise AssertionError('actual qualification disk restoration required')
            rows.append(dict(case_key=q['case_key'],profile=profile,continuous=continuous,resumed=resumed,status='EXACT_PACKET_ALL_THREE_STATES'))
    names=[v['case_key']+'/'+v['profile'] for v in rows]
    if len(names)!=18 or len(set(names))!=18:raise AssertionError('three full protocols x all3 seeds x both interpretations')
    paid=rt.db.execute("SELECT count(*) FROM attempts WHERE phase='QUALIFICATION'").fetchone()[0]
    if paid!=360:raise AssertionError('fixed full qualifier attempts, no hidden repetitions')
    value=dict(status='FULL_SCOPE_CUDA_CONTINUOUS_RESTORE_PASS',lock_fingerprint=rt.lock['fingerprint'],combinations=names,rows=rows,
        normal_qualification_attempts=360,counters_before=before,counters_after=rt.count(),at_unix=time.time())
    c.write_new(c.OUT/'QUALIFICATION_PASS.json',value);rt.put('QUALIFICATION_PASS',value);return value


def operation_start(rt,key):
    if rt.get(key) is not None or rt.get('PENDING/'+key) is not None:raise PermissionError('saved result reused; incomplete paid prediction requires explicit repair, never silent replay')
    rt.put('PENDING/'+key,dict(at_unix=time.time(),phase=rt.phase))


def target_role(rt,cache,case,record,policy):
    if policy not in ('last','selected'):raise PermissionError('fixed last primary / source-selected secondary only')
    rt.set_case(case);name=c.m.case_key(case)+'/'+record['profile'];rt.require_target(case,name)
    key='TARGET/'+name+'/'+policy;old=rt.get(key)
    if old is not None:
        check_bytes(rt,old['probability']);return old
    operation_start(rt,key);packet=load_state(rt,record['checkpoints']['800'])
    if packet['fingerprint']!=record['engine_fingerprint'] or packet['progress']['source_fingerprint']!=record['source_fingerprint'] or packet['profile']['complete_source_streams']!=c.m.profile_streams(case['seed']):raise PermissionError('actual frozen target model identity and seed streams')
    torch=torch_module(rt);rt.gpu();rt.claim('neural_model_builds')
    from qualify_ress_literature_components import PaperMOCD
    model=PaperMOCD(case['task']['classes'],len(case['task']['sources']))
    state=packet['model'] if policy=='last' else packet['selection']['best_state']
    if state is None:raise PermissionError('no source-selected state exists')
    model.load_state_dict(state,strict=True);model.cuda().requires_grad_(False).eval();before=tensor_sha(model.state_dict())
    arrays=cache.target(case['task']);cache.flush_accesses();parts=[]
    hook=model.gf.register_forward_pre_hook(lambda mod,args:rt.claim('target_prediction_windows',len(args[0])))
    with torch.no_grad():
        for start in range(0,len(arrays['y']),64):
            rt.clock(True)
            x=torch.from_numpy(np.ascontiguousarray(arrays['x'][start:start+64])).cuda()
            parts.append(torch.softmax(model.hf(model.gf(x)),1).cpu().numpy())
    hook.remove();after=tensor_sha(model.state_dict())
    if before!=after:raise AssertionError('target changes no parameters or BN buffers')
    probability=save_probability(rt,c.STORE/'targets'/c.m.case_key(case)/record['profile'],np.concatenate(parts),arrays['y'],arrays['g'])
    row=dict(protocol=case['task']['protocol'],fold=case['task']['fold'],seed=case['seed'],arm=record['profile'],role=policy,
        checkpoint=record['checkpoints']['800'],model_key='model' if policy=='last' else 'selection.best_state',selected_step=800 if policy=='last' else record['selected_step'],
        engine_fingerprint=record['engine_fingerprint'],source_fingerprint=record['source_fingerprint'],probability=probability,
        weights_sha256_before=before,weights_sha256_after=after,fresh=True,at_unix=time.time())
    rt.put(key,row);rt.clock(True);del model,packet;torch.cuda.empty_cache();return row
