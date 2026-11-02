"""Source-only canonical rendering and exact Q2-anchor checks, no neural imports."""
from collections import Counter
from pathlib import Path
import json
import sqlite3
import sys
import time

import event_component_execution as e
from compatible_set_source import Runtime as BaseRuntime
from protected_response_journal import process_identity


class Runtime(BaseRuntime):
    def __init__(self,lock):
        self.lock=lock;self.store=Path(lock['campaign']['store'])/'source';self.store.mkdir(parents=True,exist_ok=True)
        e.need(not (e.OUT/'SOURCE_CONTROL.sqlite').exists(),'source stage must not be silently replayed')
        self.db=sqlite3.connect(e.OUT/'SOURCE_CONTROL.sqlite',timeout=30)
        self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.executescript('CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT);CREATE TABLE counters(k TEXT PRIMARY KEY,n INTEGER);CREATE TABLE units(k TEXT PRIMARY KEY,state TEXT,v TEXT);')
        self.started=time.time();self.counts=Counter();self.allowed=set();self.cached=None;self.inputs_cache=None
        with self.db:
            self.db.execute('insert into meta values(?,?)',('fingerprint',json.dumps(lock['fingerprint'])))
            self.db.execute('insert into meta values(?,?)',('source_start',json.dumps(dict(at_unix=self.started))))
        e.save(e.OUT/'SOURCE_NATIVE_V1.json',dict(native=process_identity(__import__('os').getpid()),at_unix=self.started))

    def charge(self,key,count=1):
        e.need(time.time()-self.started<self.lock['campaign']['source_seconds'],'cumulative component source2h allocation')
        super().charge(key,count)


def front(rt,fn,values):
    rt.charge('source_fft_1d_transforms',len(values));rt.charge('source_fft_points',int(values.size))
    return fn['spectrum'](values)


def make_group(g,inv,ctx,anchor,rt,fn,renderer=None):
    import numpy as np
    from event_component_core import impulse_only,global_affine_moments
    if renderer is None:
        from continuous_event_core_v2 import render
        renderer=render
    side=g['source'];s,params=inv['s'][side],inv['params'][side]
    observed=ctx['observed'][side];scale=float(ctx['scale'][side])
    e.need(s.shape==(48,32768) and params.shape==(48,3) and observed.shape==(32768,) and scale>0,'original source axes/scale')
    e.need(np.isfinite(s).all() and np.isfinite(params).all() and np.isfinite(observed).all(),'finite source inverse/context')
    e.need(str(anchor['group_key'])==g['key'] and str(anchor['origin'])==g['origin'] and np.array_equal(anchor['starts'],g['starts']),'same canonical anchor identity and query offsets')
    crop=lambda x:np.stack([x[a:a+g['window']] for a in g['starts']])
    waves=[]
    for q in (1.,*rt.lock['ratios']):
        for k,v in dict(renderer_calls=1,mode_sample_updates=3*48*32768,filter_calls=3*48,
            event_grid_scans=2*48*32768,normalization_samples=48*g['kernel_length']).items():rt.charge(k,v)
        waves.append(renderer(s,params,q,g['kernel_length']))
    factual=crop(observed);clean=front(rt,fn,factual);original_response=scale*crop(waves[0])
    e.need(np.array_equal(clean,anchor['clean']),'canonical clean source frontend bytes')
    amplitude=[];moments=[];details=[]
    for j,(q,wave) in enumerate(zip(rt.lock['ratios'],waves[1:])):
        coupled=crop(observed+scale*(q*wave-waves[0]))
        reference=front(rt,fn,coupled)
        e.need(np.array_equal(reference,anchor['coupled'][:,j+2]),'every canonical Q2 edited query byte before archive reuse')
        amplitude.append(front(rt,fn,impulse_only(factual,original_response,q)))
        matched,detail=global_affine_moments(factual,coupled);moments.append(front(rt,fn,matched));details.append(detail)
    out={k:anchor[k].copy() for k in ('clean','fixed','interior_fixed','coupled')}
    out.update(amplitude_only=np.stack(amplitude,axis=1),global_moments=np.stack(moments,axis=1),
        group_key=np.array(g['key']),origin=np.array(g['origin']),starts=np.asarray(g['starts']))
    for k in ('mean','centered_rms','gain','mean_error','centered_rms_error','identity'):
        out['moment_'+k]=np.stack([v[k] for v in details],axis=1)
    for key,n in (('clean',0),('fixed',2),('interior_fixed',2),('coupled',4),('amplitude_only',2),('global_moments',2)):
        shape=(3,2048) if not n else (3,n,2048)
        e.need(out[key].dtype==np.float32 and out[key].shape==shape and np.isfinite(out[key]).all(),'finite complete source field '+key)
    return out


def assemble(case,packets):
    import numpy as np
    fields=('clean','fixed','interior_fixed','coupled','amplitude_only','global_moments',
        'moment_mean','moment_centered_rms','moment_gain','moment_mean_error','moment_centered_rms_error','moment_identity')
    parts={k:[] for k in fields};vectors={k:[] for k in ('y','g','domain','offset')}
    for item in case['bank']:
        a=packets[item['group']]
        e.need(str(a['group_key'])==item['group'] and str(a['origin'])==item['record'],'same complete bank origin/order')
        for key in fields:parts[key].append(a[key])
        for k,v in (('y',item['label']),('g',item['record']),('domain',item['domain'])):vectors[k].extend([v]*3)
        vectors['offset'].extend((a['starts']+65536).tolist())
    out={k:np.concatenate(v) for k,v in parts.items()};out.update({k:np.asarray(v) for k,v in vectors.items()})
    e.need(len(out['clean'])==case['bank_windows'],'all original bank queries kept')
    return out


def run():
    lock=e.read(e.OUT/'SOURCE_LOCK_V1.json');e.verify(lock);grant=e.read(e.OUT/'SOURCE_ACTUAL_GRANT_V1.json')
    e.need(grant['authorization']=='standing_authorization' and grant['human_message_id']==e.meta.closed.AUTH and grant['fingerprint']==lock['fingerprint'],'actual source-only grant')
    rt=Runtime(lock);sys.addaudithook(rt.guard)
    import numpy as np
    from run_continuous_event_core import old_functions
    fn=old_functions();groups={};banks={};packets={};maximum_mean_error=maximum_rms_error=0.
    try:
        for i,g in enumerate(lock['groups'],1):
            rt.begin('G/'+g['key']);inv,ctx=rt.inputs(g['inverse_identity'])
            rt.charge('canonical_anchor_group_reads');anchor=rt.payload(lock['canonical_group_anchors'][g['key']])
            a=make_group(g,inv,ctx,anchor,rt,fn)
            binding=rt.publish('G/'+g['key'],a);actual=rt.payload(binding,owned=True)
            e.need(a.keys()==actual.keys() and all(np.array_equal(a[k],actual[k]) for k in a),'exact saved group arrays')
            maximum_mean_error=max(maximum_mean_error,float(a['moment_mean_error'].max()))
            maximum_rms_error=max(maximum_rms_error,float(a['moment_centered_rms_error'].max()))
            groups[g['key']]=binding;packets[g['key']]=actual
            if i==1 or i%25==0:print('COMPONENT_SOURCE',i,'/3372','elapsed',round(time.time()-rt.started,1),flush=True)
        for key,case in lock['cases'].items():
            rt.begin('B/'+key);a=assemble(case,packets);binding=rt.publish('B/'+key,a);actual=rt.payload(binding,owned=True)
            e.need(a.keys()==actual.keys() and all(np.array_equal(a[k],actual[k]) for k in a),'exact saved complete bank')
            banks[key]=binding;print('COMPONENT_BANK',key,flush=True)
        for k in lock['limits']:e.need(rt.counts[k]==lock['forecast'][k],'exact complete source resource '+k)
        e.verify(lock);value=dict(status='ALL81_COMPONENT_BANKS_WITH_EXACT_Q2_ANCHORS',fingerprint=lock['fingerprint'],
            banks=banks,groups=groups,counts=dict(rt.counts),source_start=rt.started,source_end=time.time(),
            canonical_Q2_exact_groups=len(groups),maximum_mean_error=maximum_mean_error,maximum_rms_error=maximum_rms_error,
            new_neural_actions=0,new_inverse_fits=0,new_raw_reads=0)
        e.save(e.OUT/'SOURCE_BANKS_V1.json',value)
        with rt.db:rt.db.execute('insert into meta values(?,?)',('DONE',json.dumps(dict(at_unix=time.time()))))
        print('COMPONENT_SOURCE_COMPLETE',len(groups),len(banks),flush=True)
    except BaseException as error:
        with rt.db:rt.db.execute('insert or replace into meta values(?,?)',('STOPPED',json.dumps(dict(kind=type(error).__name__,reason=str(error),at_unix=time.time()))))
        raise
    finally:rt.db.close()


if __name__=='__main__':run()
