"""Fixed five-ratio source builder; no execution until the full entry is locked."""
from collections import Counter
from pathlib import Path
import json
import os
import sqlite3
import sys
import time
import event_full_validation_plan as p
from compatible_set_source import Runtime as BaseRuntime
from protected_response_journal import process_identity


class Runtime(BaseRuntime):
    def __init__(self,lock):
        self.lock=lock;self.store=Path(lock['campaign']['store'])/'source';self.store.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(p.OUT/'SOURCE_CONTROL.sqlite',timeout=30)
        self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.executescript('CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT);CREATE TABLE counters(k TEXT PRIMARY KEY,n INTEGER);CREATE TABLE units(k TEXT PRIMARY KEY,state TEXT,v TEXT);')
        self.started=time.time();self.counts=Counter();self.allowed=set();self.cached=None;self.inputs_cache=None
        with self.db:
            self.db.execute('INSERT INTO meta VALUES (?,?)',('fingerprint',json.dumps(lock['fingerprint'])))
            self.db.execute('INSERT INTO meta VALUES (?,?)',('source_start',json.dumps(dict(at_unix=self.started))))
        p.save(p.OUT/'SOURCE_NATIVE_V1.json',dict(native=process_identity(os.getpid()),at_unix=self.started))

    def charge(self,name,count=1):
        p.need(time.time()-self.started<self.lock['campaign']['source_seconds'],'cumulative two-hour source phase')
        super().charge(name,count)


def front(rt,fn,x):
    rt.charge('source_fft_1d_transforms',len(x));rt.charge('source_fft_points',int(x.size))
    return fn['spectrum'](x)


def make_group(g,inv,ctx,rt,fn,renderer=None):
    import numpy as np
    if renderer is None:
        from continuous_event_core_v2 import render
        renderer=render
    side=g['source'];s,params=inv['s'][side],inv['params'][side]
    observed=ctx['observed'][side];scale=float(ctx['scale'][side])
    p.need(s.shape==(48,32768) and params.shape==(48,3) and observed.shape==(32768,) and scale>0,'old source inverse/scale axes')
    p.need(np.isfinite(s).all() and np.isfinite(params).all() and np.isfinite(observed).all(),'all finite actual source inputs')
    crop=lambda x:np.stack([x[a:a+g['window']] for a in g['starts']])
    waves=[]
    for q in (1.,*rt.lock['ratios']):
        c,n=s.shape
        for k,v in dict(renderer_calls=1,mode_sample_updates=3*c*n,filter_calls=3*c,event_grid_scans=2*c*n,
                        normalization_samples=c*g['kernel_length']).items():rt.charge(k,v)
        waves.append(renderer(s,params,q,g['kernel_length']))
    base=waves[0];factual=crop(observed);clean=front(rt,fn,factual)
    fixed=[];interior=[];coupled=[];energy=[];fixed_energy=[]
    for i,(q,wave) in enumerate(zip(rt.lock['ratios'],waves[1:])):
        timing=crop(observed+scale*(wave-base));edited=crop(observed+scale*(q*wave-base))
        feature=front(rt,fn,timing)
        (fixed if i<2 else interior).append(feature)
        coupled.append(front(rt,fn,edited))
        energy.append(np.mean((edited-factual)**2,axis=1));fixed_energy.append(np.mean((timing-factual)**2,axis=1))
    out=dict(clean=clean,fixed=np.stack(fixed,axis=1),interior_fixed=np.stack(interior,axis=1),
        coupled=np.stack(coupled,axis=1),time_energy=np.stack(energy,axis=1),fixed_time_energy=np.stack(fixed_energy,axis=1),
        group_key=np.array(g['key']),origin=np.array(g['origin']),starts=np.asarray(g['starts']))
    for key in ('clean','fixed','interior_fixed','coupled'):
        p.need(out[key].dtype==np.float32 and np.isfinite(out[key]).all(),'complete finite canonical source '+key)
    return out


def assemble(case,packets):
    import numpy as np
    parts={k:[] for k in ('clean','fixed','interior_fixed','coupled','time_energy','fixed_time_energy')}
    vectors={k:[] for k in ('y','g','domain','offset')}
    for item in case['bank']:
        a=packets[item['group']]
        p.need(str(a['group_key'])==item['group'] and str(a['origin'])==item['record'],'current source-bank origin and order')
        for key in parts:parts[key].append(a[key])
        for k,v in (('y',item['label']),('g',item['record']),('domain',item['domain'])):vectors[k].extend([v]*3)
        vectors['offset'].extend((a['starts']+65536).tolist())
    out={k:np.concatenate(v) for k,v in parts.items()};out.update({k:np.asarray(v) for k,v in vectors.items()})
    p.need(len(out['clean'])==case['bank_windows'],'full original source-bank coverage')
    return out


def compatible(arrays,old):
    import numpy as np
    for k in ('y','g','domain','offset','clean','fixed'):
        p.need(np.array_equal(arrays[k],old[k]),'all actual old source-bank bytes '+k)
    p.need(np.array_equal(arrays['coupled'],old['coupled'][:,:4]),'all original endpoint/Q2 actual source input bytes')
    return dict(status='EXACT_ORIGINAL_SOURCE_INPUTS',fields=['clean','fixed','coupled_first4','y','g','domain','offset'])


def run():
    import event_full_validation_execution as e
    lock=p.read(p.OUT/'SOURCE_LOCK_V1.json');e.verify(lock);grant=p.read(p.OUT/'SOURCE_ACTUAL_GRANT_V1.json')
    p.need(grant['fingerprint']==lock['fingerprint'] and grant['authorization']=='standing_authorization' and grant['human_message_id']==lock['campaign']['human_message_id'],'actual complete-entry source grant')
    rt=Runtime(lock);sys.addaudithook(rt.guard)
    import numpy as np
    from run_continuous_event_core import old_functions
    fn=old_functions();packets={};groups={};banks={};proofs={}
    try:
        for i,g in enumerate(lock['groups'],1):
            rt.begin('G/'+g['key']);inv,ctx=rt.inputs(g['inverse_identity']);a=make_group(g,inv,ctx,rt,fn)
            binding=rt.publish('G/'+g['key'],a);actual=rt.payload(binding,owned=True)
            p.need(a.keys()==actual.keys() and all(np.array_equal(a[k],actual[k]) for k in a),'every generated group disk exact')
            groups[g['key']]=binding;packets[g['key']]=actual
            if i==1 or i%25==0:print('FULL_EVENT_SOURCE',i,'/',len(lock['groups']),'elapsed',round(time.time()-rt.started,1),flush=True)
        for case in lock['cases']:
            rt.begin('B/'+case['key']);a=assemble(case,packets)
            if case['key'] in lock['compatibility_banks']:
                rt.charge('compatibility_bank_reads');old=rt.payload(lock['compatibility_banks'][case['key']])
                proofs[case['key']]=compatible(a,old)
            binding=rt.publish('B/'+case['key'],a);actual=rt.payload(binding,owned=True)
            p.need(a.keys()==actual.keys() and all(np.array_equal(a[k],actual[k]) for k in a),'all81 full source banks disk exact')
            banks[case['key']]=binding;print('FULL_EVENT_BANK',case['key'],flush=True)
        p.need(len(proofs)==15,'all45 archive models input-equivalence prerequisites')
        for k,v in lock['forecast'].items():
            if k in lock['limits']:p.need(rt.counts[k]==v,'exact complete source resource '+k)
        e.verify(lock);value=dict(status='FULL81_SOURCE_BANKS_COMPLETE',fingerprint=lock['fingerprint'],
            banks=banks,groups=groups,archive_input_proofs=proofs,counts=dict(rt.counts),
            source_start=rt.started,source_end=time.time(),new_neural_actions=0,new_inverse_fits=0,new_raw_reads=0)
        p.save(p.OUT/'SOURCE_BANKS_V1.json',value)
        with rt.db:rt.db.execute('INSERT INTO meta VALUES (?,?)',('DONE',json.dumps(dict(at_unix=time.time()))))
        print('FULL_EVENT_SOURCE_COMPLETE',len(groups),len(banks),flush=True)
    except BaseException as error:
        with rt.db:rt.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('STOPPED',json.dumps(dict(kind=type(error).__name__,reason=str(error),at_unix=time.time()))))
        raise
    finally:rt.db.close()


if __name__=='__main__':run()
