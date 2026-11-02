"""Bounded same-kernel numerical qualification, paired timing and saved audit."""
from collections import Counter
from datetime import datetime
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import continuous_event_plan as p
from protected_response_journal import process_identity


class Runtime:
    def __init__(self,lock):
        self.lock=lock;self.store=Path(lock['campaign']['store']).resolve();p.require(self.store.drive.upper()=='D:','D-only output')
        self.store.mkdir(parents=True,exist_ok=True);self.db=sqlite3.connect(p.OUT/'CONTROL.sqlite')
        self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.executescript('CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT);CREATE TABLE counters(k TEXT PRIMARY KEY,n INTEGER);CREATE TABLE units(k TEXT PRIMARY KEY,state TEXT,v TEXT);')
        self.counts=Counter();self.allowed=set();self.started=time.time()
        with self.db:
            self.set('cpu_start',dict(at_unix=self.started));self.set('fingerprint',lock['fingerprint'])
            self.set('native',process_identity(os.getpid()))
        p.save(p.OUT/'PRODUCER_NATIVE_V1.json',dict(native=process_identity(os.getpid()),at_unix=self.started,fingerprint=lock['fingerprint']))

    def set(self,key,value):self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value,allow_nan=False)))
    def clock(self):
        p.clock(self.lock['campaign']);p.require(time.time()-self.started<self.lock['campaign']['cpu_seconds'],'original CPU60min including audit')
    def charge(self,key,n=1):
        self.clock();p.require(type(n) is int and n>=0 and key in self.lock['limits'],'registered finite numerical resource '+key)
        value=self.counts[key]+n;p.require(value<=self.lock['limits'][key],'numerical resource exhausted '+key)
        with self.db:self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)',(key,value))
        self.counts[key]=value

    def guard(self,event,args):
        if event=='import' and args[0].split('.')[0] in ('torch','jax','tensorflow'):raise PermissionError('zero neural libraries')
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() in ('.mat','.csv','.wav','.h5','.hdf5','.pt','.pth','.npz','.npy','.pkl','.pickle'):
            p.require(str(path).lower() in self.allowed,'only explicitly bound numerical payload '+str(path))

    def payload(self,b,kind):
        import numpy as np
        self.charge('source_payload_reads');self.charge('source_payload_bytes',b['size_bytes']);self.charge('source_hashes')
        path=Path(b['path']);name=str(path.resolve()).lower();self.allowed.add(name)
        try:
            raw=path.read_bytes();p.require(len(raw)==b['size_bytes'] and hashlib.sha256(raw).hexdigest()==b['sha256'],'exact legal source packet')
            fields=dict(inverse=('s','params'),context=('observed','u_normalized','mean','scale'),anchor=('clean','fixed','coupled','group_key','origin','starts'))[kind]
            with np.load(io.BytesIO(raw),allow_pickle=False) as a:return {k:a[k].copy() for k in fields}
        finally:self.allowed.remove(name)

    def begin(self,key):
        self.clock()
        with self.db:self.db.execute('INSERT INTO units VALUES (?,?,?)',(key,'PENDING',json.dumps(dict(at_unix=time.time()))))

    def publish(self,key,arrays,detail):
        import numpy as np
        path=Path(self.lock['outputs'][key]);p.require(path.is_relative_to(self.store),'registered D-only packet')
        buffer=io.BytesIO();np.savez(buffer,**arrays);raw=buffer.getvalue()
        self.charge('owned_writes');self.charge('owned_bytes',len(raw));path.parent.mkdir(parents=True,exist_ok=True);name=str(path.resolve()).lower();self.allowed.add(name)
        try:
            with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            self.charge('owned_hashes');digest=hashlib.sha256(path.read_bytes()).hexdigest()
        finally:self.allowed.remove(name)
        b=dict(path=str(path),sha256=digest,detail=detail)
        with self.db:self.db.execute('UPDATE units SET state=?,v=? WHERE k=?',('DONE',json.dumps(b),key))
        return b

    def owned(self,b):
        import numpy as np
        self.charge('owned_reads');self.charge('owned_hashes');path=Path(b['path']);name=str(path.resolve()).lower();self.allowed.add(name)
        try:
            raw=path.read_bytes();p.require(hashlib.sha256(raw).hexdigest()==b['sha256'],'owned qualification packet SHA')
            with np.load(io.BytesIO(raw),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
        finally:self.allowed.remove(name)


def old_functions():
    import numpy as np
    from fractions import Fraction
    namespace=dict(np=np,Fraction=Fraction);out={}
    for filename,names in (('clock_real_core.py',('kernel','render_phase','render_direct','support')),('clock_real_data.py',('spectrum',))):
        path=p.ROOT/'scripts/50_bearing_cnsr'/filename;tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        p.require({n.name for n in nodes}==set(names) and all(not n.decorator_list for n in nodes),'unchanged old pure functions without Torch import')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        out.update({n:namespace[n] for n in names})
    return out


def recursive(rt,s,params,q,size):
    from continuous_event_core import render
    c,n=s.shape
    for k,value in dict(renderer_calls=1,mode_sample_updates=3*c*n,filter_calls=3*c,normalization_samples=c*size,event_grid_scans=c*n).items():rt.charge(k,value)
    started=time.perf_counter();wave=render(s,params,q,size)
    return wave,time.perf_counter()-started


def legacy(rt,fn,s,params,q,size):
    from fractions import Fraction
    c,n=s.shape;a=Fraction(q).limit_denominator(12).numerator;nf=1<<(n+size-1).bit_length()
    for k,value in dict(legacy_renderer_calls=1,legacy_fft_1d_transforms=2*a*c+1,legacy_fft_points=(2*a*c+1)*nf,
        reference_kernel_values=c*a*(size+1),normalization_samples=c*a*size).items():rt.charge(k,value)
    started=time.perf_counter();wave=fn['render_phase'](s,params,q,size)
    return wave,time.perf_counter()-started


def direct(rt,fn,s,params,q,size,start,stop,full=False):
    import numpy as np
    c,n=s.shape;rt.charge('direct_renderer_calls');rt.charge('event_grid_scans',c*n);out=np.zeros(stop-start);center=(n-1)/2
    for channel,param in enumerate(params):
        ids=np.flatnonzero(s[channel]);positions=center+(ids-center)/q
        keep=(positions<=stop-1)&(positions+size-1>=start);ids,positions=ids[keep],positions[keep]
        rt.charge('normalization_samples',size)
        if full:
            times=np.floor(positions).astype(np.int64)[:,None]+np.arange(size+1)[None]
            lag=times-positions[:,None];rt.charge('reference_kernel_values',int(lag.size))
            values=s[channel,ids,None]*fn['kernel'](lag,param,size);valid=(times>=start)&(times<stop)
            np.add.at(out,times[valid]-start,values[valid])
        else:
            lag=np.arange(start,stop)[None]-positions[:,None];rt.charge('reference_kernel_values',int(lag.size))
            out+=(s[channel,ids,None]*fn['kernel'](lag,param,size)).sum(0)
    return out


def make_fixture(spec):
    import numpy as np
    c,n,size=spec['channels'],spec['n'],spec['size'];rng=np.random.default_rng(spec['seed']);s=np.zeros((c,n))
    if spec['kind']=='dense':s[:]=rng.uniform(.1,1.1,size=(c,n))
    elif not spec['key'].endswith('s00'):
        for ch in range(c):
            ids=np.unique(np.r_[0,1,n-2,n-1,rng.integers(2,n-2,size=8)]);s[ch,ids]=rng.uniform(.1,1.1,size=len(ids))
    freq=np.resize(np.array([.001,.115,.271,.449]),c);decay=np.resize(np.array([.015625,.03125,.0625,.125]),c)
    phase=((np.arange(c)+spec['seed'])%4)*np.pi/2
    return s,np.stack((freq,decay,phase),axis=1)


def checks_for_packet(a,spec,threshold):
    import numpy as np
    rows=[]
    def add(name,x,y,mode='wave'):
        diff=np.asarray(x,dtype=float)-np.asarray(y,dtype=float);ma=float(np.max(np.abs(diff),initial=0));scale=max(1.,float(np.max(np.abs(y),initial=0)))
        rms=float(np.sqrt(np.mean(np.asarray(y,dtype=float)**2)));nr=float(np.sqrt(np.mean(diff*diff))/max(1e-12,rms))
        passed=ma<=threshold['frontend_maxabs'] if mode=='front' else ma<=threshold['maxabs_scaled']*scale and (mode=='lineage' or nr<=threshold['nrmse'])
        rows.append(dict(name=name,mode=mode,maxabs=ma,nrmse=nr,maxabs_scale=scale,passed=bool(passed)))
    for i,q in enumerate(a['ratios']):
        if spec['kind']=='source':
            for j,start in enumerate(a['direct_starts']):add('direct/'+str(i)+'/'+str(j),a['rendered'][i,int(start):int(start)+8],a['direct'][i,j])
        else:add('direct/'+str(i),a['rendered'][i],a['direct'][i])
    for j,i in enumerate(a['legacy_indices']):
        lo,hi=a['legacy_ranges'][j];add('legacy/'+str(i),a['rendered'][i,lo:hi],a['legacy'][j,lo:hi])
    if spec['kind']=='source':
        add('old_component_lineage',a['rendered'][1],a['original_component'],mode='lineage')
        for i,j in ((0,0),(2,1)):
            add('fixed_front/'+str(i),a['front_fixed'][i],a['anchor_fixed'][:,j],mode='front')
            add('coupled_front/'+str(i),a['front_coupled'][i],a['anchor_coupled'][:,j],mode='front')
        for name in ('front_fixed','front_coupled'):
            p.require(np.array_equal(a[name][1],a['clean']),'exact q1 complete operator/source frontend identity')
    p.require(all(np.isfinite(v['maxabs']) and np.isfinite(v['nrmse']) for v in rows),'finite complete error arithmetic')
    return rows


def run_unit(rt,fn,spec,values):
    import numpy as np
    from continuous_event_core import reference_interval
    source=spec['kind']=='source'
    if source:
        inv=rt.payload(spec['bindings']['inverse'],'inverse');context=rt.payload(spec['bindings']['context'],'context');anchor=rt.payload(spec['bindings']['anchor'],'anchor')
        g=spec['group'];side=g['source'];p.require(inv['s'].shape==(2,48,32768) and inv['params'].shape==(2,48,3),'old inverse axes')
        p.require(str(anchor['group_key'])==g['key'] and str(anchor['origin'])==g['origin'] and np.array_equal(anchor['starts'],g['starts']),'current exact source identity')
        s,params=inv['s'][side],inv['params'][side]
    else:s,params=make_fixture(spec)
    ratios=[v['value'] for v in values];new=[];references=[];older=[];ranges=[];li=[];times=[]
    for i,q in enumerate(ratios):
        wave,seconds=recursive(rt,s,params,q,spec['size']);new.append(wave)
        if source:
            refs=[direct(rt,fn,s,params,q,spec['size'],int(a),int(a)+8) for a in spec['group']['starts']]
            references.append(np.stack(refs))
        else:references.append(direct(rt,fn,s,params,q,spec['size'],0,spec['n'],full=True))
        if values[i]['legacy']:
            old,oldseconds=legacy(rt,fn,s,params,q,spec['size']);older.append(old);li.append(i);ranges.append(reference_interval(spec['n'],spec['size'],q))
        times.append(dict(ratio=q,seconds=seconds))
    a=dict(events=s,params=params,ratios=np.asarray(ratios),rendered=np.stack(new),direct=np.stack(references),legacy=np.stack(older),
        legacy_indices=np.asarray(li),legacy_ranges=np.asarray(ranges),unit_key=np.array(spec['key']),fingerprint=np.array(rt.lock['fingerprint']))
    detail=dict(key=spec['key'],times=times,timing_pairs=[],legacy_comparison='common observation-supported range; negative-position tails separately checked against direct full synthetic sum')
    if source:
        g=spec['group'];observed=context['observed'][side];component=context['u_normalized'][side];scale=float(context['scale'][side])
        p.require(observed.shape==component.shape==(32768,) and scale>0 and np.isfinite(scale),'finite original source measurement scale')
        crop=lambda x:np.stack([x[int(start):int(start)+g['window']] for start in g['starts']])
        def spectrum(x):
            rt.charge('frontend_fft_1d_transforms',len(x));rt.charge('frontend_fft_points',int(x.size));return fn['spectrum'](x)
        from rate_impulse_core import actions
        factual=crop(observed);clean=spectrum(factual);fixed=[];coupled=[]
        for q,wave in zip(ratios,new):
            timing=factual.copy() if q==1 else crop(observed+scale*(wave-component))
            edited=actions(factual,timing,scale*crop(component),q)['C0']
            fixed.append(spectrum(timing));coupled.append(spectrum(edited))
        a.update(direct_starts=np.asarray(g['starts']),original_component=component,front_fixed=np.stack(fixed),front_coupled=np.stack(coupled),
            anchor_fixed=anchor['fixed'],anchor_coupled=anchor['coupled'],clean=clean)
        p.require(np.array_equal(clean,anchor['clean']),'unchanged measured source/FFT bytes')
        for j,i in enumerate(li):
            q=ratios[i]
            # Both time the same full context for the same legal requested queries;
            # reserve accounting before either timer to exclude journal latency.
            if j%2:
                old,ot=legacy(rt,fn,s,params,q,spec['size']);wave,nt=recursive(rt,s,params,q,spec['size'])
            else:
                wave,nt=recursive(rt,s,params,q,spec['size']);old,ot=legacy(rt,fn,s,params,q,spec['size'])
            p.require(np.array_equal(wave,new[i]) and np.array_equal(old,older[j]),'timed replay of same inputs/outputs')
            detail['timing_pairs'].append(dict(ratio=q,new_seconds=nt,legacy_seconds=ot,legacy_over_new=ot/max(nt,1e-30),materialized_n=spec['n'],legal_query_starts=g['starts']))
    detail['checks']=checks_for_packet(a,spec,rt.lock['thresholds']);return a,detail


def run():
    lock=p.read(p.OUT/'RUN_LOCK_V1.json');p.verify(lock);grant=p.read(p.OUT/'ACTUAL_GRANT_V1.json')
    p.require(grant['fingerprint']==lock['fingerprint'] and grant['authorization']=='standing_authorization','actual zero-neural finite grant')
    rt=Runtime(lock);sys.addaudithook(rt.guard)
    try:
        import numpy as np
        fn=old_functions();bindings={}
        values=lock['ratios']+[dict(name='near_one_up',value=float(np.nextafter(1.,2.)),legacy=False),dict(name='near_one_down',value=float(np.nextafter(1.,0.)),legacy=False)]
        for spec in lock['fixtures']+lock['sources']:
            rt.begin(spec['key']);a,detail=run_unit(rt,fn,spec,values[:spec['ratios']]);bindings[spec['key']]=rt.publish(spec['key'],a,detail)
            print('CONTINUOUS_CORE_UNIT',len(bindings),'/35',spec['key'],'pass',all(v['passed'] for v in detail['checks']),flush=True)
        from audit_continuous_event_core import audit
        result=audit(rt,bindings)
        for k,n in lock['forecast'].items():p.require(rt.counts[k]==n,'exact complete numerical work '+k)
        p.verify(lock);result.update(execution='COMPLETE',fingerprint=lock['fingerprint'],counts=dict(rt.counts),cpu_start=rt.started,cpu_end=time.time(),
            packets=bindings,neural_actions=0,new_raw_reads=0,target_reads=0,new_inverse_fits=0)
        p.save(p.OUT/'RESULT_V1.json',result)
        p.save(p.OUT/'AUDIT_V1.json',dict(status='PASS',result_sha256=p.sha(p.OUT/'RESULT_V1.json'),saved_packets=35,independent_error_arithmetic=True,at_unix=time.time()))
        with rt.db:rt.set('DONE',dict(scientific=result['status'],at_unix=time.time()))
        rt.db.close();closed=dict(execution='COMPLETE',scientific=result['status'],at_unix=time.time(),database=dict(path=str(p.OUT/'CONTROL.sqlite'),sha256=p.sha(p.OUT/'CONTROL.sqlite')),
            files_sha256={n:p.sha(p.OUT/n) for n in ('RUN_LOCK_V1.json','ACTUAL_GRANT_V1.json','RESULT_V1.json','AUDIT_V1.json')})
        p.save(p.OUT/'CLOSED_RECEIPT_V1.json',closed)
        print('CONTINUOUS_CORE_COMPLETE',result['status'],json.dumps(dict(rt.counts)),flush=True)
    except BaseException as error:
        try:
            with rt.db:rt.set('STOPPED',dict(kind=type(error).__name__,reason=str(error),at_unix=time.time()))
            rt.db.close()
        except sqlite3.ProgrammingError:pass
        path=p.OUT/'STOPPED_V1.json'
        if not path.exists():p.save(path,dict(kind=type(error).__name__,reason=str(error),at_unix=time.time(),automatic_replay=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','run'));args=parser.parse_args()
    if args.mode=='prepare':
        lock=p.prepare();print(json.dumps(dict(fingerprint=lock['fingerprint'],forecast=lock['forecast']),indent=2))
    elif args.mode=='test':
        import unittest
        import test_continuous_event_metadata as tests
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        p.save(p.OUT/'METADATA_TESTS_V2.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,torch_imported='torch' in sys.modules,
            code_sha256=p.local_closure([Path(__file__).with_name(n) for n in p.ENTRY]),at_unix=time.time()))
        p.require(result.wasSuccessful() and 'torch' not in sys.modules,'actual metadata tests before payload')
    else:run()
