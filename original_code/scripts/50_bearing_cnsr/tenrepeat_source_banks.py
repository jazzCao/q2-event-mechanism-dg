"""Finite seed extension of the qualified source operator, never target fitting.

Reuse exact existing per-pair queries; fit only newly required source pairs.
The qualified renderer/control functions are called, not reimplemented.
"""
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import gzip
import hashlib
import io
import json
import os
import sqlite3
import sys
import time

from tenrepeat_source_metadata import ROOT,OUT,read,sha,canonical,need,write
HERE=OUT/'mdg_banks'
STORE=Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_tenseed_all/mdg_banks')

def load_manifest():return json.loads(gzip.decompress((OUT/'MDG_SOURCE_MANIFEST_V1.json.gz').read_bytes()))

def prepare(version='V1'):
    from design_e0_natural_source_probe_v2 import centered_three
    from e0_selection_entry_spec import local_closure
    HERE.mkdir(parents=True,exist_ok=True);m=load_manifest()
    old=read(ROOT/'results/wfdg/ress_event_full_validation/SOURCE_LOCK_V1.json')
    full=read(ROOT/'results/wfdg/ress_event_full_validation/SOURCE_BANKS_V1.json')
    component=read(ROOT/'results/wfdg/ress_event_component_attribution/SOURCE_BANKS_V1.json')
    oldgroups={(g['inverse_identity'],g['source']):g for g in old['groups']}
    groups={};cases={};assets={};witness={};raw={}
    for b in m['features']:
        for v in b['source_input_records']:
            r=v['record'];bound={k:v[k] for k in ('path','sha256','samples','fs')}
            if r in raw:need(raw[r]==bound,'common raw lineage')
            raw[r]=bound
    for key,c in sorted(m['cases'].items()):
        items=[]
        for item in c['bank']:
            pk=item['pair'];pair=c['pairs'][pk];witness.setdefault(pk,key)
            if pk in m['reused_inverse_assets']:
                identity=m['reused_inverse_assets'][pk]['identity'];assets[identity]=old['assets'][identity]
            else:identity=canonical(dict(new_source_pair=pair,raw_sha256=[raw[r]['sha256'] for r in pair],context=[65536,32768],steps=500))
            side=item['source'];r=m['records'][item['record']]
            g=oldgroups.get((identity,side))
            if g is None:g=dict(key=canonical(dict(inverse_identity=identity,source=side)),inverse_identity=identity,source=side,origin=item['record'],pair=pair,fs=r['fs'],kernel_length=int(__import__('math').ceil(.004*r['fs'])),window=r['window'],starts=centered_three(32768,r['window']))
            if g['key'] not in groups:
                groups[g['key']]=dict(g,pair_key=pk,witness=key,anchor_full=full['groups'].get(g['key']),anchor_component=component['groups'].get(g['key']))
            items.append(dict(record=item['record'],label=r['label'],domain=r['domain'],group=g['key']))
        cases[key]=dict(key=key,task=c['task'],seed=c['seed'],bank=items,bank_windows=3*len(items),source_training_windows=c['training_windows'],source_validation_windows=c['validation_windows'])
    missing={pk:dict(pair=pair,witness=witness[pk],identity=canonical(dict(new_source_pair=pair,raw_sha256=[raw[r]['sha256'] for r in pair],context=[65536,32768],steps=500))) for pk,pair in m['missing_inverse_pairs'].items()}
    newgroups=[g for g in groups.values() if g['anchor_component'] is None]
    need(len(missing)==40,'exact forty additional numerical source fits')
    modules=[Path(__file__),Path(__file__).with_name('event_full_validation_source.py'),Path(__file__).with_name('event_component_source.py'),Path(__file__).with_name('clock_real_core.py'),Path(__file__).with_name('clock_real_graph_core.py')]
    code=local_closure(modules)
    paths=[OUT/'MDG_SOURCE_MANIFEST_V1.json.gz',ROOT/'results/wfdg/ress_event_full_validation/SOURCE_LOCK_V1.json',ROOT/'results/wfdg/ress_event_full_validation/SOURCE_BANKS_V1.json',ROOT/'results/wfdg/ress_event_component_attribution/SOURCE_BANKS_V1.json']
    now=datetime.now(timezone(timedelta(hours=8)))
    limits=dict(gpu_inverse_iterations=20030,graph_warmup_iterations=30,graph_capture_blueprint_steps=10,new_inverse_fits=40,
        raw_record_loads=len({r for a in missing.values() for r in a['pair']}),payload_reads=30000,owned_writes=3*40+len(newgroups)+189,
        source_fft_1d_transforms=100*len(newgroups),source_fft_points=100*5333*len(newgroups),renderer_calls=8*len(newgroups),
        mode_sample_updates=8*3*48*32768*len(newgroups),filter_calls=8*3*48*len(newgroups),event_grid_scans=8*2*48*32768*len(newgroups),normalization_samples=8*48*267*len(newgroups))
    lock=dict(kind='TEN_REPEAT_EXACT_SOURCE_BANK_EXTENSION',campaign=dict(start=now.isoformat(),deadline=(now+timedelta(hours=48)).isoformat(),source_seconds=48*3600,store=str(STORE)),
        cases=cases,groups=groups,assets=assets,new_inverse_pairs=missing,raw_bindings=raw,ratios=old['ratios'],limits=limits,
        new_groups=len(newgroups),existing_groups=len(groups)-len(newgroups),manifest_sha256=sha(OUT/'MDG_SOURCE_MANIFEST_V1.json.gz'),
        input_sha256={str(p):sha(p) for p in paths},code_sha256=code,source_only=True,new_target_predictions=0)
    lock['fingerprint']=canonical(lock);write(HERE/f'LOCK_{version}.json.gz',lock)
    print('TEN_REPEAT_SOURCE_BANK_READY',dict(groups=len(groups),new_groups=len(newgroups),new_inverse_pairs=40,cases=189),flush=True)

class Runtime:
    def __init__(self,lock):
        self.lock=lock;self.store=STORE;STORE.mkdir(parents=True,exist_ok=True)
        path=HERE/'CONTROL.sqlite';need(not path.exists(),'source preparation cannot silently replay')
        self.db=sqlite3.connect(path);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT);CREATE TABLE counters(k TEXT PRIMARY KEY,n INTEGER);CREATE TABLE units(k TEXT PRIMARY KEY,state TEXT,v TEXT);')
        self.counts=Counter();self.allowed=set();self.started=time.time();self.set('fingerprint',lock['fingerprint'])
    def set(self,k,v):
        with self.db:self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(k,json.dumps(v,allow_nan=False)))
    def clock(self):need(time.time()<datetime.fromisoformat(self.lock['campaign']['deadline']).timestamp(),'source preparation cumulative deadline')
    def charge(self,k,n=1):
        self.clock();need(k in self.lock['limits'] and self.counts[k]+n<=self.lock['limits'][k],'finite source resource '+k)
        self.counts[k]+=n
        with self.db:self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)',(k,self.counts[k]))
    claim=charge
    def begin(self,k):
        with self.db:self.db.execute('INSERT INTO units VALUES (?,?,?)',(k,'PENDING',json.dumps(dict(at=time.time()))))
    def payload(self,b):
        import numpy as np
        self.charge('payload_reads');p=Path(b['path']);self.allowed.add(str(p.resolve()).casefold())
        try:
            raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==b['sha256'],'source payload fingerprint '+str(p))
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
        finally:self.allowed.remove(str(p.resolve()).casefold())
    def save(self,k,a):
        import numpy as np
        self.charge('owned_writes');path=STORE/(hashlib.sha256(k.encode()).hexdigest()+'.npz');buffer=io.BytesIO();np.savez(buffer,**a);raw=buffer.getvalue()
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        b=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))
        with self.db:self.db.execute('INSERT OR REPLACE INTO units VALUES (?,?,?)',(k,'DONE',json.dumps(b)))
        return b
    def guard(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix.lower() in ('.npz','.npy','.pt','.pth','.mat','.wav','.h5') or p.is_relative_to(ROOT/'data/raw'):
            need(p.is_relative_to(STORE) or str(p).casefold() in self.allowed,'only per-unit legal source or own output '+str(p))

def run(version='V1'):
    lock=json.loads(gzip.decompress((HERE/f'LOCK_{version}.json.gz').read_bytes()));m=load_manifest()
    need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'source lock fingerprint')
    for p,h in {**lock['input_sha256'],**lock['code_sha256']}.items():need(sha(p)==h,'source code/input '+p)
    # Serialize GPU consumers. This runner is launched only after SDG source stage.
    need((OUT/'sdg_source/SOURCE_STAGE_COMPLETE_V1.json').is_file(),'previous source worker must finish, no competing CUDA')
    rt=Runtime(lock);sys.addaudithook(rt.guard)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from clock_real_core import design
    from clock_real_graph_core import GraphInverse
    from clock_real_data import RawStore,spectrum
    from natural_mapping_probe_plan import source_case
    from source_rate_neural import exact
    from event_full_validation_source import make_group as full_group
    from event_component_source import make_group as control_group
    from event_component_source import assemble
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rawcache={};assets=dict(lock['assets']);engine=None;proofs=[]
    class RawRuntime:
        def __init__(self):self.spec=dict(data=dict(records=m['records']));self.phase='SOURCE';self.accesses=[];self.raw_hashes={}
        def claim(self,k,n=1):rt.charge(k,n)
        def permit(self,p):rt.allowed.add(str(Path(p).resolve()).casefold())
        def revoke(self,p):rt.allowed.discard(str(Path(p).resolve()).casefold())
    rawrt=RawRuntime();rawstore=RawStore(rawrt)
    try:
        for pk,a in sorted(lock['new_inverse_pairs'].items()):
            case=m['cases'][a['witness']];pair=a['pair'];rt.begin('I/'+pk)
            for r in pair:source_case(case,m['records'],r,context=True)
            observed=np.stack([rawstore.load(r,case['task'],'source')[65536:98304] for r in pair])
            for r in pair:need(rawrt.raw_hashes[r]['sha256']==lock['raw_bindings'][r]['sha256'],'new fit existing signal SHA')
            mu=observed.mean(1);scale=np.sqrt(np.mean((observed-mu[:,None])**2,axis=1)).clip(1e-12);x=(observed-mu[:,None])/scale[:,None]
            rr=[m['records'][r] for r in pair];d=design(x,[r['shaft_hz'] for r in rr],rr[0]['fs'])
            if engine is None:engine=GraphInverse(x,d,rt.charge)
            rt.charge('new_inverse_fits');state=engine.fit(x,d,250,tick=rt.charge)
            p250=rt.save('I250/'+pk,{**state,'params':d['params']});restored=rt.payload(p250)
            for k in state:need(np.array_equal(np.asarray(state[k]),restored[k]),'actual numerical disk250 full state')
            resumed={k:restored[k].item() if np.asarray(state[k]).ndim==0 else restored[k] for k in state}
            state=engine.fit(x,d,500,state=resumed,tick=rt.charge)
            inv=rt.save('I/'+pk,dict(s=state['s'],b=state['b'],c=state['c'],params=d['params'],x=x,source_mean=mu,source_scale=scale,kernel_length=d['kernel_length'],stationary_frequencies=d['stationary_frequencies']))
            ctx=rt.save('CTX/'+pk,dict(observed=observed,mean=mu,scale=scale,u_normalized=x))
            assets[a['identity']]=dict(inverse=inv,qualified_context=ctx)
            proofs.append(dict(pair=pk,witness=a['witness'],iterations=500,actual_disk250_restore=True))
            print('TEN_REPEAT_NEW_INVERSE',len(proofs),'/40',flush=True)
        packets={};groups={}
        for i,(key,g) in enumerate(sorted(lock['groups'].items()),1):
            case=m['cases'][g['witness']]
            for r in g['pair']:source_case(case,m['records'],r,context=True)
            if g['anchor_component'] is not None:
                a=rt.payload(g['anchor_component']);anchor=rt.payload(g['anchor_full'])
                for k in ('clean','fixed','interior_fixed','coupled'):need(np.array_equal(a[k],anchor[k]),'old canonical/control shared source query bytes')
            else:
                rt.begin('G/'+key);inv=rt.payload(assets[g['inverse_identity']]['inverse']);ctx=rt.payload(assets[g['inverse_identity']]['qualified_context'])
                anchor=full_group(g,inv,ctx,rt,{'spectrum':spectrum})
                from types import SimpleNamespace
                controls=SimpleNamespace(lock=dict(rt.lock,ratios=rt.lock['ratios'][2:]),charge=rt.charge)
                a=control_group(g,inv,ctx,anchor,controls,{'spectrum':spectrum})
            a.update(time_energy=anchor['time_energy'],fixed_time_energy=anchor['fixed_time_energy'])
            need(str(a['origin'])==g['origin'] and str(a['group_key'])==g['key'],'current source group identity')
            packets[key]=a
            if g['anchor_component'] is None:groups[key]=rt.save('G/'+key,a)
            else:groups[key]=dict(component=g['anchor_component'],canonical=g['anchor_full'])
            if i%100==0:print('TEN_REPEAT_SOURCE_GROUP',i,'/',len(lock['groups']),flush=True)
        banks={}
        for key,c in sorted(lock['cases'].items()):
            rt.begin('B/'+key);a=assemble(c,packets)
            for field in ('time_energy','fixed_time_energy'):a[field]=np.concatenate([packets[v['group']][field] for v in c['bank']])
            banks[key]=rt.save('B/'+key,a);check=rt.payload(banks[key]);need(a.keys()==check.keys() and all(np.array_equal(a[k],check[k]) for k in a),'all new bank fields persisted exactly')
            print('TEN_REPEAT_SOURCE_BANK',len(banks),'/189',key,flush=True)
        need(rt.counts['new_inverse_fits']==40 and rt.counts['gpu_inverse_iterations']==20030,'exact numerical fit/warmup work')
        write(HERE/'SOURCE_BANKS_V1.json',dict(status='ALL189_NEW_SEED_SOURCE_BANKS_COMPLETE',fingerprint=lock['fingerprint'],banks=banks,groups=groups,new_assets={k:v for k,v in assets.items() if k not in lock['assets']},proofs=proofs,counts=dict(rt.counts),new_target_predictions=0,source_end=time.time()))
        rt.set('DONE',dict(at=time.time()));print('TEN_REPEAT_MDG_SOURCE_BANKS_COMPLETE',flush=True)
    except BaseException as e:
        rt.set('STOPPED',dict(type=type(e).__name__,message=str(e),counts=dict(rt.counts)));raise
    finally:rt.db.close()

if __name__=='__main__':{'prepare':prepare,'run':run,'prepare-v2':lambda:prepare('V2'),'run-v2':lambda:run('V2')}[sys.argv[1]]()
