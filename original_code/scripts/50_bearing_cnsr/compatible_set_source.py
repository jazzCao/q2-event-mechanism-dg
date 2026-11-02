"""Source-only complete set generation, durable precharged resource accounting."""
from collections import Counter
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import compatible_set_plan as p


class Runtime:
    def __init__(self, lock):
        self.lock=lock; self.store=Path(lock['campaign']['store'])/'source'; self.store.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(p.OUT/'SOURCE_CONTROL.sqlite',timeout=30)
        self.db.execute('PRAGMA synchronous=FULL'); self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.executescript('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);CREATE TABLE IF NOT EXISTS counters(k TEXT PRIMARY KEY,n INTEGER);CREATE TABLE IF NOT EXISTS units(k TEXT PRIMARY KEY,state TEXT,v TEXT);')
        p.require(self.db.execute("SELECT count(*) FROM meta").fetchone()[0] == 0,'no implicit replay or source clock reset')
        with self.db: self.db.execute('INSERT INTO meta VALUES (?,?)',('fingerprint',lock['fingerprint']))
        self.counts=Counter(); self.allowed=set(); self.cached=None; self.inputs_cache=None

    def charge(self,name,count=1):
        p.clock(self.lock['campaign']); value=self.counts[name]+int(count)
        p.require(name in self.lock['limits'] and value <= self.lock['limits'][name], 'source quota before action: '+name)
        with self.db: self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)',(name,value))
        self.counts[name]=value

    def fft(self,name,value,n=None,axis=-1,**kwargs):
        import numpy as np
        a=np.asarray(value); axis%=a.ndim; rows=a.size//a.shape[axis]
        length=n if n is not None else (2*(a.shape[axis]-1) if name=='irfft' else a.shape[axis])
        self.charge('source_fft_1d_transforms',rows); self.charge('source_fft_points',rows*length)
        return getattr(np.fft,name)(value,n=n,axis=axis,**kwargs)

    def payload(self,binding,owned=False):
        import numpy as np
        path=Path(binding['path']); key=str(path.resolve()).lower()
        if owned:
            self.charge('owned_reads'); self.charge('owned_hashes')
        else:
            self.charge('source_payload_decodes'); self.charge('source_payload_bytes',path.stat().st_size)
            p.require(path.stat().st_size == self.lock['payload_sizes'][str(path.resolve())], 'bound source bytes')
        self.allowed.add(key)
        try:
            raw=path.read_bytes(); p.require(hashlib.sha256(raw).hexdigest()==binding['sha256'],'source packet fingerprint')
            with np.load(io.BytesIO(raw),allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
        finally: self.allowed.remove(key)

    def inputs(self,identity):
        if identity != self.cached:
            b=self.lock['assets'][identity]
            self.inputs_cache=(self.payload(b['inverse']),self.payload(b['qualified_context'])); self.cached=identity
        return self.inputs_cache

    def begin(self,name):
        p.clock(self.lock['campaign'])
        with self.db: self.db.execute('INSERT INTO units VALUES (?,?,?)',(name,'PENDING',json.dumps(dict(at=time.time()))))

    def publish(self,name,arrays):
        import numpy as np
        self.charge('owned_writes'); path=self.store/(hashlib.sha256(name.encode()).hexdigest()+'.npz'); key=str(path.resolve()).lower()
        self.allowed.add(key)
        try:
            with path.open('xb') as stream: np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
            self.charge('owned_hashes'); digest=hashlib.sha256(path.read_bytes()).hexdigest()
        finally: self.allowed.remove(key)
        b=dict(path=str(path),sha256=digest)
        with self.db: self.db.execute('UPDATE units SET state=?,v=? WHERE k=?',('DONE',json.dumps(b),name))
        return b

    def guard(self,event,args):
        if event=='import' and isinstance(args[0],str) and args[0].split('.')[0] in ('torch','tensorflow','jax'): raise PermissionError('zero-neural source stage')
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() in ('.mat','.csv','.wav','.h5','.pt','.pth','.npz','.npy','.pkl'):
            p.require(str(path).lower() in self.allowed,'only current explicitly granted scientific payload: '+str(path))


def run():
    lock=p.read(p.OUT/'SOURCE_LOCK_V2.json'); p.verify(lock)
    qualification=p.read(p.OUT/'SOURCE_CPU_QUALIFICATION_V1.json')
    p.require(qualification['status']=='PASS' and qualification['fingerprint']==lock['fingerprint'], 'current synthetic source tests passed before payload')
    import numpy as np
    from compatible_set_core import make_group,assemble
    from run_abduction_action import physics
    rt=Runtime(lock); sys.addaudithook(rt.guard); functions=physics(rt)
    packets={}; groups={}; coverage=[]; started=time.time()
    for number,g in enumerate(lock['groups'],1):
        rt.begin('G/'+g['key']); inv,context=rt.inputs(g['inverse_identity'])
        a=make_group(g,inv,context,rt,functions); anchor=rt.payload(lock['anchors'][g['key']])
        by_q={str(q):v for q,v in zip(anchor['ratios'],anchor['event'])}
        p.require(np.array_equal(a['clean'],anchor['clean']) and np.array_equal(a['fixed'],np.stack([by_q[q] for q in ('2/3','3/2')],axis=1)),'bitwise original E0 anchor on every source group')
        b=rt.publish('G/'+g['key'],a); actual=rt.payload(b,owned=True)
        p.require(a.keys()==actual.keys() and all(np.array_equal(a[k],actual[k]) for k in a),'all group arrays persisted exactly')
        packets[g['key']]=actual; groups[g['key']]=b
        coverage.append(dict(group=g['key'],alias=bool(a['alias']),pair=a['selected_pair'].tolist(),gram_error=float(a['gram_error']),factual_error=float(a['factual_error'])))
        if number==1 or number%25==0:
            elapsed=time.time()-started; print('COMPATIBLE_SOURCE',number,'/',len(lock['groups']),'elapsed_seconds',round(elapsed,2),'projected_seconds',round(elapsed/number*len(lock['groups']),2),flush=True)
    banks={}
    for c in lock['cases']:
        rt.begin('B/'+c['key']); a=assemble(c,packets); b=rt.publish('B/'+c['key'],a); actual=rt.payload(b,owned=True)
        p.require(a.keys()==actual.keys() and all(np.array_equal(a[k],actual[k]) for k in a),'complete case bank byte identity')
        banks[c['key']]=b
    p.require(rt.counts['source_payload_decodes']==lock['forecast']['source_payload_decodes'] and rt.counts['source_payload_bytes']==lock['forecast']['source_payload_bytes'],'exact old source read coverage')
    value=dict(status='SOURCE_BANKS_COMPLETE_CLASSIFICATION_UNTESTED',fingerprint=lock['fingerprint'],banks=banks,groups=groups,coverage=coverage,counts=dict(rt.counts),start_unix=started,end_unix=time.time(),neural_actions=0,target_predictions=0)
    p.save(p.OUT/'SOURCE_BANKS_V1.json',value)
    with rt.db: rt.db.execute('INSERT INTO meta VALUES (?,?)',('DONE',json.dumps(dict(at=time.time()))))
    rt.db.close(); print('COMPATIBLE_SOURCE_DONE',json.dumps(value['counts']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('prepare','qualify','run')); args=parser.parse_args()
    if args.mode=='prepare':
        value=p.source_lock(); print(json.dumps(dict(fingerprint=value['fingerprint'],forecast=value['forecast']),indent=2))
    elif args.mode=='qualify':
        import unittest
        import test_compatible_set
        lock=p.read(p.OUT/'SOURCE_LOCK_V2.json'); p.verify(lock)
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_compatible_set))
        p.save(p.OUT/'SOURCE_CPU_QUALIFICATION_V1.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
            fingerprint=lock['fingerprint'],at_unix=time.time(),actual_source_payload_reads=0,actual_FFTs=0,actual_neural_actions=0))
        p.require(result.wasSuccessful(),'source synthetic qualification failed')
    else: run()
