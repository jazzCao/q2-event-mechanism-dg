"""Small FULL SQLite control for finite repeat stages; D-only big artifacts."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib
import io
import json
import os
import sqlite3
import time

from tenrepeat_source_metadata import need,canonical

class Runtime:
    def __init__(self,lock,here):
        self.lock=lock;self.here=Path(here);self.store=Path(lock['campaign']['store']).resolve();self.store.mkdir(parents=True,exist_ok=True)
        need(self.store.drive.upper()=='D:','new model/array D destination')
        path=self.here/'CONTROL.sqlite';need(not path.exists(),'no implicit second producer/replay')
        self.db=sqlite3.connect(path);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT);CREATE TABLE counters(k TEXT PRIMARY KEY,n INTEGER);CREATE TABLE steps(chain TEXT,n INTEGER,done INTEGER,PRIMARY KEY(chain,n));CREATE TABLE artifacts(path TEXT PRIMARY KEY,v TEXT);CREATE TABLE models(chain TEXT PRIMARY KEY,v TEXT);')
        self.counts=Counter();self.permitted=set();self.active=None;self.cursor={};self.pending=None;self.phase='SOURCE';self.gpu_start=None
        self.set('fingerprint',lock['fingerprint']);self.set('pid',os.getpid())
    def set(self,k,v):
        with self.db:self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(k,json.dumps(v,allow_nan=False)))
    def get(self,k):
        r=self.db.execute('SELECT v FROM meta WHERE k=?',(k,)).fetchone();return None if r is None else json.loads(r[0])
    def clock(self):
        need(time.time()<datetime.fromisoformat(self.lock['campaign']['deadline']).timestamp(),'repeat stage total deadline')
        need(self.gpu_start is None or time.time()<self.gpu_start+self.lock['campaign']['neural_seconds'],'repeat cumulative neural deadline')
    def before_neural(self):
        if self.gpu_start is None:self.gpu_start=time.time();self.set('gpu_start',self.gpu_start)
        self.clock()
    gpu=before_neural
    def claim(self,k,n=1):
        self.clock();need(k in self.lock['limits'] and self.counts[k]+n<=self.lock['limits'][k],'repeat stage resource '+k)
        self.counts[k]+=n
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)',(k,self.counts[k]))
            if k=='actual_optimizer_updates':
                need(self.pending==(self.active,self.cursor.get(self.active,0)+1),'complete only actual paid native step')
                self.db.execute('UPDATE steps SET done=1 WHERE chain=? AND n=?',self.pending)
                self.cursor[self.active]=self.pending[1];self.pending=None
    def attempt(self,n):
        self.before_neural();need(self.active and self.pending is None and n==self.cursor.get(self.active,0)+1,'sequential paid update only')
        self.claim('optimizer_attempts')
        with self.db:self.db.execute('INSERT INTO steps VALUES (?,?,0)',(self.active,n))
        self.pending=(self.active,n)
    def permit(self,p):self.permitted.add(str(Path(p).resolve()).lower())
    def revoke(self,p):self.permitted.discard(str(Path(p).resolve()).lower())
    def hash(self,p,k):
        self.claim(k);self.permit(p)
        try:return hashlib.sha256(Path(p).read_bytes()).hexdigest()
        finally:self.revoke(p)
    def guard(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix.lower() in ('.mat','.csv','.wav','.h5','.hdf5'):raise PermissionError('no unregistered raw signal')
        if p.suffix.lower() in ('.pt','.pth','.npz','.npy','.pkl','.pickle'):need(str(p).lower() in self.permitted or p.is_relative_to(self.store),'only explicitly granted source/own arrays '+str(p))
    def save(self,relative,value,*,packet=False):
        path=self.store/relative;need(path.is_relative_to(self.store),'own store path');path.parent.mkdir(parents=True,exist_ok=True)
        buffer=io.BytesIO()
        if packet:
            import torch
            torch.save(value,buffer)
        else:
            import numpy as np
            np.savez_compressed(buffer,**value)
        b=buffer.getvalue();self.claim('checkpoint_writes' if packet else 'probability_writes')
        with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
        detail=dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),at=time.time(),packet=packet)
        with self.db:self.db.execute('INSERT INTO artifacts VALUES (?,?)',(str(path),json.dumps(detail)))
        return detail
    def load(self,b):
        self.claim('checkpoint_reads' if b['packet'] else 'probability_reads');raw=Path(b['path']).read_bytes()
        need(hashlib.sha256(raw).hexdigest()==b['sha256'],'saved actual payload SHA')
        if b['packet']:
            import torch
            return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        import numpy as np
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def finish(self,name,value):
        need(self.pending is None,'no unpaid/incomplete step on source completion')
        with self.db:self.db.execute('INSERT INTO models VALUES (?,?)',(name,json.dumps(value,allow_nan=False)))
    def models(self):return {k:json.loads(v) for k,v in self.db.execute('SELECT chain,v FROM models')}

class NativePermit:
    def __init__(self,rt,case_key,config,qualification=False):
        self.rt=rt;self.fingerprint=canonical(dict(campaign=rt.lock['fingerprint'],case=case_key,config=config,qualification=qualification))
    def require_neural(self,fingerprint):
        need(fingerprint==self.fingerprint,'exact native permit fingerprint');self.rt.before_neural()
    def claim(self,fingerprint,n,phase):
        self.require_neural(fingerprint);need(phase=='B','native existing one-stage recipe');self.rt.attempt(n)
