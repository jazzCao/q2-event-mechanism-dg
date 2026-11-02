"""One six-view forward, common factual supervision and compatible-set risk."""
import json
import os
from pathlib import Path
import sqlite3
import time
import compatible_set_plan as p
from source_rate_screen_runtime import Runtime as BaseRuntime, Guard
from source_rate_screen_data import Files as BaseFiles, check_vectors
from source_rate_neural import Chain as BaseChain, libraries, tensor_sha
from compatible_set_screen_plan import ARMS


class Runtime(BaseRuntime):
    def __init__(self,lock):
        from protected_response_journal import process_identity
        self.lock=lock;self.phase='SOURCE';self.permitted=set()
        self.store=Path(lock['campaign']['store']).resolve();self.store.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(p.OUT/'NEURAL_CONTROL.sqlite',timeout=30)
        self.db.execute('PRAGMA journal_mode=DELETE');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);CREATE TABLE IF NOT EXISTS counters(k TEXT PRIMARY KEY,n INTEGER);CREATE TABLE IF NOT EXISTS steps(chain TEXT,step INTEGER,done INTEGER,PRIMARY KEY(chain,step));CREATE TABLE IF NOT EXISTS artifacts(path TEXT PRIMARY KEY,kind TEXT,detail TEXT);CREATE TABLE IF NOT EXISTS finished(chain TEXT PRIMARY KEY,detail TEXT);')
        self.db.execute('BEGIN IMMEDIATE');owner=self.get('owner')
        if owner and owner['state']=='RUNNING': p.require(process_identity(owner['pid']) != owner['native'],'existing native producer still alive')
        old=self.get('fingerprint');p.require(old is None or old==lock['fingerprint'],'same durable five-arm comparison')
        p.require(self.get('DONE') is None,'closed comparison cannot restart')
        self.set('fingerprint',lock['fingerprint']);self.native=process_identity(os.getpid())
        self.set('owner',dict(state='RUNNING',pid=os.getpid(),native=self.native,at_unix=time.time()))
        self.db.commit();self.counts={k:n for k,n in self.db.execute('SELECT k,n FROM counters')};self.clock()


class Files(BaseFiles):
    def bank(self,case):
        import numpy as np
        from design_e0_natural_source_probe_v2 import centered_three
        key=case['key'];rt=self.rt
        # Role checks are repeated before cache reuse, not only on first decode.
        vectors=dict(y=[],g=[],domain=[],offset=[])
        for item in case['bank']:
            p.require(item['domain'] in case['task']['sources'],'current-case source bank only')
            record=self.lock['records'][item['record']]
            for name,value in (('y',record['label']),('g',item['record']),('domain',record['domain'])):vectors[name].extend([value]*3)
            vectors['offset'].extend([65536+v for v in centered_three(32768,case['task']['window'])])
        if key not in self.banks:
            b=self.lock['banks'][key];p.require(rt.hash(b['path'],'bank_hashes')==b['sha256'],'complete current bank fingerprint')
            rt.claim('bank_decodes');rt.permit(b['path'])
            try:
                with np.load(b['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            finally:rt.revoke(b['path'])
            for name in ('clean','fixed','plus','minus','random_plus','random_minus'):
                shape=(case['bank_windows'],2048) if name=='clean' else (case['bank_windows'],2,2048)
                p.require(a[name].shape==shape and a[name].dtype==np.float32 and np.isfinite(a[name]).all(),'finite exact full bank '+name)
            absolute=relative=0.;zero=0
            for lo in range(0,len(a['y']),256):
                base=a['fixed'][lo:lo+256].astype(float)
                mechanical=np.stack([a[n][lo:lo+256].astype(float)-base for n in ('plus','minus')],axis=-2)
                random=np.stack([a[n][lo:lo+256].astype(float)-base for n in ('random_plus','random_minus')],axis=-2)
                g=np.einsum('...id,...jd->...ij',mechanical,mechanical)
                h=np.einsum('...id,...jd->...ij',random,random)
                p.require(np.allclose(g,h,atol=5e-6,rtol=1e-4),'full saved float32 bank Gram within prespecified tolerance')
                difference=np.linalg.norm(g-h,axis=(-2,-1));norm=np.linalg.norm(g,axis=(-2,-1));active=norm>0
                absolute=max(absolute,float(abs(g-h).max()))
                if active.any():relative=max(relative,float((difference[active]/norm[active]).max()))
                zero+=int((~active).sum())
                p.require(np.array_equal(g[~active],h[~active]),'zero set remains exact random alias')
            audits=rt.get('SOURCE_BANK_GRAM_AUDITS') or {}
            audits[key]=dict(max_gram_absolute_error=absolute,max_relative_frobenius_error=relative,
                zero_query_sets=zero,query_sets=len(a['y'])*2,absolute_tolerance=5e-6,relative_tolerance=1e-4,
                boundary='tolerance qualified after float32 centering, not exact algebraic Gram identity')
            with rt.db:rt.set('SOURCE_BANK_GRAM_AUDITS',audits)
            from compatible_set_control import global_members,mixed_geometry
            # One Q is shared by every view and record in the current legal source pool.
            identity='COMPATIBLE_GLOBAL_ORTHOGONAL_V2|'+p.canonical(sorted({v['group'] for v in case['bank']}))
            rt.claim('orthogonal_feature_rows',4*len(a['y']))
            a['global_plus'],a['global_minus']=global_members(a['fixed'],a['plus'],a['minus'],identity)
            rt.claim('control_mixed_pairs',640)
            geometry=mixed_geometry(a['fixed'],a['plus'],a['minus'],a['global_plus'],a['global_minus'])
            records=rt.get('GLOBAL_CONTROL_MIXED_AUDITS') or {}
            records[key]=dict(identity_sha256=p.canonical(identity),**geometry)
            with rt.db:rt.set('GLOBAL_CONTROL_MIXED_AUDITS',records)
            self.banks[key]=a
        check_vectors(self.banks[key],vectors)
        return self.banks[key]


class Data:
    def __init__(self,case,source,bank,rt):
        np,torch,_,_=libraries(rt)
        self.rt,self.case,self.source,self.task=rt,case,source,case['task']
        self.per_cell=120//(len(self.task['sources'])*self.task['classes'])
        self.batch_size=self.per_cell*len(self.task['sources'])*self.task['classes']
        self.x=torch.as_tensor(source['tr']['x'],device='cuda');self.y=torch.as_tensor(source['tr']['y'],device='cuda')
        self.bx={k:torch.as_tensor(bank[k],device='cuda') for k in ('fixed','plus','minus','global_plus','global_minus')}
        self.by=torch.as_tensor(bank['y'],device='cuda');self.pools=[];self.bpools=[]
        for d in self.task['sources']:
            for y in range(self.task['classes']):
                self.pools.append(np.flatnonzero((source['tr']['domain']==d)&(source['tr']['y']==y)))
                self.bpools.append(np.flatnonzero((bank['domain']==d)&(bank['y']==y)))
        p.require(all(len(v) for v in self.pools+self.bpools),'full class/domain source coverage')

    def batch(self,arm,sampler,augmentation):
        import numpy as np
        import torch
        from source_rate_core import shared_edit_mask
        full=np.concatenate([sampler.choice(v,self.per_cell,replace=True) for v in self.pools])
        bank=np.concatenate([sampler.choice(v,self.per_cell,replace=True) for v in self.bpools])
        mask=shared_edit_mask(sampler.uniform(size=self.batch_size),len(self.task['sources']),self.task['classes'],self.per_cell)
        factors=augmentation.randint(0,2,size=self.batch_size)
        fi,bi,mi,qi=[torch.as_tensor(v,device='cuda') for v in (full,bank,mask,factors)]
        factual,yf,ya=self.x[fi],self.y[fi],self.by[bi]
        p.require(torch.equal(yf,ya),'same balanced full/auxiliary label order')
        names={'A0_STRONG_IN_MIXUP':('fixed',)*3,'A1_MATCHED_E0':('fixed',)*3,
            'A2_COMPATIBLE_MEAN':('fixed','plus','minus'),'C0_COMPATIBLE_WORST':('fixed','plus','minus'),
            'N1_GLOBAL_ORTHOGONAL_WORST':('fixed','global_plus','global_minus')}[arm]
        aux=[factual if arm==ARMS[0] else torch.where(mi[:,None],self.bx[name][bi,qi],factual) for name in names]
        auxiliary_y=torch.where(mi,ya,yf)
        lam=float(augmentation.beta(.2,.2));permutation=torch.as_tensor(augmentation.permutation(self.batch_size),device='cuda')
        import hashlib
        trace=hashlib.sha256()
        for value in (full,bank,mask,factors,permutation.detach().cpu().numpy()):trace.update(np.asarray(value).tobytes())
        trace.update(lam.hex().encode());self.last_trace=trace.hexdigest()
        mixed=[lam*a+(1-lam)*a[permutation] for a in aux]
        return torch.cat([factual,factual,factual,*mixed]),(yf,auxiliary_y),permutation,lam


class Chain(BaseChain):
    def step(self,inspect=False):
        from compatible_set_core import risk_loss
        torch=self.torch;started=time.perf_counter();self.rt.attempt(self.name,self.step_n+1);self.model.train()
        x,labels,permutation,lam=self.data.batch(self.arm,self.sampler,self.augmentation)
        self.rt.claim('training_forward_windows',len(x))
        loss=risk_loss(torch,self.model(x)['logits'],labels,permutation,lam,self.arm in ARMS[3:])
        p.require(all(torch.isfinite(v) for v in loss.values()),'finite six-view loss')
        self.optimizer.zero_grad(set_to_none=True);loss['total'].backward()
        if inspect:
            grads={k:v.grad for k,v in self.model.named_parameters()}
            p.require(all(v is not None and torch.isfinite(v).all() for v in grads.values()),'actual finite gradients on every parameter')
            norm=float(sum(v.square().sum() for v in grads.values()));p.require(norm>1e-24,'nonzero actual gradients')
            self.gradient_audit=dict(squared_norm=norm,sha256=tensor_sha(grads))
        self.optimizer.step();self.step_n+=1;self.rt.completed(self.name,self.step_n)
        values={k:float(v.detach()) for k,v in loss.items()}
        if self.step_n%100==0:self.losses.append(dict(step=self.step_n,**values))
        if self.name.startswith('P0/'):
            times=self.rt.get('P0_UPDATE_TIMES') or [];times.append(time.perf_counter()-started)
            draws=self.rt.get('P0_DRAWS') or {};draws.setdefault(self.name,[]).append(self.data.last_trace)
            with self.rt.db:self.rt.set('P0_UPDATE_TIMES',times);self.rt.set('P0_DRAWS',draws)
        return values
