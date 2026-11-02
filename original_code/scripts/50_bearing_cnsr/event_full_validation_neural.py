"""Matched five-arm adapters and provenance-checked archived probabilities."""
from pathlib import Path
import hashlib
import time
import event_full_validation_plan as p
from source_rate_screen_runtime import Runtime as BaseRuntime,Guard
from source_rate_screen_data import Files as BaseFiles,check_vectors
from source_rate_neural import libraries,exact
from interval_coverage_neural import Chain
from rate_impulse_screen_neural import draw_batch


def selection(arm):
    return {'A0_STRONG_IN_MIXUP':('factual',0),'F0_UNEDITED_BANK':('clean',0),'A1_MATCHED_E0':('fixed',0),
            'Q2_FIXED_IMPULSE':('interior_fixed',0),'Q2':('coupled',2)}[arm]


class Runtime(BaseRuntime):
    def archive(self,b):
        p.need(self.get('SOURCE_FREEZE') and self.phase=='AUDIT','archive probabilities only after all new source freeze')
        self.claim('archive_probability_reads');self.claim('archive_prediction_windows_read',b['n'])
        p.need(self.hash(b['path'],'archive_probability_hashes')==b['sha256'],'old saved probability SHA')
        events=self.get('ARCHIVE_READS') or [];events.append(dict(path=b['path'],at_unix=time.time(),parent_fingerprint=b['parent_fingerprint']))
        with self.db:self.set('ARCHIVE_READS',events)
        self.permit(b['path'])
        try:
            import numpy as np
            with np.load(b['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
        finally:self.revoke(b['path'])


class Files(BaseFiles):
    def bank(self,case):
        import numpy as np
        key=case['key'];rt=self.rt
        if key not in self.banks:
            b=self.lock['banks'][key];p.need(rt.hash(b['path'],'bank_hashes')==b['sha256'],'whole actual source-bank SHA')
            rt.claim('bank_decodes');rt.permit(b['path'])
            try:
                with np.load(b['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            finally:rt.revoke(b['path'])
            for k,n in (('clean',0),('fixed',2),('interior_fixed',2),('coupled',4)):
                shape=(case['bank_windows'],2048) if not n else (case['bank_windows'],n,2048)
                p.need(a[k].shape==shape and a[k].dtype==np.float32 and np.isfinite(a[k]).all(),'full finite source '+k)
            self.banks[key]=a
        vectors={k:[] for k in ('y','g','domain','offset')}
        from design_e0_natural_source_probe_v2 import centered_three
        for item in case['bank']:
            p.need(item['domain'] in case['task']['sources'] and self.lock['records'][item['record']].get('bearing') not in ('K006','KA08','KA09'),'current source case even when cached')
            for k,v in (('y',item['label']),('g',item['record']),('domain',item['domain'])):vectors[k].extend([v]*3)
            vectors['offset'].extend([65536+v for v in centered_three(32768,case['task']['window'])])
        check_vectors(self.banks[key],vectors);return self.banks[key]


class Data:
    def __init__(self,case,source,bank,rt):
        np,torch,_,_=libraries(rt);self.rt=rt;self.task=case['task'];self.radial_rng=None
        self.per_cell=120//(len(self.task['sources'])*self.task['classes']);self.batch_size=self.per_cell*len(self.task['sources'])*self.task['classes']
        self.x=torch.as_tensor(source['tr']['x'],device='cuda');self.y=torch.as_tensor(source['tr']['y'],device='cuda')
        self.bx={k:torch.as_tensor(bank[k],device='cuda') for k in ('clean','fixed','interior_fixed','coupled')};self.by=torch.as_tensor(bank['y'],device='cuda')
        self.pools=[];self.bpools=[]
        for d in self.task['sources']:
            for y in range(self.task['classes']):
                self.pools.append(np.flatnonzero((source['tr']['domain']==d)&(source['tr']['y']==y)))
                self.bpools.append(np.flatnonzero((bank['domain']==d)&(bank['y']==y)))
        p.need(all(len(v) for v in self.pools+self.bpools),'all original source domain/class sampling cells')

    def batch(self,arm,sampler,augmentation):
        import torch
        full,bank,mask,direction,lam,permutation,trace=draw_batch(self.pools,self.bpools,self.per_cell,len(self.task['sources']),self.task['classes'],sampler,augmentation)
        p.need(self.radial_rng is not None,'saved dedicated auxiliary RNG')
        radial=self.radial_rng.uniform(size=len(full));self.original_trace=trace;self.last_trace=hashlib.sha256(trace.encode()+radial.tobytes()).hexdigest()
        fi,bi,mi,qi=[torch.as_tensor(v,device='cuda') for v in (full,bank,mask,direction)]
        factual,yf,ya=self.x[fi],self.y[fi],self.by[bi];p.need(torch.equal(yf,ya),'independent same-class source views')
        field,offset=selection(arm)
        if field=='factual':auxiliary=factual
        else:
            changed=self.bx[field][bi] if field=='clean' else self.bx[field][bi,qi+offset]
            auxiliary=torch.where(mi[:,None],changed,factual)
        permutation=torch.as_tensor(permutation,device='cuda');mixed=lam*auxiliary+(1-lam)*auxiliary[permutation]
        return torch.cat([factual,factual,factual,mixed,mixed,mixed]),(yf,torch.where(mi,ya,yf)),permutation,lam

    def original_batch_check(self,chain):
        import numpy as np
        from interval_coverage_neural import Data as Old
        p.need(chain.arm in ('A0_STRONG_IN_MIXUP','A1_MATCHED_E0','Q2'),'only three unchanged reference adapters')
        streams=[]
        for _ in (0,1):
            s,a=np.random.RandomState(),np.random.RandomState();s.set_state(chain.sampler.get_state());a.set_state(chain.augmentation.get_state());streams.append((s,a))
        self.rt.claim('batch_reference_rows',12*self.batch_size)
        def radial():
            value=np.random.RandomState();value.set_state(chain.auxiliary.get_state());return value
        self.radial_rng=radial();one=self.batch(chain.arm,*streams[0]);trace=self.last_trace
        self.radial_rng=radial();two=Old.batch(self,chain.arm,*streams[1])
        exact(one,two,chain.torch,np,'actual old/new adapter full6B tensors and labels')
        for i in (0,1):exact(streams[0][i].get_state(),streams[1][i].get_state(),chain.torch,np,'unchanged actual random stream')
        p.need(trace==self.last_trace,'same actual sampler/Mixup/radial trace')
