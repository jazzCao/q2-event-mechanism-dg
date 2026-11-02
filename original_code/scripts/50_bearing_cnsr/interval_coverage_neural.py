"""Shared six-B mean-risk adapters; isolated radial RNG saved in original state."""
import hashlib
import interval_coverage_plan as p
from source_rate_screen_data import Files as BaseFiles,check_vectors
from rate_impulse_screen_neural import Chain as BaseChain,draw_batch
from source_rate_neural import libraries,exact


class Files(BaseFiles):
    def bank(self,case):
        import numpy as np
        key=case['key'];rt=self.rt
        if key not in self.banks:
            b=rt.lock['banks'][key];p.require(rt.hash(b['path'],'bank_hashes')==b['sha256'],'whole calibrated source bank SHA')
            rt.claim('bank_decodes');rt.permit(b['path'])
            try:
                with np.load(b['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            finally:rt.revoke(b['path'])
            for k in ('clean','fixed','coupled','ntime','nfront'):
                shape=(case['bank_windows'],2048) if k=='clean' else (case['bank_windows'],8 if k=='coupled' else 2,2048)
                p.require(a[k].shape==shape and a[k].dtype==np.float32 and np.isfinite(a[k]).all(),'complete calibrated classifier bank '+k)
            self.banks[key]=a
        vectors={k:[] for k in ('y','g','domain','offset')}
        for item in case['bank']:
            p.require(item['domain'] in case['task']['sources'] and self.lock['records'][item['record']].get('bearing') not in ('K006','KA08','KA09'),'current source recording before any bank reuse')
            from design_e0_natural_source_probe_v2 import centered_three
            for k,v in (('y',item['label']),('g',item['record']),('domain',item['domain'])):vectors[k].extend([v]*3)
            vectors['offset'].extend([65536+s for s in centered_three(32768,case['task']['window'])])
        check_vectors(self.banks[key],vectors);return self.banks[key]


class Data:
    def __init__(self,case,source,bank,rt):
        np,torch,_,_=libraries(rt);self.task=case['task'];self.rt=rt
        self.per_cell=120//(len(self.task['sources'])*self.task['classes']);self.batch_size=self.per_cell*len(self.task['sources'])*self.task['classes']
        self.x=torch.as_tensor(source['tr']['x'],device='cuda');self.y=torch.as_tensor(source['tr']['y'],device='cuda')
        self.bx={k:torch.as_tensor(bank[k],device='cuda') for k in ('fixed','coupled','ntime','nfront')};self.by=torch.as_tensor(bank['y'],device='cuda')
        self.pools=[];self.bpools=[];self.radial_rng=None
        for d in self.task['sources']:
            for y in range(self.task['classes']):
                self.pools.append(np.flatnonzero((source['tr']['domain']==d)&(source['tr']['y']==y)))
                self.bpools.append(np.flatnonzero((bank['domain']==d)&(bank['y']==y)))
        p.require(all(len(v) for v in self.pools+self.bpools),'entire balanced source pools')

    def batch(self,arm,sampler,augmentation):
        import numpy as np
        import torch
        from interval_coverage_core import choice
        full,bank,mask,direction,lam,permutation,original_trace=draw_batch(self.pools,self.bpools,self.per_cell,len(self.task['sources']),self.task['classes'],sampler,augmentation)
        p.require(self.radial_rng is not None,'dedicated restored radial RNG')
        radial=self.radial_rng.uniform(size=len(full));indices=choice(arm,direction,radial)
        self.original_trace=original_trace;self.last_trace=hashlib.sha256(original_trace.encode()+radial.tobytes()).hexdigest()
        fi,bi,mi,qi=[torch.as_tensor(v,device='cuda') for v in (full,bank,mask,indices)]
        factual,yf,ya=self.x[fi],self.y[fi],self.by[bi];p.require(torch.equal(yf,ya),'same balanced auxiliary class order')
        if arm==p.ARMS[0]:auxiliary=factual
        else:
            key='fixed' if arm==p.ARMS[1] else 'ntime' if arm=='NTIME' else 'nfront' if arm=='NFRONT' else 'coupled'
            auxiliary=torch.where(mi[:,None],self.bx[key][bi,qi],factual)
        permutation=torch.as_tensor(permutation,device='cuda');mixed=lam*auxiliary+(1-lam)*auxiliary[permutation]
        return torch.cat([factual,factual,factual,mixed,mixed,mixed]),(yf,torch.where(mi,ya,yf)),permutation,lam

    def original_batch_check(self,chain):
        import numpy as np
        from rate_impulse_repeat_neural import Data as Original
        p.require(chain.arm in p.ARMS[:3],'three unchanged adapter anchors')
        streams=[]
        for _ in (0,1):
            s,a=np.random.RandomState(),np.random.RandomState();s.set_state(chain.sampler.get_state());a.set_state(chain.augmentation.get_state());streams.append((s,a))
        radial=np.random.RandomState();radial.set_state(chain.auxiliary.get_state());self.radial_rng=radial
        self.rt.claim('batch_reference_rows',12*self.batch_size)
        one=self.batch(chain.arm,*streams[0]);newtrace=self.original_trace
        oldarm=chain.arm if chain.arm in p.ARMS[:2] else 'C0_RATE_IMPULSE'
        two=Original.batch(self,oldarm,*streams[1])
        exact(one,two,chain.torch,np,'same current bank anchor adapter exact original six-B batch')
        exact(streams[0][0].get_state(),streams[1][0].get_state(),chain.torch,np,'sampler state unchanged by radial stream')
        exact(streams[0][1].get_state(),streams[1][1].get_state(),chain.torch,np,'augmentation state unchanged by radial stream')
        p.require(newtrace==self.last_trace,'original sampler/bank/direction/Mixup trace matches')


class Chain(BaseChain):
    def step(self,inspect=False):
        self.data.radial_rng=self.auxiliary
        return super().step(inspect)
