"""Three matched6B arms; same sampler draws and original mean-risk Chain."""
from rate_impulse_screen_neural import Chain,Files,draw_batch
from source_rate_neural import libraries,exact
import rate_impulse_repeat_plan as p


class Data:
    def __init__(self,case,source,bank,rt):
        np,torch,_,_=libraries(rt);self.task=case['task'];self.rt=rt
        self.per_cell=120//(len(self.task['sources'])*self.task['classes']);self.batch_size=self.per_cell*len(self.task['sources'])*self.task['classes']
        self.x=torch.as_tensor(source['tr']['x'],device='cuda');self.y=torch.as_tensor(source['tr']['y'],device='cuda')
        self.bx={k:torch.as_tensor(bank[k],device='cuda') for k in ('fixed','coupled')};self.by=torch.as_tensor(bank['y'],device='cuda')
        self.pools=[];self.bpools=[]
        for d in self.task['sources']:
            for y in range(self.task['classes']):
                self.pools.append(np.flatnonzero((source['tr']['domain']==d)&(source['tr']['y']==y)))
                self.bpools.append(np.flatnonzero((bank['domain']==d)&(bank['y']==y)))
        p.require(all(len(v) for v in self.pools+self.bpools),'all original balanced full/bank cells')

    def batch(self,arm,sampler,augmentation):
        import torch
        full,bank,mask,factors,lam,permutation,self.last_trace=draw_batch(self.pools,self.bpools,self.per_cell,len(self.task['sources']),self.task['classes'],sampler,augmentation)
        fi,bi,mi,qi=[torch.as_tensor(v,device='cuda') for v in (full,bank,mask,factors)]
        factual,yf,ya=self.x[fi],self.y[fi],self.by[bi];p.require(torch.equal(yf,ya),'same balanced auxiliary class order')
        p.require(arm in p.ARMS,'only fixed three arms')
        auxiliary=factual if arm==p.ARMS[0] else torch.where(mi[:,None],self.bx['fixed' if arm==p.ARMS[1] else 'coupled'][bi,qi],factual)
        permutation=torch.as_tensor(permutation,device='cuda');mixed=lam*auxiliary+(1-lam)*auxiliary[permutation]
        return torch.cat([factual,factual,factual,mixed,mixed,mixed]),(yf,torch.where(mi,ya,yf)),permutation,lam

    def original_batch_check(self,chain):
        import numpy as np
        from compatible_set_neural import Data as Original
        p.require(chain.arm in p.ARMS[:2],'two unchanged anchors only')
        streams=[]
        for _ in (0,1):
            s,a=np.random.RandomState(),np.random.RandomState();s.set_state(chain.sampler.get_state());a.set_state(chain.augmentation.get_state());streams.append((s,a))
        self.rt.claim('batch_reference_rows',12*self.batch_size)
        one=self.batch(chain.arm,*streams[0]);two=Original.batch(self,chain.arm,*streams[1])
        exact(one,two,chain.torch,np,'real source seed2 six-view batch equals original anchor implementation')
