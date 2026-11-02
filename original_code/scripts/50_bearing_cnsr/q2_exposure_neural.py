"""Unchanged current Q2 banks, network, loss and optimizer; exposure adapter only."""
import hashlib
import time
import q2_exposure_plan as p
from q2_exposure_core import replacement
from event_full_validation_neural import Data as BaseData,Files,Guard
from interval_coverage_neural import Chain as BaseChain
from source_rate_screen_runtime import Runtime as BaseRuntime
from rate_impulse_screen_neural import draw_batch
from source_rate_neural import exact

class Runtime(BaseRuntime):
    def clock(self):
        p.clock(self.lock['campaign']);start=self.get('neural_start')
        p.need(start is None or self.get('NEURAL_END') is not None or
            time.time()<start['at_unix']+self.lock['campaign']['neural_seconds'],'same cumulative2h neural clock')

class Data(BaseData):
    def batch(self,arm,sampler,augmentation):
        import torch
        full,bank,mask,direction,lam,permutation,trace=draw_batch(self.pools,self.bpools,self.per_cell,
            len(self.task['sources']),self.task['classes'],sampler,augmentation)
        p.need(self.radial_rng is not None,'saved existing private radial stream')
        radial=self.radial_rng.uniform(size=len(full));self.original_trace=trace
        self.last_trace=hashlib.sha256(trace.encode()+radial.tobytes()).hexdigest()
        fi,bi,mi,qi=[torch.as_tensor(v,device='cuda') for v in (full,bank,mask,direction)]
        factual,yf,ya=self.x[fi],self.y[fi],self.by[bi]
        p.need(torch.equal(yf,ya),'same full/bank class order')
        changed,keep=replacement(arm,self.bx['clean'][bi],self.bx['coupled'][bi,qi+2],radial,
            len(self.task['sources']),self.task['classes'],self.per_cell,
            to_mask=lambda v:torch.as_tensor(v,device='cuda'),where=torch.where)
        self.last_exposure_counts=dict(batch_slots=len(full),bank_slots=int(mask.sum()),edited_slots=int((mask&keep).sum()))
        auxiliary=torch.where(mi[:,None],changed,factual)
        permutation=torch.as_tensor(permutation,device='cuda');mixed=lam*auxiliary+(1-lam)*auxiliary[permutation]
        return torch.cat([factual,factual,factual,mixed,mixed,mixed]),(yf,torch.where(mi,ya,yf)),permutation,lam

    def reference_batch_check(self,chain):
        import numpy as np
        streams=[]
        for _ in (0,1):
            a,b=np.random.RandomState(),np.random.RandomState();a.set_state(chain.sampler.get_state());b.set_state(chain.augmentation.get_state());streams.append((a,b))
        def radial():
            value=np.random.RandomState();value.set_state(chain.auxiliary.get_state());return value
        self.rt.claim('batch_reference_rows',12*self.batch_size)
        self.radial_rng=radial();one=self.batch('Q2',*streams[0]);trace=self.last_trace;after=self.radial_rng.get_state()
        self.radial_rng=radial();two=BaseData.batch(self,'Q2',*streams[1])
        exact(one,two,chain.torch,np,'accept-all adapter exact unchanged current Q2 six-B tensors/labels')
        for i in (0,1):exact(streams[0][i].get_state(),streams[1][i].get_state(),chain.torch,np,'same sampler/Mixup RNG')
        exact(after,self.radial_rng.get_state(),chain.torch,np,'same private radial RNG')
        p.need(trace==self.last_trace,'same full original draw trace')

class Chain(BaseChain):
    def step(self,inspect=False):
        value=super().step(inspect)
        if self.step_n%100==0:self.losses[-1].update(self.data.last_exposure_counts)
        return value
