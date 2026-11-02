"""Two control adapters; original classifier, Adam, six-B risk and draw streams."""
from datetime import datetime
import hashlib
import time

import event_component_execution as e
from source_rate_screen_runtime import Runtime as BaseRuntime,Guard
from event_full_validation_neural import Files as BaseFiles,Data as BaseData,Chain
from source_rate_neural import libraries,exact


class Runtime(BaseRuntime):
    def clock(self):
        now=time.time();campaign=self.lock['campaign']
        e.need(now<datetime.fromisoformat(campaign['deadline']).timestamp(),'component original12h preparation-inclusive deadline')
        start=self.get('neural_start')
        e.need(start is None or self.get('NEURAL_END') is not None or now<start['at_unix']+campaign['neural_seconds'],'component original cumulative5h neural deadline')


class Files(BaseFiles):
    def bank(self,case):
        import numpy as np
        value=super().bank(case)
        for key in ('amplitude_only','global_moments'):
            e.need(value[key].shape==(case['bank_windows'],2,2048) and value[key].dtype==np.float32 and np.isfinite(value[key]).all(),'complete current control bank '+key)
        return value


class Data(BaseData):
    def __init__(self,case,source,bank,rt):
        super().__init__(case,source,bank,rt)
        import torch
        for key in ('amplitude_only','global_moments'):self.bx[key]=torch.as_tensor(bank[key],device='cuda')

    def batch(self,arm,sampler,augmentation):
        import torch
        from rate_impulse_screen_neural import draw_batch
        full,bank,mask,direction,lam,permutation,trace=draw_batch(self.pools,self.bpools,self.per_cell,len(self.task['sources']),self.task['classes'],sampler,augmentation)
        e.need(self.radial_rng is not None,'same saved dedicated auxiliary random stream')
        radial=self.radial_rng.uniform(size=len(full));self.original_trace=trace
        self.last_trace=hashlib.sha256(trace.encode()+radial.tobytes()).hexdigest()
        fi,bi,mi,qi=[torch.as_tensor(v,device='cuda') for v in (full,bank,mask,direction)]
        factual,yf,ya=self.x[fi],self.y[fi],self.by[bi]
        e.need(torch.equal(yf,ya),'same independent source-bank class-balanced order')
        field,offset={'A0_IMPULSE_ONLY':('amplitude_only',0),'M0_GLOBAL_AFFINE_MOMENTS':('global_moments',0),'Q2':('coupled',2)}[arm]
        auxiliary=torch.where(mi[:,None],self.bx[field][bi,qi+offset],factual)
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
        exact(one,two,chain.torch,np,'actual fixed Q2 reference full six-B batch/labels/permutation/lambda')
        for i in (0,1):exact(streams[0][i].get_state(),streams[1][i].get_state(),chain.torch,np,'original sampler/Mixup stream state')
        exact(after,self.radial_rng.get_state(),chain.torch,np,'same auxiliary stream state')
        e.need(trace==self.last_trace,'actual original Q2 trace')
