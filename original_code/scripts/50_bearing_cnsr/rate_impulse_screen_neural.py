"""New bank adapter and shared6B mean risk; exact original classifier/optimizer."""
import hashlib
import time
from source_rate_screen_runtime import Runtime,Guard
from source_rate_screen_data import Files as BaseFiles,check_vectors
from source_rate_neural import Chain as BaseChain,libraries,tensor_sha
import rate_impulse_screen_plan as p


def draw_batch(pools,bpools,per_cell,domains,classes,sampler,augmentation):
    import numpy as np
    from source_rate_core import shared_edit_mask
    full=np.concatenate([sampler.choice(v,per_cell,replace=True) for v in pools])
    bank=np.concatenate([sampler.choice(v,per_cell,replace=True) for v in bpools])
    mask=shared_edit_mask(sampler.uniform(size=len(full)),domains,classes,per_cell)
    factors=augmentation.randint(0,2,size=len(full))
    lam=float(augmentation.beta(.2,.2));permutation=augmentation.permutation(len(full))
    trace=hashlib.sha256()
    for v in (full,bank,mask,factors,permutation):trace.update(np.asarray(v).tobytes())
    trace.update(lam.hex().encode())
    return full,bank,mask,factors,lam,permutation,trace.hexdigest()


class Files(BaseFiles):
    def bank(self,case):
        import numpy as np
        vectors=dict(y=[],g=[],domain=[],offset=[])
        for item in case['bank']:
            record=self.lock['records'][item['record']]
            p.require(item['domain'] in case['task']['sources'] and record.get('bearing') not in ('K006','KA08','KA09'),'legal source bank')
            from design_e0_natural_source_probe_v2 import centered_three
            for k,v in (('y',record['label']),('g',item['record']),('domain',record['domain'])):vectors[k].extend([v]*3)
            vectors['offset'].extend([65536+s for s in centered_three(32768,case['task']['window'])])
        key=case['key']
        if key not in self.banks:
            b=self.lock['banks'][key];rt=self.rt
            p.require(rt.hash(b['path'],'bank_hashes')==b['sha256'],'current entire source bank SHA')
            rt.claim('bank_decodes');rt.permit(b['path'])
            try:
                with np.load(b['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            finally:rt.revoke(b['path'])
            for k in ('clean','fixed','coupled','global_gain','event_gain','inverse_amplitude'):
                shape=(case['bank_windows'],2048) if k=='clean' else (case['bank_windows'],2,2048)
                p.require(a[k].shape==shape and a[k].dtype==np.float32 and np.isfinite(a[k]).all(),'all finite complete bank '+k)
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
        self.bx={k:torch.as_tensor(bank[k],device='cuda') for k in ('clean','coupled','global_gain','event_gain','inverse_amplitude')}
        self.by=torch.as_tensor(bank['y'],device='cuda');self.pools=[];self.bpools=[]
        for d in self.task['sources']:
            for y in range(self.task['classes']):
                self.pools.append(np.flatnonzero((source['tr']['domain']==d)&(source['tr']['y']==y)))
                self.bpools.append(np.flatnonzero((bank['domain']==d)&(bank['y']==y)))
        p.require(all(len(v) for v in self.pools+self.bpools),'all original domain/class full/bank coverage')

    def batch(self,arm,sampler,augmentation):
        import numpy as np
        import torch
        full,bank,mask,factors,lam,permutation,self.last_trace=draw_batch(self.pools,self.bpools,self.per_cell,
            len(self.task['sources']),self.task['classes'],sampler,augmentation)
        fi,bi,mi,qi=[torch.as_tensor(v,device='cuda') for v in (full,bank,mask,factors)]
        factual,yf,ya=self.x[fi],self.y[fi],self.by[bi]
        p.require(torch.equal(yf,ya),'same independent full/aux balanced class order')
        name=dict(zip(p.ARMS,('coupled','global_gain','event_gain','inverse_amplitude','clean')))[arm]
        changed=self.bx[name][bi] if name=='clean' else self.bx[name][bi,qi]
        auxiliary=torch.where(mi[:,None],changed,factual);auxiliary_y=torch.where(mi,ya,yf)
        permutation=torch.as_tensor(permutation,device='cuda')
        mixed=lam*auxiliary+(1-lam)*auxiliary[permutation]
        return torch.cat([factual,factual,factual,mixed,mixed,mixed]),(yf,auxiliary_y),permutation,lam


class Chain(BaseChain):
    def step(self,inspect=False):
        from compatible_set_core import risk_loss
        torch=self.torch;started=time.perf_counter();self.rt.attempt(self.name,self.step_n+1);self.model.train()
        x,labels,permutation,lam=self.data.batch(self.arm,self.sampler,self.augmentation)
        self.rt.claim('training_forward_windows',len(x))
        loss=risk_loss(torch,self.model(x)['logits'],labels,permutation,lam,False)
        p.require(all(torch.isfinite(v) for v in loss.values()),'finite common mean risk')
        self.optimizer.zero_grad(set_to_none=True);loss['total'].backward()
        if inspect:
            grads={k:v.grad for k,v in self.model.named_parameters()}
            p.require(all(v is not None and torch.isfinite(v).all() for v in grads.values()),'finite all parameter gradients')
            norm=float(sum(v.square().sum() for v in grads.values()));p.require(norm>1e-24,'positive total gradient norm')
            self.gradient_audit=dict(squared_norm=norm,sha256=tensor_sha(grads))
        self.optimizer.step();self.step_n+=1;self.rt.completed(self.name,self.step_n)
        values={k:float(v.detach()) for k,v in loss.items()}
        if self.step_n%100==0:self.losses.append(dict(step=self.step_n,**values))
        if self.name.startswith('P0/'):
            times=self.rt.get('P0_UPDATE_TIMES') or [];times.append(time.perf_counter()-started)
            draws=self.rt.get('P0_DRAWS') or {};draws.setdefault(self.name,[]).append(self.data.last_trace)
            with self.rt.db:self.rt.set('P0_UPDATE_TIMES',times);self.rt.set('P0_DRAWS',draws)
        return values
