"""Native CCDG-family engine component; a valid finite permit is mandatory.

Not a CLI and not authorization. No model is constructed by importing this file.
"""
from __future__ import annotations
import copy
import hashlib
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from wfdg.ress_ccdg_matched.author import author_net,contrast
from mocd_audit_cpu_view_v2 import cpu_view_preserving_type

CONFIGS=('A0_AUTHOR_CE','A2_AUTHOR_IN_MIXUP','C0_CCDG_CODE','C1_CCDG_LOGITS_SENS')


def seed_for(text,nibbles=8):return int(hashlib.sha256(text.encode()).hexdigest()[:nibbles],16)


class NativeNetwork(nn.Module):
    def __init__(self,classes,with_in=False):
        super().__init__();self.featurizer,self.classifier=author_net(464,num_classes=classes,with_in=with_in)
    def forward(self,x):return dict(logits=self.classifier(self.featurizer(x)))


class NativeAuthorEngine:
    def __init__(self,sources,task,seed,config,fingerprint,permit,rt,maximum_steps=3001,validation_steps=(0,1000,2000,3001)):
        if permit is None:raise PermissionError('new finite native authorization required before model construction')
        permit.require_neural(fingerprint)
        if config not in CONFIGS or type(seed) is not int or not fingerprint:raise ValueError('explicit native configuration/seed/fingerprint')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU-training fallback')
        if tuple(task['sources'])!=sources.allowed or task['target'] not in sources.forbidden:raise ValueError('bound source task')
        if maximum_steps not in (10,3001) or tuple(validation_steps)!=(0,10) and tuple(validation_steps)!=(0,1000,2000,3001):raise ValueError('registered formal/P0 native schedule only')
        if max(validation_steps)!=maximum_steps:raise ValueError('validation schedule end')
        self.sources=sources;self.task=copy.deepcopy(task);self.seed=seed;self.config=config;self.fingerprint=fingerprint;self.permit=permit;self.rt=rt
        self.maximum_steps=maximum_steps;self.validation_steps=tuple(validation_steps);self.step_n=0;self.logs=[];self.curve=[];self.best_score=None;self.best_step=None
        initial=seed_for(f'EVENT_AUTHOR_V1|INIT|{task["fold"]}|{seed}')
        random.seed(initial);np.random.seed(initial);torch.manual_seed(initial);torch.cuda.manual_seed_all(initial)
        self.model=NativeNetwork(sources.classes,config=='A2_AUTHOR_IN_MIXUP').cuda()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=5e-5)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=2400,gamma=.1)
        self.batch_rng=np.random.Generator(np.random.PCG64(seed_for(f'EVENT_AUTHOR_V1|BATCH|{task["fold"]}|{seed}',16)))
        self.mix_rng=np.random.Generator(np.random.PCG64(seed_for(f'EVENT_AUTHOR_V1|MIX|{task["fold"]}|{seed}',16)))
        self.draws=0

    def step(self,inspect=False):
        if self.step_n>=self.maximum_steps:raise PermissionError('native chain complete; no extra update')
        self.permit.require_neural(self.fingerprint);self.permit.claim(self.fingerprint,self.step_n+1,'B')
        self.rt.claim('native_optimizer_attempts');self.model.train();samples=[];labels=[];indices=[]
        for domain in self.sources.allowed:
            part=self.sources.get(domain,'train');pick=self.batch_rng.integers(0,len(part.y),size=32);indices.append(pick.tolist());self.draws+=32
            samples.append(part.x[pick]);labels.append(part.y[pick])
        x=torch.cat(samples).cuda();y=torch.as_tensor(np.concatenate(labels),device='cuda',dtype=torch.long);n=len(y)
        contrastive=x.new_zeros(());mixed_ce=x.new_zeros(())
        if self.config=='A2_AUTHOR_IN_MIXUP':
            lam=float(self.mix_rng.beta(.2,.2));perm=torch.as_tensor(self.mix_rng.permutation(n),device='cuda');mixed=lam*x+(1-lam)*x[perm]
            logits=self.model(torch.cat((x,mixed)))['logits'];clean_ce=F.cross_entropy(logits[:n],y)
            mixed_ce=lam*F.cross_entropy(logits[n:],y)+(1-lam)*F.cross_entropy(logits[n:],y[perm]);loss=.5*(clean_ce+mixed_ce);windows=2*n
        else:
            feat=self.model.featurizer(x);logits=self.model.classifier(feat);clean_ce=F.cross_entropy(logits,y);windows=n
            if self.config.startswith('C'):
                contrastive=contrast(feat if self.config=='C0_CCDG_CODE' else logits,y,.7);loss=.9*clean_ce+.1*contrastive
            else:loss=clean_ce
        self.rt.claim('native_training_forward_windows',windows)
        if not torch.isfinite(loss):raise FloatingPointError('native loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        if inspect:
            gradients=[p.grad for p in self.model.parameters()]
            if any(g is None or not torch.isfinite(g).all() for g in gradients) or sum(float(g.square().sum()) for g in gradients)<=1e-24:raise AssertionError('actual finite nonzero native gradients')
        lr=float(self.optimizer.param_groups[0]['lr']);self.optimizer.step();self.scheduler.step();self.step_n+=1;self.rt.claim('actual_optimizer_updates')
        row=dict(step=self.step_n,ce=float(clean_ce.detach()),mixup_ce=float(mixed_ce.detach()),contrast=float(contrastive.detach()),total=float(loss.detach()),lr=lr)
        if self.step_n%100==0 or self.maximum_steps==10 or self.step_n==self.maximum_steps:self.logs.append(row)
        return row,indices

    def consider(self,row):
        if row['step']!=self.step_n or self.step_n not in self.validation_steps or any(v['step']==self.step_n for v in self.curve):raise ValueError('registered unique source validation point')
        if not 0<=row['score']<=1 or not np.isfinite(row['score']) or row.get('target_used_for_selection',False):raise ValueError('finite source-only selection score')
        if self.best_score is None or row['score']>self.best_score:self.best_score=float(row['score']);self.best_step=self.step_n
        self.curve.append(copy.deepcopy(row))

    def packet(self):
        value=dict(schema='EVENT_NATIVE_AUTHOR_STATE_V1',fingerprint=self.fingerprint,source_fingerprint=self.sources.fingerprint,task=self.task,seed=self.seed,config=self.config,
                   maximum_steps=self.maximum_steps,validation_steps=self.validation_steps,step=self.step_n,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),
                   model_modes={k:m.training for k,m in self.model.named_modules()},batch_rng=self.batch_rng.bit_generator.state,mix_rng=self.mix_rng.bit_generator.state,draws=self.draws,
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),logs=self.logs,curve=self.curve,best_score=self.best_score,best_step=self.best_step)
        return cpu_view_preserving_type(copy.deepcopy(value))

    def restore(self,packet):
        self.permit.require_neural(self.fingerprint)
        identity=dict(schema='EVENT_NATIVE_AUTHOR_STATE_V1',fingerprint=self.fingerprint,source_fingerprint=self.sources.fingerprint,task=self.task,seed=self.seed,config=self.config,maximum_steps=self.maximum_steps,validation_steps=self.validation_steps)
        if any(packet.get(k)!=v for k,v in identity.items()):raise ValueError('native complete state identity')
        step=packet['step']
        if type(step) is not int or not 0<=step<=self.maximum_steps or packet['draws']!=32*len(self.sources.allowed)*step:raise ValueError('native step/draw cursor')
        expected=[v for v in self.validation_steps if v<=step]
        if [r['step'] for r in packet['curve']]!=expected:raise ValueError('complete registered source selection curve')
        chosen=min(packet['curve'],key=lambda r:(-r['score'],r['step']))
        if (packet['best_step'],packet['best_score'])!=(chosen['step'],chosen['score']):raise ValueError('earliest strict source-best native state')
        if packet['scheduler']['last_epoch']!=step:raise ValueError('native scheduler cursor')
        if step and (not packet['optimizer']['state'] or any(int(v['step'])!=step for v in packet['optimizer']['state'].values())):raise ValueError('native Adam cursor')
        if set(packet['model_modes'])!=set(dict(self.model.named_modules())):raise ValueError('native module behavior coverage')
        self.model.load_state_dict(packet['model']);self.optimizer.load_state_dict(packet['optimizer']);self.scheduler.load_state_dict(packet['scheduler'])
        for k,m in self.model.named_modules():m.training=packet['model_modes'][k]
        self.batch_rng.bit_generator.state=copy.deepcopy(packet['batch_rng']);self.mix_rng.bit_generator.state=copy.deepcopy(packet['mix_rng']);self.draws=packet['draws'];self.step_n=step
        self.logs=copy.deepcopy(packet['logs']);self.curve=copy.deepcopy(packet['curve']);self.best_score=packet['best_score'];self.best_step=packet['best_step']
        torch.set_rng_state(packet['torch_rng']);torch.cuda.set_rng_state_all(packet['cuda_rng']);np.random.set_state(packet['numpy_rng']);random.setstate(packet['python_rng'])
