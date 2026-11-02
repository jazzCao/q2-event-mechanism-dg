"""Portable tensor adapter for the three primary matched recipes.

This exposes training mathematics, not the original campaign's filesystem,
authorization ledger, or distributed recovery controller. No training starts
on import. Bank views must have been made from source-training contexts.
"""
import hashlib
import numpy as np
import torch
from .sampling import draw_batch


class SourceBatch:
    def __init__(self, source, bank, domains, classes, device='cpu'):
        self.domains, self.classes = list(domains), classes
        self.per_cell = 120//(len(domains)*classes)
        if self.per_cell < 1:
            raise ValueError('nonempty domain/class batch cells required')
        self.x = torch.as_tensor(source['x'],device=device)
        self.y = torch.as_tensor(source['y'],device=device)
        self.clean = torch.as_tensor(bank['clean'],device=device)
        self.coupled = torch.as_tensor(bank['coupled'],device=device)
        self.by = torch.as_tensor(bank['y'],device=device)
        if self.x.ndim!=2 or self.x.shape[1]!=2048 or self.coupled.shape!=(len(self.by),2,2048):
            raise ValueError('2048 inputs and exactly the two Q2 rates in the compact bank')
        if self.clean.shape!=(len(self.by),2048):
            raise ValueError('one clean bank view per label')
        self.pools=[];self.bpools=[]
        for d in self.domains:
            for c in range(classes):
                self.pools.append(np.flatnonzero((np.asarray(source['domain'])==d)&(np.asarray(source['y'])==c)))
                self.bpools.append(np.flatnonzero((np.asarray(bank['domain'])==d)&(np.asarray(bank['y'])==c)))
        if any(not len(v) for v in self.pools+self.bpools):
            raise ValueError('every source domain/class cell must be represented')

    def draw(self, recipe, sampler, augmentation, auxiliary_rng):
        if recipe not in ('IN_MIXUP','UNEDITED_BANK','Q2'):
            raise ValueError('only the three named primary recipes')
        full,bank,mask,direction,lam,perm,trace=draw_batch(
            self.pools,self.bpools,self.per_cell,len(self.domains),self.classes,sampler,augmentation)
        radial=auxiliary_rng.uniform(size=len(full))  # preserve original random-stream advancement
        self.last_trace=hashlib.sha256(trace.encode()+radial.tobytes()).hexdigest()
        device=self.x.device
        fi,bi,mi,qi=[torch.as_tensor(v,device=device) for v in (full,bank,mask,direction)]
        factual,yf,ya=self.x[fi],self.y[fi],self.by[bi]
        if not torch.equal(yf,ya):raise ValueError('matched class order')
        if recipe=='IN_MIXUP':auxiliary=factual
        else:
            changed=self.clean[bi] if recipe=='UNEDITED_BANK' else self.coupled[bi,qi]
            auxiliary=torch.where(mi[:,None],changed,factual)
        perm=torch.as_tensor(perm,device=device)
        mixed=lam*auxiliary+(1-lam)*auxiliary[perm]
        return torch.cat([factual,factual,factual,mixed,mixed,mixed]),(yf,torch.where(mi,ya,yf)),perm,lam
