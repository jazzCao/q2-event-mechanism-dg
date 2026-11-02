"""Uniform original source pool; consume, but never use, legacy bank draws."""
from pathlib import Path
import time
import numpy as np
import torch
import clock_real_data as parent

ROOT = parent.ROOT
OUT = ROOT/'results/wfdg/ress_clock_uniform'
STORE = Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_clock_uniform')
OUTER_GPU_DEADLINE = 1792210843.6547704


class Runtime(parent.Runtime):
    def clock(self):
        now = time.time()
        if now >= self.spec['deadline_unix'] or now >= OUTER_GPU_DEADLINE or self.gpu_start is not None and now >= self.gpu_start+1800:
            raise TimeoutError('uniform original 60min/30min/outer clocks')
    def gpu(self):
        self.clock()
        if self.gpu_start is None:
            self.gpu_start = time.time()
            parent.save(STORE/'GPU_START.json', dict(at_unix=self.gpu_start, deadline=min(self.gpu_start+1800, OUTER_GPU_DEADLINE), outer_deadline=OUTER_GPU_DEADLINE))
    def claim(self, key, n=1):
        if key == 'raw_record_loads':
            super().claim('source_record_loads' if self.phase == 'SOURCE' else 'target_only_record_loads', n)
        super().claim(key, n)


class UniformData:
    def __init__(self, case, source, rt):
        self.task = case['task']; self.classes = self.task['classes']; self.rt = rt
        self.per_cell = 120//(len(self.task['sources'])*self.classes)
        self.batch_size = self.per_cell*len(self.task['sources'])*self.classes
        self.vy = source['va']['y']; self.vg = source['va']['g']; self.vdomain = source['va']['domain']
        rt.gpu()
        self.x = torch.from_numpy(source['tr']['x']).cuda(); self.y = torch.from_numpy(source['tr']['y']).cuda()
        self.vx = torch.from_numpy(source['va']['x']).cuda(); self.pools = []; self.bpools = []
        bank_label = []; bank_domain = []
        for item in case['bank']:
            record = rt.spec['data']['records'][item['record']]
            bank_label.extend([record['label']]*4); bank_domain.extend([record['domain']]*4)
        bank_label = np.asarray(bank_label); bank_domain = np.asarray(bank_domain)
        for domain in self.task['sources']:
            for cls in range(self.classes):
                self.pools.append(np.flatnonzero((source['tr']['domain'] == domain) & (source['tr']['y'] == cls)))
                self.bpools.append(np.flatnonzero((bank_domain == domain) & (bank_label == cls)))
        if any(not len(v) for v in self.pools+self.bpools): raise ValueError('uniform/full-bank role support')
    def batch(self, arm, sampler, augmentation):
        if arm not in ('B0_ID_MIX', 'B1_IN_MIX'): raise ValueError('only two standard uniform baselines')
        full = np.concatenate([sampler.choice(pool, self.per_cell, replace=True) for pool in self.pools])
        # Match legacy RNG call order without reading/using any bank waveform.
        unused_bank = np.concatenate([sampler.choice(pool, self.per_cell, replace=True) for pool in self.bpools])
        unused_mask = sampler.uniform(size=self.batch_size) < .5
        unused_factors = augmentation.randint(0, 2, size=self.batch_size)
        idx = torch.as_tensor(full, device='cuda'); x = self.x[idx]; y = self.y[idx]
        lam = float(augmentation.beta(.2, .2)); perm = torch.as_tensor(augmentation.permutation(len(y)), device='cuda')
        return torch.cat((x, lam*x+(1-lam)*x[perm])), y, None, perm, lam
