"""Same Table-3 classifier in every arm; generation is source-only offline."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from wfdg.mocd_model import EncoderGTable3, ClfHead


class Net(nn.Module):
    def __init__(self, classes, arm):
        super().__init__(); self.arm = arm; self.enc = EncoderGTable3(2048); self.clf = ClfHead(classes)
    def forward(self, x):
        h = self.enc.net(x[:, None] if x.ndim == 2 else x)
        if self.arm != 'B0_ID_MIX':
            h = (h-h.mean(-1, keepdim=True))/torch.sqrt(h.var(-1, unbiased=False, keepdim=True)+1e-5)
        logits = self.clf(self.enc.fc(h))
        return dict(logits=logits, r=None, z=None)
    def losses(self, out, y, c, permutation, lam):
        n = len(y); q = out['logits']; clean = F.cross_entropy(q[:n], y)
        mixed = lam*F.cross_entropy(q[n:], y)+(1-lam)*F.cross_entropy(q[n:], y[permutation])
        return dict(total=.5*(clean+mixed), clean_ce=clean, mixup_ce=mixed)


class Data:
    def __init__(self, case, source, bank, rt):
        self.task = case['task']; self.classes = self.task['classes']; self.rt = rt
        self.eligible = case['eligible']; self.vy = source['va']['y']; self.vg = source['va']['g']; self.vdomain = source['va']['domain']
        self.per_cell = 120//(len(self.task['sources'])*self.classes)
        self.batch_size = self.per_cell*len(self.task['sources'])*self.classes
        self.x = torch.from_numpy(source['tr']['x']).cuda(); self.y = torch.from_numpy(source['tr']['y']).cuda()
        self.vx = torch.from_numpy(source['va']['x']).cuda()
        self.bx = torch.from_numpy(bank['clean']).cuda(); self.by = torch.from_numpy(bank['y']).cuda()
        self.transforms = {k: torch.from_numpy(bank[k]).cuda() for k in ('warp', 'joint', 'event')}
        self.pools = []; self.bpools = []
        for domain in self.task['sources']:
            for cls in range(self.classes):
                self.pools.append(np.flatnonzero((source['tr']['domain'] == domain) & (source['tr']['y'] == cls)))
                self.bpools.append(np.flatnonzero((bank['domain'] == domain) & (bank['y'] == cls)))
        if any(not len(v) for v in self.pools+self.bpools): raise ValueError('empty balanced full/bank pool')
    def batch(self, arm, sampler, augmentation):
        full = np.concatenate([sampler.choice(v, self.per_cell, replace=True) for v in self.pools])
        bank = np.concatenate([sampler.choice(v, self.per_cell, replace=True) for v in self.bpools])
        use_bank = sampler.uniform(size=self.batch_size) < .5
        device = lambda v: torch.as_tensor(v, device='cuda')
        mask = device(use_bank); bi = device(bank); fi = device(full)
        x = torch.where(mask[:, None], self.bx[bi], self.x[fi]); y = torch.where(mask, self.by[bi], self.y[fi])
        factors = device(augmentation.randint(0, 2, size=self.batch_size))
        kind = {'N0_WARP': 'warp', 'C0_CLOCK': 'joint', 'E0_EVENTS': 'event'}.get(arm)
        # All arms consume identical random draws, including non-augmented arms.
        changed = self.transforms[kind][bi, factors] if kind is not None else self.bx[bi]
        augmented = torch.where(mask[:, None], changed, x)
        lam = float(augmentation.beta(.2, .2)); permutation = device(augmentation.permutation(len(y)))
        mixed = lam*augmented+(1-lam)*augmented[permutation]
        return torch.cat((x, mixed)), y, None, permutation, lam
