"""Balanced sampler and six-copy mean risk from the frozen recipe."""
import hashlib
import numpy as np

def require(value, message):
    if not value: raise ValueError(message)

def shared_edit_mask(uniforms, domains, classes, per_cell):
    values = np.asarray(uniforms)
    require(values.shape == (domains * classes * per_cell,), 'same old full-length uniform draw')
    first = (values.reshape(domains, classes, per_cell)[0] < .5)
    return np.broadcast_to(first[None], (domains, classes, per_cell)).reshape(-1).copy()

def draw_batch(pools,bpools,per_cell,domains,classes,sampler,augmentation):
    import numpy as np
    full=np.concatenate([sampler.choice(v,per_cell,replace=True) for v in pools])
    bank=np.concatenate([sampler.choice(v,per_cell,replace=True) for v in bpools])
    mask=shared_edit_mask(sampler.uniform(size=len(full)),domains,classes,per_cell)
    factors=augmentation.randint(0,2,size=len(full))
    lam=float(augmentation.beta(.2,.2));permutation=augmentation.permutation(len(full))
    trace=hashlib.sha256()
    for v in (full,bank,mask,factors,permutation):trace.update(np.asarray(v).tobytes())
    trace.update(lam.hex().encode())
    return full,bank,mask,factors,lam,permutation,trace.hexdigest()

def initial_seed(task, seed):
    token = f'CONTEXT_EVIDENCE|{task["protocol"]}|{task["fold"]}|{seed}'
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)

def risk_loss(torch, logits, labels, permutation, lam, worst):
    yf, ya = labels; n = len(yf)
    require(logits.shape[0] == 6*n and len(ya) == n, 'common single six-view forward')
    ce = torch.nn.functional.cross_entropy
    factual = torch.stack([ce(v, yf, reduction='none') for v in logits[:3*n].split(n)]).mean(0)
    mixed = torch.stack([lam*ce(v, ya, reduction='none') + (1-lam)*ce(v, ya[permutation], reduction='none')
                         for v in logits[3*n:].split(n)])
    chosen = mixed.max(dim=0).values if worst else mixed.mean(dim=0)
    return dict(total=.5*factual.mean()+.5*chosen.mean(), factual=factual.mean(),
                mixed=chosen.mean(), mixed_mean=mixed.mean(), mixed_max=mixed.max(dim=0).values.mean())
