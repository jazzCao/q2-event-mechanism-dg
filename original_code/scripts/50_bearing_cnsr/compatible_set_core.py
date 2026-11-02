"""Factual-compatible numerical sets and matched risk; no payload IO or torch import."""
from fractions import Fraction
import hashlib
import numpy as np
from compatible_set_plan import require
from abduction_family_core import prepare, batched, old_positive_direction, PAIRS, query_crops


def gram_random(plus, minus, identity):
    """Preserve each two-vector Gram; random orientation uses no class/score."""
    plus, minus = np.asarray(plus, float), np.asarray(minus, float)
    require(plus.shape == minus.shape and plus.ndim == 1 and len(plus) >= 2, 'two finite feature deltas')
    require(np.isfinite(plus).all() and np.isfinite(minus).all(), 'finite Gram input')
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed); u, v = rng.normal(size=(2, len(plus)))
    u /= np.linalg.norm(u); v -= np.dot(u, v)*u; v /= np.linalg.norm(v)
    a, b = np.linalg.norm(plus), np.linalg.norm(minus)
    if a == 0: return np.zeros_like(plus), b*u
    if b == 0: return a*u, np.zeros_like(minus)
    cos = np.clip(np.dot(plus, minus)/(a*b), -1., 1.)
    return a*u, b*(cos*u + np.sqrt(max(0., 1.-cos*cos))*v)


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


def make_group(group, inverse, context, rt, functions):
    source = group['source']; size = group['kernel_length']; window = group['window']; starts = group['starts']
    s, params = inverse['s'], inverse['params']
    require(s.shape == (2, 48, 32768) and params.shape == (2,48,3) and np.isfinite(s).all() and np.all(s >= 0), 'original complete inverse')
    observed, component = context['observed'][source], context['u_normalized'][source]
    scale = float(context['scale'][source]); require(observed.shape == component.shape == (32768,) and scale > 0 and np.isfinite(scale), 'physical context')
    crop = lambda v: query_crops(v, starts, window)
    clean = functions['spectrum'](crop(observed)); basewaves, fixed = {}, []
    for text in ('2/3','3/2'):
        for start in starts: functions['support'](32768,size,float(Fraction(text)),start,start+window)
        u = functions['render_phase'](s[source],params[source],float(Fraction(text)),size)
        basewaves[text] = crop(observed+scale*(u-component)); fixed.append(functions['spectrum'](basewaves[text]))
    seed = s[source].sum(0).copy(); seed[:8192] = 0.; seed[24576-2*size:] = 0.
    alias = float(seed.sum()) == 0.; scores = np.zeros(len(PAIRS)); selected = -1
    if not alias:
        family = prepare(s[source],params[source],size,functions['kernel'])
        for text in ('2/3','3/2'):
            d = batched(family,text,functions['kernel'],rt.fft)
            queries = query_crops(d,starts,window).reshape(len(PAIRS),3,window)
            scores += np.square(queries).sum(axis=(1,2))*scale**2
        selected = int(np.argmax(scores)); alias = bool(scores[selected] == 0.)
    plus, minus = [], []
    max_factual_error = 0.; added_fractions = []
    if alias:
        plus, minus = [v.copy() for v in fixed], [v.copy() for v in fixed]
    else:
        pair = PAIRS[selected]
        for sign, destination in ((1,plus),(-1,minus)):
            delta, pars = old_positive_direction(family,params[source],pair,sign)
            fraction = float(delta.sum()/s[source].sum()); added_fractions.append(fraction)
            require(abs(fraction-.01) < 1e-12 and np.all(delta >= 0), 'same1% nonnegative sparse addition')
            d1 = functions['render_phase'](delta,pars,1.,size)
            max_factual_error = max(max_factual_error,float(abs(d1).max()))
            require(float(abs(d1).max()) < 1e-10, 'same factual convolution within numerical tolerance')
            for text in ('2/3','3/2'):
                dq = functions['render_phase'](delta,pars,float(Fraction(text)),size)
                destination.append(functions['spectrum'](basewaves[text]+scale*crop(dq-d1)))
    fixed, plus, minus = [np.stack(v,axis=1) for v in (fixed,plus,minus)]
    rp, rm = np.empty_like(fixed), np.empty_like(fixed); gram_error = 0.
    for i in range(3):
        for q in range(2):
            v = fixed[i,q].astype(float); p, m = plus[i,q]-v, minus[i,q]-v
            a,b = gram_random(p,m,'COMPATIBLE_SET_GRAM_V1|'+group['key']+'|'+str(i)+'|'+str(q))
            rp[i,q],rm[i,q] = v+a,v+b
            original = np.stack((p,m)); actual = np.stack((rp[i,q].astype(float)-v,rm[i,q].astype(float)-v))
            g, h = original@original.T, actual@actual.T
            require(np.allclose(g,h,atol=5e-6,rtol=1e-4), 'float32 random control preserves full Gram within locked tolerance')
            gram_error = max(gram_error,float(abs(g-h).max()))
    return dict(clean=clean,fixed=fixed,plus=plus,minus=minus,random_plus=rp,random_minus=rm,
        group_key=np.array(group['key']),origin=np.array(group['origin']),starts=np.asarray(starts),
        selected_pair=np.asarray((-1,-1) if alias else PAIRS[selected]),scores=scores,
        alias=np.array(alias),gram_error=np.array(gram_error),factual_error=np.array(max_factual_error),
        added_l1_fraction=np.asarray(added_fractions))


def assemble(case, packets):
    names = ('clean','fixed','plus','minus','random_plus','random_minus')
    result = {k: [] for k in names+('y','g','domain','offset')}
    for item in case['bank']:
        a = packets[item['group']]
        require(str(a['group_key']) == item['group'] and str(a['origin']) == item['record'], 'exact group origin')
        for k in names: result[k].append(a[k])
        for k,v in (('y',item['label']),('g',item['record']),('domain',item['domain'])): result[k].extend([v]*3)
        result['offset'].extend((a['starts']+65536).tolist())
    return {k: np.concatenate(v) if k in names else np.asarray(v) for k,v in result.items()}
