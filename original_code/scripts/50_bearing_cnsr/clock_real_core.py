"""Physical-unit source-clock inverse and exact rational-phase packet edits.

The numerical analyzer is not a neural network or an identified latent model.
"""
from fractions import Fraction
import numpy as np
import torch
from shared_clock_core import stationary_candidates, harmonic_basis, group_prox


def kernel(lag, parameter, size):
    f, decay, phase = parameter
    if not (0 < f <= .45 and 0 < decay < 1 and size >= 8):
        raise ValueError('physical packet parameters')
    grid = np.arange(size, dtype=float)
    norm = np.linalg.norm(np.exp(-decay*grid)*np.sin(2*np.pi*f*grid+phase)*np.cos(np.pi*grid/(2*(size-1)))**2)
    u = np.clip(lag, 0, size-1)
    value = np.exp(-decay*u)*np.sin(2*np.pi*f*u+phase)*np.cos(np.pi*u/(2*(size-1)))**2
    return np.where((lag >= 0) & (lag <= size-1), value/max(norm, 1e-12), 0.)


def design(x, shaft_hz, fs):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != 2 or not np.isfinite(x).all():
        raise ValueError('two finite source observations')
    n = x.shape[1]; size = int(np.ceil(.004*fs))
    clock = np.asarray(shaft_hz, dtype=float)/fs
    params = []
    for v in x:
        frequency = np.fft.rfftfreq(n, 1/fs)
        ranked = np.argsort(abs(np.fft.rfft(v)))[::-1]; peaks = []
        for k in ranked:
            f = float(frequency[k])
            if 4*max(shaft_hz) <= f <= .44*fs and all(abs(f-q) >= max(100., 3*fs/n) for q in peaks):
                peaks.append(f)
                if len(peaks) == 4: break
        if len(peaks) != 4: raise ValueError('source carrier support')
        params.append([(f/fs, 1/(fs*tau), phase) for f in peaks
                       for tau in (.00025, .0005, .001) for phase in (0., np.pi/2, np.pi, 3*np.pi/2)])
    params = np.asarray(params)
    h = np.stack([[kernel(np.arange(size), p, size) for p in ps] for ps in params])
    nfft = 1 << (n+size-2).bit_length()
    H = np.fft.rfft(h, nfft, axis=-1)
    fixed = stationary_candidates(x); F, fnorm = harmonic_basis(n, fixed)
    A, anorm = zip(*(harmonic_basis(n, rate*np.arange(1, 5)) for rate in clock))
    A = np.stack(A); anorm = np.stack(anorm)
    f = np.fft.rfftfreq(n)
    sigma = np.array([max(float(np.median(abs(np.fft.rfft(v))[(f >= .45) & (f <= .5)])/np.sqrt(n*np.log(2))),
                          .005*float(np.sqrt(np.mean(v*v)))) for v in x])
    bound = lambda groups: float(np.mean(sigma)*np.sqrt(4+4*np.sqrt(np.log(max(groups, 2)))+2*np.log(max(groups, 2))))
    L = float(np.max(np.sum(abs(H)**2, axis=1))) + max(float(np.linalg.eigvalsh(a.T@a).max()) for a in A)
    if F.shape[1]: L += float(np.linalg.eigvalsh(F.T@F).max())
    return dict(params=params, h=h, H=H, nfft=nfft, A=A, anorm=anorm, F=F, fnorm=fnorm,
                stationary_frequencies=fixed, clock=clock, fs=float(fs), kernel_length=size,
                L=L, penalty_s=sigma*np.sqrt(2*np.log(2*48*n)), penalty_b=bound(4),
                penalty_n=bound(max(len(fixed), 1)), eligible=bool(abs(clock[0]-clock[1]) > 1/n))


def fit_cpu(x, d, steps, tick=None):
    n = x.shape[-1]; s = np.zeros((2, 48, n)); b = np.zeros((2, 8)); c = np.zeros((2, d['F'].shape[1]))
    zs, zb, zc = s.copy(), b.copy(), c.copy(); momentum = 1.
    for _ in range(steps):
        if tick: tick('cpu_inverse_iterations')
        pred = np.fft.irfft(np.sum(d['H']*np.fft.rfft(zs, d['nfft'], axis=-1), axis=1), d['nfft'], axis=-1)[:, :n]
        residual = pred + np.einsum('dlk,dk->dl', d['A'], zb) + zc@d['F'].T - x
        gs = np.fft.irfft(d['H'].conj()*np.fft.rfft(residual, d['nfft'], axis=-1)[:, None], d['nfft'], axis=-1)[:, :, :n]
        nb = group_prox(zb-np.einsum('dlk,dl->dk', d['A'], residual)/d['L'], d['penalty_b']/d['L'])
        nc = group_prox(zc-residual@d['F']/d['L'], d['penalty_n']/d['L'])
        ns = np.maximum(zs-gs/d['L']-d['penalty_s'][:, None, None]/d['L'], 0)
        mt = (1+np.sqrt(1+4*momentum**2))/2; ratio = (momentum-1)/mt
        zs, zb, zc = ns+ratio*(ns-s), nb+ratio*(nb-b), nc+ratio*(nc-c)
        s, b, c, momentum = ns, nb, nc, mt
    return dict(s=s, b=b, c=c, zs=zs, zb=zb, zc=zc, momentum=momentum, step=steps)


@torch.no_grad()
def fit_gpu(x, d, steps, state=None, tick=None, dtype=torch.float32):
    convert = lambda v: torch.as_tensor(v, device='cuda', dtype=dtype)
    tx = convert(x); n = x.shape[-1]; h = convert(d['h']); H = torch.fft.rfft(h, d['nfft'], dim=-1)
    A, F = convert(d['A']), convert(d['F']); penalty = convert(d['penalty_s'])[:, None, None]/d['L']
    if state is None:
        s = torch.zeros((2, 48, n), device='cuda', dtype=dtype); b = torch.zeros((2, 8), device='cuda', dtype=dtype)
        c = torch.zeros((2, F.shape[1]), device='cuda', dtype=dtype); zs, zb, zc = s.clone(), b.clone(), c.clone()
        momentum = 1.; begin = 0
    else:
        s, b, c, zs, zb, zc = [convert(state[k]).clone() for k in ('s', 'b', 'c', 'zs', 'zb', 'zc')]
        momentum = float(state['momentum']); begin = int(state['step'])
    def prox(v, threshold):
        if v.shape[1] == 0: return v
        a = v.reshape(2, -1, 2); norm = torch.sqrt((a*a).sum((0, 2)))
        return (a*(1-threshold/norm.clamp_min(1e-30)).clamp_min(0)[None, :, None]).reshape(v.shape)
    for _ in range(begin, steps):
        if tick: tick('gpu_inverse_iterations')
        pred = torch.fft.irfft((H*torch.fft.rfft(zs, d['nfft'], dim=-1)).sum(1), d['nfft'], dim=-1)[:, :n]
        residual = pred+torch.einsum('dlk,dk->dl', A, zb)+zc@F.T-tx
        gs = torch.fft.irfft(H.conj()*torch.fft.rfft(residual, d['nfft'], dim=-1)[:, None], d['nfft'], dim=-1)[:, :, :n]
        ns = (zs-gs/d['L']-penalty).clamp_min(0)
        nb = prox(zb-torch.einsum('dlk,dl->dk', A, residual)/d['L'], d['penalty_b']/d['L'])
        nc = prox(zc-residual@F/d['L'], d['penalty_n']/d['L'])
        mt = (1+np.sqrt(1+4*momentum**2))/2; ratio = (momentum-1)/mt
        zs, zb, zc = ns+ratio*(ns-s), nb+ratio*(nb-b), nc+ratio*(nc-c)
        s, b, c, momentum = ns, nb, nc, mt
    torch.cuda.synchronize()
    return dict(s=s.cpu().numpy(), b=b.cpu().numpy(), c=c.cpu().numpy(), zs=zs.cpu().numpy(),
                zb=zb.cpu().numpy(), zc=zc.cpu().numpy(), momentum=momentum, step=steps)


def render_phase(events, params, ratio, size):
    """Exact sampled analytic fractional packets for rational ratios, no splat.

    Source indices within one residue class map to unique integer destinations.
    Thus there is no nondeterministic floating scatter accumulation.
    """
    events = np.asarray(events); n = events.shape[-1]; frac = Fraction(float(ratio)).limit_denominator(12)
    if abs(float(frac)-ratio) > 1e-12: raise ValueError('unqualified rational factor')
    a, b = frac.numerator, frac.denominator; center = (n-1)/2
    nfft = 1 << (n+size-1).bit_length(); total = np.zeros(nfft//2+1, complex)
    for residue in range(a):
        ids = np.arange(residue, n, a); positions = center+(ids-center)*b/a
        floor = np.floor(positions+1e-10).astype(int); fraction = float(positions[0]-floor[0])
        if not np.allclose(positions-floor, fraction, atol=1e-9, rtol=0): raise AssertionError('fractional phase grouping')
        valid = (floor >= 0) & (floor < n); grid = np.zeros_like(events)
        grid[:, floor[valid]] = events[:, ids[valid]]
        h = np.stack([kernel(np.arange(size+1)-fraction, p, size) for p in params])
        total += np.sum(np.fft.rfft(grid, nfft, axis=-1)*np.fft.rfft(h, nfft, axis=-1), axis=0)
    return np.fft.irfft(total, nfft)[:n]


def render_direct(events, params, ratio, size, start, stop):
    n = events.shape[-1]; center = (n-1)/2; out = np.zeros(stop-start)
    for channel, p in enumerate(params):
        ids = np.flatnonzero(events[channel]); positions = center+(ids-center)/ratio
        keep = (positions <= stop-1) & (positions+size-1 >= start); ids, positions = ids[keep], positions[keep]
        times = np.floor(positions).astype(int)[:, None]+np.arange(size+1)[None]
        values = events[channel, ids, None]*kernel(times-positions[:, None], p, size)
        valid = (times >= start) & (times < stop)
        np.add.at(out, times[valid]-start, values[valid])
    return out


def support(n, size, ratio, start, stop):
    center = (n-1)/2
    if center+ratio*(start-(size-1)-center) < 0 or center+ratio*(stop-1-center) > n-1:
        raise ValueError('unknown outside-context events required')


def edits(x, state, d, source, ratio, starts, window):
    n = x.shape[-1]; size = d['kernel_length']; center = (n-1)/2
    for start in starts: support(n, size, ratio, start, start+window)
    original = render_phase(state['s'][source], d['params'][source], 1., size)
    changed = render_phase(state['s'][source], d['params'][source], ratio, size)
    times = np.arange(n, dtype=float)
    A, _ = harmonic_basis(n, d['clock'][source]*np.arange(1, 5), times, d['anorm'][source])
    Ar, _ = harmonic_basis(n, d['clock'][source]*np.arange(1, 5), center+ratio*(times-center), d['anorm'][source])
    event = changed-original; shaft = (Ar-A)@state['b'][source]
    result = []
    for start in starts:
        t = np.arange(start, start+window); raw = x[source, t]
        result.append(dict(event=raw+event[t], joint=raw+event[t]+shaft[t],
                           warp=np.interp(center+ratio*(t-center), np.arange(n), x[source])))
    return result
