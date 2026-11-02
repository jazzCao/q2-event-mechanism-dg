"""Source-pair numerical analyzer; CPU reference, no neural imports."""
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

def harmonic_basis(length,frequencies,times=None,norms=None):
    t=np.arange(length,dtype=float) if times is None else np.asarray(times,dtype=float)
    centered=t-(length-1)/2
    columns=[]
    for f in frequencies:
        columns += [np.cos(2*np.pi*f*centered),np.sin(2*np.pi*f*centered)]
    raw=np.stack(columns,axis=1) if columns else np.zeros((len(t),0))
    if norms is None:norms=np.maximum(np.linalg.norm(raw,axis=0),1e-12)
    return raw/norms[None],norms


def stationary_candidates(x):
    """Source-consensus narrow lines, inferred from both observed waveforms."""
    n=x.shape[1];power=abs(np.fft.rfft(x,axis=1))**2
    peak_sets=[]
    for d in range(2):
        ids,_=find_peaks(power[d]);keep=[]
        for k in ids:
            if not .001<=k/n<=.44 or k<13 or k+13>=power.shape[1]:continue
            side=np.r_[power[d,k-12:k-3],power[d,k+4:k+13]]
            if power[d,k]>5*max(np.mean(side),1e-20):keep.append(int(k))
        peak_sets.append(keep)
    possible=[]
    for k in peak_sets[0]:
        match=[j for j in peak_sets[1] if abs(j-k)<=1]
        if match:possible.append((min(power[0,k],max(power[1,j] for j in match)),k))
    possible.sort(reverse=True);selected=[]
    for score,k in possible:
        if any(abs(k/n-f)<2/n for f in selected):continue
        t=np.arange(n)-(n-1)/2
        def objective(f):
            a=np.stack([np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t)],axis=1)
            gram=a.T@a+np.eye(2)*1e-12
            projections=np.array([v@a@np.linalg.solve(gram,a.T@v) for v in x])
            return -float(np.min(projections/np.maximum(np.sum(x*x,axis=1),1e-20)))
        fit=minimize_scalar(objective,bounds=(max(.001,(k-1)/n),min(.44,(k+1)/n)),method='bounded',options={'maxiter':30,'xatol':1e-10})
        selected.append(float(fit.x))
        if len(selected)==8:break
    return np.array(selected)


def group_prox(value,threshold):
    if value.shape[1]==0:return value
    shaped=value.reshape(2,-1,2)
    norm=np.sqrt(np.sum(shaped*shaped,axis=(0,2)))
    return (shaped*np.maximum(1-threshold/np.maximum(norm,1e-30),0)[None,:,None]).reshape(value.shape)

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
