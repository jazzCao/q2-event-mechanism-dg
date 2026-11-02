"""Source-clock/group-sparse numerical inverse; no neural models or true fields."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

# Reuse the actually qualified NumPy packet code, without importing its Torch API.
_source=Path(__file__).with_name('event_packet_core.py')
_names={'support_guard','kernel_np','dictionary_np','render_np','fit_dictionary','parameters_from_wave'}
_tree=ast.parse(_source.read_text(encoding='utf8'))
_ns={'np':np,'torch':SimpleNamespace(Tensor=type(None))}
exec(compile(ast.Module(body=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in _names],type_ignores=[]),str(_source),'exec'),_ns)
packet=SimpleNamespace(**{n:_ns[n] for n in _names})


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


def design(x,clock):
    n=x.shape[1];params=np.stack([packet.parameters_from_wave(v) for v in x])
    h=np.stack([packet.dictionary_np(p,96) for p in params]);nfft=1<<(n+94).bit_length()
    H=np.fft.rfft(h,nfft,axis=2)
    frequencies=stationary_candidates(x)
    F,fnorm=harmonic_basis(n,frequencies)
    A=[];anorm=[]
    for rate in clock:
        a,norm=harmonic_basis(n,rate*np.arange(1,5));A.append(a);anorm.append(norm)
    A=np.stack(A);anorm=np.stack(anorm)
    sigma=[]
    for v in x:
        f=np.fft.rfftfreq(n);high=abs(np.fft.rfft(v))[(f>=.45)&(f<=.5)]
        sigma.append(max(float(np.median(high)/np.sqrt(n*np.log(2))),.005*float(np.sqrt(np.mean(v*v)))))
    sigma=np.asarray(sigma)
    lb=max(float(np.linalg.eigvalsh(a.T@a).max()) for a in A)
    lf=0. if F.shape[1]==0 else float(np.linalg.eigvalsh(F.T@F).max())
    lipschitz=float(np.max(np.sum(abs(H)**2,axis=1)))+lb+lf
    def group_bound(groups):
        z=np.log(max(groups,2));return float(np.mean(sigma)*np.sqrt(4+4*np.sqrt(z)+2*z))
    return dict(params=params,h=h,H=H,A=A,F=F,anorm=anorm,fnorm=fnorm,stationary_frequencies=frequencies,
                clock=np.asarray(clock),nfft=nfft,L=lipschitz,penalty_s=sigma*np.sqrt(2*np.log(2*48*n)),
                penalty_b=group_bound(4),penalty_n=group_bound(max(len(frequencies),1)),
                eligible=bool(abs(clock[0]-clock[1])>1/n))


def group_prox(value,threshold):
    if value.shape[1]==0:return value
    shaped=value.reshape(2,-1,2)
    norm=np.sqrt(np.sum(shaped*shaped,axis=(0,2)))
    return (shaped*np.maximum(1-threshold/np.maximum(norm,1e-30),0)[None,:,None]).reshape(value.shape)


def smooth_terms(s,b,c,x,d):
    n=x.shape[1];A=d['A'];F=d['F'];H=d['H'];nfft=int(d['nfft'])
    pred=np.fft.irfft(np.sum(H*np.fft.rfft(s,nfft,axis=2),axis=1),nfft,axis=1)[:,:n]
    pred+=np.einsum('dlk,dk->dl',A,b,optimize=True)
    if F.shape[1]:pred+=c@F.T
    residual=pred-x
    gs=np.fft.irfft(H.conj()*np.fft.rfft(residual,nfft,axis=1)[:,None],nfft,axis=2)[:,:,:n]
    gb=np.einsum('dlk,dl->dk',A,residual,optimize=True)
    gc=residual@F if F.shape[1] else np.zeros_like(c)
    return float(.5*np.sum(residual*residual)),gs,gb,gc


def joint_fit(x,d,steps=500,state=None,tick=None):
    n=x.shape[1];A=d['A'];F=d['F'];H=d['H'];nfft=int(d['nfft']);lip=float(d['L'])
    if not d['eligible']:
        return dict(s=np.zeros((2,48,n)),b=np.zeros((2,8)),c=np.zeros((2,F.shape[1])),
                    zs=np.zeros((2,48,n)),zb=np.zeros((2,8)),zc=np.zeros((2,F.shape[1])),momentum=1.,step=0)
    if state is None:
        s=np.zeros((2,48,n));b=np.zeros((2,8));c=np.zeros((2,F.shape[1]));zs=s.copy();zb=b.copy();zc=c.copy();momentum=1.;begin=0
    else:
        s=state['s'].copy();b=state['b'].copy();c=state['c'].copy();zs=state['zs'].copy();zb=state['zb'].copy();zc=state['zc'].copy();momentum=float(state['momentum']);begin=int(state['step'])
    for iteration in range(begin,steps):
        if tick is not None:tick('joint_iterations')
        _,gs,gb,gc=smooth_terms(zs,zb,zc,x,d)
        ns=np.maximum(zs-gs/lip-d['penalty_s'][:,None,None]/lip,0)
        nb=group_prox(zb-gb/lip,d['penalty_b']/lip);nc=group_prox(zc-gc/lip,d['penalty_n']/lip)
        mt=(1+np.sqrt(1+4*momentum**2))/2;ratio=(momentum-1)/mt
        zs=ns+ratio*(ns-s);zb=nb+ratio*(nb-b);zc=nc+ratio*(nc-c)
        s,b,c,momentum=ns,nb,nc,mt
    return dict(s=s,b=b,c=c,zs=zs,zb=zb,zc=zc,momentum=momentum,step=steps)


def harmonic_edit(state,d,source,ratio,start,stop):
    n=state['s'].shape[-1];t=np.arange(start,stop,dtype=float);center=(n-1)/2
    old,_=harmonic_basis(n,d['clock'][source]*np.arange(1,5),t,d['anorm'][source])
    new,_=harmonic_basis(n,d['clock'][source]*np.arange(1,5),center+ratio*(t-center),d['anorm'][source])
    return (new-old)@state['b'][source]
