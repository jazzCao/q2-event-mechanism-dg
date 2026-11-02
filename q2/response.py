"""Finite analytic response renderer, extracted from the experimental source."""

def render(events,params,ratio,size):
    import numpy as np
    from scipy.signal import lfilter
    s=np.asarray(events,dtype=np.float64);p=np.asarray(params,dtype=np.float64)
    if s.ndim!=2 or p.shape!=(len(s),3) or s.shape[1]<1 or size<8:raise ValueError('same C,N events and C,3 kernels')
    if not np.isfinite(s).all() or (s<0).any() or not np.isfinite(p).all():raise ValueError('finite nonnegative events')
    if not (np.isfinite(ratio) and ratio>0) or not ((p[:,0]>0)&(p[:,0]<=.45)&(p[:,1]>0)&(p[:,1]<1)).all():raise ValueError('finite ratio and old kernel support')
    n=s.shape[1];center=(n-1)/2;grid=np.arange(size,dtype=np.float64);out=np.zeros(n,np.float64)
    alpha=np.pi/(size-1)
    for channel,(frequency,decay,phase) in enumerate(p):
        original=np.exp(-decay*grid)*np.sin(2*np.pi*frequency*grid+phase)*np.cos(np.pi*grid/(2*(size-1)))**2
        norm=max(float(np.linalg.norm(original)),1e-12)
        ids=np.flatnonzero(s[channel]);tau=center+(ids-center)/ratio
        if not np.isfinite(tau).all():raise ValueError('finite transformed locations')
        keep=(tau<=n-1)&(tau+size-1>=0);tau=tau[keep];mass=s[channel,ids[keep]]
        begin=np.maximum(0,np.ceil(tau)).astype(np.int64);end=np.floor(tau+size-1).astype(np.int64)+1
        keep=begin<end;begin,end,tau,mass=[v[keep] for v in (begin,end,tau,mass)]
        for offset,weight in ((0.,.5),(alpha,.25),(-alpha,.25)):
            exponent=-decay+1j*(2*np.pi*frequency+offset);a=np.exp(exponent)
            drive=np.zeros(n,np.complex128)
            np.add.at(drive,begin,mass*np.exp(exponent*(begin-tau)))
            active=end<n
            np.add.at(drive,end[active],-mass[active]*np.exp(exponent*(end[active]-tau[active])))
            z=lfilter(np.array([1.],np.complex128),np.array([1.,-a],np.complex128),drive)
            out+=(weight/norm)*(np.exp(1j*phase)*z).imag
    return out


def reference_interval(n,size,q):
    import math
    center=(n-1)/2
    # Extra sample margins avoid a floating boundary being called observed support.
    lo=max(0,math.ceil(center-center/q+size-1)+1)
    hi=min(n,math.floor(center+(n-1-center)/q))
    if lo>=hi:raise ValueError('nonempty common observation-supported interval')
    return lo,hi
