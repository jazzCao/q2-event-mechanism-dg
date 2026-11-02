"""Same analytic renderer, exact finite-support zeros; no magnitude threshold."""
from continuous_event_core import render as previous_render
from continuous_event_core import reference_interval as previous_interval


def render(events,params,ratio,size):
    import numpy as np
    result=previous_render(events,params,ratio,size)
    s=np.asarray(events);n=s.shape[1];center=(n-1)/2;change=np.zeros(n+1,np.int64)
    for row in s:
        ids=np.flatnonzero(row);tau=center+(ids-center)/ratio
        begin=np.maximum(0,np.ceil(tau)).astype(np.int64);end=np.minimum(n,np.floor(tau+size-1).astype(np.int64)+1)
        keep=(begin<end)&(begin<n)&(end>0)
        np.add.at(change,begin[keep],1);np.add.at(change,end[keep],-1)
    active=np.cumsum(change[:-1])>0
    result[~active]=0.
    return result


def reference_interval(n,size,q):
    lo,hi=previous_interval(n,size,q)
    lo=max(size-1,lo)
    if lo>=hi:raise ValueError('nonempty common support excluding all negative-event tails')
    return lo,hi
