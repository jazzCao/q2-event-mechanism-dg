"""Fixed-node choices and energy matching; no Torch, fitting, targets or I/O."""
import math
import numpy as np

OUTER=(18-math.sqrt(30))/36


def choice(arm,direction,radial):
    d=np.asarray(direction);r=np.asarray(radial)
    if d.shape!=r.shape or not np.isin(d,[0,1]).all() or not ((r>=0)&(r<1)).all():raise ValueError('shared binary direction and isolated uniform')
    if arm=='Q2':return 2+d
    if arm=='Q4':return np.where(d==0,np.where(r<OUTER,4,5),np.where(r<OUTER,7,6))
    return d


def weights(domain,y):
    domain,y=np.asarray(domain),np.asarray(y);cells=sorted(set(zip(domain.tolist(),y.tolist())));w=np.zeros(len(y))
    for d,c in cells:
        mask=(domain==d)&(y==c);w[mask]=1/(len(cells)*int(mask.sum()))
    if not np.isfinite(w).all() or not np.isclose(w.sum(),1):raise ValueError('complete positive balanced source sampling measure')
    return w


def calibrate(energies,w):
    e=np.asarray(energies,np.float64);w=np.asarray(w,np.float64)
    if e.shape!=(len(w),8) or not np.isfinite(e).all() or (e<0).any():raise ValueError('finite all eight source edit energies')
    end=w@e[:,:2];q4=np.array([w@(OUTER*e[:,4]+(1-OUTER)*e[:,5]),w@(OUTER*e[:,7]+(1-OUTER)*e[:,6])])
    scale=[];aliases=[]
    for a,b in zip(end,q4):
        if a==0 and b>0:raise ValueError('SOURCE_STRENGTH_CONTROL_UNDEFINED')
        scale.append(1. if a==b==0 else math.sqrt(b/a));aliases.append(bool(a==b==0))
    if not np.isfinite(scale).all():raise ValueError('SOURCE_STRENGTH_CONTROL_UNDEFINED')
    return np.asarray(scale),dict(endpoint=end.tolist(),q4=q4.tolist(),scale=scale,zero_alias=aliases)


def matching(measured,desired,tolerance):
    a,b=np.asarray(measured,np.float64),np.asarray(desired,np.float64)
    error=np.where(b>0,np.abs(a-b)/np.maximum(b,np.finfo(float).tiny),np.abs(a-b))
    if not (np.isfinite(error).all() and np.all(error<=tolerance) and np.all(a[b==0]==0)):raise ValueError('source energy match outside fixed tolerance')
    return error.tolist()
