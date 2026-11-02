"""One source-pool isometry preserves cross-example geometry through Mixup."""
import hashlib
import numpy as np


def require(ok,why):
    if not ok:raise ValueError(why)


def orthogonal(values,identity):
    a=np.asarray(values,dtype=np.float64);d=a.shape[-1]
    require(d>0 and d&(d-1)==0 and np.isfinite(a).all(),'finite power-of-two feature dimension')
    rng=np.random.RandomState(int(hashlib.sha256(identity.encode()).hexdigest()[:8],16))
    permutation=rng.permutation(d);before=rng.choice([-1.,1.],size=d);after=rng.choice([-1.,1.],size=d)
    result=np.take(a,permutation,axis=-1)*before
    width=1
    while width<d:
        blocks=result.reshape(-1,d//(2*width),2*width)
        left=blocks[...,:width].copy();right=blocks[...,width:].copy()
        blocks[...,:width]=left+right;blocks[...,width:]=left-right;width*=2
    return result*(after/np.sqrt(d))


def global_members(fixed,plus,minus,identity):
    fixed=np.asarray(fixed);require(fixed.shape==plus.shape==minus.shape and fixed.dtype==np.float32,'same float32 member axes')
    base=fixed.astype(float);delta=np.stack((plus.astype(float)-base,minus.astype(float)-base),axis=-2)
    rotated=orthogonal(delta,identity)
    return (base+rotated[...,0,:]).astype(np.float32),(base+rotated[...,1,:]).astype(np.float32)


def mixed_geometry(fixed,plus,minus,random_plus,random_minus):
    """640 prespecified source-only pairs, including zero edits, all five lambdas."""
    arrays=[np.asarray(a).reshape(-1,2048) for a in (fixed,plus,minus,random_plus,random_minus)]
    n=len(arrays[0]);require(all(a.shape==(n,2048) and a.dtype==np.float32 for a in arrays),'same saved feature axes')
    index=np.arange(640);i=index%n;j=(index*37+17)%n
    lam=np.tile(np.asarray([0.,.2,.5,.8,1.],np.float32),128)[:,None]
    base,pp,mm,rp,rm=arrays
    # A masked-out edit is zero for either operand. The factual base can be arbitrary.
    def choose(member,indices,zero):return np.where(zero[:,None],base[indices],member[indices])
    mixbase=lam*base[i]+(1-lam)*base[j]
    def delta(member):
        mixed=lam*choose(member,i,index%3==0)+(1-lam)*choose(member,j,index%5==0)
        return mixed.astype(float)-mixbase.astype(float)
    mechanical=np.stack((delta(pp),delta(mm)),axis=1);random=np.stack((delta(rp),delta(rm)),axis=1)
    g=np.einsum('nid,njd->nij',mechanical,mechanical);h=np.einsum('nid,njd->nij',random,random)
    require(np.allclose(g,h,atol=5e-6,rtol=1e-4),'shared isometry preserves actual float32 mixed Gram within fixed tolerance')
    norms=np.linalg.norm(g,axis=(1,2));error=np.linalg.norm(g-h,axis=(1,2));active=norms>0
    zi=(index%3==0)|(np.all(pp[i]==base[i],axis=1)&np.all(mm[i]==base[i],axis=1))
    zj=(index%5==0)|(np.all(pp[j]==base[j],axis=1)&np.all(mm[j]==base[j],axis=1))
    exact_zero=(zi|(lam[:,0]==0))&(zj|(lam[:,0]==1))
    require(np.count_nonzero(g[exact_zero])==np.count_nonzero(h[exact_zero])==0,'structurally zero mixed edits remain exact aliases')
    return dict(pairs=640,max_absolute_gram_error=float(abs(g-h).max()),
        max_relative_frobenius_error=float((error[active]/norms[active]).max()) if active.any() else 0.,
        absolute_tolerance=5e-6,relative_tolerance=1e-4,structural_zero_pairs=int(exact_zero.sum()),
        boundary='full cross-example isometry algebra;640 fixed source float32 mixed pairs, not all possible float32 minibatches')
