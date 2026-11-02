"""All-candidate blocked L2 search, with direct refinement; no model code.

Numerical qualification is empirical, not a claim of universal floating-point
proof. No candidate cap is introduced; uncertain near-ties are refined directly.
The previously qualified scalar implementation remains unmodified.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from mocd_matching_state import IndexedMatching


def _direct_winner(query, candidates, block):
    best=-np.inf;index=-1
    for start in range(0,len(candidates),block):
        distances=((candidates[start:start+block]-query)**2).sum(1)
        if not np.isfinite(distances).all():
            raise ValueError("nonfinite squared distance")
        local=int(np.argmax(distances));value=distances[local]
        if value>best:
            best=value;index=start+local
    return index


def farthest_all_candidates(queries,candidates,query_block=64,candidate_block=512):
    q=np.asarray(queries,dtype=np.float64);c=np.asarray(candidates,dtype=np.float64)
    if q.ndim!=2 or c.ndim!=2 or q.shape[1]!=c.shape[1] or not len(c) or not q.shape[1]:
        raise ValueError("invalid distance matrix dimensions")
    if not np.isfinite(q).all() or not np.isfinite(c).all() or query_block<1 or candidate_block<1:
        raise ValueError("finite inputs and positive blocks required")
    winners=np.empty(len(q),np.int64)
    stats={"full_pairs":len(q)*len(c),"refined_pairs":0,"norm_overflow_fallback_queries":0,
           "max_pair_matrix_elements":0,"query_block":query_block,"candidate_block":candidate_block}
    with np.errstate(over="ignore",invalid="ignore"):
        cnorm=(c*c).sum(1)
    for start in range(0,len(q),query_block):
        qs=q[start:start+query_block]
        with np.errstate(over="ignore",invalid="ignore"):
            qnorm=(qs*qs).sum(1)
        if not np.isfinite(qnorm).all() or not np.isfinite(cnorm).all():
            for offset,query in enumerate(qs):
                winners[start+offset]=_direct_winner(query,c,candidate_block)
            stats["norm_overflow_fallback_queries"]+=len(qs)
            stats["refined_pairs"]+=len(qs)*len(c)
            continue
        lower=np.full(len(qs),-np.inf)
        factor=64*np.finfo(np.float64).eps*(q.shape[1]+2)
        def interval(cs,csnorm):
            approximate=qnorm[:,None]+csnorm[None,:]-2*(qs@cs.T)
            envelope=factor*np.maximum((np.sqrt(qnorm)[:,None]+np.sqrt(csnorm)[None,:])**2,np.finfo(np.float64).tiny)
            return approximate,envelope
        for column in range(0,len(c),candidate_block):
            approximate,error=interval(c[column:column+candidate_block],cnorm[column:column+candidate_block])
            if not np.isfinite(approximate).all() or not np.isfinite(error).all():
                raise ValueError("nonfinite distance interval")
            lower=np.maximum(lower,(approximate-error).max(1))
            stats["max_pair_matrix_elements"]=max(stats["max_pair_matrix_elements"],approximate.size)
        best=np.full(len(qs),-np.inf);indices=np.full(len(qs),-1,np.int64)
        for column in range(0,len(c),candidate_block):
            cs=c[column:column+candidate_block]
            approximate,error=interval(cs,cnorm[column:column+candidate_block])
            possible=(approximate+error)>=lower[:,None]
            for offset in range(len(qs)):
                positions=np.flatnonzero(possible[offset])
                if not len(positions):continue
                # Same subtraction/square/axis reduction as the scalar oracle.
                distances=((cs[positions]-qs[offset])**2).sum(1)
                if not np.isfinite(distances).all():raise ValueError("nonfinite direct refined distance")
                stats["refined_pairs"]+=len(positions)
                local=int(np.argmax(distances));value=distances[local]
                index=column+positions[local]
                if value>best[offset] or (value==best[offset] and index<indices[offset]):
                    best[offset]=value;indices[offset]=index
        if (indices<0).any():raise RuntimeError("refinement produced no winner")
        winners[start:start+len(qs)]=indices
    return winners,stats


class BlockedIndexedMatching(IndexedMatching):
    """Same persisted index schema; backend identity belongs in outer lock."""
    def refresh(self,embeddings_by_domain,branch,space,query_block=64,candidate_block=512):
        if branch not in ("f","d") or space not in ("projection","encoder"):
            raise ValueError("explicit branch and embedding space required")
        z=[np.asarray(a,dtype=np.float64) for a in embeddings_by_domain]
        if len(z)!=len(self.labels) or any(a.ndim!=2 or len(a)!=len(y) or not np.isfinite(a).all() for a,y in zip(z,self.labels)) or len({a.shape[1] for a in z})!=1:
            raise ValueError("embedding provenance/finite/dimensions")
        self.validate()
        key="fault" if branch=="f" else "domain"
        updated=copy.deepcopy(self.state[key]);groups={}
        for matrix_key,rows in updated.items():
            for row_index,row in enumerate(rows):
                anchor_dom,anchor_i=row[0]
                for column,(target_dom,target_i) in enumerate(row[1:],1):
                    target_class=int(self.labels[target_dom][target_i])
                    group=(int(anchor_dom),int(target_dom),target_class)
                    groups.setdefault(group,[]).append((matrix_key,row_index,column,int(anchor_i)))
        summaries=[]
        for (anchor_dom,target_dom,target_class),entries in groups.items():
            candidates=np.flatnonzero(self.labels[target_dom]==target_class)
            anchors=np.asarray([entry[3] for entry in entries],np.int64)
            winners,stats=farthest_all_candidates(z[anchor_dom][anchors],z[target_dom][candidates],query_block,candidate_block)
            for entry,winner in zip(entries,winners):
                matrix_key,row,column,_=entry
                updated[matrix_key][row,column]=(target_dom,candidates[winner])
            summaries.append(stats)
        self.state[key]=updated;self.refreshes+=1;self.state["last_refresh_space"]=space
        self.validate()
        return summaries


@dataclass(frozen=True)
class FullSchedule:
    A_outer:int=20
    A_inner:int=20
    B_outer:int=20
    B_inner:int=20

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in (self.A_outer,self.A_inner,self.B_outer,self.B_inner)):
            raise ValueError("positive integer stage loops required")

    @property
    def A_steps(self):return self.A_outer*self.A_inner

    @property
    def total_steps(self):return self.A_steps+self.B_outer*self.B_inner

    def event(self,completed_step):
        if type(completed_step) is not int or not 1<=completed_step<=self.total_steps:
            raise ValueError("completed step outside locked schedule")
        stage="A" if completed_step<=self.A_steps else "B"
        local=completed_step if stage=="A" else completed_step-self.A_steps
        inner=self.A_inner if stage=="A" else self.B_inner
        return {"stage":stage,"outer":(local-1)//inner+1,"inner":(local-1)%inner+1,
                "refresh":local%inner==0,"start_B":completed_step==self.A_steps+1,
                "source_val_diagnostic":stage=="B" and local%inner==0,
                "primary_final":completed_step==self.total_steps}
