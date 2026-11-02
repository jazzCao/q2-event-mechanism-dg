"""Protocol-specific source binding; NumPy metadata, no Torch/model imports.

PHM preserves recording-group splits. JNU preserves disjoint temporal blocks,
not fictitious recording/entity separation. No old qualified source file edited.
"""
from types import SimpleNamespace
import numpy as np
from pcr_campaign_core import canonical


def role_indices(records,case,domain,side):
    task=case['task']
    if side not in ('tr','va') or domain not in task['sources'] or domain==task['target']:raise PermissionError('declared source train/validation only')
    if any(k not in records or records[k]['protocol']!=task['protocol'] or records[k]['domain'] not in task['sources'] for k in case['roles']):raise PermissionError('foreign/target record in source roles')
    ids=[];ys=[];gs=[];os=[];cursor=0
    for key,r in sorted(records.items()):
        if (r['protocol'],r['domain'])!=(task['protocol'],domain):continue
        all_offsets=r['offsets'];parts=case['roles'].get(key)
        if len(set(all_offsets))!=len(all_offsets) or parts is None:raise ValueError('complete canonical source records required')
        if set(parts)!=set(('tr','va')) or any(len(set(v))!=len(v) for v in parts.values()):raise ValueError('unique two-way source roles')
        if set(parts['tr'])&set(parts['va']) or not set(parts['tr']+parts['va'])<=set(all_offsets):raise ValueError('shared/unknown source window')
        positions={offset:cursor+i for i,offset in enumerate(all_offsets)}
        for offset in parts[side]:ids.append(positions[offset]);ys.append(r['label']);gs.append(key);os.append(offset)
        cursor+=len(all_offsets)
    if not ids:raise ValueError('nonempty registered source split')
    return dict(indices=np.asarray(ids,np.int64),y=np.asarray(ys,np.int64),g=np.asarray(gs),offset=np.asarray(os,np.int64))


def temporal_disjoint(train,validation,window):
    """Full sampled intervals, not merely distinct starting offsets."""
    left=sorted(train);right=sorted(validation);i=j=0
    while i<len(left) and j<len(right):
        if left[i]+window<=right[j]:i+=1
        elif right[j]+window<=left[i]:j+=1
        else:return False
    return True


def inventory(records,case):
    task=case['task'];protocol=task['protocol']
    if protocol not in ('phm09_w3','jnu_speed') or len(task['sources'])!=2 or len(set(task['sources']))!=2 or task['target'] in task['sources'] or task['classes']!=(8 if protocol=='phm09_w3' else 4):raise ValueError('fixed two-protocol case schema')
    result={}
    for d in task['sources']:
        tr=role_indices(records,case,d,'tr');va=role_indices(records,case,d,'va')
        if set(tr['y'])!=set(range(task['classes'])):raise ValueError('all actual source training classes required')
        common=set(tr['g'])&set(va['g'])
        if protocol=='phm09_w3' and common:raise ValueError('PHM must retain recording-disjoint source split')
        for key in common:
            r=records[key];parts=case['roles'][key]
            if not temporal_disjoint(parts['tr'],parts['va'],r['window']):raise ValueError('overlapping JNU full train/validation intervals')
        result[d]=dict(rule='GROUPED_RECORDING_DISJOINT' if protocol=='phm09_w3' else 'TEMPORAL_BLOCK_FULL_INTERVAL_DISJOINT',
            same_recordings_across_roles=len(common),not_new_entity_validation=True)
        for side,v in (('tr',tr),('va',va)):
            count=np.bincount(v['y'],minlength=task['classes']).tolist()
            result[d][side]=dict(windows=len(v['y']),recordings=len(np.unique(v['g'])),class_counts=count,
                missing_classes=[i for i,n in enumerate(count) if n==0],role_sha256=canonical({k:x.tolist() for k,x in v.items()}))
    return result


class NativeSources:
    def __init__(self,allowed,forbidden,train,val,classes,fingerprint):
        self.allowed=tuple(allowed);self.forbidden=set(forbidden);self.train=train;self.val=val;self.classes=classes;self.fingerprint=fingerprint
        if len(self.allowed)!=2 or len(set(self.allowed))!=2 or set(self.allowed)&self.forbidden or not fingerprint or type(classes) is not int or classes not in (4,8):raise PermissionError('explicit source-only object')
        if set(train)!=set(self.allowed) or set(val)!=set(self.allowed):raise PermissionError('all source domains have both roles')
    def get(self,domain,role):
        if domain in self.forbidden or domain not in self.allowed or role not in ('train','val'):raise PermissionError('source object refuses target/test domain')
        return (self.train if role=='train' else self.val)[domain]


def bind_sources(records,case,arrays,fingerprint,tensor_factory=None):
    """Actual run passes torch.from_numpy; fixtures use identity, zero neural work."""
    inventory(records,case)
    if set(arrays)!=set(('tr','va')):raise ValueError('only source train/validation arrays')
    build=(lambda x:x) if tensor_factory is None else tensor_factory;train={};val={}
    for side,destination in (('tr',train),('va',val)):
        a=arrays[side]
        if set(a)!=set(('x','y','g','domain')):raise ValueError('explicit four-field source arrays')
        n=len(a['y'])
        if a['x'].dtype!=np.float32 or a['x'].shape!=(n,2048) or not np.isfinite(a['x']).all():raise ValueError('same FP32 logFFT2048, not raw5333/4000 network input')
        if a['y'].dtype.kind not in 'iu' or any(len(a[k])!=n for k in ('g','domain')) or set(a['domain'])!=set(case['task']['sources']):raise ValueError('exact source classes/domain/row support')
        for d in case['task']['sources']:
            v=role_indices(records,case,d,side);mask=a['domain']==d
            if not np.array_equal(a['y'][mask],v['y']) or not np.array_equal(a['g'][mask],v['g']):raise ValueError('source row labels/record order drift')
            destination[d]=SimpleNamespace(domain=d,role='train' if side=='tr' else 'val',
                x=build(np.ascontiguousarray(a['x'][mask])),y=v['y'],groups=v['g'],indices=v['indices'])
    return NativeSources(case['task']['sources'],[case['task']['target']],train,val,case['task']['classes'],fingerprint)
