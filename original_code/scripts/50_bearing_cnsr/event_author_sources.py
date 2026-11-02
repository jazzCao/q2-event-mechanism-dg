"""Source-role adapter for the existing event protocol; no model or I/O at import."""
from __future__ import annotations

import hashlib
import json
import numpy as np

PROTECTED=frozenset(('K006','KA08','KA09'))
FOLDS=('BCDtoA','ACDtoB','ABDtoC','ABCtoD')


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def physical_case(records,fold,seed):
    if fold not in FOLDS or type(seed) is not int or seed not in (0,1,2):raise ValueError('fixed physical four-fold natural seed0/1/2 only')
    pu={k:r for k,r in records.items() if r['protocol']=='pu_clock4'}
    if not pu or {r['bearing'] for r in pu.values()}&PROTECTED:raise ValueError('PU/protected-bearing boundary')
    runs={b:set((np.random.RandomState(int(hashlib.sha256(f'CLOCK4_SPLIT|{b}|{seed}'.encode()).hexdigest()[:8],16)).permutation(20)[:16]+1).tolist()) for b in sorted({r['bearing'] for r in pu.values()})}
    sources=list(fold.split('to')[0]);target=fold.split('to')[1]
    roles={k:dict(tr=r['offsets'] if r['run'] in runs[r['bearing']] else [],va=[] if r['run'] in runs[r['bearing']] else r['offsets']) for k,r in pu.items() if r['domain'] in sources}
    return dict(task=dict(protocol='pu_clock4',fold=fold,sources=sources,target=target,classes=3,window=5120),seed=seed,roles=roles)


def role_indices(records,case,domain,side):
    task=case['task']
    if side not in ('tr','va') or domain not in task['sources'] or domain==task['target']:raise PermissionError('source-only role requested')
    if any(k not in records or records[k]['protocol']!=task['protocol'] or records[k]['domain'] not in task['sources'] for k in case['roles']):raise PermissionError('target/foreign record in source role manifest')
    indices=[];labels=[];groups=[];offsets=[];cursor=0
    for k,r in sorted(records.items()):
        if (r['protocol'],r['domain'])!=(task['protocol'],domain):continue
        if r.get('bearing') in PROTECTED:raise PermissionError('protected bearing')
        all_offsets=r['offsets']
        if len(set(all_offsets))!=len(all_offsets):raise ValueError('duplicate canonical offsets')
        sides=case['roles'].get(k)
        if sides is None:raise ValueError('incomplete source record coverage')
        if set(sides)!=set(('tr','va')) or len(set(sides['tr']))!=len(sides['tr']) or len(set(sides['va']))!=len(sides['va']):raise ValueError('role keys/duplicates')
        if set(sides['tr'])&set(sides['va']) or not set(sides['tr']+sides['va'])<=set(all_offsets):raise ValueError('window overlap/unknown offset')
        positions={off:cursor+i for i,off in enumerate(all_offsets)}
        for off in sides[side]:indices.append(positions[off]);labels.append(r['label']);groups.append(k);offsets.append(off)
        cursor+=len(all_offsets)
    if not indices:raise ValueError('empty source role')
    return dict(indices=np.asarray(indices,dtype=np.int64),y=np.asarray(labels,dtype=np.int64),g=np.asarray(groups),offset=np.asarray(offsets,dtype=np.int64))


def inventory(records,case):
    result={}
    for domain in case['task']['sources']:
        tr=role_indices(records,case,domain,'tr');va=role_indices(records,case,domain,'va')
        if set(tr['g'])&set(va['g']):raise ValueError('recording-disjoint source roles required')
        if set(tr['y'])!=set(range(case['task']['classes'])):raise ValueError('missing actual source training class')
        result[domain]={side:dict(windows=len(v['y']),recordings=len(np.unique(v['g'])),class_counts=np.bincount(v['y'],minlength=case['task']['classes']).tolist(),role_sha256=digest({k:a.tolist() for k,a in v.items()})) for side,v in (('tr',tr),('va',va))}
    return result


def bind_sources(records,case,source_arrays,fingerprint):
    """Consume CachedStore.source_case arrays; never accept a target loader/model."""
    import torch
    from mocd_full_engine import BoundSources,SourceWindows
    if not fingerprint or set(source_arrays)!=set(('tr','va')):raise ValueError('bound source arrays only')
    inventory(records,case);train={};val={}
    for side,target in (('tr',train),('va',val)):
        arrays=source_arrays[side]
        if set(arrays)!=set(('x','y','g','domain')):raise ValueError('explicit common source schema')
        if not isinstance(arrays['x'],np.ndarray) or arrays['x'].dtype!=np.float32 or arrays['x'].ndim!=2 or arrays['x'].shape!=(len(arrays['y']),2048) or not np.isfinite(arrays['x']).all():raise ValueError('finite FP32 logFFT2048 required')
        if any(len(arrays[k])!=len(arrays['y']) for k in ('g','domain')) or set(arrays['domain'])!=set(case['task']['sources']):raise ValueError('source domain array coverage')
        for domain in case['task']['sources']:
            expected=role_indices(records,case,domain,side);mask=arrays['domain']==domain
            if not np.array_equal(arrays['y'][mask],expected['y']) or not np.array_equal(arrays['g'][mask],expected['g']):raise ValueError('label/record order differs from bound source roles')
            target[domain]=SourceWindows(domain,'train' if side=='tr' else 'val',torch.from_numpy(np.ascontiguousarray(arrays['x'][mask])),expected['y'],expected['g'],expected['indices'])
    return BoundSources(case['task']['sources'],[case['task']['target']],train,val,case['task']['classes'],fingerprint)


def matching_pair_counts(domain_inventory):
    """Exact pair arithmetic before constructing embeddings or any NN."""
    n=np.asarray([v['tr']['class_counts'] for v in domain_inventory.values()],dtype=np.int64)
    fault=0
    for cls in range(n.shape[1]):
        ref=int(np.argmax(n[:,cls]));fault+=int(n[ref,cls])*int(n[:,cls].sum()-n[ref,cls])
    domain=0
    for row in n:
        ref=int(np.argmax(row));domain+=int(row[ref])*int(row.sum()-row[ref])
    return dict(fault_pairs_per_refresh=fault,domain_pairs_per_refresh=domain,total_pairs_per_refresh=fault+domain)
