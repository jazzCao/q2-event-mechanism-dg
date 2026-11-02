"""Reproduce all old source roles before extending the frozen panel to seed9.

Only CSV/JSON and original integer split NPZ metadata are read. No signal,
feature, model or prediction arrays are opened by this compiler.
"""
from collections import Counter
from pathlib import Path
import ast
import copy
import csv
import gzip
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/wfdg/ress_tenseed_all'

def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def need(v,m):
    if not v:raise AssertionError(m)
def write(p,v):
    b=(json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    if str(p).endswith('.gz'):b=gzip.compress(b,mtime=0)
    with Path(p).open('xb') as f:f.write(b)

def split_function():
    import numpy as np
    path=Path(__file__).with_name('wfdg')/'data.py'
    tree=ast.parse(path.read_text('utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='grouped_indices_class_guard')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[node.name]

def rowmaps(records):
    maps={};bindings={}
    for proto in ('phm09_w3','jnu_speed'):
        p=ROOT/'data/processed'/proto/'provenance.csv';bindings[str(p)]=sha(p)
        with p.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                k=f'{proto}/{r["domain"]}/{r["file"]}'
                need(k in records,'known provenance recording '+k)
                need(records[k]['label']==int(r['class']) and records[k]['window']==int(r['window']),'source metadata label/window')
                maps.setdefault((proto,r['domain']),[]).append((k,int(r['offset']),r['file'],int(r['class'])))
    return maps,bindings

def roles_for(task,seed,records,maps,grouped,jnu_roles):
    import numpy as np
    proto=task['protocol'];roles={}
    if proto=='pu_clock4':
        bearings=sorted({r['bearing'] for r in records.values() if r['protocol']==proto})
        need(len(bearings)==18 and not set(bearings)&{'K006','KA08','KA09'},'historical 18 bearing boundary')
        runs={b:set((np.random.RandomState(int(hashlib.sha256(f'CLOCK4_SPLIT|{b}|{seed}'.encode()).hexdigest()[:8],16)).permutation(20)[:16]+1).tolist()) for b in bearings}
        for k,r in records.items():
            if r['protocol']==proto and r['domain'] in task['sources']:
                tr=r['run'] in runs[r['bearing']]
                roles[k]=dict(tr=copy.copy(r['offsets']) if tr else [],va=[] if tr else copy.copy(r['offsets']))
    elif proto=='jnu_speed':return copy.deepcopy(jnu_roles[task['fold']])
    else:
        for d in task['sources']:
            rows=maps[proto,d];groups=np.array([r[2] for r in rows]);labels=np.array([r[3] for r in rows])
            tr,va,_=grouped(groups,labels,seed)
            for side,ids in (('tr',tr),('va',va)):
                for index in ids:
                    k,offset,_,_=rows[int(index)];roles.setdefault(k,dict(tr=[],va=[]))[side].append(offset)
    return roles

def case_for(base,seed,roles,records):
    from inventory_protected_response import covered
    task=base['task'];proto=task['protocol'];window=task['window'];bank=[];pairs={}
    speeds=[records[k]['shaft_hz'] for k in roles];diversity=max(speeds)-min(speeds)>records[next(iter(roles))]['fs']/32768
    for k,v in sorted(roles.items()):
        if not v['tr']:continue
        r=records[k];need(covered(v['tr'],window),'TRAIN covers full inverse context '+k)
        need(not v['va'] or max(v['tr'])+window<=min(v['va']),'physical source train/val intervals')
        if proto=='pu_clock4' and not diversity:
            opts=[q for q,sides in roles.items() if sides['tr'] and records[q]['bearing']==r['bearing'] and records[q]['run']==r['run'] and records[q]['domain']!=r['domain']]
            opts.sort(key=lambda q:(records[q]['domain']!='C',q))
        elif diversity:
            opts=[q for q,sides in roles.items() if sides['tr'] and records[q]['label']==r['label'] and abs(records[q]['shaft_hz']-r['shaft_hz'])>r['fs']/32768]
            if proto=='pu_clock4':
                opts=[q for q in opts if records[q]['bearing']==r['bearing']]
                opts.sort(key=lambda q:(records[q]['run']!=r['run'],records[q]['domain']!='C',records[q]['run'],q))
            else:opts.sort()
        else:
            opts=[q for q,sides in roles.items() if sides['tr'] and records[q]['label']==r['label'] and records[q]['configuration']==r['configuration'] and records[q]['domain']!=r['domain']]
            opts.sort(key=lambda q:(records[q]['rep']!=r['rep'],q))
        need(bool(opts),'same frozen pairing rule has support '+k)
        pair=sorted((k,opts[0]))
        if proto=='pu_clock4' and not diversity:
            # Historical same-clock IDs used default JSON separators/ASCII.
            pk=hashlib.sha256(json.dumps(dict(pair=pair,context=[65536,32768],mode='SAME_CLOCK_LOAD_PAIR_V1'),sort_keys=True,ensure_ascii=True).encode()).hexdigest()[:24]
        else:pk=hashlib.sha256(('|'.join(pair)+'|65536|32768'+('' if diversity else '|SAME_CLOCK_CONFIG_PAIR_V1')).encode()).hexdigest()[:24]
        pairs[pk]=pair;bank.append(dict(record=k,starts=[5792+i*window for i in range(4)],pair=pk,source=pair.index(k)))
    for d in task['sources']:
        need({records[k]['label'] for k,sides in roles.items() if sides['tr'] and records[k]['domain']==d}==set(range(task['classes'])),'all source TRAIN classes '+d)
    c=copy.deepcopy(base)
    for field in ('inventory','reused','already_has_core_results'):c.pop(field,None)
    c.update(seed=seed,roles=roles,bank=bank,pairs=pairs,eligible=True,source_speed_diversity=diversity,intervention_available=True,
        training_windows=sum(len(v['tr']) for v in roles.values()),validation_windows=sum(len(v['va']) for v in roles.values()),bank_windows=4*len(bank))
    return c

def run():
    from protected_response_run_spec import load_plan
    old=load_plan();records=old['records'];maps,inputs=rowmaps(records);grouped=split_function()
    bases={c['task']['protocol']+'/'+c['task']['fold']:c for c in old['cases'].values() if c['seed']==0}
    jnu={c['task']['fold']:c['roles'] for c in bases.values() if c['task']['protocol']=='jnu_speed'}
    cases={};golden=[];fields=('task','seed','roles','bank','pairs','eligible','training_windows','validation_windows','bank_windows')
    for taskkey,base in sorted(bases.items()):
        for seed in range(10):
            key=taskkey+'/s'+str(seed);roles=roles_for(base['task'],seed,records,maps,grouped,jnu);c=case_for(base,seed,roles,records)
            if seed<3:
                expected=old['cases'][key]
                for field in fields:need(c[field]==expected[field],'golden old source reproduction '+key+' '+field)
                golden.append(key)
            else:cases[key]=c
    need(len(golden)==81 and len(cases)==189,'81 exact old role checks before189 new seed cases')
    oldpairs={tuple(v['binding']['pair']):dict(identity=k,binding=v['binding']) for k,v in old['inverses'].items()}
    union={pk:pair for c in cases.values() for pk,pair in c['pairs'].items()}
    reusable={pk:oldpairs[tuple(pair)] for pk,pair in union.items() if tuple(pair) in oldpairs}
    missing={pk:pair for pk,pair in union.items() if pk not in reusable}
    nnpath=ROOT/'results/wfdg/ress_event_full_validation/NEURAL_LOCK_V1.json.gz';nn=json.loads(gzip.decompress(nnpath.read_bytes()))
    inputs.update(old['input_sha256']);inputs[str(nnpath)]=sha(nnpath)
    result=dict(status='OLD81_SOURCE_ROLES_EXACT_NEW189_MANIFEST',cases=cases,records=records,features=nn['feature_bindings'],
        FFT_code_sha256=nn['expected_FFT_code_sha256'],golden_source_checks=golden,reused_inverse_assets=reusable,
        missing_inverse_pairs=missing,all_pairs=union,input_sha256=inputs,code_sha256={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('wfdg')/'data.py'):sha(Path(__file__).with_name('wfdg')/'data.py')},
        counts=dict(cases=189,tasks=27,new_seeds=7,source_pairs=len(union),reusable_inverse_pairs=len(reusable),new_inverse_pairs=len(missing)),
        new_signal_feature_weight_prediction_reads=0,new_neural_actions=0)
    result['fingerprint']=canonical(result);write(OUT/'MDG_SOURCE_MANIFEST_V1.json.gz',result)
    write(OUT/'MDG_SOURCE_MANIFEST_CHECKS_V1.json',dict(status='PASS',old_source_cases_exact=81,fields=list(fields),counts=result['counts'],fingerprint=result['fingerprint'],new_neural_actions=0))
    print('TEN_REPEAT_SOURCE_MANIFEST',result['counts'],flush=True)

if __name__=='__main__':run()
