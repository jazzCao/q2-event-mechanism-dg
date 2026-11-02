"""Fixed component questions; the closed full27 performance terminal stays intact."""
from collections import defaultdict
from statistics import fmean
import event_component_execution as e

FIELDS=('acc','macro_f1','recording_ba')
ALL_ARMS=(*e.old.ARMS,*e.ARMS)


def aggregate(rows,policy):
    output={}
    for arm in ALL_ARMS:
        tasks=defaultdict(list)
        for r in rows:
            if r['arm']==arm and r['role']==policy:tasks[r['protocol'],r['fold']].append(r['metrics'])
        means={k:{f:fmean(v[f] for v in vs) for f in FIELDS} for k,vs in tasks.items()}
        protocols=sorted({p for p,_ in means})
        per={p:{f:fmean(v[f] for (ds,_),v in means.items() if ds==p) for f in FIELDS} for p in protocols}
        output[arm]=dict(per_protocol=per,dataset_equal={f:fmean(v[f] for v in per.values()) for f in FIELDS},
            task_equal={f:fmean(v[f] for v in means.values()) for f in FIELDS})
    return output


def decide(rows,cases,aggregator=aggregate):
    wanted={(c['task']['protocol'],c['task']['fold'],c['seed'],a,role) for c in cases for a in ALL_ARMS for role in ('last','selected')}
    actual=[(r['protocol'],r['fold'],r['seed'],r['arm'],r['role']) for r in rows]
    e.need(len(actual)==len(wanted)==1134 and set(actual)==wanted,'all seven fixed rows/27tasks/three seeds/two policies')
    primary=aggregator(rows,'last');secondary=aggregator(rows,'selected')
    per_seed={str(s):aggregator([r for r in rows if r['seed']==s],'last') for s in (0,1,2)}
    targets={(c['task']['protocol'],c['task']['fold']):c['task']['target'] for c in cases}
    leave={p+'/'+t:aggregator([r for r in rows if not (r['protocol']==p and targets[r['protocol'],r['fold']]==t)],'last')
        for p,t in sorted({(p,t) for (p,_),t in targets.items()})}
    checks=[];labels={}
    def add(control,name,value,bound=0.,strict=False):
        passed=value>bound+1e-12 if strict else value+1e-12>=bound
        checks.append(dict(control=control,name=name,value=value,bound=bound,strict=strict,passed=bool(passed)))
    for control in e.ARMS:
        q,b=primary['Q2']['dataset_equal'],primary[control]['dataset_equal']
        add(control,'Acc_increment_0p10pp',q['acc']-b['acc'],.001)
        for field in ('macro_f1','recording_ba'):add(control,field+'_nonnegative',q[field]-b[field])
        add(control,'strict_positive_seed_count',sum(v['Q2']['dataset_equal']['acc']>v[control]['dataset_equal']['acc']+1e-12 for v in per_seed.values()),2)
        for target,v in leave.items():add(control,'without_'+target+'_Acc',v['Q2']['dataset_equal']['acc']-v[control]['dataset_equal']['acc'],strict=True)
        labels[control]=dict(status='STABLE_INCREMENT_OVER_CONTROL' if all(v['passed'] for v in checks if v['control']==control) else 'COMPONENT_SPECIFICITY_UNRESOLVED',
            delta={f:q[f]-b[f] for f in FIELDS},control_covers_Q2_mean_Acc=b['acc']+1e-12>=q['acc'])
    # Effects are contrasts, not an added objective or an assertion of positive interaction.
    factorial={}
    for policy,panel in (('last',primary),('selected',secondary)):
        factorial[policy]={}
        for weight in ('dataset_equal','task_equal'):
            factorial[policy][weight]={}
            for field in FIELDS:
                f,t,a,q=[panel[arm][weight][field] for arm in ('F0_UNEDITED_BANK','Q2_FIXED_IMPULSE','A0_IMPULSE_ONLY','Q2')]
                factorial[policy][weight][field]=dict(timing_given_amplitude=q-a,amplitude_given_timing=q-t,
                    timing_alone=t-f,amplitude_alone=a-f,interaction=q-t-a+f)
    per_protocol_factorial={};paired_factorial={}
    def effects(values):
        f,t,a,q=values
        return dict(timing_given_amplitude=q-a,amplitude_given_timing=q-t,timing_alone=t-f,amplitude_alone=a-f,interaction=q-t-a+f)
    row_index={(r['protocol'],r['fold'],r['seed'],r['arm'],r['role']):r['metrics'] for r in rows}
    for policy,panel in (('last',primary),('selected',secondary)):
        per_protocol_factorial[policy]={p:{field:effects([panel[arm]['per_protocol'][p][field] for arm in ('F0_UNEDITED_BANK','Q2_FIXED_IMPULSE','A0_IMPULSE_ONLY','Q2')]) for field in FIELDS} for p in panel['Q2']['per_protocol']}
        paired_factorial[policy]=[dict(protocol=c['task']['protocol'],fold=c['task']['fold'],seed=c['seed'],
            **{field:effects([row_index[c['task']['protocol'],c['task']['fold'],c['seed'],arm,policy][field] for arm in ('F0_UNEDITED_BANK','Q2_FIXED_IMPULSE','A0_IMPULSE_ONLY','Q2')]) for field in FIELDS}) for c in cases]
    paired={policy:[dict(protocol=r['protocol'],fold=r['fold'],seed=r['seed'],
        **{field:next(x for x in rows if x['role']==policy and x['arm']=='Q2' and (x['protocol'],x['fold'],x['seed'])==(r['protocol'],r['fold'],r['seed']))['metrics'][field]-r['metrics'][field] for field in FIELDS},control=r['arm'])
        for r in rows if r['role']==policy and r['arm'] in e.ARMS] for policy in ('last','selected')}
    return dict(status='FIXED_Q2_COMPONENT_SPECIFICITY_SUPPORTED' if all(v['status']=='STABLE_INCREMENT_OVER_CONTROL' for v in labels.values()) else 'FIXED_Q2_COMPONENT_SPECIFICITY_UNRESOLVED',
        original_scientific_status='FULL_SCOPE_EVENT_OPERATOR_SUPPORTED',candidate='Q2',primary=primary,secondary=secondary,
        per_seed=per_seed,leave_one_target_domain=leave,control_evidence=labels,checks=checks,factorial=factorial,
        per_protocol_factorial=per_protocol_factorial,paired_factorial=paired_factorial,paired_control_differences=paired,
        global_SOTA_established=False,publication_ready=False,causal_latents_identified=False)
