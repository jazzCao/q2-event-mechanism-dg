"""Prospective full-panel performance and matched-component decisions."""
import event_full_validation_plan as p


def decide(rows,cases,aggregate):
    wanted={(c['task']['protocol'],c['task']['fold'],c['seed'],a,role) for c in cases for a in p.ARMS for role in ('last','selected')}
    actual=[(r['protocol'],r['fold'],r['seed'],r['arm'],r['role']) for r in rows]
    p.need(len(actual)==len(wanted)==810 and set(actual)==wanted,'all405 models/two policies before a scientific decision')
    primary=aggregate(rows,'last');secondary=aggregate(rows,'selected')
    per_seed={str(s):aggregate([r for r in rows if r['seed']==s],'last') for s in (0,1,2)}
    targets={(c['task']['protocol'],c['task']['fold']):c['task']['target'] for c in cases}
    leave={d+'/'+t:aggregate([r for r in rows if not (r['protocol']==d and targets[r['protocol'],r['fold']]==t)],'last')
           for d,t in sorted({(d,t) for (d,_),t in targets.items()})}
    checks=[];q=p.CANDIDATE

    def check(stage,name,value,bound=0.,strict=False):
        passed=value>bound+1e-12 if strict else value+1e-12>=bound
        checks.append(dict(stage=stage,name=name,value=value,bound=bound,strict=strict,passed=bool(passed)))

    for a in p.ARMS[:2]:
        for f,b in (('acc',.003),('macro_f1',0.),('recording_ba',0.)):
            check('performance',f+'_over_'+a,primary[q]['dataset_equal'][f]-primary[a]['dataset_equal'][f],b)
    for s,v in per_seed.items():
        check('performance','seed'+s+'_Acc_over_A0',v[q]['dataset_equal']['acc']-v[p.ARMS[0]]['dataset_equal']['acc'],strict=True)
    check('performance','F0_nonnegative_seed_count',sum(v[q]['dataset_equal']['acc']+1e-12>=v[p.ARMS[1]]['dataset_equal']['acc'] for v in per_seed.values()),2)
    for protocol,v in primary[q]['per_protocol'].items():
        check('performance',protocol+'_Acc_A0_floor',v['acc']-primary[p.ARMS[0]]['per_protocol'][protocol]['acc'],-.002 if protocol=='jnu_speed' else 0.)
    for domain,v in leave.items():
        for a in p.ARMS[:2]:check('performance','without_'+domain+'_Acc_over_'+a,v[q]['dataset_equal']['acc']-v[a]['dataset_equal']['acc'],strict=True)
    effect=all(v['passed'] for v in checks)
    for a in p.ARMS[2:4]:
        check('component','Acc_over_'+a,primary[q]['dataset_equal']['acc']-primary[a]['dataset_equal']['acc'],strict=True)
        check('component','macro_F1_over_'+a,primary[q]['dataset_equal']['macro_f1']-primary[a]['dataset_equal']['macro_f1'])
        check('component','nonnegative_seed_count_over_'+a,sum(v[q]['dataset_equal']['acc']+1e-12>=v[a]['dataset_equal']['acc'] for v in per_seed.values()),2)
    component=all(v['passed'] for v in checks if v['stage']=='component')
    status='FULL_SCOPE_EVENT_OPERATOR_NOT_SUPPORTED' if not effect else 'FULL_SCOPE_PERFORMANCE_SUPPORTED_COMPONENT_UNRESOLVED' if not component else 'FULL_SCOPE_EVENT_OPERATOR_SUPPORTED'
    return dict(status=status,primary=primary,secondary=secondary,per_seed=per_seed,leave_one_target_domain=leave,checks=checks,
        performance_gate=effect,component_gate=component,candidate=q,
        selection='existing_Q2_selected_from_exposed_development_comparison_not_retroactive_Q4_success',
        publication_ready=False,global_SOTA_established=False,causal_latents_identified=False)
