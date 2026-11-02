"""Independent saved-output arithmetic, no re-rendering or source payload reads."""
import math


def audit(rt,bindings):
    import numpy as np
    from statistics import median
    import continuous_event_plan as p
    reported=[];pairs=[];all_rows=[]
    specs={v['key']:v for v in rt.lock['fixtures']+rt.lock['sources']}
    p.require(set(bindings)==set(specs) and len(bindings)==35,'all32 synthetic and3 source packets')
    for key,b in bindings.items():
        spec=specs[key];a=rt.owned(b);rows=[]
        p.require(str(a['unit_key'])==key and str(a['fingerprint'])==rt.lock['fingerprint'],'saved numerical identity')
        p.require(a['events'].shape==(spec['channels'],spec['n']) and a['params'].shape==(spec['channels'],3),'saved full production dimensions')
        expected=[q['value'] for q in rt.lock['ratios']]
        if spec['ratios']==9:expected += [math.nextafter(1.,2.),math.nextafter(1.,0.)]
        p.require(np.array_equal(a['ratios'],expected) and a['rendered'].shape==(spec['ratios'],spec['n']),'all requested real-valued ratios, no rounding')
        def check(name,x,y,mode='wave'):
            x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64);err=(x-y).ravel();ref=y.ravel()
            ma=float(np.max(np.abs(err),initial=0));scale=max(1.,float(np.max(np.abs(ref),initial=0)))
            nr=float(np.linalg.norm(err)/max(math.sqrt(len(ref))*1e-12,float(np.linalg.norm(ref))))
            t=rt.lock['thresholds'];passed=ma<=t['frontend_maxabs'] if mode=='front' else ma<=t['maxabs_scaled']*scale and (mode=='lineage' or nr<=t['nrmse'])
            rows.append(dict(name=name,mode=mode,maxabs=ma,nrmse=nr,maxabs_scale=scale,passed=bool(passed)))
        for i,q in enumerate(expected):
            if spec['kind']=='source':
                p.require(np.array_equal(a['direct_starts'],spec['group']['starts']),'exact predeclared source query starts')
                for j,start in enumerate(a['direct_starts']):check('direct/'+str(i)+'/'+str(j),a['rendered'][i,int(start):int(start)+8],a['direct'][i,j])
            else:check('direct/'+str(i),a['rendered'][i],a['direct'][i])
        p.require(np.array_equal(a['legacy_indices'],[0,1,2]),'all three original rational anchors')
        for j,i in enumerate(a['legacy_indices']):
            lo,hi=[int(v) for v in a['legacy_ranges'][j]];n=spec['n'];q=expected[int(i)];center=(n-1)/2
            p.require(0<=lo<hi<=n and center+q*(lo-spec['size']+1-center)>=-1e-9 and center+q*(hi-1-center)<=n-1+1e-9,'legacy comparison only uses original observed support')
            check('legacy/'+str(i),a['rendered'][i,lo:hi],a['legacy'][j,lo:hi])
        if spec['kind']=='source':
            check('old_component_lineage',a['rendered'][1],a['original_component'],mode='lineage')
            for i,j in ((0,0),(2,1)):
                check('fixed_front/'+str(i),a['front_fixed'][i],a['anchor_fixed'][:,j],mode='front')
                check('coupled_front/'+str(i),a['front_coupled'][i],a['anchor_coupled'][:,j],mode='front')
            p.require(np.array_equal(a['front_fixed'][1],a['clean']) and np.array_equal(a['front_coupled'][1],a['clean']),'source q1 exact operator identity')
        original=b['detail']['checks'];p.require(len(rows)==len(original),'complete independent checks')
        for x,y in zip(rows,original):
            for f in ('name','mode','passed'):p.require(x[f]==y[f],'same independent error decision '+f)
            for f in ('maxabs','nrmse','maxabs_scale'):p.require(abs(x[f]-y[f])<=1e-12*max(1.,abs(y[f])),'independent numerical error '+f)
        for v in b['detail']['timing_pairs']:
            p.require(v['new_seconds']>0 and v['legacy_seconds']>0 and v['materialized_n']==spec['n'],'same objective paired timing')
            p.require(abs(v['legacy_over_new']-v['legacy_seconds']/v['new_seconds'])<1e-12,'paired speed ratio arithmetic')
            pairs.append(dict(key=key,**v))
        reported.append(dict(key=key,checks=rows));all_rows.extend(rows)
    p.require(len(all_rows)==469 and len(pairs)==9,'all469 numerical comparisons and9 paired timing cases')
    passed=all(v['passed'] for v in all_rows)
    return dict(status='ARBITRARY_REAL_RATIO_SAME_KERNEL_QUALIFIED_NOT_CLASSIFICATION_EVIDENCE' if passed else 'REAL_RATIO_SAME_KERNEL_NUMERICAL_QUALIFICATION_FAILED',
        numerical_gate=passed,checks=len(all_rows),failed=[dict(unit=u['key'],**v) for u in reported for v in u['checks'] if not v['passed']],
        max_wave_nrmse=max(v['nrmse'] for v in all_rows if v['mode']=='wave'),max_frontend_abs=max(v['maxabs'] for v in all_rows if v['mode']=='front'),
        independent_saved_errors=reported,timing_pairs=pairs,median_legacy_over_new=median(v['legacy_over_new'] for v in pairs),
        timing_boundary='same float64 full-context materialization for same legal source queries; SQLite reservation excluded from both timers; unsupported old edge samples excluded from equivalence, not direct synthetic validation',
        novelty='KNOWN_EXPONENTIAL_KERNEL_RECURRENCE_NOT_AN_OPTIMIZER_INVENTION',classification_tested=False,global_SOTA_established=False)
