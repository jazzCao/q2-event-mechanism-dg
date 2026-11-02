"""Complete zero-neural numerical renderer qualification, metadata-only compile."""
from datetime import datetime
from pathlib import Path
import importlib.metadata
import math
import source_rate_plan as old
from e0_selection_entry_spec import local_closure
ROOT=old.ROOT
OUT=ROOT/'results/wfdg/ress_continuous_event_operator'
PARENT=ROOT/'results/wfdg/ress_rate_impulse_repeat'
read,save,sha,canonical,require=old.read,old.save,old.sha,old.canonical,old.require
ENTRY=('continuous_event_core.py','continuous_event_plan.py','run_continuous_event_core.py','audit_continuous_event_core.py','test_continuous_event_metadata.py','clock_real_core.py','clock_real_data.py')


def ratios():
    a=math.log(1.5);hi=math.sqrt((3+2*math.sqrt(6/5))/7);lo=math.sqrt((3-2*math.sqrt(6/5))/7)
    return [dict(name=n,value=q,legacy=True) for n,q in (('two_thirds',2/3),('identity',1.),('three_halves',1.5))]+[
        dict(name='gauss4_'+str(i),value=math.exp(a*x),legacy=False) for i,x in enumerate((-hi,-lo,lo,hi))]


def fixtures():
    out=[dict(key='synthetic/s'+str(i).zfill(2),n=2048,channels=4,size=64,kind='regular',seed=461+i,ratios=9) for i in range(27)]
    out += [dict(key='synthetic/d'+str(i),n=64,channels=3,size=size,kind='dense',seed=810+i,ratios=9) for i,size in enumerate((8,16,24,32))]
    out += [dict(key='synthetic/production',n=32768,channels=48,size=256,kind='production',seed=910,ratios=7)]
    return out


def clock(scope):require(datetime.now().astimezone()<datetime.fromisoformat(scope['deadline']),'original90min total clock')


def verify(lock):
    clock(lock['campaign']);require(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'immutable numerical qualification lock')
    for path,h in {**lock['input_sha256'],**lock['code_sha256']}.items():require(sha(path)==h,'changed bound code/input '+path)


def metadata():
    from natural_mapping_probe_plan import source_case
    source=read(PARENT/'SOURCE_LOCK_V1.json');banks=read(PARENT/'SOURCE_BANKS_V1.json');closed=read(PARENT/'CLOSED_RECEIPT_V1.json')
    require(read(PARENT/'CLOSED_POSTCHECK_V1.json')['status']=='PASS_ALL90_ROLES_NEW_SEED_ONLY_AND_TIMING','actual completed source parent')
    for n in ('SOURCE_BANKS_V1.json','SCREEN_RESULT_V1.json'):require(sha(PARENT/n)==closed['files_sha256'][n],'closed actual parent '+n)
    groups={g['key']:g for g in source['groups']};cases={c['key']:c for c in source['cases']};units=[]
    for key in ('pu_clock4/ABCtoD/s2','phm09_w3/adj_AC_B/s2','jnu_speed/A/s2'):
        case=cases[key];item=case['bank'][0];g=groups[item['group']];original=source['original_cases'][key]
        for record in g['pair']:source_case(original,source['records'],record,context=True)
        require(g['origin']==item['record'] and g['origin'] in g['pair'],'deterministic first legal bank group')
        a=source['assets'][g['inverse_identity']]
        bindings=dict(inverse=a['inverse'],context=a['qualified_context'],anchor=banks['groups'][g['key']])
        bindings={k:dict(b,size_bytes=Path(b['path']).stat().st_size) for k,b in bindings.items()}
        units.append(dict(key='source/'+key,case_key=key,group=g,n=32768,channels=48,size=g['kernel_length'],kind='source',ratios=7,bindings=bindings))
    return units


def forecast(source):
    units=fixtures()+source;new_calls=sum(v['ratios']+(3 if v['kind']=='source' else 0) for v in units)
    modes=sum(3*v['n']*v['channels']*(v['ratios']+(3 if v['kind']=='source' else 0)) for v in units)
    filters=sum(3*v['channels']*(v['ratios']+(3 if v['kind']=='source' else 0)) for v in units)
    legacy_calls=32*3+3*6;fft_count=fft_points=0
    for v in units:
        repeat=2 if v['kind']=='source' else 1
        count=repeat*(12*v['channels']+3);nfft=1<<(v['n']+v['size']-1).bit_length()
        fft_count+=count;fft_points+=count*nfft
    front=45*3;frontpoints=45*sum(v['group']['window'] for v in source)
    direct_calls=sum(v['ratios']*(3 if v['kind']=='source' else 1) for v in units)
    scans=modes//3+sum(v['ratios']*v['channels']*v['n']*(3 if v['kind']=='source' else 1) for v in units)
    return dict(renderer_calls=new_calls,mode_sample_updates=modes,filter_calls=filters,legacy_renderer_calls=legacy_calls,
        direct_renderer_calls=direct_calls,event_grid_scans=scans,
        legacy_fft_1d_transforms=fft_count,legacy_fft_points=fft_points,frontend_fft_1d_transforms=front,frontend_fft_points=frontpoints,
        source_payload_reads=9,source_payload_bytes=sum(b['size_bytes'] for u in source for b in u['bindings'].values()),
        source_hashes=9,owned_writes=35,owned_reads=35,owned_hashes=70)


def prepare():
    scope=read(OUT/'CAMPAIGN_SCOPE_V1.json');clock(scope);source=metadata();tests=read(OUT/'METADATA_TESTS_V2.json')
    require(tests['status']=='PASS' and not tests['torch_imported'],'actual metadata-only readiness tests')
    code=local_closure([Path(__file__).with_name(n) for n in ENTRY]);require(code==tests['code_sha256'],'exact tested full implementation')
    resources=forecast(source)
    for k,n in resources.items():require(n<=scope['limits'][k],'entire numerical comparison fits '+k)
    require(resources['mode_sample_updates']==180580608 and resources['filter_calls']==8568 and resources['renderer_calls']==316,'fixed full32synthetic/3source scope')
    outputs={u['key']:str(Path(scope['store'])/'packets'/(canonical(u['key'])+'.npz')) for u in fixtures()+source}
    paths=(OUT/'CAMPAIGN_SCOPE_V1.json',OUT/'METADATA_TESTS_V2.json',OUT/'RESEARCH_CARD_V1.md',OUT/'IMPLEMENTATION_SPEC_V1.md',
        PARENT/'SOURCE_LOCK_V1.json',PARENT/'SOURCE_BANKS_V1.json',PARENT/'CLOSED_RECEIPT_V1.json',PARENT/'CLOSED_POSTCHECK_V1.json',
        ROOT/'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md')
    lock=dict(kind='REAL_RATIO_SAME_FINITE_KERNEL_QUALIFICATION_V1',campaign=scope,fixtures=fixtures(),sources=source,ratios=ratios(),outputs=outputs,
        thresholds=dict(nrmse=1e-9,maxabs_scaled=1e-8,frontend_maxabs=1e-5),forecast=resources,limits=scope['limits'],
        input_sha256={str(p):sha(p) for p in paths},code_sha256=code,neural_actions=0,new_raw_reads=0,new_inverse_fits=0,target_reads=0,
        environment={n:importlib.metadata.version(n) for n in ('numpy','scipy')})
    lock['fingerprint']=canonical(lock);save(OUT/'RUN_LOCK_V1.json',lock);return lock
