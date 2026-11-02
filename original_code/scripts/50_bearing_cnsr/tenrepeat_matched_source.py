"""Repeat the nine frozen matched recipes on seed3..9, source stage only.

The original classifier, optimizer, six-B loss, random draws and batch
adapters are reused verbatim. Campaign identities and sizes are extended.
"""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import copy
import gzip
import json
import sys
import time

from tenrepeat_source_metadata import ROOT,OUT,read,sha,canonical,need,write

ARMS=('A0_STRONG_IN_MIXUP','F0_UNEDITED_BANK','A1_MATCHED_E0','Q2_FIXED_IMPULSE','Q2','A0_IMPULSE_ONLY','M0_GLOBAL_AFFINE_MOMENTS','T0_HALF_EXPOSURE','M0_EXPECTED_ENERGY_SHRINK')
PROTO=('jnu_speed','pu_clock4','phm09_w3')
P0_FOLD=dict(jnu_speed='A',pu_clock4='ABCtoD',phm09_w3='adj_AC_B')

def load_manifest():return json.loads(gzip.decompress((OUT/'MDG_SOURCE_MANIFEST_V1.json.gz').read_bytes()))

def initials():
    import torch
    from clock_real_model import Net
    from source_rate_neural import initial_seed,tensor_sha
    from protected_response_run_spec import load_plan
    old=load_plan();m=load_manifest()
    with gzip.open(ROOT/'results/wfdg/ress_event_full_validation/NEURAL_LOCK_V1.json.gz','rt',encoding='utf-8') as f:parent=json.load(f)
    hashes={};checks=0
    for key,c in {**old['cases'],**m['cases']}.items():
        torch.manual_seed(initial_seed(c['task'],c['seed']));net=Net(c['task']['classes'],'E0_EVENTS')
        value=tensor_sha(net.enc.state_dict())
        if key in parent['initial_encoder_sha256']:
            need(value==parent['initial_encoder_sha256'][key],'CPU initialization exactly reproduces old encoder '+key);checks+=1
        else:hashes[key]=value
        del net
    need(checks==81 and len(hashes)==189,'all old/new initialization identities')
    write(OUT/'INITIAL_ENCODERS_V1.json',dict(status='PASS',old81_exact=True,new189=hashes,scope='CPU encoder initialization only; not training/CUDA',new_optimizer_updates=0))
    print('TEN_REPEAT_CPU_INITIALS_PASS_81_NEW189',flush=True)

def prepare(proto):
    from source_rate_screen_plan import expected,paths
    from e0_selection_entry_spec import local_closure
    need(proto in PROTO,'registered existing dataset only')
    here=OUT/'matched_source'/proto;here.mkdir(parents=True,exist_ok=True)
    m=load_manifest();source=read(OUT/'mdg_banks/SOURCE_BANKS_V1.json');initial=read(OUT/'INITIAL_ENCODERS_V1.json')
    need(source['status']=='ALL189_NEW_SEED_SOURCE_BANKS_COMPLETE','all source bank inputs actual and frozen before neural lock')
    selected={k:c for k,c in m['cases'].items() if c['task']['protocol']==proto}
    source_lock=json.loads(gzip.decompress((OUT/'mdg_banks/LOCK_V2.json.gz').read_bytes()))
    cases={k:dict(source_lock['cases'][k],real_arms=list(ARMS)) for k in selected}
    need(len(cases)==dict(jnu_speed=21,pu_clock4=28,phm09_w3=140)[proto],'exact dataset case count')
    store=Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_tenseed_all/matched_source')/proto
    chains={};checkpoints={};predictions={};vectors={}
    p0=f'{proto}/{P0_FOLD[proto]}/s3'
    def add(k,arm,qualification=False,branch=None):
        name=('P0/' if qualification else 'MAIN/')+k+'/'+arm+('/'+branch if qualification else '')
        chains[name]=dict(case_key=k,arm=arm,qualification=qualification,branch=branch,steps=5 if qualification else 2000)
        for step in ((5,) if branch=='continuous' else (2,5) if qualification else range(0,2001,250)):
            checkpoints[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        vector=expected(m['records'],selected[k],'va');vk=('qualification_source/' if qualification else 'source/')+k
        vectors[vk]={q:v[:32] for q,v in vector.items()} if qualification else vector
        for step in ((0,5) if qualification else (0,1000,2000)):
            predictions[paths(store,name,'source',step)]=dict(chain=name,case_key=k,side='source',step=step,qualification=qualification,expected_key=vk)
    for arm in ARMS:
        for branch in ('continuous','resume'):add(p0,arm,True,branch)
    for k in cases:
        for arm in ARMS:add(k,arm)
    n=len(cases)*9;batch=120//(len(next(iter(cases.values()))['task']['sources'])*next(iter(cases.values()))['task']['classes'])*len(next(iter(cases.values()))['task']['sources'])*next(iter(cases.values()))['task']['classes']
    limits=dict(optimizer_attempts=n*2000+90,formal_updates=n*2000,qualification_updates=90,
        training_forward_windows=(n*2000+90)*6*batch,
        source_prediction_windows=sum(len(vectors[p['expected_key']]['y']) for p in predictions.values()),target_prediction_windows=0,
        neural_model_builds=n*5+128,checkpoint_writes=n*9+27,checkpoint_reads=n*4+64,checkpoint_hashes=n*32+512,
        probability_writes=n*3+36,probability_reads=n*3+64,probability_hashes=n*9+256,
        feature_hashes=17,feature_decodes=17,bank_hashes=len(cases),bank_decodes=len(cases),batch_reference_rows=3*12*batch)
    now=datetime.now(timezone(timedelta(hours=8)))
    hours=dict(jnu_speed=8,pu_clock4=12,phm09_w3=48)[proto]
    inputs=[OUT/'MDG_SOURCE_MANIFEST_V1.json.gz',OUT/'mdg_banks/SOURCE_BANKS_V1.json',OUT/'mdg_banks/LOCK_V2.json.gz',OUT/'INITIAL_ENCODERS_V1.json',OUT/'HUMAN_SCOPE_V1.md']
    code=local_closure([Path(__file__)])
    lock=dict(kind='TEN_REPEAT_NINE_MATCHED_RECIPES_SOURCE_ONLY',campaign=dict(start=now.isoformat(),deadline=(now+timedelta(hours=hours+8)).isoformat(),neural_seconds=hours*3600,store=str(store)),
        cases=cases,original_cases=selected,records=m['records'],banks={k:source['banks'][k] for k in cases},feature_bindings=m['features'],
        initial_encoder_sha256={k:initial['new189'][k] for k in cases},expected_FFT_code_sha256=m['FFT_code_sha256'],
        chains=chains,checkpoints=checkpoints,predictions=predictions,vectors=vectors,limits=limits,P0_case=p0,arms=list(ARMS),models=n,
        control_filename=str(here/'CONTROL.sqlite'),input_sha256={str(p):sha(p) for p in inputs},code_sha256=code,
        source_only=True,target_requires_all3822_source_models=True)
    lock['fingerprint']=canonical(lock);write(here/'LOCK_V1.json.gz',lock)
    need(len(checkpoints)==limits['checkpoint_writes'] and len(predictions)==limits['probability_writes'],'complete schedule counts')
    write(here/'STATIC_CHECKS_V1.json',dict(status='PASS',models=n,formal_updates=2000*n,P0_updates=90,source_windows=limits['source_prediction_windows'],target_windows=0,actual_inputs_ready=True,matching_math='unchanged original adapters and Chain',new_neural_actions=0))
    print('TEN_REPEAT_MATCHED_SOURCE_READY',proto,n,limits['source_prediction_windows'],flush=True)

def classes():
    from event_component_neural import Data as ComponentData
    from event_full_validation_neural import Data as CoreData
    from q2_exposure_neural import Data as ExposureData
    from interval_coverage_neural import Chain as CoreChain
    from source_rate_screen_runtime import Runtime as BaseRuntime
    class Data(ComponentData):
        def batch(self,arm,sampler,augmentation):
            if arm in ARMS[:5]:return CoreData.batch(self,arm,sampler,augmentation)
            if arm in ARMS[5:7]:return ComponentData.batch(self,arm,sampler,augmentation)
            return ExposureData.batch(self,arm,sampler,augmentation)
    class Chain(CoreChain):
        def step(self,inspect=False):
            v=super().step(inspect)
            if self.arm in ARMS[7:] and self.step_n%100==0:self.losses[-1].update(self.data.last_exposure_counts)
            return v
    return Data,Chain,BaseRuntime

def qualify(rt,files,here):
    from source_rate_neural import exact
    from run_source_rate_screen import source_score,save_chain,scientific_packet
    Data,Chain,_=classes();case=rt.lock['cases'][rt.lock['P0_case']];source=files.source(case);data=Data(case,source,files.bank(case),rt)
    need(rt.get('P0_PASS') is None and not rt.db.execute('SELECT 1 FROM steps LIMIT 1').fetchone(),'new bounded qualification, no replay')
    proofs=[]
    for arm in ARMS:
        outputs=[]
        for branch in ('continuous','resume'):
            name='P0/'+case['key']+'/'+arm+'/'+branch;chain=Chain(rt,case,arm,name,data)
            if branch=='continuous' and arm in ('A0_STRONG_IN_MIXUP','A1_MATCHED_E0','Q2'):data.original_batch_check(chain)
            first=source_score(rt,chain,source,True)
            for _ in range(5):
                chain.step(inspect=chain.step_n==0)
                if branch=='resume' and chain.step_n==2:
                    path=save_chain(rt,chain);before=chain.packet();del chain
                    chain=Chain(rt,case,arm,name,data,rt.reopen(path));exact(before,chain.packet(),chain.torch,chain.np,'new-seed real CUDA disk2 restore')
            last=source_score(rt,chain,source,True);save_chain(rt,chain)
            outputs.append((scientific_packet(chain.packet()),first['prob'],last['prob']));torch,np=chain.torch,chain.np;del chain
        for k in range(3):exact(outputs[0][k],outputs[1][k],torch,np,'ten-repeat actual continuous5/disk2+3 state/probability equality')
        proofs.append(dict(arm=arm,resume_exact=True,nonzero_finite_gradients=True))
    draws=rt.get('P0_DRAWS');need(len(draws)==18,'all nine-arm qualification draw logs')
    logs=list(draws.values());need(all(v==logs[0] for v in logs),'all nine original sampler/augmentation streams equal')
    need(rt.counts['qualification_updates']==90,'ninety qualification updates actual')
    value=dict(status='PASS',updates=90,proofs=proofs,source_only=True,target_predictions=0,at=time.time())
    write(here/'CUDA_P0_V1.json',value)
    with rt.db:rt.set('P0_PASS',value)
    del data;torch.cuda.empty_cache();print('TEN_REPEAT_MATCHED_CUDA_PASS',here.name,flush=True)

def train(rt,files,here):
    from run_source_rate_screen import source_score,save_chain
    from source_rate_screen_plan import paths
    from source_rate_neural import exact
    Data,Chain,_=classes()
    for k,case in sorted(rt.lock['cases'].items()):
        source=files.source(case);data=Data(case,source,files.bank(case),rt)
        for arm in ARMS:
            name='MAIN/'+k+'/'+arm
            need(name not in rt.finished(),'no automatic finished-model retraining')
            chain=Chain(rt,case,arm,name,data);source_score(rt,chain,source);save_chain(rt,chain)
            while chain.step_n<2000:
                chain.step(inspect=chain.step_n==0)
                if chain.step_n in (1000,2000):source_score(rt,chain,source)
                if chain.step_n%250==0:
                    path=save_chain(rt,chain)
                    if chain.step_n==1000:
                        packet=chain.packet();del chain;chain=Chain(rt,case,arm,name,data,rt.reopen(path));exact(packet,chain.packet(),chain.torch,chain.np,'formal disk1000 complete continuation')
            detail=dict(last=paths(rt.store,name,'checkpoint',2000),selected=paths(rt.store,name,'checkpoint',chain.selected),selected_step=chain.selected,
                case_key=k,arm=arm,curve=chain.curve,losses=chain.losses,actual_disk1000_restore=True,completed_at_unix=time.time())
            rt.finish(name,detail);torch=chain.torch;del chain
            print('TEN_REPEAT_MATCHED_MODEL',here.name,len(rt.finished()),'/',rt.lock['models'],name,flush=True)
        del data;torch.cuda.empty_cache()
    need(len(rt.finished())==rt.lock['models'] and rt.counts['formal_updates']==rt.lock['models']*2000,'complete fixed repeat panel')
    # Independent source selection reconstruction and actual saved state links.
    states={};audit=[]
    for name,detail in rt.finished().items():
        chosen=min(detail['curve'],key=lambda r:(-r['score'],r['step']))
        need(chosen['step']==detail['selected_step'],'strict earliest source-only selection')
        scores=[]
        for row in detail['curve']:
            a=rt.reopen(row['probability']);prob=a['prob'];labels=a['y'];domains=a['domain']
            score=float(__import__('numpy').mean([(prob[domains==d].argmax(1)==labels[domains==d]).mean() for d in rt.lock['cases'][detail['case_key']]['task']['sources']]))
            need(score==row['score'],'saved source probability reconstruction')
            path=paths(rt.store,name,'checkpoint',row['step']);packet=rt.reopen(path)
            from source_rate_neural import tensor_sha
            need(tensor_sha(packet['model'])==str(a['model_tensor_sha256']),'actual source probability/weight binding')
            need(packet['step']==row['step'] and packet['fingerprint']==rt.lock['fingerprint'],'source state cursor/fingerprint')
            states[path]=rt.artifact(path)['sha256'];scores.append([row['step'],score])
        audit.append(dict(chain=name,selection=detail['selected_step'],saved_source_scores=scores))
    write(here/'SOURCE_AUDIT_V1.json',dict(status='PASS',models=len(audit),checks=audit,target_predictions=0))
    write(here/'SOURCE_STAGE_COMPLETE_V1.json',dict(status='SOURCE_COMPLETE_TARGET_PENDING_MASTER',fingerprint=rt.lock['fingerprint'],models=rt.finished(),states=states,counts=rt.counts,at=time.time()))
    with rt.db:rt.set('SOURCE_STAGE_DONE',dict(at=time.time(),models=len(audit)))

def run(proto):
    from event_full_validation_neural import Files
    from source_rate_screen_runtime import Guard
    from source_rate_neural import libraries
    here=OUT/'matched_source'/proto;lock=json.loads(gzip.decompress((here/'LOCK_V1.json.gz').read_bytes()))
    need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'immutable repeat lock')
    for p,h in {**lock['input_sha256'],**lock['code_sha256']}.items():need(sha(p)==h,'bound repeat code/input '+p)
    _,_,Runtime=classes();rt=Runtime(lock);sys.addaudithook(Guard(rt))
    try:
        libraries(rt);files=Files(rt);qualify(rt,files,here);train(rt,files,here)
        print('TEN_REPEAT_MATCHED_SOURCE_COMPLETE',proto,lock['models'],'NO_TARGET',flush=True)
    except BaseException as e:
        with rt.db:rt.set('STOPPED',dict(type=type(e).__name__,message=str(e),counts=rt.counts,at=time.time()))
        raise
    finally:rt.db.close()

if __name__=='__main__':
    if sys.argv[1]=='initials':initials()
    elif sys.argv[1]=='prepare':prepare(sys.argv[2])
    elif sys.argv[1]=='run':run(sys.argv[2])
