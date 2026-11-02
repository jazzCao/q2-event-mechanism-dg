"""378 complete MOCD repeats, preserving both A400+B400 interpretations.

Only the accepted seed set in metadata/stream checks is extended. All model,
matching, AMP, objective, transition and source-selection code stays qualified.
"""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import ast
import gzip
import json
import os
import sys
import time
from tenrepeat_source_metadata import ROOT,OUT,read,sha,canonical,need,write

HERE=OUT/'mocd_source';PROFILES=('EQ_LITERAL','PROSE_STOPGRAD_SENSITIVITY')

def seed_adapter():
    import mocd_complete_source_adapter as original
    ns=dict(original.__dict__);tree=ast.parse(Path(original.__file__).read_text('utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('source_metadata','random_stream_spec','bind_sources')]
    changes=0
    for n in nodes:
        for v in ast.walk(n):
            if isinstance(v,ast.Tuple) and [q.value if isinstance(q,ast.Constant) else None for q in v.elts]==[0,1,2]:
                v.elts=[ast.Constant(value=i) for i in range(10)];changes+=1
    need(changes==2,'exactly two seed guards extended, no mathematics changed')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::metadata_seed_extension','exec'),ns)
    return ns

def prepare(version='V1'):
    from e0_selection_entry_spec import local_closure
    HERE.mkdir(parents=True,exist_ok=True);m=json.loads(gzip.decompress((OUT/'MDG_SOURCE_MANIFEST_V1.json.gz').read_bytes()));a=seed_adapter();cases={}
    for k,c in m['cases'].items():
        inv=a['source_metadata'](m['records'],c);cases[k]=dict(c,key=k,source_inventory=inv,source_training_windows=c['training_windows'],source_validation_windows=c['validation_windows'])
    qkeys=[k for k,c in cases.items() if c['task']['fold']==dict(pu_clock4='ABCtoD',phm09_w3='adj_AC_B',jnu_speed='A')[c['task']['protocol']]]
    need(len(qkeys)==21,'all seven new seeds and three source/class shapes qualified')
    training=sum(2*400*(14*len(c['task']['sources'])*c['task']['classes']+256) for c in cases.values())
    training+=sum(2*(128*len(cases[k]['task']['sources'])*cases[k]['task']['classes']+3072) for k in qkeys)
    matching=sum(160*c['training_windows'] for c in cases.values())+sum(16*cases[k]['training_windows'] for k in qkeys)
    source=sum(40*c['validation_windows'] for c in cases.values())+sum(4*cases[k]['validation_windows'] for k in qkeys)
    limits=dict(optimizer_attempts=378*800+840,actual_optimizer_updates=378*800+840,amp_skips=378*800+840,
        training_encoder_windows=training,matching_encoder_windows=matching,source_prediction_windows=source,target_prediction_windows=0,
        feature_hashes=17,feature_decodes=17,checkpoint_writes=378*11+126,checkpoint_reads=378*5+42,probability_writes=378*20+84,probability_reads=378*20+84)
    profiles_path=ROOT/'results/wfdg/ress_mocd_state_qualification/INTERPRETATION_PROFILES.json'
    parent=read(ROOT/'results/wfdg/ress_mocd_complete/PROPOSED_LOCK.json')
    for p,h in parent['math_code_sha256'].items():need(sha(p)==h,'original complete MOCD mathematics '+p)
    now=datetime.now(timezone(timedelta(hours=8)))
    lock=dict(kind='TEN_REPEAT_COMPLETE_MOCD_SOURCE_ONLY',campaign=dict(start=now.isoformat(),deadline=(now+timedelta(hours=120)).isoformat(),neural_seconds=72*3600,store='D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_tenseed_all/mocd_source'),
        cases=cases,original_cases=m['cases'],records=m['records'],feature_bindings=m['features'],expected_FFT_code_sha256=m['FFT_code_sha256'],
        profiles=read(profiles_path),models=378,P0_cases=qkeys,limits=limits,input_sha256={str(OUT/'MDG_SOURCE_MANIFEST_V1.json.gz'):sha(OUT/'MDG_SOURCE_MANIFEST_V1.json.gz'),str(profiles_path):sha(profiles_path)},
        code_sha256=local_closure([Path(__file__)]),math_code_sha256=parent['math_code_sha256'],seed_guard_changes=2,source_only=True)
    lock['fingerprint']=canonical(lock);write(HERE/f'LOCK_{version}.json.gz',lock);print('TEN_REPEAT_COMPLETE_MOCD_READY_378',limits['optimizer_attempts'],flush=True)

def run(version='V1'):
    lock=json.loads(gzip.decompress((HERE/f'LOCK_{version}.json.gz').read_bytes()))
    need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'MOCD repeat lock')
    for p,h in {**lock['input_sha256'],**lock['code_sha256'],**lock['math_code_sha256']}.items():need(sha(p)==h,'MOCD repeat input/code '+p)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from tenrepeat_durable import Runtime as BaseRuntime
    class Runtime(BaseRuntime):
        def optimizer_finished(self,name,step,actual):
            need(self.pending==(name,step) and name==self.active and type(actual)is bool,'actual complete-MOCD optimizer result')
            # Every attempted step completes; GradScaler skips remain visible.
            key='actual_optimizer_updates' if actual else 'amp_skips';self.counts[key]+=1
            need(self.counts[key]<=self.lock['limits'][key],'finite MOCD actual/AMP outcome')
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)',(key,self.counts[key]));self.db.execute('UPDATE steps SET done=1 WHERE chain=? AND n=?',(name,step))
            self.cursor[name]=step;self.pending=None
        def source_probe_begin(self,name,step):need(name==self.active and self.get('probe/'+name+'/'+str(step)) is None,'unique source-only MOCD probe')
        def source_probe_finished(self,name,step,receipt):self.set('probe/'+name+'/'+str(step),receipt)
    rt=Runtime(lock,HERE);sys.addaudithook(rt.guard)
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from source_rate_screen_data import Files
    from mocd_full_source_engine import RecordedFullSourceEngine
    from mocd_full_matching import BlockedIndexedMatching,FullSchedule
    from mocd_complete_neural import exact,qualifier_packet
    from source_rate_neural import tensor_sha
    from tenrepeat_native_source import sources as native_sources
    import mocd_complete_factory as factory
    adapter=seed_adapter()
    tree=ast.parse(Path(factory.__file__).read_text('utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='engine_class')
    ns=dict(factory.__dict__);ns['profile_streams']=lambda s:{k:v for k,v in adapter['random_stream_spec'](s).items() if k!='status'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),__file__+'::original_complete_factory','exec'),ns)
    Engine=ns['engine_class'](RecordedFullSourceEngine,BlockedIndexedMatching,torch)
    class Permit:
        def __init__(self,c,profile,mini):
            self.name=rt.active;self.case=c;self.profile=profile;self.mini=mini;self.fingerprint=canonical(dict(lock=lock['fingerprint'],case=c['key'],profile=profile,mini=mini))
        def require_neural(self,fp):need(fp==self.fingerprint,'complete-MOCD exact permit');rt.before_neural()
        check=require_neural
        def validate_request(self,s,p,seed,profiles,schedule):
            need(p==self.profile and seed==self.case['seed'] and tuple(s.allowed)==tuple(self.case['task']['sources']) and profiles==lock['profiles'],'full source/profile/seed permit')
            need((schedule.A_steps,schedule.total_steps)==((4,10) if self.mini else (400,800)),'unchanged full or qualification schedule')
        def claim(self,fp,n,phase):
            self.require_neural(fp);need(phase==('A' if n<=(4 if self.mini else 400) else 'B'),'actual A/B boundary');rt.attempt(n)
    def data(c):
        a=files.source(c);arrays={k:{q:v for q,v in val.items() if q!='offset'} for k,val in a.items()}
        fp=canonical(dict(case=c['key'],roles=c['roles'],FFT=lock['expected_FFT_code_sha256'],source_feature_sha={d:next(b['sha256'] for b in lock['feature_bindings'] if b['protocol']==c['task']['protocol'] and b['domain']==d) for d in c['task']['sources']}))
        return adapter['bind_sources'](lock['records'],c,arrays,fp,tensor_factory=torch.from_numpy)
    def make(c,s,profile,mini=False):
        permit=Permit(c,profile,mini);schedule=FullSchedule(A_outer=1,A_inner=4,B_outer=1,B_inner=6) if mini else FullSchedule()
        def observer(step,prob,y,g,row):
            b=rt.save(Path(rt.active)/f'source{step}.npz',dict(prob=prob,y=y,g=g,model_tensor_sha256=np.array(tensor_sha(e.model.state_dict()))))
            return dict(step=step,score=row['source_score'],probability=b,target_used_for_selection=False)
        e=Engine(s,profile,lock['profiles'],permit.fingerprint,seed=c['seed'],permit=permit,rt=rt,on_source_probe=observer,device='cuda',schedule=schedule)
        return e
    try:
        rt.before_neural();files=Files(rt);proofs=[]
        for k in lock['P0_cases']:
            c=lock['cases'][k];s=data(c)
            for profile in PROFILES:
                outputs=[]
                for branch in ('continuous','resume'):
                    name='P0/'+k+'/'+profile+'/'+branch;rt.active=name;e=make(c,s,profile,True)
                    if branch=='resume':
                        e.run(5);p=e.packet();b=rt.save(Path(name)/'state5.pt',p,packet=True);del e;e=make(c,s,profile,True);e.restore(rt.load(b));exact(p,e.packet(),'MOCD actual mini disk5 complete state')
                    e.run(10);outputs.append(e.packet());qualifier_packet(outputs[-1]);rt.save(Path(name)/'state10.pt',outputs[-1],packet=True);del e
                exact(outputs[0],outputs[1],'MOCD complete continuous10/disk5+5 equality');proofs.append(dict(case=k,profile=profile,complete_state_exact=True))
            del s;torch.cuda.empty_cache()
        need(rt.counts['optimizer_attempts']==840,'actual bounded MOCD qualification840 attempts')
        write(HERE/'CUDA_P0_V1.json',dict(status='PASS',proofs=proofs,updates=840,target_predictions=0));rt.set('P0_PASS',True)
        for k,c in sorted(lock['cases'].items()):
            s=data(c)
            for profile in PROFILES:
                name='MAIN/'+k+'/'+profile;rt.active=name;e=make(c,s,profile);states={0:rt.save(Path(name)/'state0.pt',e.packet(),packet=True)}
                for stop in (100,200,300,400,421,500,600,700,800):
                    e.run(stop);states[stop]=rt.save(Path(name)/f'state{stop}.pt',e.packet(),packet=True)
                    if stop==421:
                        packet=e.packet();del e;e=make(c,s,profile);e.restore(rt.load(states[421]));exact(packet,e.packet(),'complete A/B/AMP/matching/RNG disk421 continuation')
                packet=e.packet();need(len(packet['sampler']['logs'])==800 and e.controller.completed==800,'all800 complete attempted steps')
                curve=[]
                for row in packet['sampler']['source_scores']:
                    v=rt.get('probe/'+name+'/'+str(row['step']));a=rt.load(v['probability']);need(float(np.mean(a['prob'].argmax(1)==a['y']))==row['source_score'],'independent saved MOCD source score');curve.append(v)
                selected=min(curve,key=lambda v:(-v['score'],v['step']));need(selected['step']==packet['selection']['best_step'],'source-selected state supported by complete saved curve')
                rt.finish(name,dict(case_key=k,profile=profile,last=states[800],selected_container=states[800],selected_model_key='selection.best_state',selected_step=selected['step'],states={str(q):b for q,b in states.items()},curve=curve,actual_disk421_restore=True))
                del e;print('TEN_REPEAT_COMPLETE_MOCD_MODEL',len(rt.models()),'/378',name,flush=True)
            del s;torch.cuda.empty_cache()
        need(len(rt.models())==378 and rt.counts['optimizer_attempts']==303240,'all378 original complete source recipes')
        write(HERE/'SOURCE_STAGE_COMPLETE_V1.json',dict(status='SOURCE_COMPLETE_TARGET_PENDING_MASTER',models=rt.models(),counts=dict(rt.counts),fingerprint=lock['fingerprint'],at=time.time(),target_predictions=0));rt.set('DONE',True)
    except BaseException as e:rt.set('STOPPED',dict(type=type(e).__name__,message=str(e),counts=dict(rt.counts)));raise
    finally:rt.db.close()

if __name__=='__main__':{'prepare':prepare,'run':run,'prepare-v2':lambda:prepare('V2'),'run-v2':lambda:run('V2')}[sys.argv[1]]()
