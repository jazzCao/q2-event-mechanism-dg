"""945 fixed archived Table3 source repeats, preserving their two-B loss."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import ast
import gzip
import hashlib
import json
import pickle
import sys
import time
from tenrepeat_source_metadata import ROOT,OUT,read,sha,canonical,need,write

ARMS=('B0_ID_MIX','B1_IN_MIX','S0_INTACT','E0_EVENTS','N0_WARP')
HERE=OUT/'legacy_source'

def prepare():
    from source_rate_screen_plan import expected,paths
    from e0_selection_entry_spec import local_closure
    HERE.mkdir(parents=True,exist_ok=True);m=json.loads(gzip.decompress((OUT/'MDG_SOURCE_MANIFEST_V1.json.gz').read_bytes()));b=read(OUT/'legacy_banks/SOURCE_BANKS_V1.json');initial=read(OUT/'INITIAL_ENCODERS_V1.json')
    need(b['status']=='ALL189_LEGACY_FOUR_QUERY_BANKS_COMPLETE','actual original four-query source banks first')
    store=Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_tenseed_all/legacy_source');cases={k:dict(c,key=k,source_training_windows=c['training_windows'],source_validation_windows=c['validation_windows'],real_arms=list(ARMS)) for k,c in m['cases'].items()}
    chains={};checkpoints={};predictions={};vectors={};p0='pu_clock4/ABCtoD/s3'
    def add(k,arm,qualification=False,branch=None):
        name=('P0/' if qualification else 'MAIN/')+k+'/'+arm+('/'+branch if qualification else '')
        chains[name]=dict(case_key=k,arm=arm,qualification=qualification,branch=branch,steps=5 if qualification else 2000)
        for step in ((5,) if branch=='continuous' else (2,5) if qualification else range(0,2001,250)):checkpoints[paths(store,name,'checkpoint',step)]=dict(chain=name,step=step)
        vector=expected(m['records'],m['cases'][k],'va');vk=('qualification_source/' if qualification else 'source/')+k;vectors[vk]={q:v[:32] for q,v in vector.items()} if qualification else vector
        for step in ((0,5) if qualification else (0,1000,2000)):predictions[paths(store,name,'source',step)]=dict(chain=name,case_key=k,side='source',step=step,qualification=qualification,expected_key=vk)
    for arm in ARMS:
        for branch in ('continuous','resume'):add(p0,arm,True,branch)
    for k in cases:
        for arm in ARMS:add(k,arm)
    batch=lambda c:120//(len(c['task']['sources'])*c['task']['classes'])*len(c['task']['sources'])*c['task']['classes']
    n=945;limits=dict(optimizer_attempts=n*2000+50,formal_updates=n*2000,qualification_updates=50,
        training_forward_windows=sum(5*2000*2*batch(c) for c in cases.values())+50*2*batch(cases[p0]),
        source_prediction_windows=sum(len(vectors[p['expected_key']]['y']) for p in predictions.values()),target_prediction_windows=0,
        neural_model_builds=5*n+128,checkpoint_writes=9*n+15,checkpoint_reads=4*n+64,checkpoint_hashes=32*n+512,
        probability_writes=3*n+20,probability_reads=3*n+64,probability_hashes=9*n+256,
        feature_hashes=17,feature_decodes=17,bank_hashes=189,bank_decodes=189,batch_reference_rows=0)
    now=datetime.now(timezone(timedelta(hours=8)))
    inputs=[OUT/'MDG_SOURCE_MANIFEST_V1.json.gz',OUT/'legacy_banks/SOURCE_BANKS_V1.json',OUT/'INITIAL_ENCODERS_V1.json',OUT/'HUMAN_SCOPE_V1.md']
    lock=dict(kind='TEN_REPEAT_ARCHIVED_TABLE3_SOURCE_ONLY',campaign=dict(start=now.isoformat(),deadline=(now+timedelta(hours=64)).isoformat(),neural_seconds=40*3600,store=str(store)),
        cases=cases,original_cases=m['cases'],records=m['records'],banks=b['banks'],feature_bindings=m['features'],expected_FFT_code_sha256=m['FFT_code_sha256'],
        initial_encoder_sha256=initial['new189'],chains=chains,checkpoints=checkpoints,predictions=predictions,vectors=vectors,limits=limits,P0_case=p0,arms=list(ARMS),models=n,
        control_filename=str(HERE/'CONTROL.sqlite'),input_sha256={str(p):sha(p) for p in inputs},code_sha256=local_closure([Path(__file__)]),source_only=True)
    lock['fingerprint']=canonical(lock);write(HERE/'LOCK_V1.json.gz',lock);need(len(checkpoints)==limits['checkpoint_writes'] and len(predictions)==limits['probability_writes'],'original two-B source schedule counts')
    print('TEN_REPEAT_LEGACY_SOURCE_READY_945',flush=True)

def classes():
    from clock_factual_data import Data as OriginalData
    from source_rate_neural import Chain as OriginalChain
    from source_rate_screen_runtime import Runtime as OriginalRuntime
    class Data(OriginalData):
        def batch(self,arm,sampler,augmentation):
            x,labels,unused,permutation,lam=OriginalData.batch(self,arm,sampler,augmentation)
            self.last_trace=hashlib.sha256(pickle.dumps((sampler.get_state(),augmentation.get_state()),protocol=4)).hexdigest()
            return x,labels,permutation,lam
    class Chain(OriginalChain):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.model.arm='B0_ID_MIX' if self.arm=='B0_ID_MIX' else self.arm
        def step(self,inspect=False):
            v=super().step(inspect)
            if self.name.startswith('P0/'):
                draws=self.rt.get('P0_DRAWS') or {};draws.setdefault(self.name,[]).append(self.data.last_trace)
                with self.rt.db:self.rt.set('P0_DRAWS',draws)
            return v
    class Runtime(OriginalRuntime):
        def __init__(self,lock):
            super().__init__(lock);self.spec=dict(data=dict(records=lock['records']))
        gpu=OriginalRuntime.before_neural
    return Data,Chain,Runtime

def functions():
    # Reuse the complete new-seed resume/source auditor; change only P0 counts.
    import tenrepeat_matched_source as original
    ns=dict(original.__dict__);ns.update(ARMS=ARMS,classes=classes)
    tree=ast.parse(Path(original.__file__).read_text('utf-8'))
    nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in ('qualify','train')]
    for node in nodes:
        if node.name=='qualify':
            for v in ast.walk(node):
                if isinstance(v,ast.Constant) and type(v.value)is int and v.value in (90,18):v.value={90:50,18:10}[v.value]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::shared_exact_source_orchestration','exec'),ns)
    return ns

def run():
    from source_rate_screen_data import Files as BaseFiles
    from source_rate_screen_runtime import Guard
    from source_rate_neural import libraries
    lock=json.loads(gzip.decompress((HERE/'LOCK_V1.json.gz').read_bytes()))
    need(canonical({k:v for k,v in lock.items() if k!='fingerprint'})==lock['fingerprint'],'legacy source lock fingerprint')
    for p,h in {**lock['input_sha256'],**lock['code_sha256']}.items():need(sha(p)==h,'legacy exact code/input '+p)
    class Files(BaseFiles):
        def bank(self,c):
            import numpy as np
            k=c['key']
            if k not in self.banks:
                b=lock['banks'][k];need(self.rt.hash(b['path'],'bank_hashes')==b['sha256'],'old-interface newseed bank SHA');self.rt.claim('bank_decodes');self.rt.permit(b['path'])
                try:
                    with np.load(b['path'],allow_pickle=False) as z:a={q:z[q].copy() for q in z.files}
                finally:self.rt.revoke(b['path'])
                y=[];g=[];domains=[]
                for v in c['bank']:
                    r=lock['records'][v['record']];y.extend([r['label']]*4);g.extend([v['record']]*4);domains.extend([r['domain']]*4)
                for q,wanted in (('y',y),('g',g),('domain',domains)):need(np.array_equal(a[q],np.asarray(wanted)),'exact original four-query roles '+q)
                self.banks[k]=a
            return self.banks[k]
    _,_,Runtime=classes();rt=Runtime(lock);sys.addaudithook(Guard(rt));ns=functions()
    try:
        libraries(rt);files=Files(rt);ns['qualify'](rt,files,HERE);ns['train'](rt,files,HERE)
        print('TEN_REPEAT_LEGACY_SOURCE_COMPLETE_945_NO_TARGET',flush=True)
    except BaseException as e:
        with rt.db:rt.set('STOPPED',dict(type=type(e).__name__,message=str(e),counts=rt.counts))
        raise
    finally:rt.db.close()

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
