"""Finite complete-MOCD control. Metadata only until real scoped authority.

No synthetic fixture is accepted by Runtime. Tests use the ledger and explicit
metadata objects without making a model, GPU call or production ledger.
"""
import ast
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time
import uuid
import mocd_complete_metadata as m
from pcr_campaign_core import BudgetLedger,canonical,read,sha,write_new,windows_process_live

ROOT=m.ROOT;OUT=m.OUT
STORE=Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_mocd_complete')
VERSION='COMPLETE_MOCD154_TWO_INTERPRETATIONS_V1'
NEURAL_SECONDS=36000
TOTAL_SECONDS=64800
LIMITS=dict(optimizer_attempts=123680,formal_attempts=123320,qualification_attempts=480,
    actual_optimizer_updates=123680,amp_skips=123680,training_encoder_windows=28300000,
    matching_encoder_windows=100000000,source_prediction_windows=6400000,target_prediction_windows=1125216,
    neural_model_builds=800,cache_hash_checks=12000,source_feature_NPZ_loads=120,target_feature_NPZ_loads=17,
    archive_byte_checks=4000,raw_record_loads=0,new_inverse_fits=0,FFT_source_windows=0)
PUBLIC_COSTS=set(LIMITS)-{'optimizer_attempts','formal_attempts','qualification_attempts','actual_optimizer_updates','amp_skips'}
FULL_ENTRIES=('run_mocd_complete.py','mocd_complete_neural.py','audit_mocd_complete.py',
    'mocd_complete_control.py','mocd_complete_factory.py','mocd_complete_metadata.py','mocd_complete_source_adapter.py')
MATH_ENTRIES=('mocd_complete_factory.py','mocd_complete_source_adapter.py','mocd_full_source_engine.py')


def schedule_spec(qualification=False):return dict(A_outer=1,A_inner=4,B_outer=1,B_inner=6) if qualification else dict(A_outer=20,A_inner=20,B_outer=20,B_inner=20)
def chain_key(case,profile,trajectory=None):
    if profile not in m.PROFILES or trajectory not in (None,'continuous','resumed'):raise ValueError('two fixed profiles and qualification trajectories')
    return ('formal/' if trajectory is None else 'qualification/'+trajectory+'/')+m.case_key(case)+'/'+profile
def expected_new_names(scope):return {v['case_key']+'/'+v['profile'] for v in scope['models'] if v['action']=='MISSING_NOT_AUTHORIZED'}
def expected_all_names(scope):return {v['case_key']+'/'+v['profile'] for v in scope['models']}
def math_fingerprint(scope,closure):return canonical(dict(code_sha256=closure,profiles=scope['profiles'],stream_binding='COMPLETE_SOURCE_STREAMS_V1'))
def engine_fingerprint(lock,case,profile,qualification=False):return canonical(dict(math_fingerprint=lock['math_fingerprint'],task=case['task'],seed=case['seed'],source_fingerprint=case['new_source_fingerprint'],profile=profile,schedule=schedule_spec(qualification)))


def approved_scope(scope):
    return dict(protocols={'pu_clock4':4,'phm09_w3':20,'jnu_speed':3},seeds=[0,1,2],profiles=list(m.PROFILES),
        new_models=154,archive_models=8,formal_attempts=123200,qualification_planned_attempts=360,repair_reserve=120,
        maximum_attempts=LIMITS['optimizer_attempts'],normal_forecast=scope['normal_total_forecast'],limits=LIMITS,
        A_steps=400,B_steps=400,source_scientific_steps=[0,400,421,800],source_score_steps=scope['source_score_steps'],
        primary='fixed800',secondary='earliest_strict_source_best',new_target_roles=308,archive_target_roles=16,
        current_eight_method_roles=1296,complete_ten_method_roles=1620,
        common_freeze='all154 new and8 byte-verified archived before308 new target predictions',
        neural_seconds=NEURAL_SECONDS,total_seconds=TOTAL_SECONDS,
        no_new_dataset_split_seed_operator_hparam_search=True,new_FFT_windows=0,new_inverse_fits=0,raw_loads=0,
        new_models_arrays='D',FULL_control='C',no_silent_paid_replay=True)


def closure(entries):
    """Bind transitive repository/declared author imports, without importing them."""
    folder=Path(__file__).parent;author=ROOT/'related projects/CCDG-main';todo=[Path(p).resolve() for p in entries];found={}
    def local(base,parts):
        path=base.joinpath(*parts).with_suffix('.py')
        if not path.is_file():path=base.joinpath(*parts,'__init__.py')
        if path.is_file():
            todo.append(path)
            for ancestor in path.parents:
                if ancestor==ROOT or not ancestor.is_relative_to(ROOT):break
                if (ancestor/'__init__.py').is_file():todo.append(ancestor/'__init__.py')
    while todo:
        path=todo.pop().resolve()
        if str(path) in found:continue
        found[str(path)]=sha(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            if isinstance(node,ast.ImportFrom) and node.level:
                base=path.parent
                for _ in range(node.level-1):base=base.parent
                stem=node.module.split('.') if node.module else [];local(base,stem)
                for name in node.names:local(base,stem+name.name.split('.'))
            else:
                names=[node.module] if isinstance(node,ast.ImportFrom) else [v.name for v in node.names] if isinstance(node,ast.Import) else []
                for name in names:
                    if name:
                        for base in (folder,folder.parent,ROOT,author):local(base,name.split('.'))
    return found


def verify_lock(lock):
    if canonical({k:v for k,v in lock.items() if k!='fingerprint'})!=lock.get('fingerprint') or lock.get('version')!=VERSION or lock.get('limits')!=LIMITS:raise PermissionError('exact final complete execution lock')
    if lock.get('runtime_status')!='FULL_STATIC_READY_NOT_YET_AUTHORIZED' or lock.get('new_models')!=154:raise PermissionError('PART1/partial plan is not executable')
    folder=Path(__file__).parent
    if any(not (folder/p).is_file() for p in FULL_ENTRIES):raise PermissionError('complete runner, neural worker and independent auditor must exist first')
    if lock['code_sha256']!=closure([folder/p for p in FULL_ENTRIES]) or lock['math_code_sha256']!=closure([folder/p for p in MATH_ENTRIES]):raise PermissionError('actual fixed-entry transitive execution and mathematical closures')
    test_path=OUT/'FULL_STATIC_TESTS.json'
    if not test_path.exists():raise PermissionError('whole current-code synthetic receipt before complete execution lock')
    tests=read(test_path)
    if tests.get('status')!='FULL_SCOPE_SYNTHETIC_STATIC_PASS_NOT_CUDA' or tests.get('code_sha256')!=lock['code_sha256'] or tests.get('new_neural_actions')!=0 or tests.get('full_scope_mock_models')!=154 or lock['input_sha256'].get(str(test_path))!=sha(test_path):raise PermissionError('complete current-code tests, not a partial preparatory PASS')
    for path,h in tests['test_code_sha256'].items():
        if lock['input_sha256'].get(path)!=h:raise PermissionError('actual full test code remains bound')
    for path,h in {**lock['code_sha256'],**lock['input_sha256']}.items():
        if sha(path)!=h:raise PermissionError('complete-MOCD code/input drift '+path)
    if set(lock['dependency_versions'])!={'torch','numpy','scipy','torchvision','tqdm'}:raise PermissionError('all actual neural and numeric dependency versions bound')
    for package,expected in lock['dependency_versions'].items():
        if version(package)!=expected:raise PermissionError('bound dependency drift')
    scope=read(OUT/'PART1_SOURCE_MANIFEST_V2.json')
    if lock.get('approved_scope')!=approved_scope(scope):raise PermissionError('exact full protocols/seeds/profiles/steps/policies/budget, not a weakened scope')
    if canonical({k:v for k,v in scope.items() if k!='fingerprint'})!=scope['fingerprint'] or lock['input_sha256'].get(str(OUT/'PART1_SOURCE_MANIFEST_V2.json'))!=sha(OUT/'PART1_SOURCE_MANIFEST_V2.json'):raise PermissionError('current complete manifest byte binding')
    for path,h in scope['input_sha256'].items():
        if lock['input_sha256'].get(path)!=h:raise PermissionError('original source references must remain inside full input closure')
    if lock['math_fingerprint']!=math_fingerprint(scope,lock['math_code_sha256']):raise PermissionError('actual mathematical closure')
    for path,h in lock['math_code_sha256'].items():
        if lock['code_sha256'].get(path)!=h:raise PermissionError('math closure must be part of complete execution closure')
    return scope


def authorization(auth,lock,now=None,purpose='control'):
    now=time.time() if now is None else now
    if auth.get('kind')!='HUMAN_FINITE_APPROVAL' or auth.get('answer')!='批准这轮完整MOCD比较':raise PermissionError('new actual complete-MOCD human reply required')
    qid=auth.get('question_item_id','')
    if not isinstance(qid,str) or 'request_user_input_async' not in qid or 'call_' not in qid:raise PermissionError('actual scoped question/reply identity')
    if auth.get('lock_fingerprint')!=lock['fingerprint'] or auth.get('approved_scope')!=lock['approved_scope'] or auth.get('limits')!=LIMITS:raise PermissionError('old approval is not this complete comparison')
    if (auth.get('neural_seconds'),auth.get('total_seconds'))!=(NEURAL_SECONDS,TOTAL_SECONDS):raise PermissionError('new independent finite clocks')
    start=auth.get('total_start_unix')
    if type(start) not in (int,float) or not math.isfinite(start) or purpose not in ('control','audit') or not start<=now<start+TOTAL_SECONDS:raise PermissionError('original new preparation/total clock expired or unstarted')
    return auth


class Ledger(BudgetLedger):
    def __init__(self,path,identity,limits):
        super().__init__(path,identity,limits)
        fields={v[1] for v in self.db.execute('PRAGMA table_info(attempts)')}
        with self.db:
            if 'optimizer_done' not in fields:self.db.execute('ALTER TABLE attempts ADD COLUMN optimizer_done INTEGER NOT NULL DEFAULT 0')
            if 'actual_update' not in fields:self.db.execute('ALTER TABLE attempts ADD COLUMN actual_update INTEGER')
    def attempt(self,name,step,phase,stage,qualification=False):
        maximum,Asteps=(10,4) if qualification else (800,400)
        if phase!=('QUALIFICATION' if qualification else 'SOURCE') or type(step) is not int or not 1<=step<=maximum or stage!=('A' if step<=Asteps else 'B'):raise PermissionError('actual locked stage/step')
        with self.db:
            last,done,n=self.cursor(name)
            if (last,done,n)!=(last,)*3 or step!=last+1:raise PermissionError('no implicit paid replay, gap or unfinished controller step')
            self._add('optimizer_attempts',1);self._add('qualification_attempts' if qualification else 'formal_attempts',1)
            self.db.execute('INSERT INTO attempts(chain,step,phase,at,completed,optimizer_done,actual_update) VALUES (?,?,?,?,0,0,NULL)',(name,step,phase,time.time()))
    def optimizer_finished(self,name,step,actual):
        if type(actual) is not bool:raise ValueError('actual scaler result must be boolean')
        with self.db:
            row=self.db.execute('UPDATE attempts SET optimizer_done=1,actual_update=? WHERE chain=? AND step=? AND optimizer_done=0 AND completed=0',(int(actual),name,step))
            if row.rowcount!=1:raise PermissionError('one actual result for one paid attempt')
            self._add('actual_optimizer_updates',int(actual));self._add('amp_skips',int(not actual))
    def controller_finished(self,name,step):
        with self.db:
            row=self.db.execute('UPDATE attempts SET completed=1 WHERE chain=? AND step=? AND optimizer_done=1 AND completed=0',(name,step))
            if row.rowcount!=1:raise PermissionError('controller completion follows actual optimizer and all callbacks')


class Permit:
    def __init__(self,rt,case,profile,trajectory=None):
        self.rt=rt;self.case=case;self.profile=profile;self.qualification=trajectory is not None;self.name=chain_key(case,profile,trajectory)
        self.fingerprint=engine_fingerprint(rt.lock,case,profile,self.qualification)
    def require_neural(self,fingerprint):
        rt=self.rt
        if not rt.authority_verified:raise PermissionError('synthetic object is not a real finite permit')
        rt.clock(rt.gpu_start is not None);rt.require_case(self.case)
        if fingerprint!=self.fingerprint or rt.phase!=('QUALIFICATION' if self.qualification else 'SOURCE') or rt.get('SOURCE_FREEZE') is not None:raise PermissionError('source phase, exact fingerprint and pre-freeze only')
    def check(self,fingerprint):self.require_neural(fingerprint)
    def validate_request(self,sources,profile,seed,profiles,schedule):
        rt=self.rt;self.require_neural(self.fingerprint)
        actual={k:getattr(schedule,k) for k in schedule_spec(self.qualification)}
        if type(seed) is not int or seed!=self.case['seed'] or profile!=self.profile or canonical(profiles)!=canonical(rt.scope['profiles']) or actual!=schedule_spec(self.qualification) or any(type(v) is not int for v in actual.values()):raise PermissionError('actual profile/seed/equations/schedule request')
        task=self.case['task']
        if sources.fingerprint!=self.case['new_source_fingerprint'] or tuple(sources.allowed)!=tuple(task['sources']) or type(sources.classes) is not int or sources.classes!=task['classes'] or tuple(sources.forbidden)!=(task['target'],):raise PermissionError('actual bound source roles/class/domain order')
    def claim(self,fingerprint,step,stage):
        self.require_neural(fingerprint);self.rt.watch(step)
        if self.rt.pending_optimizer is not None:raise PermissionError('only one current source optimizer action')
        self.rt.ledger.attempt(self.name,step,self.rt.phase,stage,self.qualification);self.rt.pending_optimizer=(self.name,step)


class Runtime:
    def __init__(self,lock,auth):
        self.scope=verify_lock(lock);authorization(auth,lock);self.lock=lock;self.approval=auth;self.authority_verified=True
        self.ledger=Ledger(OUT/'control/CONTROL.sqlite',canonical(dict(lock=lock['fingerprint'],approval=auth)),LIMITS);self.db=self.ledger.db
        self.phase='SOURCE_ADMISSION';self.active_case=None;self.pending_optimizer=None;self.allow=set()
        self.by_case={m.case_key(c):c for c in self.scope['cases']};self.new_names=expected_new_names(self.scope)
        self.qual_cases={v['case_key'] for v in self.scope['qualification_forecast']['cases']}
        self.spec=dict(data=dict(records=self.scope['records']),feature_bindings=self.scope['feature_bindings'],raw_bindings=self.scope['raw_bindings'],FFT_code_sha256=self.scope['FFT_code_sha256'])
        rows=self.db.execute('SELECT start FROM neural_clock').fetchall()
        if len(rows)>1:raise PermissionError('one original first-neural clock')
        self.gpu_start=rows[0][0] if rows else None
        self.watched={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in lock['code_sha256']};self.clock()
    def get(self,name):
        if name=='SOURCE_FREEZE' and hasattr(self,'_freeze_value'):return self._freeze_value
        value=self.ledger.get(name)
        if name=='SOURCE_FREEZE' and value is not None:
            # The producer rejects every subsequent source/freeze mutation.
            # Decode/hash once, not the entire 162-model JSON for every target
            # encoder chunk. The independent final audit rereads raw SQLite.
            self._freeze_value=value;self._freeze_digest=canonical(value)
        return value
    def put(self,name,value):
        if self.get('SOURCE_FREEZE') is not None and (name=='SOURCE_FREEZE' or name in ('QUALIFICATION_PASS','ARCHIVE_PASS') or name.startswith(('formal/','qualification/','PROBE/','PENDING/PROBE/'))):raise PermissionError('frozen source journal is immutable')
        self.ledger.put(name,value)
    def count(self):return self.ledger.snapshot()
    def cursor(self,name):return self.ledger.cursor(name)
    def clock(self,neural=False):
        at=time.time();start=self.approval['total_start_unix']
        if not start<=at<start+TOTAL_SECONDS:raise TimeoutError('same new approved total clock, never reset')
        if neural and (self.gpu_start is None or not self.gpu_start<=at<self.gpu_start+NEURAL_SECONDS):raise TimeoutError('same first-neural clock, never reset')
    def gpu(self):
        self.clock()
        if self.gpu_start is None:
            at=time.time()
            with self.db:self.db.execute('INSERT INTO neural_clock VALUES (?)',(at,))
            self.gpu_start=at;write_new(OUT/'NEURAL_CLOCK.json',dict(first_neural_unix=at,deadline=min(at+NEURAL_SECONDS,self.approval['total_start_unix']+TOTAL_SECONDS)))
        self.clock(True)
    def watch(self,step):
        if step%20!=1:return
        for p,before in self.watched.items():
            stat=Path(p).stat()
            if (stat.st_size,stat.st_mtime_ns)!=before:raise PermissionError('bound source changed during execution '+p)
    def require_case(self,case):
        key=m.case_key(case)
        if self.active_case is None or m.case_key(self.active_case)!=key or key not in self.by_case or canonical(case)!=canonical(self.by_case[key]):raise PermissionError('exact active source case')
    def set_case(self,case):
        if m.case_key(case) not in self.by_case or canonical(case)!=canonical(self.by_case[m.case_key(case)]):raise PermissionError('known source case only')
        self.active_case=case
    def permit_for(self,case,profile,trajectory=None):
        self.require_case(case)
        if profile not in m.PROFILES:raise PermissionError('fixed two interpretation profiles only')
        if trajectory is None and m.case_key(case)+'/'+profile not in self.new_names:raise PermissionError('old8 archive models must not be retrained')
        if trajectory is not None and m.case_key(case) not in self.qual_cases:raise PermissionError('fixed full-shape qualification cases only')
        return Permit(self,case,profile,trajectory)
    def claim(self,name,n=1):
        if name not in PUBLIC_COSTS:raise PermissionError('paid attempt/outcome counters require bound ledger methods')
        neural=name in ('neural_model_builds','training_encoder_windows','matching_encoder_windows','source_prediction_windows','target_prediction_windows')
        self.clock(neural)
        if neural:
            if self.active_case is None:raise PermissionError('neural action needs an active exact case')
            self.require_case(self.active_case)
            if self.phase=='TARGET':
                if name not in ('neural_model_builds','target_prediction_windows'):raise PermissionError('no source action after common freeze')
                self.require_target(self.active_case)
            elif self.phase in ('SOURCE','QUALIFICATION'):
                if self.get('SOURCE_FREEZE') is not None or name=='target_prediction_windows':raise PermissionError('no target inference before whole source freeze')
                if name=='training_encoder_windows' and self.pending_optimizer is None:raise PermissionError('training encoder requires a prepaid actual attempt')
            else:raise PermissionError('no neural admission/audit/preparation action')
        self.ledger.claim(name,n)
    def optimizer_finished(self,name,step,actual):
        if self.pending_optimizer!=(name,step):raise PermissionError('one already-paid optimizer outcome')
        self.ledger.optimizer_finished(name,step,actual);self.pending_optimizer=None;self.clock(True)
    def controller_finished(self,name,step):
        if self.phase not in ('SOURCE','QUALIFICATION') or self.get('SOURCE_FREEZE') is not None:raise PermissionError('source controller completes only before freeze')
        self.ledger.controller_finished(name,step);self.clock(True)
    def probe_identity(self,name,step):
        if self.active_case is None or self.phase not in ('SOURCE','QUALIFICATION') or self.get('SOURCE_FREEZE') is not None:raise PermissionError('pre-freeze exact source probe phase/case')
        self.require_case(self.active_case)
        qualification=self.phase=='QUALIFICATION'
        trajectories=('continuous','resumed') if qualification else (None,)
        allowed={chain_key(self.active_case,p,t) for p in m.PROFILES for t in trajectories}
        steps=[10] if qualification else self.scope['source_score_steps']
        if name not in allowed or type(step) is not int or step not in steps or self.cursor(name)!=(step,step-1,step):raise PermissionError('probe follows paid optimizer before current controller completion')
        row=self.db.execute('SELECT optimizer_done FROM attempts WHERE chain=? AND step=?',(name,step)).fetchone()
        if row!=(1,):raise PermissionError('probe must follow actual recorded optimizer outcome')
    def source_probe_begin(self,name,step):
        self.probe_identity(name,step)
        key='PROBE/'+name+'/'+str(step)
        if self.get(key) is not None or self.get('PENDING/'+key) is not None:raise PermissionError('no free replay of paid source inference')
        self.put('PENDING/'+key,dict(at_unix=time.time(),phase=self.phase))
    def source_probe_finished(self,name,step,receipt):
        self.probe_identity(name,step)
        key='PROBE/'+name+'/'+str(step)
        if self.get('PENDING/'+key) is None or self.get(key) is not None:raise PermissionError('paid pending source probe before receipt')
        self.put(key,receipt)
    def validate_new_record(self,name,record):
        case=self.by_case[name.rsplit('/',1)[0]];profile=name.rsplit('/',1)[1]
        if record.get('case_key')!=m.case_key(case) or record.get('profile')!=profile or record.get('seed')!=case['seed'] or record.get('source_fingerprint')!=case['new_source_fingerprint'] or record.get('engine_fingerprint')!=engine_fingerprint(self.lock,case,profile):raise PermissionError('actual completed model seed/profile/source/engine identity')
        curve=record.get('source_curve',[])
        if [v.get('step') for v in curve]!=self.scope['source_score_steps']:raise PermissionError('all20 original source selection points')
        for item in curve:
            score=item.get('source_score')
            if set(item)!={'step','source_score','target_used_for_selection'} or type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1 or item['target_used_for_selection'] is not False:raise PermissionError('source-only finite curve')
        best=min(curve,key=lambda v:(-v['source_score'],v['step']))
        if (record.get('selected_step'),record.get('selected_score'))!=(best['step'],best['source_score']):raise PermissionError('earliest strict source best, not target selected')
        states=record.get('checkpoints',{})
        if set(states)!={str(v) for v in self.scope['source_scientific_steps']} or any(states[str(step)].get('step')!=step for step in self.scope['source_scientific_steps']) or record.get('resume')!=states['800']:raise PermissionError('full0/400/421/800 scientific and terminal resume bindings')
        if record.get('restored421',{}).get('status')!='EXACT_DISK_RESTORE' or record['restored421'].get('checkpoint_sha256')!=states['421'].get('sha256'):raise PermissionError('actual421 disk restoration evidence')
        source_states=record.get('source_checkpoints',{})
        if set(source_states)!={str(s) for s in self.scope['source_score_steps']} or any(source_states[str(s)].get('step')!=s for s in self.scope['source_score_steps']) or source_states['800']!=states['800']:raise PermissionError('all source-score checkpoint references retained without new computation')
        for item in curve:
            probe=self.get('PROBE/formal/'+name+'/'+str(item['step']))
            if not probe or probe.get('source_score')!=item['source_score'] or probe.get('step')!=item['step'] or probe.get('target_used_for_selection') is not False:raise PermissionError('saved probability receipt for every source score')
        outcome=self.db.execute('SELECT sum(actual_update),sum(optimizer_done),sum(completed) FROM attempts WHERE chain=?',('formal/'+name,)).fetchone()
        if outcome[1:]!=(800,800) or record.get('actual_updates')!=outcome[0] or record.get('amp_skips')!=800-outcome[0]:raise PermissionError('actual updates/AMP skips and all source callbacks complete')
        return record
    def validate_freeze(self,freeze):
        if not isinstance(freeze,dict) or freeze.get('lock_fingerprint')!=self.lock['fingerprint'] or freeze.get('math_fingerprint')!=self.lock['math_fingerprint'] or set(freeze.get('models',{}))!=expected_all_names(self.scope):raise PermissionError('complete 162-model source barrier with exact lock/math')
        q=self.get('QUALIFICATION_PASS');archive=self.get('ARCHIVE_PASS')
        if not q or q.get('lock_fingerprint')!=self.lock['fingerprint'] or not archive or archive.get('lock_fingerprint')!=self.lock['fingerprint']:raise PermissionError('actual qualification and fresh archive verification before common freeze')
        expected_qual={v['case_key']+'/'+p for v in self.scope['qualification_forecast']['cases'] for p in m.PROFILES}
        expected_archives=expected_all_names(self.scope)-self.new_names
        if q.get('status')!='FULL_SCOPE_CUDA_CONTINUOUS_RESTORE_PASS' or set(q.get('combinations',[]))!=expected_qual or len(q.get('combinations',[]))!=18 or archive.get('status')!='FRESH_ARCHIVE_BYTES_PASS' or set(archive.get('model_bindings',{}))!=expected_archives:raise PermissionError('all18 actual qualification combinations and exact8 archive sources')
        for name,item in freeze['models'].items():
            if name in self.new_names:
                record=self.get('formal/'+name)
                if not record or record.get('status')!='COMPLETE' or record.get('resume_step')!=800 or self.cursor('formal/'+name)!=(800,)*3 or canonical(item)!=canonical(record):raise PermissionError('every new model fully complete and same exact journal record')
                self.validate_new_record(name,record)
            else:
                expected=next(v for v in self.scope['models'] if v['case_key']+'/'+v['profile']==name)
                if canonical(item)!=canonical(expected) or archive.get('model_bindings',{}).get(name)!=canonical(expected):raise PermissionError('historical8 retain original identity and freshly checked bytes')
        return freeze
    def require_target(self,case,name=None):
        self.require_case(case);freeze=self.get('SOURCE_FREEZE')
        if self.phase!='TARGET' or freeze is None:raise PermissionError('target phase and actual complete source freeze required')
        # Source journal becomes immutable after freeze. Cache the digest only
        # after fully checking all 154 paid cursors and eight archive bindings.
        digest=getattr(self,'_freeze_digest',None)
        if digest is None or getattr(self,'validated_freeze',None)!=digest:self.validate_freeze(freeze);self.validated_freeze=digest
        if name is not None and (name not in self.new_names or not name.startswith(m.case_key(case)+'/')):raise PermissionError('new target only for active new frozen model; old16 reused without inference')
    def permit(self,path):self.allow.add(str(Path(path).resolve()).casefold())
    def revoke(self,path):self.allow.discard(str(Path(path).resolve()).casefold())
    def install_guard(self):
        import sys
        prefix=str(STORE.resolve()).casefold()+os.sep
        def hook(event,args):
            if event!='open' or not isinstance(args[0],str):return
            path=Path(args[0]).resolve();value=str(path).casefold()
            if value in self.allow or value.startswith(prefix):return
            if path.suffix.lower() in ('.pt','.pth','.mat','.npz','.npy') or '/data/raw/' in path.as_posix().lower():raise PermissionError('array/state/raw outside explicit active-role permit '+str(path))
        sys.addaudithook(hook)
    def acquire(self):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE');rows=self.db.execute('SELECT pid,status FROM producer').fetchall()
            if rows and rows[0][1]=='RUNNING':raise PermissionError('unique producer, inspect actual PID before recovery')
            if rows:self.db.execute("UPDATE producer SET pid=?,status='RUNNING'",(os.getpid(),))
            else:self.db.execute('INSERT INTO producer VALUES (?,?)',(os.getpid(),'RUNNING'))
    def release(self):
        with self.db:self.db.execute("UPDATE producer SET status='STOPPED'")


def recover_exact(rt):
    """No rewind/retirement exists here: only exact fully durable same-clock state."""
    rows=rt.db.execute('SELECT pid,status FROM producer').fetchall()
    if len(rows)!=1 or rows[0][1]!='RUNNING' or windows_process_live(rows[0][0]):raise PermissionError('prove absent actual producer, not a timeout')
    if rt.db.execute('SELECT count(*) FROM attempts WHERE completed=0 OR optimizer_done=0').fetchone()[0]:raise PermissionError('incomplete paid control state cannot be replayed')
    for name,step in rt.db.execute('SELECT chain,max(step) FROM attempts GROUP BY chain'):
        record=rt.get(name)
        if not record or record['resume_step']!=step or rt.cursor(name)!=(step,)*3 or sha(record['resume']['path'])!=record['resume']['sha256']:raise PermissionError('paid tail without exact complete state needs explicit repair, never free rewind')
    for key, in rt.db.execute("SELECT name FROM journal WHERE name LIKE 'PENDING/%'"):
        if rt.get(key[8:]) is None:raise PermissionError('incomplete paid prediction cannot be retried silently')
    rt.release();return dict(status='EXACT_SAME_CLOCK_RECOVERY_READY',old_pid=rows[0][0],counters=rt.count())


def fresh_path(folder,suffix):
    root=STORE.resolve();folder=Path(folder).resolve()
    if root.drive.upper()!='D:' or not folder.is_relative_to(root):raise PermissionError('new models/arrays in named D root only')
    folder.mkdir(parents=True,exist_ok=True);return folder/(uuid.uuid4().hex+suffix)
