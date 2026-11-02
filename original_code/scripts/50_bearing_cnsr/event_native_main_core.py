"""Finite native PHM/JNU controls. Stdlib/metadata only; no neural imports."""
import ast
from importlib.metadata import version
import os
from pathlib import Path
import time
import uuid
import event_native_main_metadata as m
from pcr_campaign_core import BudgetLedger,canonical,read,sha,write_new,windows_process_live

ROOT=m.ROOT;OUT=m.OUT;STORE=m.STORE;CONFIGS=m.CONFIGS
VERSION='NATIVE_PHM_JNU207_FULL_MAIN_V1'
NEURAL_SECONDS=21600
TOTAL_SECONDS=43200
SOURCE_STEPS=(0,1000,2000,3001)
LIMITS=dict(optimizer_attempts=621387,formal_updates=621207,qualification_updates=180,
    training_forward_windows=53030000,source_prediction_windows=500000,target_prediction_windows=651024,
    neural_model_builds=1300,cache_hash_checks=12000,source_feature_NPZ_loads=104,
    target_feature_NPZ_loads=26,raw_record_loads=0,new_inverse_fits=0)
ENTRY_NAMES=('event_native_main_core.py','event_native_main_neural.py','event_native_main_metadata.py',
    'event_native_main_sources.py','run_event_native_main.py','audit_event_native_main.py','test_event_native_main.py')


def case_key(case):return m.case_key(case)
def forward_count(config):
    if config not in CONFIGS:raise ValueError('only three fixed native configurations')
    return 128 if config=='A2_AUTHOR_IN_MIXUP' else 64
def chain_key(case,config,trajectory=None):
    if config not in CONFIGS:raise ValueError('fixed native config')
    if trajectory is None:return m.model_key(case,config)
    if trajectory not in ('continuous','resumed') or (case['task']['protocol'],case['task']['fold'],case['seed']) not in m.QUAL_CASES:raise PermissionError('registered qualifier only')
    return 'qualification/'+trajectory+'/'+case_key(case)+'/'+config
def engine_fingerprint(lock,case,config,qualification=False):
    return canonical(dict(lock=lock['fingerprint'],case=case_key(case),source=case['source_fingerprint'],
        config=config,maximum_steps=10 if qualification else 3001))
def store_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to(STORE.resolve()):raise PermissionError('only this named D artifact store')
    return path
def fresh_path(folder,suffix):
    folder=store_path(folder);folder.mkdir(parents=True,exist_ok=True)
    return folder/(uuid.uuid4().hex+suffix)


def code_closure():
    folder=Path(__file__).parent;author=ROOT/'related projects/CCDG-main'
    todo=[folder/x for x in ENTRY_NAMES]+[Path(p) for p in m.author_math_bindings()['bindings']];found={}
    def local(base,parts):
        p=base.joinpath(*parts).with_suffix('.py')
        if not p.is_file():p=base.joinpath(*parts,'__init__.py')
        if p.is_file():
            todo.append(p)
            for ancestor in p.parents:
                if ancestor==ROOT or not ancestor.is_relative_to(ROOT):break
                if (ancestor/'__init__.py').is_file():todo.append(ancestor/'__init__.py')
    while todo:
        p=todo.pop().resolve()
        if str(p) in found:continue
        source=p.read_text(encoding='utf8');found[str(p)]=sha(p)
        for n in ast.walk(ast.parse(source)):
            if isinstance(n,ast.ImportFrom) and n.level:
                base=p.parent
                for _ in range(n.level-1):base=base.parent
                stem=n.module.split('.') if n.module else [];local(base,stem)
                for a in n.names:local(base,stem+a.name.split('.'))
            else:
                names=[n.module] if isinstance(n,ast.ImportFrom) else [a.name for a in n.names] if isinstance(n,ast.Import) else []
                for name in names:
                    if name:
                        for base in (folder,folder.parent,ROOT,author):local(base,name.split('.'))
    return found


def verify_lock(lock,files=True):
    if canonical({k:v for k,v in lock.items() if k!='fingerprint'})!=lock.get('fingerprint'):raise PermissionError('execution fingerprint')
    if lock.get('version')!=VERSION or lock.get('limits')!=LIMITS or lock.get('formal_models')!=207 or lock.get('steps')!=3001:raise PermissionError('fixed complete native scope')
    if files:
        for p,h in {**lock['code_sha256'],**lock['input_sha256']}.items():
            if sha(p)!=h:raise PermissionError('locked code/metadata drift '+p)
        for p,v in lock['dependency_versions'].items():
            if version(p)!=v:raise PermissionError('dependency drift '+p)


def authorization(value,lock,now=None):
    now=time.time() if now is None else now;qid=value.get('question_item_id','')
    if value.get('kind')!='HUMAN_FINITE_APPROVAL' or value.get('answer')!='批准这轮PHM/JNU作者基线补齐':raise PermissionError('new actual scoped native approval required')
    if not isinstance(qid,str) or 'request_user_input_async' not in qid or 'call_' not in qid:raise PermissionError('actual new question/reply identity')
    if value.get('lock_fingerprint')!=lock['fingerprint'] or value.get('approved_scope')!=lock['approved_scope'] or value.get('limits')!=LIMITS:raise PermissionError('old campaign or altered scope is not authority')
    if value.get('neural_seconds')!=NEURAL_SECONDS or value.get('total_seconds')!=TOTAL_SECONDS:raise PermissionError('exact finite six/twelve-hour allocation')
    start=value.get('total_start_unix')
    if type(start) not in (int,float) or not start<=now<start+TOTAL_SECONDS:raise PermissionError('same original approved preparation clock')
    return value


class Ledger(BudgetLedger):
    def attempt(self,name,step,phase,config):
        if phase not in ('SOURCE','QUALIFICATION') or type(step) is not int or not 1<=step<=(3001 if phase=='SOURCE' else 10):raise PermissionError('registered native update shape')
        windows=forward_count(config)
        with self.db:
            last,done,n=self.cursor(name)
            if (last,done,n)!=(last,)*3 or step!=last+1:raise PermissionError('no gap, duplicate or ambiguous paid replay')
            for key,amount in (('optimizer_attempts',1),('formal_updates' if phase=='SOURCE' else 'qualification_updates',1),('training_forward_windows',windows)):self._add(key,amount)
            self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,0)',(name,step,phase,time.time()))


class NativePermit:
    def __init__(self,rt,case,config,trajectory=None):
        self.rt=rt;self.case=case;self.config=config;self.name=chain_key(case,config,trajectory)
        self.qualification=trajectory is not None;self.fingerprint=engine_fingerprint(rt.lock,case,config,self.qualification)
    def require_neural(self,fingerprint):
        rt=self.rt;rt.clock(True)
        if fingerprint!=self.fingerprint or rt.phase!=('QUALIFICATION' if self.qualification else 'SOURCE'):raise PermissionError('native source permit phase/fingerprint')
        rt.require_case(self.case)
    def claim(self,fingerprint,step,stage):
        self.require_neural(fingerprint);rt=self.rt
        if stage!='B' or rt.pending is not None:raise PermissionError('one complete native update at a time')
        rt.watch(step);rt.ledger.attempt(self.name,step,rt.phase,self.config)
        rt.pending=dict(name=self.name,step=step,windows=forward_count(self.config),attempt_seen=False,forward_seen=False)


class Runtime:
    def __init__(self,lock,approval):
        verify_lock(lock);authorization(approval,lock)
        self.lock=lock;self.approval=approval;self.scope=read(OUT/'SOURCE_SCOPE.json')
        self.spec=dict(data=dict(records=self.scope['records']),feature_bindings=self.scope['feature_bindings'],
            raw_bindings=self.scope['raw_bindings'],FFT_code_sha256=self.scope['FFT_code_sha256'])
        self.ledger=Ledger(OUT/'control/CONTROL.sqlite',canonical(dict(lock=lock['fingerprint'],approval=approval)),LIMITS)
        self.db=self.ledger.db;self.phase='SOURCE_ADMISSION';self.pending=None;self.active_case=None;self.allow=set()
        self.by_case={case_key(x):x for x in self.scope['cases']};self.watched={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in lock['code_sha256']}
        rows=self.db.execute('SELECT start FROM neural_clock').fetchall()
        if len(rows)>1:raise PermissionError('one first-neural clock')
        self.gpu_start=rows[0][0] if rows else None;self.clock()
    def get(self,name):return self.ledger.get(name)
    def put(self,name,value):self.ledger.put(name,value)
    def count(self):return self.ledger.snapshot()
    def cursor(self,name):return self.ledger.cursor(name)
    def clock(self,neural=False):
        at=time.time()
        if not self.approval['total_start_unix']<=at<self.approval['total_start_unix']+TOTAL_SECONDS:raise TimeoutError('same approved total clock')
        if neural and (self.gpu_start is None or not self.gpu_start<=at<self.gpu_start+NEURAL_SECONDS):raise TimeoutError('same first actual GPU clock')
    def gpu(self):
        self.clock()
        if self.gpu_start is None:
            at=time.time()
            with self.db:self.db.execute('INSERT INTO neural_clock VALUES (?)',(at,))
            self.gpu_start=at
            write_new(OUT/'NEURAL_CLOCK.json',dict(first_neural_unix=at,deadline=min(at+NEURAL_SECONDS,self.approval['total_start_unix']+TOTAL_SECONDS)))
        self.clock(True)
    def watch(self,step):
        if step%250!=1:return
        for p,before in self.watched.items():
            st=Path(p).stat()
            if (st.st_size,st.st_mtime_ns)!=before:raise PermissionError('bound code changed during producer '+p)
    def require_case(self,case):
        key=case_key(case)
        if self.active_case is None or case_key(self.active_case)!=key or key not in self.by_case or canonical(case)!=canonical(self.by_case[key]):raise PermissionError('exact active source task/seed')
    def set_case(self,case):
        key=case_key(case)
        if key not in self.by_case or canonical(case)!=canonical(self.by_case[key]):raise PermissionError('known immutable native case')
        self.active_case=case
    def claim(self,name,n=1):
        if name in ('native_optimizer_attempts','native_training_forward_windows','actual_optimizer_updates'):
            # Completion reports an action that ALREADY happened. Record it even
            # if that action crossed the deadline, then stop; never hide its cost.
            if name!='actual_optimizer_updates':self.clock(True)
            p=self.pending
            if p is None:raise PermissionError('native callback without paid update')
            if name=='native_optimizer_attempts':
                if n!=1 or p['attempt_seen']:raise PermissionError('one paid native attempt callback')
                p['attempt_seen']=True
            elif name=='native_training_forward_windows':
                if n!=p['windows'] or not p['attempt_seen'] or p['forward_seen']:raise PermissionError('actual forward matches prepaid windows')
                p['forward_seen']=True
            else:
                if n!=1 or not p['attempt_seen'] or not p['forward_seen']:raise PermissionError('only actual complete optimizer/scheduler update')
                self.ledger.completed(p['name'],p['step']);self.pending=None
                self.clock(True)
            return
        self.clock(name in ('neural_model_builds','source_prediction_windows','target_prediction_windows'))
        self.ledger.claim(name,n)
    def permit_for(self,case,config,trajectory=None):return NativePermit(self,case,config,trajectory)
    def permit(self,path):self.allow.add(str(Path(path).resolve()).casefold())
    def revoke(self,path):self.allow.discard(str(Path(path).resolve()).casefold())
    def require_target(self,case,name=None):
        self.require_case(case);freeze=self.get('SOURCE_FREEZE')
        if self.phase!='TARGET' or freeze is None or freeze['fingerprint']!=self.lock['fingerprint'] or set(freeze['models'])!=expected_names(self.scope):raise PermissionError('complete common207-source freeze before target')
        if name is not None and name not in freeze['models']:raise PermissionError('target uses own frozen model')
    def install_guard(self):
        import sys
        prefix=str(STORE.resolve()).casefold()+os.sep;metadata={str(Path(p).resolve()).casefold() for p in self.lock['input_sha256']}
        def hook(event,args):
            if event!='open' or not isinstance(args[0],str):return
            p=Path(args[0]).resolve();s=str(p).casefold()
            if s in self.allow or s in metadata or s.startswith(prefix):return
            if p.suffix.lower() in ('.pt','.pth','.mat','.npz','.npy') or '/data/raw/' in p.as_posix().lower():raise PermissionError('array/model/raw outside current declared role '+str(p))
        sys.addaudithook(hook)
    def acquire(self):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            rows=self.db.execute('SELECT pid,status FROM producer').fetchall()
            if rows and rows[0][1]=='RUNNING':raise PermissionError('one producer; recover only proven missing PID')
            if rows:self.db.execute("UPDATE producer SET pid=?,status='RUNNING'",(os.getpid(),))
            else:self.db.execute('INSERT INTO producer VALUES (?,?)',(os.getpid(),'RUNNING'))
    def release(self):
        with self.db:self.db.execute("UPDATE producer SET status='STOPPED'")


def write_equivalent(path,value):
    """Resume a non-neural terminal audit without overwriting prior evidence."""
    path=Path(path)
    if path.exists():
        if canonical(read(path))!=canonical(value):raise PermissionError('existing immutable derived output differs '+str(path))
    else:write_new(path,value)


def recover(lock,approval):
    rt=Runtime(lock,approval)
    try:
        rows=rt.db.execute('SELECT pid,status FROM producer').fetchall()
        if len(rows)!=1 or rows[0][1]!='RUNNING' or windows_process_live(rows[0][0]):raise PermissionError('actual absent producer, not observation timeout')
        if rt.db.execute('SELECT count(*) FROM attempts WHERE completed=0').fetchone()[0]:raise PermissionError('ambiguous paid attempts cannot replay')
        for name,step in rt.db.execute('SELECT chain,max(step) FROM attempts GROUP BY chain'):
            if name.startswith('qualification/'):
                if rt.get('P0_PASS') is None:raise PermissionError('incomplete paid qualifier needs explicit repair')
            else:
                saved=rt.get(name)
                if not saved or saved['resume_step']!=step or rt.cursor(name)!=(step,)*3 or sha(saved['resume']['path'])!=saved['resume']['sha256']:raise PermissionError('exact durable paid state required')
        for name, in rt.db.execute("SELECT name FROM journal WHERE name LIKE 'PENDING/%'"):
            if rt.get(name[8:]) is None:raise PermissionError('partial paid inference not auto-replayed')
        rt.release();return dict(status='SAFE_SAME_CLOCK_CONTINUATION',pid=rows[0][0],counters=rt.count())
    finally:rt.db.close()


def expected_names(scope):return {chain_key(c,a) for c in scope['cases'] for a in CONFIGS}


def validate_models(rt,models,files=True):
    if set(models)!=expected_names(rt.scope) or len(models)!=207:raise PermissionError('all207 source models, never a passing subset')
    for c in rt.scope['cases']:
        initial=[]
        for a in CONFIGS:
            name=chain_key(c,a);row=models[name];initial.append(row['initial_parameter_sha256'])
            if row['name']!=name or row['case']!=case_key(c) or row['task']!=c['task'] or row['seed']!=c['seed'] or row['config']!=a:raise PermissionError('complete registered model identity')
            if rt.cursor(name)!=(3001,)*3 or row['resume_step']!=3001 or row['source_fingerprint']!=c['source_fingerprint'] or row['engine_fingerprint']!=engine_fingerprint(rt.lock,c,a):raise PermissionError('fully paid frozen native source chain')
            if set(row['states'])!={'0','1000','2000','3001'} or [v['step'] for v in row['curve']]!=list(SOURCE_STEPS) or [v['step'] for v in row['logs']]!=list(range(100,3001,100))+[3001]:raise PermissionError('four scientific states and31 objective logs')
            chosen=min(row['curve'],key=lambda v:(-v['score'],v['step']))
            if row['selected']!=chosen['step'] or row['resume']!=row['states']['3001'] or not row['completed_unix']:raise PermissionError('fixed last / earliest strict source selection')
            if set(row['restores'])!={'1000','2000'} or any(not v.get('actual_disk_restore') or v['binding']!=row['states'][s] for s,v in row['restores'].items()):raise PermissionError('both actual complete-state disk restorations')
            if files:
                for b in row['states'].values():
                    store_path(b['path'])
                    if sha(b['path'])!=b['sha256']:raise PermissionError('frozen complete state bytes')
        if len(set(initial))!=1:raise PermissionError('same-case parameter initialization across native configs')


def freeze_sources(rt,models):
    if rt.phase!='SOURCE':raise PermissionError('freeze immediately after full source completion')
    validate_models(rt,models)
    path=OUT/'SOURCE_FREEZE.json'
    value=dict(fingerprint=rt.lock['fingerprint'],models=models,at_unix=time.time(),counters=rt.count())
    if path.exists():
        old=read(path)
        if old['fingerprint']!=value['fingerprint'] or canonical(old['models'])!=canonical(models):raise PermissionError('immutable full source freeze')
        value=old
    else:write_new(path,value)
    rt.put('SOURCE_FREEZE',value);rt.phase='TARGET';return value
