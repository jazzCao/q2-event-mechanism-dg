"""Selection experiment's stdlib-only grant and durable execution primitives.

No production lock exists yet. These components do not authorize execution;
the future full runner must bind its own code, qualifications and finite caps.
"""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/wfdg/ress_e0_source_selection'
STORE = Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_e0_source_selection')
KIND = 'E0_SOURCE_SELECTION_G_R_U_V1'
GRANT_KIND = 'EXPLICIT_HUMAN_E0_SOURCE_SELECTION_GRANT'
ARMS = ('G_PAIRED', 'R_COUNT', 'U_ENERGY')
COUNTERS = ('optimizer_attempts', 'auxiliary_attempts', 'main_attempts',
            'qualification_attempts', 'training_forward_windows',
            'source_prediction_windows', 'target_prediction_windows',
            'neural_model_builds', 'cache_read_requests', 'cache_hash_checks',
            'cache_payload_decodes')
FORBIDDEN = ('new_raw_reads', 'new_FFT', 'new_inverse_fits')
REQUIRED_CODE = ('e0_selection_control.py', 'e0_selection_files.py',
                 'e0_selection_runtime.py', 'e0_selection_neural.py', 'e0_selection_driver.py',
                 'run_e0_selection.py', 'audit_e0_selection.py',
                 'e0_selection_source_arrays.py', 'e0_source_selection.py',
                 'clock_factual_data.py')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def finite_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def compile_models(registry):
    models = {}
    for t in registry['teachers']:
        key = 'AUX/' + t['key']
        if key in models or t['fixed_steps'] != 2000:raise ValueError('unique fixed auxiliary model')
        models[key] = dict(stage='TEACHERS', kind='auxiliary', steps=2000,
                           forward_windows=240, origin=t['key'])
    for case in registry['cases']:
        for arm in ARMS:
            key = 'MAIN/' + arm + '/' + case['key']
            if key in models:raise ValueError('unique fixed main model')
            models[key] = dict(stage='MAIN', kind='main', steps=2000,
                               forward_windows=2 * case['main_batch'], origin=case['key'], arm=arm)
    return models


def authorize(lock, grant, registry, now=None):
    """Pure fail-closed checks, before production mkdir/SQLite/cache/Torch.

    Presence of a JSON grant is not human verification. Its recorded message
    must be an actual new human response; the runner cannot manufacture one.
    """
    if not grant or grant.get('kind') != GRANT_KIND:raise PermissionError('new actual selection grant required; old72 grant is closed')
    if lock.get('kind') != KIND or lock.get('execution_ready') is not True:
        raise PermissionError('complete selection entry not yet locked')
    if lock.get('fingerprint') != canonical({k:v for k,v in lock.items() if k != 'fingerprint'}):
        raise PermissionError('selection lock fingerprint')
    if registry.get('fingerprint') != canonical({k:v for k,v in registry.items() if k != 'fingerprint'}):
        raise PermissionError('input registry fingerprint')
    if lock.get('registry_fingerprint') != registry['fingerprint']:raise PermissionError('exact input registry required')
    if (len(registry['cases']), len(registry['slots']), len(registry['teachers']), len(registry['caches'])) != (81, 99, 297, 98):
        raise PermissionError('unchanged full81/99/297/98 scope')
    if lock.get('models') != compile_models(registry):raise PermissionError('fixed three arms and all auxiliary models')
    if len({x['key'] for x in registry['cases']}) != 81 or len({x['slot'] for x in registry['slots']}) != 99:
        raise PermissionError('duplicate case or slot')
    for slot in registry['slots']:
        members = [t['ensemble_member'] for t in registry['teachers'] if t['slot'] == slot['slot']]
        if sorted(members) != [0, 1, 2]:raise PermissionError('exact three paired-vote members per slot')
    if not set(REQUIRED_CODE) <= {Path(p).name for p in lock.get('code_sha256', {})}:
        raise PermissionError('full runtime, neural entry and independent audit must be bound')
    caps = lock.get('limits', {})
    if set(caps) != set(COUNTERS) or any(type(v) is not int or v < 0 for v in caps.values()):
        raise PermissionError('complete finite integer resource caps')
    if caps['auxiliary_attempts'] != 594000 or caps['main_attempts'] != 486000:
        raise PermissionError('unchanged formal update scope')
    if caps['optimizer_attempts'] != caps['auxiliary_attempts'] + caps['main_attempts'] + caps['qualification_attempts']:
        raise PermissionError('attempt categories must close exactly')
    if any(lock.get(k) != 0 or grant.get(k) != 0 for k in FORBIDDEN):raise PermissionError('zero new raw/FFT/inverse fits')
    if grant.get('lock_fingerprint') != lock['fingerprint'] or grant.get('limits') != caps:
        raise PermissionError('actual grant must bind this exact lock and caps')
    if not all(isinstance(grant.get(k), str) and grant[k].strip() for k in ('human_message_id', 'answer', 'question_id')):
        raise PermissionError('recorded actual new human answer required')
    for name in ('total_seconds', 'neural_seconds'):
        if not finite_seconds(lock.get(name)) or grant.get(name) != lock[name]:raise PermissionError('finite approved clocks')
    if lock['neural_seconds'] > lock['total_seconds']:raise PermissionError('neural clock cannot exceed total clock')
    now = time.time() if now is None else now
    start = grant.get('total_start_unix')
    if type(start) not in (int, float) or not math.isfinite(start) or not start <= now < start + lock['total_seconds']:
        raise TimeoutError('total grant clock expired or not started')
    qualifications = lock.get('qualification_models', {})
    if not qualifications or any(not k.startswith('P0/') or v.get('kind') != 'qualification' or v.get('stage') != 'P0'
        or type(v.get('steps')) is not int or v['steps'] < 1 or type(v.get('forward_windows')) is not int or v['forward_windows'] < 1
        for k,v in qualifications.items()):raise PermissionError('explicit finite P0 chains required')
    if sum(v['steps'] for v in qualifications.values()) > caps['qualification_attempts']:
        raise PermissionError('P0 over fixed qualification cap')
    return dict(grant)


class Ledger:
    """FULL pre-charge, no paid-tail replay, immutable packet/freeze identities.

    An in-memory connection with clocks=None is exclusively a synthetic test.
    The production factory always supplies both a bound grant and a D root.
    """
    def __init__(self, db, models, prediction_specs, limits, *, qualification_models=None,
                 now=None, clocks=None, store_root=None):
        self.db = db;self.now = time.time if now is None else now
        self.models = dict(models);self.qualifications = dict(qualification_models or {})
        self.prediction_specs = dict(prediction_specs);self.limits = dict(limits)
        self.clocks = clocks;self.store_root = Path(store_root).resolve() if store_root is not None else None
        if set(self.limits) != set(COUNTERS):raise ValueError('all resource caps required')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY,n INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts(chain TEXT NOT NULL,step INTEGER NOT NULL,kind TEXT NOT NULL,
          at REAL NOT NULL,done INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(chain,step));
        CREATE TABLE IF NOT EXISTS packets(chain TEXT PRIMARY KEY,step INTEGER NOT NULL,binding TEXT NOT NULL,complete INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS packet_history(chain TEXT NOT NULL,step INTEGER NOT NULL,binding TEXT NOT NULL,PRIMARY KEY(chain,step));
        CREATE TABLE IF NOT EXISTS predictions(key TEXT PRIMARY KEY,spec TEXT NOT NULL,done INTEGER NOT NULL DEFAULT 0,binding TEXT);
        CREATE TABLE IF NOT EXISTS io(id INTEGER PRIMARY KEY,at REAL NOT NULL,detail TEXT NOT NULL);
        ''')
        identity = canonical(dict(models=self.models, qualifications=self.qualifications,
                                  predictions=self.prediction_specs, limits=self.limits, clocks=clocks))
        if self.get('identity') is None:self.put('identity', identity)
        if self.get('identity') != identity:raise PermissionError('ledger scope or original clocks changed')
        if self.get('stage') is None:self.put('stage', 'P0')

    def get(self, key):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key, value):
        self._set(key,value);self.db.commit()

    def _set(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value, allow_nan=False)))

    def _once(self, key, value):
        self.db.execute('INSERT INTO meta VALUES (?,?)', (key, json.dumps(value, allow_nan=False)))

    def count(self, key):
        row = self.db.execute('SELECT n FROM counters WHERE name=?', (key,)).fetchone()
        return 0 if row is None else row[0]

    def _add(self, key, n):
        old = self.count(key)
        if key not in self.limits or type(n) is not int or n < 0 or old + n > self.limits[key]:
            raise PermissionError('finite resource cap: ' + key)
        self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)', (key, old+n))

    def check_total(self):
        if self.clocks is not None and not self.clocks['total_start_unix'] <= self.now() < self.clocks['total_start_unix'] + self.clocks['total_seconds']:
            raise TimeoutError('total clock; restart cannot reset')

    def neural(self):
        self.check_total()
        if self.get('stage') == 'DONE' or self.get('NEURAL_END') is not None:raise PermissionError('neural phase closed')
        if self.clocks is None:return
        start = self.get('NEURAL_START')
        if start is None:start = self.now();self.put('NEURAL_START', start)
        if not start <= self.now() < start + self.clocks['neural_seconds']:raise TimeoutError('neural clock; restart cannot reset')

    def model(self, chain):
        if chain not in self.models and chain not in self.qualifications:raise PermissionError('unregistered model')
        return self.models.get(chain, self.qualifications.get(chain))

    def construction(self, chain, *, purpose='train'):
        self.neural();model = self.model(chain);stage = self.get('stage')
        if purpose == 'train':allowed = stage == model['stage']
        elif purpose == 'predict':allowed = ((stage == 'SOURCE_QUERY' and model['kind'] == 'auxiliary')
            or (stage == 'TARGET' and model['kind'] == 'main') or (stage == 'P0' and model['kind'] == 'qualification'))
        else:allowed = False
        if not allowed:raise PermissionError('model construction phase/purpose')
        with self.db:self._add('neural_model_builds', 1)

    def attempt(self, chain, step):
        self.neural();m = self.model(chain)
        if self.get('stage') != m['stage'] or type(step) is not int or not 1 <= step <= m['steps']:
            raise PermissionError('model stage or exact step range')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            n, done = self.db.execute('SELECT count(*),coalesce(sum(done),0) FROM attempts WHERE chain=?', (chain,)).fetchone()
            if n != done or step != n + 1:raise PermissionError('no duplicate, gapped or unfinished paid step')
            self._add('optimizer_attempts', 1);self._add(m['kind']+'_attempts', 1)
            self._add('training_forward_windows', m['forward_windows'])
            self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,0)', (chain, step, m['kind'], self.now()));self.db.commit()
        except BaseException:self.db.rollback();raise

    def completed(self, chain, step):
        with self.db:
            row = self.db.execute('UPDATE attempts SET done=1 WHERE chain=? AND step=? AND done=0', (chain,step))
            if row.rowcount != 1:raise PermissionError('one completion for each paid attempt')

    def check_binding(self, b, suffix):
        if not isinstance(b, dict) or not isinstance(b.get('path'), str) or len(b.get('sha256', '')) != 64:
            raise ValueError('artifact byte binding required')
        p = Path(b['path']).resolve()
        if p.suffix != suffix or (self.store_root is not None and not p.is_relative_to(self.store_root)):
            raise PermissionError('owned D artifact path and suffix')

    def checkpoint(self, chain, step, binding, *, complete=False):
        m = self.model(chain)
        n, done = self.db.execute('SELECT count(*),coalesce(sum(done),0) FROM attempts WHERE chain=?', (chain,)).fetchone()
        if n != step or done != step or binding.get('step') != step:raise PermissionError('packet cannot omit any paid tail')
        if complete and step != m['steps']:raise PermissionError('final packet is exact fixed step')
        self.check_binding(binding, '.pt')
        old = self.db.execute('SELECT binding FROM packet_history WHERE chain=? AND step=?', (chain,step)).fetchone()
        if old and json.loads(old[0]) != binding:raise PermissionError('immutable scientific packet')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO packet_history VALUES (?,?,?)', (chain,step,json.dumps(binding)))
            self.db.execute('INSERT OR REPLACE INTO packets VALUES (?,?,?,?)', (chain,step,json.dumps(binding),int(complete)))

    def resume(self, chain):
        self.model(chain)
        p = self.db.execute('SELECT step,binding FROM packets WHERE chain=?', (chain,)).fetchone()
        n, done = self.db.execute('SELECT count(*),coalesce(sum(done),0) FROM attempts WHERE chain=?', (chain,)).fetchone()
        if p is None:
            if n:raise PermissionError('paid orphan needs new explicit repair decision')
            return None
        if p[0] != n or done != n:raise PermissionError('paid tail replay is not automatic')
        return json.loads(p[1])

    def _all_final(self, ids):
        if not ids:raise PermissionError('nonempty fixed model set')
        for chain in ids:
            p = self.db.execute('SELECT step,complete FROM packets WHERE chain=?', (chain,)).fetchone()
            if not p or p != (self.model(chain)['steps'], 1):raise PermissionError('all fixed models need final scientific packet')
            self.resume(chain)

    def _predictions_done(self, stage):
        keys = {k for k,v in self.prediction_specs.items() if v['stage'] == stage}
        done = {k for k, in self.db.execute('SELECT key FROM predictions WHERE done=1')}
        if not keys <= done:raise PermissionError('all fixed phase predictions required')

    def p0_pass(self, evidence):
        if self.get('stage') != 'P0':raise PermissionError('P0 only once')
        self._all_final(self.qualifications);self._predictions_done('P0')
        flags = ('cuda', 'complete_state_equal', 'gradient_finite', 'gradient_nonzero', 'timing_admitted')
        if evidence.get('status') != 'CUDA_CONTINUOUS_DISK_RESUME_PASS' or any(evidence.get(k) is not True for k in flags):
            raise PermissionError('real CUDA, complete state, gradients and timing required')
        with self.db:
            self._once('P0', evidence);self._once('STAGE/P0', self.now());self._set('stage','TEACHERS')

    def freeze(self, kind):
        source, target, token = ('TEACHERS', 'SOURCE_QUERY', 'TEACHER_FREEZE') if kind == 'auxiliary' else ('MAIN', 'TARGET', 'MAIN_FREEZE') if kind == 'main' else (None,None,None)
        if self.get('stage') != source or self.get('P0') is None:raise PermissionError('correct freeze stage after P0')
        ids = {k for k,v in self.models.items() if v['kind'] == kind};self._all_final(ids)
        if kind == 'main':self._predictions_done('MAIN')
        value = dict(at_unix=self.now(), models={k:self.resume(k) for k in sorted(ids)})
        with self.db:self._once(token, value);self._set('stage',target)
        return value

    def claim_prediction(self, key):
        self.neural()
        spec = self.prediction_specs.get(key)
        if not spec or spec['stage'] != self.get('stage'):raise PermissionError('exact predeclared prediction role')
        if type(spec.get('windows')) is not int or spec['windows'] < 1:raise PermissionError('positive fixed prediction window count')
        if spec['stage'] == 'SOURCE_QUERY' and self.get('TEACHER_FREEZE') is None:raise PermissionError('common teacher freeze first')
        if spec['stage'] == 'TARGET' and self.get('MAIN_FREEZE') is None:raise PermissionError('common main freeze first')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM predictions WHERE key=?', (key,)).fetchone():raise PermissionError('no prediction replay')
            counter = 'target_prediction_windows' if spec['stage'] == 'TARGET' else 'source_prediction_windows'
            self._add(counter, spec['windows'])
            self.db.execute('INSERT INTO predictions(key,spec) VALUES (?,?)', (key,json.dumps(spec)));self.db.commit()
        except BaseException:self.db.rollback();raise

    def prediction_done(self, key, binding):
        self.check_binding(binding, '.npz');spec = self.prediction_specs[key]
        if binding.get('windows') != spec['windows']:raise ValueError('exact paid prediction window count')
        with self.db:
            row = self.db.execute('UPDATE predictions SET done=1,binding=? WHERE key=? AND done=0', (json.dumps(binding),key))
            if row.rowcount != 1:raise PermissionError('one actual prediction completion')

    def queries_complete(self):
        if self.get('stage') != 'SOURCE_QUERY':raise PermissionError('source query stage')
        self._predictions_done('SOURCE_QUERY');self.put('stage', 'SELECT')

    def selection(self, case, binding):
        if self.get('stage') != 'SELECT':raise PermissionError('selection after all source queries')
        if case not in {v['origin'] for v in self.models.values() if v['kind'] == 'main'}:raise PermissionError('fixed main case')
        self.check_binding(binding, '.npz')
        with self.db:self._once('SELECTION/'+case, binding)

    def selections_complete(self):
        cases = {v['origin'] for v in self.models.values() if v['kind'] == 'main'}
        if self.get('stage') != 'SELECT' or not cases or any(self.get('SELECTION/'+k) is None for k in cases):
            raise PermissionError('every fixed case selection required')
        self.put('stage', 'MAIN')

    def io(self, detail, *, hash_checks=0, decodes=0):
        self.check_total()
        if self.get('stage') == 'DONE':raise PermissionError('no new cache action after finish')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._add('cache_read_requests', 1);self._add('cache_hash_checks',hash_checks);self._add('cache_payload_decodes',decodes)
            self.db.execute('INSERT INTO io(at,detail) VALUES (?,?)', (self.now(),json.dumps(detail,allow_nan=False)));self.db.commit()
        except BaseException:self.db.rollback();raise

    def finish(self):
        if self.get('stage') != 'TARGET':raise PermissionError('target completion only')
        self._predictions_done('TARGET')
        if self.db.execute('SELECT count(*) FROM predictions WHERE done=0').fetchone()[0]:raise PermissionError('partial prediction')
        with self.db:self._set('NEURAL_END',self.now());self._set('stage','DONE')


def open_production(lock_path=OUT/'RUN_LOCK_V1.json', grant_path=OUT/'HUMAN_APPROVAL_V1.json'):
    """Future full entry only; no new production lock/grant is made here."""
    grant = read(grant_path) if Path(grant_path).is_file() else None
    if not grant or grant.get('kind') != GRANT_KIND:raise PermissionError('new actual selection grant required before production I/O')
    lock = read(lock_path);registry = read(ROOT/lock['registry_path'])
    auth = authorize(lock,grant,registry)
    for name in ('code_sha256', 'input_sha256'):
        for path,digest in lock[name].items():
            if sha(ROOT/path) != digest:raise PermissionError('locked code/input drift: '+path)
    if sha(ROOT/lock['registry_path']) != lock['registry_sha256']:raise PermissionError('exact registry bytes required')
    control = OUT/'control';control.mkdir(parents=True,exist_ok=True)
    db = sqlite3.connect(control/'CONTROL.sqlite');db.execute('PRAGMA journal_mode=WAL')
    ledger = Ledger(db,lock['models'],lock['prediction_specs'],lock['limits'],
        qualification_models=lock['qualification_models'],clocks=auth,store_root=STORE)
    if ledger.get('grant') is None:ledger.put('grant',auth)
    elif ledger.get('grant') != auth:raise PermissionError('existing grant must never be rebound')
    return lock,auth,registry,ledger


class AccessContext:
    """Bound metadata/cache capability used by the future execution runtime."""
    def __init__(self, lock, grant, registry, ledger):
        authorize(lock,grant,registry)
        if ledger.get('grant') != grant:raise PermissionError('exact durable grant required')
        self.lock, self.grant, self.registry, self.ledger = lock, grant, registry, ledger
        self._lock_hash, self._grant_hash = canonical(lock), canonical(grant)

    def assert_authorized(self):
        if canonical(self.lock) != self._lock_hash or canonical(self.grant) != self._grant_hash:
            raise PermissionError('in-memory lock or grant mutated')
        authorize(self.lock,self.grant,self.registry);self.ledger.check_total()

    def read_locked_json(self, path):
        self.assert_authorized()
        if path not in self.lock['input_sha256']:raise PermissionError('unregistered metadata JSON')
        digest = self.lock['input_sha256'][path];p = ROOT/path
        if sha(p) != digest:raise PermissionError('metadata byte drift')
        value = read(p)
        if sha(p) != digest:raise PermissionError('metadata changed during read')
        return value
