"""Durable execution boundary for the proposed PCR comparison. No Torch import.

Creating a proposal or passing synthetic tests is never neural authorization.
"""
from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256
import json
from importlib.metadata import version
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/wfdg/ress_pcr_comparison'
STORE = Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_pcr_comparison')
VERSION = 'PCR_AUTHOR_CORE_MDG_COMPARISON_V1'
PROFILES = ('AUTHOR_CE', 'CE_PLUS_REAL_HEALTHY', 'PCR_FAULT_TO_HEALTHY')
FRONTENDS = ('NATIVE40', 'SHORT_PHYSICAL_WIDTH')
LIMITS = dict(optimizer_attempts=144300, formal_updates=144000,
              qualification_updates=300, training_forward_windows=48104000,
              source_prediction_windows=2509608, target_prediction_windows=2785040,
              raw_record_loads=1440)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def chain_name(case, frontend, profile, *, qualification=None):
    if frontend not in FRONTENDS or profile not in PROFILES:
        raise ValueError('fixed comparator identity')
    stem = f"{case['task']['fold']}/s{case['seed']}/{frontend}/{profile}"
    if qualification is None:
        return 'formal/' + stem
    if qualification not in ('continuous', 'resumed'):
        raise ValueError('qualification trajectory')
    return 'qualification/' + qualification + '/' + stem


def expected_model_names(proposal):
    names = [chain_name(c, f, a) for c in proposal['cases'] for f in FRONTENDS for a in PROFILES]
    if len(names) != len(set(names)) or len(names) != 72:
        raise ValueError('exact 72 planned comparator models')
    return names


def verify_lock(lock, *, files=True):
    if canonical({k: v for k, v in lock.items() if k != 'fingerprint'}) != lock.get('fingerprint'):
        raise PermissionError('lock fingerprint')
    if lock.get('version') != VERSION or lock.get('limits') != LIMITS or lock.get('formal_models') != 72 or lock.get('steps') != 2000:
        raise PermissionError('fixed finite comparison scope')
    if files:
        for path, h in {**lock['code_sha256'], **lock['input_sha256']}.items():
            if sha(path) != h:
                raise PermissionError('locked source/metadata drift: ' + path)
        for package, expected in lock.get('dependency_versions', {}).items():
            if version(package) != expected:
                raise PermissionError('locked dependency drift: ' + package)


def authorization(value, lock, *, now=None):
    """Accept an ACTUAL, new scoped human reply receipt, never old card residue."""
    now = time.time() if now is None else now
    if value.get('kind') != 'HUMAN_FINITE_APPROVAL' or value.get('answer') != '批准这轮PCR完整比较':
        raise PermissionError('a new explicit PCR comparison approval is required')
    qid = value.get('question_item_id', '')
    if not isinstance(qid, str) or 'request_user_input_async' not in qid or 'call_' not in qid:
        raise PermissionError('actual question/reply identity required')
    if any(old in qid for old in ('call_c7cef07368e94f31b2d6175a6098da6c', 'call_1db61173002b4a10aa7ff13760a48c6a')):
        raise PermissionError('a completed older campaign is not authority')
    if value.get('lock_fingerprint') != lock['fingerprint'] or value.get('approved_scope') != lock['approved_scope']:
        raise PermissionError('human reply belongs to another code/scope')
    if value.get('limits') != LIMITS or value.get('neural_seconds') != 21600 or value.get('total_seconds') != 43200:
        raise PermissionError('exact proposed 6h neural/12h total finite budget required')
    start = value.get('total_start_unix')
    if type(start) not in (int, float) or not start <= now < start + 43200:
        raise PermissionError('human-approved original preparation clock expired/not started')
    return value


class BudgetLedger:
    """FULL SQLite counters. Unit fixtures use a separate ledger, never a NN."""
    def __init__(self, path, identity, limits):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10)
        if self.db.execute('PRAGMA journal_mode=WAL').fetchone()[0] != 'wal':
            raise RuntimeError('WAL unavailable')
        self.db.execute('PRAGMA synchronous=FULL')
        self.limits = dict(limits)
        with self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS identity (binding TEXT NOT NULL)')
            rows = self.db.execute('SELECT binding FROM identity').fetchall()
            if rows and rows != [(identity,)]:
                raise PermissionError('ledger cannot be rebound')
            if not rows:
                self.db.execute('INSERT INTO identity VALUES (?)', (identity,))
            self.db.execute('CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY,n INTEGER NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS attempts (chain TEXT NOT NULL,step INTEGER NOT NULL,phase TEXT NOT NULL,at REAL NOT NULL,completed INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(chain,step))')
            self.db.execute('CREATE TABLE IF NOT EXISTS journal (name TEXT PRIMARY KEY,value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS neural_clock (start REAL NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS producer (pid INTEGER NOT NULL,status TEXT NOT NULL)')

    def _add(self, name, n):
        if name not in self.limits and name not in ('actual_optimizer_updates', 'neural_model_constructions'):
            raise PermissionError('unrecognized counter')
        if type(n) is not int or n < 0:
            raise ValueError('nonnegative integer count')
        row = self.db.execute('SELECT n FROM counters WHERE name=?', (name,)).fetchone()
        new = (row[0] if row else 0) + n
        if new > self.limits.get(name, float('inf')):
            raise PermissionError('finite counter cap: ' + name)
        self.db.execute('INSERT INTO counters VALUES (?,?) ON CONFLICT(name) DO UPDATE SET n=excluded.n', (name, new))

    def claim(self, name, n=1):
        with self.db:
            self._add(name, n)

    def snapshot(self):
        return dict(self.db.execute('SELECT name,n FROM counters'))

    def cursor(self, name):
        row = self.db.execute('SELECT MAX(step),SUM(completed),COUNT(*) FROM attempts WHERE chain=?', (name,)).fetchone()
        return row[0] or 0, row[1] or 0, row[2]

    def attempt(self, name, step, phase, batch, profile):
        if phase not in ('SOURCE', 'QUALIFICATION') or profile not in PROFILES or type(step) is not int or type(batch) is not int or batch != 200:
            raise PermissionError('actual fixed training attempt identity')
        if not 1 <= step <= (2000 if phase == 'SOURCE' else 10):
            raise PermissionError('fixed chain step range')
        with self.db:
            last, completed, count = self.cursor(name)
            if (last, completed, count) != (last,) * 3 or step != last + 1:
                raise PermissionError('ambiguous/duplicate/gapped paid cursor; no replay')
            self._add('optimizer_attempts', 1)
            self._add('formal_updates' if phase == 'SOURCE' else 'qualification_updates', 1)
            self._add('training_forward_windows', batch * (1 if profile == 'AUTHOR_CE' else 2))
            self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,0)', (name, step, phase, time.time()))

    def completed(self, name, step):
        with self.db:
            cur = self.db.execute('UPDATE attempts SET completed=1 WHERE chain=? AND step=? AND completed=0', (name, step))
            if cur.rowcount != 1:
                raise PermissionError('completion needs one unique paid attempt')
            self._add('actual_optimizer_updates', 1)

    def get(self, name):
        row = self.db.execute('SELECT value FROM journal WHERE name=?', (name,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, name, value):
        with self.db:
            self.db.execute('INSERT INTO journal VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value',
                            (name, json.dumps(value, ensure_ascii=False, allow_nan=False)))


class RoleBoundary:
    def __init__(self, proposal):
        self.proposal = proposal
        self.records = proposal['records']
        self.cases = {(c['task']['fold'], c['seed']): c for c in proposal['cases']}
        self.frozen = False

    def require(self, case, key, role, *, counterfactual=False):
        pair = case['task']['fold'], case['seed']
        if pair not in self.cases or canonical(case) != canonical(self.cases[pair]):
            raise PermissionError('case metadata identity')
        if key not in self.records or role not in ('source_train', 'source_validation', 'target'):
            raise PermissionError('known record/role required')
        r = self.records[key]
        if r['bearing'] in {'K006', 'KA08', 'KA09'} or key not in case['roles'][role]:
            raise PermissionError('record has no legal role in this case')
        if role == 'target' and not self.frozen:
            raise PermissionError('target before full common source freeze')
        if counterfactual and role != 'source_train':
            raise PermissionError('CF labels only belong to source training')
        return r


class Runtime:
    def __init__(self, lock, approval):
        verify_lock(lock)
        authorization(approval, lock)
        self.lock, self.approval = lock, approval
        self.proposal = read(lock['proposal_path'])
        if sha(lock['proposal_path']) != lock['proposal_sha256']:
            raise PermissionError('proposal binding')
        self.roles = RoleBoundary(self.proposal)
        self.ledger = BudgetLedger(OUT / 'control/CONTROL.sqlite', canonical(dict(lock=lock['fingerprint'], approval=approval)), LIMITS)
        self.phase = 'SOURCE'
        self.allow = set()
        self.accesses = []
        row = self.ledger.db.execute('SELECT start FROM neural_clock').fetchall()
        if len(row) > 1:
            raise PermissionError('multiple neural clocks')
        self.neural_start = row[0][0] if row else None
        frozen = self.ledger.get('FROZEN_SOURCE')
        if frozen is not None:
            self.validate_frozen(frozen)
            self.roles.frozen = True
            self.phase = 'TARGET'

    def clock(self, neural=False):
        now = time.time()
        if not self.approval['total_start_unix'] <= now < self.approval['total_start_unix'] + 43200:
            raise TimeoutError('same human-approved total clock, no reset')
        if neural and (self.neural_start is None or now >= self.neural_start + 21600):
            raise TimeoutError('same first-neural clock, no reset')

    def neural(self):
        self.clock()
        if self.neural_start is None:
            at = time.time()
            with self.ledger.db:
                self.ledger.db.execute('INSERT INTO neural_clock VALUES (?)', (at,))
            self.neural_start = at
            write_new(OUT / 'NEURAL_CLOCK.json', dict(first_neural_unix=at, deadline=min(at+21600, self.approval['total_start_unix']+43200)))
        self.clock(True)

    def claim(self, name, n=1):
        self.clock(name in ('source_prediction_windows', 'target_prediction_windows', 'neural_model_constructions'))
        self.ledger.claim(name, n)

    def attempt(self, name, step, profile):
        self.clock(True)
        if step % 100 == 1:
            verify_lock(self.lock)
        self.ledger.attempt(name, step, self.phase, 200, profile)

    def authorize_record(self, case, key, role, *, counterfactual=False):
        self.clock()
        if role == 'target' and self.phase != 'TARGET':
            raise PermissionError('target stage not active')
        r = self.roles.require(case, key, role, counterfactual=counterfactual)
        self.accesses.append(dict(case=[case['task']['fold'], case['seed']], record=key, role=role,
                                  counterfactual=counterfactual, phase=self.phase, at_unix=time.time()))
        return r

    @contextmanager
    def permit(self, path):
        key = str(Path(path).resolve()).casefold()
        self.allow.add(key)
        try:
            yield
        finally:
            self.allow.discard(key)

    def install_access_guard(self):
        prefix = str(STORE.resolve()).casefold() + os.sep
        def hook(event, args):
            if event != 'open' or not isinstance(args[0], str):
                return
            path = Path(args[0]).resolve()
            name = str(path).casefold()
            if name in self.allow or name.startswith(prefix):
                return
            if path.suffix.lower() in ('.mat', '.npz', '.npy', '.pt', '.pth'):
                raise PermissionError('raw/array/state access outside declared role: ' + str(path))
        sys.addaudithook(hook)

    def acquire(self):
        with self.ledger.db:
            rows = self.ledger.db.execute('SELECT pid,status FROM producer').fetchall()
            if rows and rows[0][1] == 'RUNNING':
                raise PermissionError('producer marked running; inspect actual PID and exact paid checkpoint before explicit recovery')
            if rows:
                self.ledger.db.execute("UPDATE producer SET pid=?,status='RUNNING'", (os.getpid(),))
            else:
                self.ledger.db.execute("INSERT INTO producer VALUES (?,'RUNNING')", (os.getpid(),))

    def release(self):
        with self.ledger.db:
            self.ledger.db.execute("UPDATE producer SET status='STOPPED'")

    def validate_frozen(self, frozen):
        if frozen['fingerprint'] != self.lock['fingerprint'] or set(frozen['models']) != set(expected_model_names(self.proposal)):
            raise PermissionError('full common source freeze identity')
        for name, row in frozen['models'].items():
            if self.ledger.cursor(name) != (2000, 2000, 2000) or row['step'] != 2000 or [c['step'] for c in row['curve']] != [0, 1000, 2000]:
                raise PermissionError('every model must complete paid training and all source states')
            if not row.get('actual_disk_restore1000'):
                raise PermissionError('actual complete-state restore was not performed')
            for binding in (row['last'], row['selected']):
                require_store_path(binding['path'])
                if sha(binding['path']) != binding['sha256']:
                    raise PermissionError('frozen state bytes drift')

    def freeze(self, models):
        value = dict(fingerprint=self.lock['fingerprint'], models=models, at_unix=time.time())
        self.validate_frozen(value)
        path = OUT / 'FROZEN_SOURCE.json'
        if path.exists():
            prior = read(path)
            if prior['fingerprint'] != value['fingerprint'] or canonical(prior['models']) != canonical(models):
                raise PermissionError('existing freeze differs; no overwrite or changed timestamp')
            self.validate_frozen(prior)
            value = prior
        else:
            write_new(path, value)
        self.ledger.put('FROZEN_SOURCE', value)
        self.roles.frozen = True
        self.phase = 'TARGET'
        return value


def require_store_path(path):
    value = Path(path).resolve()
    if not value.is_relative_to(STORE.resolve()):
        raise PermissionError('new arrays/models must remain in the named D store')
    return value


def fresh_path(folder, suffix):
    folder = require_store_path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (uuid.uuid4().hex + suffix)


def windows_process_live(pid):
    """Read-only Win32 process query. Never use os.kill(pid,0) on Windows."""
    if os.name != 'nt' or type(pid) is not int or pid <= 0:
        raise ValueError('an actual positive Windows PID is required')
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.GetExitCodeProcess.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION only
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists
            return False
        raise OSError(error, 'cannot establish producer absence; do not restart')
    try:
        code = wintypes.DWORD()
        if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError(ctypes.get_last_error(), 'producer status unknown; do not restart')
        return code.value == 259  # STILL_ACTIVE
    finally:
        api.CloseHandle(handle)


def recover_missing_producer(lock, approval):
    """Explicit, zero-neural recovery only after actual process absence proof."""
    rt = Runtime(lock, approval)
    rows = rt.ledger.db.execute('SELECT pid,status FROM producer').fetchall()
    if len(rows) != 1 or rows[0][1] != 'RUNNING':
        raise PermissionError('only an existing marked producer can be recovered')
    pid = rows[0][0]
    if windows_process_live(pid):
        raise PermissionError('actual producer still live: poll/wait it, never restart')
    attempts = rt.ledger.db.execute('SELECT COUNT(*) FROM attempts WHERE completed=0').fetchone()[0]
    if attempts:
        raise PermissionError('ambiguous paid work cannot be replayed')
    for name, paid_step in rt.ledger.db.execute('SELECT chain,MAX(step) FROM attempts GROUP BY chain'):
        if name.startswith('qualification/'):
            if rt.ledger.get('P0_PASS') is None:
                raise PermissionError('partial qualification needs explicit repair, not fresh replay')
            continue
        journal = rt.ledger.get(name)
        if journal is None or journal.get('resume_step') != paid_step or rt.ledger.cursor(name) != (paid_step,)*3:
            raise PermissionError('all paid formal cursors require exact durable state')
        binding = journal['resume']
        require_store_path(binding['path'])
        if sha(binding['path']) != binding['sha256']:
            raise PermissionError('recovery complete-state bytes differ')
    for name, value in rt.ledger.db.execute("SELECT name,value FROM journal WHERE name LIKE 'PENDING/TARGET/%'"):
        target_name = name[len('PENDING/'):]
        if rt.ledger.get(target_name) is None:
            raise PermissionError('partial target evaluation cannot be silently repeated')
    rt.release()
    receipt = dict(status='MISSING_PRODUCER_VERIFIED_SAFE_TO_CONTINUE_SAME_CLOCKS', pid=pid,
                   fingerprint=lock['fingerprint'], at_unix=time.time(), counters=rt.ledger.snapshot(),
                   new_neural=0, new_updates=0, deadlines_reset=False)
    write_new(OUT/('RECOVERY_'+str(time.time_ns())+'.json'), receipt)
    return receipt
