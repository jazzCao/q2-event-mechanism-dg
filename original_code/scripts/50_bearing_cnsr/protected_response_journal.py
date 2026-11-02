"""Finite CPU source-work journal; no arrays, imports of ML or production CLI.

Reserve a whole immutable unit before work; FFT/read counters are in memory
within the unit and committed once. A lost incomplete unit is NOT replayed.
Only a complete immutable packet+receipt can be adopted without recomputation.
The eventual entry still has to authenticate the real human reply and enforce
its source-path allowlist; this primitive alone does NOT authorize real data.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

FORBIDDEN = ('neural_models', 'optimizer_updates', 'target_predictions', 'new_inverse_fits')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def process_identity(pid):
    """Windows native creation timestamp, decimal string, never JS float."""
    if os.name != 'nt':
        raise RuntimeError('this journal is qualified only on Windows')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD); api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME))
    api.GetProcessTimes.restype = wintypes.BOOL
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = api.OpenProcess(0x1000, False, int(pid))
    if not handle:
        # Access denied is not evidence that a producer has disappeared.
        if ctypes.get_last_error() == 5:
            raise PermissionError('cannot verify existing producer identity')
        return None
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)):
            raise PermissionError('cannot read producer creation time')
        birth = (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
        return dict(pid=int(pid), creation_filetime=str(birth))
    finally:
        api.CloseHandle(handle)


class Journal:
    def __init__(self, database, store, *, scope, units, limits, grant, now=time.time):
        self.now, self.scope, self.units, self.limits = now, scope, units, limits
        self.database, self.store = Path(database), Path(store).resolve()
        self.active, self.usage, self.sealed = None, {}, False
        if not grant or grant.get('scope_fingerprint') != canonical(dict(scope=scope, units=units, limits=limits)):
            raise PermissionError('exact new finite scope grant required before mkdir/database')
        if grant.get('kind') not in ('PROTECTED_RESPONSE_SOURCE_GRANT_V1', 'SYNTHETIC_ONLY_SOURCE_GRANT'):
            raise PermissionError('not a source qualification grant')
        if grant['kind'] == 'SYNTHETIC_ONLY_SOURCE_GRANT' and not any(v.startswith('SYNTHETIC_') for v in self.store.parts):
            raise PermissionError('test authorization only inside retained SYNTHETIC fixture')
        if any(type(v) is not int or v < 0 for v in limits.values()) or any(limits.get(k) != 0 for k in FORBIDDEN):
            raise PermissionError('finite integer quotas and exactly zero neural/target/fitting authority')
        if limits.get('adoption_hash_checks', 0) < 1:
            raise PermissionError('bounded proof-only adoption accounting required')
        for bound in units.values():
            if any(k not in limits or type(v) is not int or v < 0 or v > limits[k] for k, v in bound.items()):
                raise PermissionError('predeclared finite unit reservations')
        self.grant = grant
        self.clock()
        self.database.parent.mkdir(parents=True, exist_ok=True); self.store.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.database, timeout=10)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS units(key TEXT PRIMARY KEY,status TEXT NOT NULL,detail TEXT NOT NULL); CREATE TABLE IF NOT EXISTS counters(key TEXT PRIMARY KEY,n INTEGER NOT NULL);')
        identity = canonical(dict(scope=scope, units=units, limits=limits, grant=grant, store=str(self.store)))
        old = self.get('identity')
        if old is None:
            self.put('identity', identity)
        if self.get('identity') != identity:
            self.db.close(); raise PermissionError('scope/grant/original clock changed')
        if self.get('DONE') is not None:
            self.db.close(); raise PermissionError('source stage closed; use read-only verification, do not reenter')
        self.owner = dict(process_identity(os.getpid()), token=uuid.uuid4().hex)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.get('owner')
            if prior and prior['status'] == 'RUNNING' and process_identity(prior['pid']) == {k: prior[k] for k in ('pid', 'creation_filetime')}:
                raise PermissionError('another actual producer is still live')
            self._put('owner', dict(self.owner, status='RUNNING')); self.db.commit()
        except BaseException:
            self.db.rollback(); self.db.close(); raise

    def clock(self):
        start, seconds = self.grant.get('start_unix'), self.grant.get('total_seconds')
        if type(start) not in (int, float) or type(seconds) not in (int, float) or not seconds > 0 or not start <= self.now() < start+seconds:
            raise TimeoutError('original finite total clock; recovery does not reset')

    def get(self, key):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value, allow_nan=False)))

    def put(self, key, value):
        with self.db:
            self._put(key, value)

    def _delta(self, key, amount):
        old = self.db.execute('SELECT n FROM counters WHERE key=?', (key,)).fetchone()
        total = (old[0] if old else 0)+amount
        if key not in self.limits or not 0 <= total <= self.limits[key]:
            raise PermissionError('source quota '+key)
        self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)', (key, total))

    def counts(self):
        return dict(self.db.execute('SELECT key,n FROM counters'))

    def paths(self, key):
        name = hashlib.sha256(key.encode()).hexdigest()
        return self.store/(name+'.npz'), self.store/(name+'.receipt.json')

    def state(self, key):
        row = self.db.execute('SELECT status,detail FROM units WHERE key=?', (key,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def begin(self, key):
        self.clock()
        if self.sealed or self.active is not None or key not in self.units:
            raise PermissionError('one registered active source unit')
        row = self.state(key)
        if row and row[0] == 'DONE':
            return False
        if row:
            return self._adopt(key)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for name, maximum in self.units[key].items():
                self._delta(name, maximum)
            self.db.execute('INSERT INTO units VALUES (?,?,?)', (key, 'PENDING', json.dumps(dict(reserved=self.units[key], started_unix=self.now()))))
            self.db.commit()
        except BaseException:
            self.db.rollback(); raise
        self.active, self.usage = key, {k: 0 for k in self.units[key]}
        return True

    def spend(self, name, n=1):
        self.clock()
        if self.active is None or name not in self.usage or type(n) is not int or n < 0 or self.usage[name]+n > self.units[self.active][name]:
            raise PermissionError('active source unit resource '+name)
        self.usage[name] += n

    def finish(self):
        self.clock(); key = self.active
        if key is None:
            raise PermissionError('no active unit to finish')
        packet, receipt_path = self.paths(key)
        self.spend('owned_hash_checks', 1)
        digest = file_sha(packet)
        self.clock()
        receipt = dict(unit=key, scope_fingerprint=self.grant['scope_fingerprint'], packet=dict(path=str(packet), sha256=digest), usage=dict(self.usage), finished_unix=self.now())
        with receipt_path.open('x', encoding='utf8') as stream:
            json.dump(receipt, stream, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
        self._commit(key, receipt)
        self.active, self.usage = None, {}
        return receipt['packet']

    def _commit(self, key, receipt):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for name, reserved in self.units[key].items():
                self._delta(name, receipt['usage'][name]-reserved)
            row = self.db.execute('UPDATE units SET status=?,detail=? WHERE key=? AND status=?', ('DONE', json.dumps(receipt, allow_nan=False), key, 'PENDING'))
            if row.rowcount != 1:
                raise PermissionError('one source completion per reservation')
            self.db.commit()
        except BaseException:
            self.db.rollback(); raise

    def _adopt(self, key):
        packet, receipt_path = self.paths(key)
        if not packet.is_file() or not receipt_path.is_file():
            raise PermissionError('paid incomplete source unit: no automatic replay or grant-clock reset')
        value = json.loads(receipt_path.read_text(encoding='utf8'))
        if value.get('unit') != key or value.get('scope_fingerprint') != self.grant['scope_fingerprint'] or value.get('packet', {}).get('path') != str(packet):
            raise PermissionError('immutable source receipt identity')
        if not self.grant['start_unix'] <= value.get('finished_unix', -1) < self.grant['start_unix']+self.grant['total_seconds']:
            raise PermissionError('original packet finished outside original total clock')
        usage = value.get('usage', {})
        if set(usage) != set(self.units[key]) or any(type(n) is not int or not 0 <= n <= self.units[key][k] for k, n in usage.items()):
            raise PermissionError('complete finite original source-unit accounting')
        with self.db:
            self._delta('adoption_hash_checks', 1)
        if file_sha(packet) != value['packet']['sha256']:
            raise PermissionError('orphan packet bytes do not prove completed source work')
        self.clock()
        self._commit(key, value)
        return False

    def seal(self):
        self.clock()
        if self.active is not None or any(not self.state(k) or self.state(k)[0] != 'DONE' for k in self.units):
            raise PermissionError('all source units must be complete before closure')
        with self.db:
            self._put('DONE', dict(at_unix=self.now(), counts=self.counts()))
            self._put('owner', dict(self.owner, status='DONE'))
        self.sealed = True

    def close(self):
        if not self.sealed:
            with self.db:
                self._put('owner', dict(self.owner, status='STOPPED', pending_unit=self.active, observed_uncommitted_usage=self.usage))
        self.db.close()
