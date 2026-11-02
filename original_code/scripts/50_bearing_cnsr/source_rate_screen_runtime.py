"""Native single-producer, durable paid steps and bounded immutable artifacts."""
from datetime import datetime
from pathlib import Path
import json
import os
import sqlite3
import time
import source_rate_plan as p
from protected_response_journal import process_identity


class Runtime:
    def __init__(self, lock):
        self.lock = lock; self.phase = 'SOURCE'; self.permitted = set()
        self.store = Path(lock['campaign']['store']).resolve(); self.store.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(p.OUT / lock.get('control_filename', 'CONTROL.sqlite'), timeout=30)
        self.db.execute('PRAGMA journal_mode=DELETE'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);'
            'CREATE TABLE IF NOT EXISTS counters(k TEXT PRIMARY KEY,n INTEGER);'
            'CREATE TABLE IF NOT EXISTS steps(chain TEXT,step INTEGER,done INTEGER,PRIMARY KEY(chain,step));'
            'CREATE TABLE IF NOT EXISTS artifacts(path TEXT PRIMARY KEY,kind TEXT,detail TEXT);'
            'CREATE TABLE IF NOT EXISTS finished(chain TEXT PRIMARY KEY,detail TEXT);')
        self.db.execute('BEGIN IMMEDIATE')
        owner = self.get('owner')
        if owner and owner['state'] == 'RUNNING':
            p.require(process_identity(owner['pid']) != owner['native'], 'existing producer still alive; do not restart')
        old = self.get('fingerprint')
        p.require(old is None or old == lock['fingerprint'], 'same durable neural comparison')
        p.require(self.get('DONE') is None, 'closed experiment cannot restart')
        self.set('fingerprint', lock['fingerprint']); self.native = process_identity(os.getpid())
        self.set('owner', dict(state='RUNNING', pid=os.getpid(), native=self.native, at_unix=time.time()))
        self.db.commit(); self.counts = {k: n for k, n in self.db.execute('SELECT k,n FROM counters')}
        self.clock()

    def get(self, key):
        row = self.db.execute('SELECT v FROM meta WHERE k=?', (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value, allow_nan=False)))

    def clock(self):
        now = time.time(); scope = self.lock['campaign']
        p.require(now < datetime.fromisoformat(scope['deadline']).timestamp(), 'original8h total expired')
        start = self.get('neural_start')
        p.require(start is None or self.get('NEURAL_END') is not None or
            now < start['at_unix'] + scope['neural_seconds'], 'original2h neural stage expired; no reset')

    def before_neural(self):
        self.clock()
        p.require(self.get('NEURAL_END') is None, 'no new neural work after sealed neural end')
        if self.get('neural_start') is None:
            with self.db:
                self.set('neural_start', dict(at_unix=time.time(), native=self.native))

    def _charge(self, key, count):
        p.require(type(count) is int and count >= 0 and key in self.lock['limits'], 'declared finite resource ' + key)
        value = self.counts.get(key, 0) + count
        p.require(value <= self.lock['limits'][key], 'resource exhausted: ' + key)
        self.counts[key] = value
        self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)', (key, value))

    def claim(self, key, count=1):
        self.clock()
        with self.db: self._charge(key, count)

    def attempt(self, chain, step):
        self.before_neural(); spec = self.lock['chains'][chain]
        p.require(self.phase == 'SOURCE' and 1 <= step <= spec['steps'], 'source-only registered update')
        rows = self.db.execute('SELECT count(*),coalesce(sum(done),0) FROM steps WHERE chain=?', (chain,)).fetchone()
        p.require(rows == (step - 1, step - 1), 'paid/done sequential; no automatic lost-step replay')
        with self.db:
            self._charge('optimizer_attempts', 1)
            self._charge('qualification_updates' if spec['qualification'] else 'formal_updates', 1)
            self.db.execute('INSERT INTO steps VALUES (?,?,0)', (chain, step))

    def completed(self, chain, step):
        with self.db:
            cur = self.db.execute('UPDATE steps SET done=1 WHERE chain=? AND step=? AND done=0', (chain, step))
            p.require(cur.rowcount == 1, 'one actual optimizer completion for paid update')

    def permit(self, path): self.permitted.add(str(Path(path).resolve()).lower())
    def revoke(self, path): self.permitted.discard(str(Path(path).resolve()).lower())

    def artifact(self, path):
        row = self.db.execute('SELECT kind,detail FROM artifacts WHERE path=?', (str(path),)).fetchone()
        return None if row is None else json.loads(row[1])

    def hash(self, path, kind):
        self.claim(kind); self.permit(path)
        try: return p.sha(path)
        finally: self.revoke(path)

    def publish(self, path, arrays=None, packet=None):
        path = str(path); kind = 'checkpoint' if packet is not None else 'probability'
        schedule = self.lock['checkpoints'] if kind == 'checkpoint' else self.lock['predictions']
        p.require(path in schedule and self.artifact(path) is None and not Path(path).exists(), 'registered immutable new artifact')
        p.require(Path(path).resolve().is_relative_to(self.store), 'D-only scientific output')
        self.claim(kind + '_writes'); Path(path).parent.mkdir(parents=True, exist_ok=True); self.permit(path)
        try:
            with Path(path).open('xb') as stream:
                if packet is not None:
                    import torch
                    torch.save(packet, stream)
                else:
                    import numpy as np
                    np.savez_compressed(stream, **arrays)
                stream.flush(); os.fsync(stream.fileno())
        finally: self.revoke(path)
        detail = dict(path=path, sha256=self.hash(path, kind + '_hashes'), at_unix=time.time(), **schedule[path])
        # Avoid duplicating full label vectors in the small journal.
        detail.pop('expected', None)
        with self.db: self.db.execute('INSERT INTO artifacts VALUES (?,?,?)', (path, kind, json.dumps(detail)))
        return detail

    def reopen(self, path):
        path = str(path); value = self.artifact(path)
        p.require(value is not None, 'registered completed artifact, not file-exists reuse')
        kind = 'checkpoint' if path in self.lock['checkpoints'] else 'probability'
        self.claim(kind + '_reads')
        p.require(self.hash(path, kind + '_hashes') == value['sha256'], 'owned artifact fingerprint')
        self.permit(path)
        try:
            if kind == 'checkpoint':
                import torch
                with Path(path).open('rb') as stream: return torch.load(stream, map_location='cpu', weights_only=False)
            import numpy as np
            with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
        finally: self.revoke(path)

    def finish(self, chain, detail):
        with self.db: self.db.execute('INSERT INTO finished VALUES (?,?)', (chain, json.dumps(detail)))

    def finished(self): return {k: json.loads(v) for k, v in self.db.execute('SELECT chain,detail FROM finished')}


class Guard:
    def __init__(self, rt): self.rt = rt
    def __call__(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        try: path = Path(os.fsdecode(args[0])).resolve()
        except (ValueError, OSError): return
        suffix = path.suffix.lower(); text = str(path).lower()
        if suffix in ('.mat', '.csv', '.wav', '.h5', '.hdf5'):
            raise PermissionError('no new raw or reserved signals')
        if suffix in ('.pt', '.pth', '.npz', '.npy', '.pkl', '.pickle') and text not in self.rt.permitted:
            raise PermissionError('scientific payload outside current explicit read/write grant: ' + str(path))
