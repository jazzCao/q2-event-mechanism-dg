"""Explicit raw roles and full-source + source-only intervention-bank inputs."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/wfdg/ress_clock_real'
STORE = Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_clock_real')
CONDITIONS = dict(zip('ABCD', ('N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10', 'N09_M07_F10')))
PROTECTED = {'K006', 'KA08', 'KA09'}
ARMS = ('B0_ID_MIX', 'B1_IN_MIX', 'N0_WARP', 'C0_CLOCK', 'E0_EVENTS')
CONTEXT_START = 65536
CONTEXT_LENGTH = 32768


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def spectrum(x):
    magnitude = np.log1p(abs(np.fft.rfft(np.asarray(x, dtype=np.float64), axis=-1)))
    old = np.linspace(0., 1., magnitude.shape[-1]); new = np.linspace(0., 1., 2048)
    return np.stack([np.interp(new, old, row) for row in magnitude]).astype(np.float32)


def tasks():
    return [dict(protocol='pu_clock4', fold='BCDtoA', sources=list('BCD'), target='A', classes=3, window=5120),
            dict(protocol='pu_clock4', fold='ABCtoD', sources=list('ABC'), target='D', classes=3, window=5120),
            dict(protocol='phm09_w3', fold='adj_FH_G', sources=list('FH'), target='G', classes=8, window=5333),
            dict(protocol='jnu_speed', fold='A', sources=list('BC'), target='A', classes=4, window=4000)]


def metadata():
    files = [ROOT/'data/processed/pu_ccdg/manifest.csv', ROOT/'data/processed/phm09_w3/provenance.csv',
             ROOT/'data/processed/jnu_speed/provenance.csv']
    pu, phm, jnu = [table(p) for p in files]; labels = {}
    for v in pu:
        b = v['bearing_id']; lab = int(v['label'])
        if b in labels and labels[b] != lab: raise ValueError('bearing coarse label changed')
        labels[b] = lab
    if len(labels) != 18 or set(labels) & PROTECTED: raise ValueError('historical bearing boundary')
    records = {}; domain_rows = {}; bindings = {str(p): sha(p) for p in files}
    for domain, condition in CONDITIONS.items():
        for bearing, label in sorted(labels.items()):
            for run in range(1, 21):
                name = f'{condition}_{bearing}_{run}.mat'; key = f'pu_clock4/{domain}/{bearing}/{run}'
                path = ROOT/'Paderborn'/bearing/bearing/name
                records[key] = dict(key=key, protocol='pu_clock4', domain=domain, label=label, bearing=bearing, run=run,
                                    fs=64000., shaft_hz=15. if domain == 'D' else 25., window=5120,
                                    path=str(path), stat=stat(path), expected_samples=None,
                                    offsets=[i*4096 for i in range(60)])
    phm_paths = {p.name: p for p in (ROOT/'data/raw/phm09/raw').rglob('*.txt')}
    for protocol, rows in (('phm09_w3', phm), ('jnu_speed', jnu)):
        for v in rows:
            domain = v['domain']; key = f'{protocol}/{domain}/{v["file"]}'
            if protocol == 'phm09_w3' and domain not in 'FHG': continue
            path = phm_paths[v['file']] if protocol == 'phm09_w3' else ROOT/'data/raw/jnu'/v['file']
            if key not in records:
                records[key] = dict(key=key, protocol=protocol, domain=domain, label=int(v['class']), file=v['file'],
                                    fs=float(v['fs']), shaft_hz=float(v['hz']) if protocol == 'phm09_w3' else float(v['rpm'])/60,
                                    window=int(v['window']), path=str(path), stat=stat(path),
                                    expected_samples=int(v['file_n_samples']), offsets=[])
            records[key]['offsets'].append(int(v['offset']))
            domain_rows.setdefault((protocol, domain), []).append(dict(key=key, offset=int(v['offset'])))
    cases = []; split_reads = 0
    for task in tasks():
        for seed in (0, 1):
            roles = {}
            if task['protocol'] == 'pu_clock4':
                train_runs = {}
                for bearing in sorted(labels):
                    init = int(hashlib.sha256(f'CLOCK4_SPLIT|{bearing}|{seed}'.encode()).hexdigest()[:8], 16)
                    train_runs[bearing] = set((np.random.RandomState(init).permutation(20)[:16]+1).tolist())
                for key, record in records.items():
                    if record['protocol'] != 'pu_clock4' or record['domain'] not in task['sources']: continue
                    side = 'tr' if record['run'] in train_runs[record['bearing']] else 'va'
                    roles[key] = dict(tr=record['offsets'] if side == 'tr' else [], va=record['offsets'] if side == 'va' else [])
            else:
                folder = ROOT/'results/wfdg/splits'
                mode = 'grouped_class_guard'
                if task['protocol'] == 'jnu_speed': folder /= 'jnu_speed'; mode = 'temporal_block'
                path = folder/f'{task["protocol"]}_{mode}_{task["fold"]}_seed{seed}.npz'
                bindings[str(path)] = sha(path); split_reads += 1
                with np.load(path, allow_pickle=False) as z:
                    for domain in task['sources']:
                        rows = domain_rows[(task['protocol'], domain)]
                        indices = {side: np.asarray(z[f'{domain}_{side}'], dtype=np.int64) for side in ('tr', 'va')}
                        if np.intersect1d(indices['tr'], indices['va']).size: raise ValueError('window train/val intersection')
                        for side, idx in indices.items():
                            if len(np.unique(idx)) != len(idx) or (len(idx) and (idx.min() < 0 or idx.max() >= len(rows))): raise ValueError('split indices')
                            for i in idx:
                                v = rows[int(i)]; roles.setdefault(v['key'], dict(tr=[], va=[]))[side].append(v['offset'])
            bank = []; pairs = {}; eligible = max(records[k]['shaft_hz'] for k in roles)-min(records[k]['shaft_hz'] for k in roles) > records[next(iter(roles))]['fs']/CONTEXT_LENGTH
            for key, sides in sorted(roles.items()):
                if not sides['tr']: continue
                record = records[key]
                if sides['va']:
                    # Temporal JNU may share a file, never an interval.
                    if max(sides['tr'])+record['window'] > min(sides['va']): raise ValueError('temporal source overlap')
                intervals = []
                for offset in sorted(sides['tr']):
                    if intervals and offset <= intervals[-1][1]: intervals[-1][1] = max(intervals[-1][1], offset+record['window'])
                    else: intervals.append([offset, offset+record['window']])
                if not any(lo <= CONTEXT_START and hi >= CONTEXT_START+CONTEXT_LENGTH for lo, hi in intervals): raise ValueError('context outside contiguous source train interval')
                starts = [5792+i*record['window'] for i in range(4)]
                if starts[-1]+record['window'] > 27800: raise ValueError('query support')
                item = dict(record=key, starts=starts, pair=None, source=None)
                if eligible:
                    candidates = [k for k, side in roles.items() if side['tr'] and records[k]['label'] == record['label']
                                  and abs(records[k]['shaft_hz']-record['shaft_hz']) > record['fs']/CONTEXT_LENGTH]
                    if task['protocol'] == 'pu_clock4':
                        candidates = [k for k in candidates if records[k]['bearing'] == record['bearing']]
                        candidates.sort(key=lambda k: (records[k]['run'] != record['run'], records[k]['domain'] != 'C', records[k]['run'], k))
                    else: candidates.sort()
                    if not candidates: raise ValueError('no same-class legal source clock partner')
                    pair = tuple(sorted((key, candidates[0])))
                    pair_key = hashlib.sha256(('|'.join(pair)+'|65536|32768').encode()).hexdigest()[:24]
                    pairs[pair_key] = list(pair); item.update(pair=pair_key, source=pair.index(key))
                bank.append(item)
            for domain in task['sources']:
                for cls in range(task['classes']):
                    if not any(records[k]['domain'] == domain and records[k]['label'] == cls and v['tr'] for k, v in roles.items()): raise ValueError('source train missing class/domain')
            case = dict(task=task, seed=seed, roles=roles, bank=bank, pairs=pairs, eligible=eligible,
                        validation_windows=sum(len(v['va']) for v in roles.values()),
                        training_windows=sum(len(v['tr']) for v in roles.values()), bank_windows=4*len(bank))
            cases.append(case)
    unique_pairs = {}
    for case in cases:
        for key, pair in case['pairs'].items():
            if key in unique_pairs and unique_pairs[key] != pair: raise ValueError('pair collision')
            unique_pairs[key] = pair
    target_counts = {t['protocol']+'/'+t['fold']: sum(len(v['offsets']) for v in records.values()
                     if v['protocol'] == t['protocol'] and v['domain'] == t['target']) for t in tasks()}
    return dict(records=records, cases=cases, unique_pairs=unique_pairs, inputs=bindings, metadata_split_NPZ_reads=split_reads,
                target_counts=target_counts, discovery_raw_signal_reads=0)


class Runtime:
    def __init__(self, spec):
        self.spec = spec; self.counts = {k: 0 for k in spec['limits']}; self.gpu_start = None
        self.phase = 'SOURCE'; self.allow = set(); self.accesses = []; self.raw_hashes = {}
    def clock(self):
        now = time.time()
        if now >= self.spec['deadline_unix'] or self.gpu_start is not None and now-self.gpu_start >= 7200: raise TimeoutError('original total/GPU clock')
    def gpu(self):
        self.clock()
        if self.gpu_start is None:
            self.gpu_start = time.time(); save(STORE/'GPU_START.json', dict(at_unix=self.gpu_start, deadline=min(self.spec['deadline_unix'], self.gpu_start+7200)))
    def claim(self, key, n=1):
        self.clock()
        if self.counts[key]+n > self.spec['limits'][key]: raise PermissionError('budget '+key)
        self.counts[key] += n
    def permit(self, path): self.allow.add(Path(path).resolve().as_posix().casefold())
    def revoke(self, path): self.allow.discard(Path(path).resolve().as_posix().casefold())
    def install_guard(self):
        prefix = STORE.resolve().as_posix().casefold()+'/'
        def hook(event, args):
            if event != 'open' or not isinstance(args[0], (str, bytes)): return
            p = Path(args[0]).resolve(); name = p.as_posix().casefold()
            if name in self.allow or name.startswith(prefix): return
            if p.suffix.casefold() in ('.mat', '.pt', '.pth', '.npy', '.npz') or '/data/raw/' in name: raise PermissionError('outside lawful input role '+str(p))
        sys.addaudithook(hook)


class RawStore:
    def __init__(self, runtime): self.rt = runtime; self.records = runtime.spec['data']['records']; self.raw = {}; self.features = {}
    def load(self, key, task, role):
        rt = self.rt; r = self.records[key]
        if r['protocol'] != task['protocol'] or role == 'source' and r['domain'] not in task['sources'] or role == 'target' and (rt.phase != 'TARGET' or r['domain'] != task['target']): raise PermissionError('raw per-fold source/target role')
        rt.accesses.append(dict(record=key, fold=task['fold'], role=role, phase=rt.phase, at_unix=time.time()))
        if key not in self.raw:
            rt.claim('raw_record_loads'); path = r['path']; rt.permit(path)
            if stat(path) != r['stat']: raise ValueError('raw stat changed')
            digest = sha(path)
            if r['protocol'] == 'pu_clock4':
                values = loadmat(path, struct_as_record=False, squeeze_me=True); names = [k for k in values if not k.startswith('__')]
                channels = np.atleast_1d(values[names[0]].Y); channel = channels[6]
                name = channel.Name if isinstance(channel.Name, str) else ''.join(str(v) for v in np.asarray(channel.Name).reshape(-1))
                name = name.strip()
                if name.lower() != 'vibration_1': raise ValueError('PU Y(7) is not vibration_1')
                x = np.asarray(channel.Data, dtype=float).reshape(-1)
            elif r['protocol'] == 'jnu_speed': x = np.loadtxt(path, delimiter=',', dtype=float).reshape(-1)
            else:
                values = np.loadtxt(path, dtype=float)
                if values.ndim != 2: raise ValueError('PHM raw channel shape')
                x = values[:, 0]
            rt.revoke(path)
            if not np.isfinite(x).all() or r['expected_samples'] is not None and len(x) != r['expected_samples'] or max(r['offsets'])+r['window'] > len(x): raise ValueError('actual raw length/nonfinite')
            rt.raw_hashes[key] = dict(path=path, sha256=digest, samples=len(x), fs=r['fs'], first_role=role)
            self.raw[key] = x
        return self.raw[key]
    def record_features(self, key, task, role):
        x = self.load(key, task, role); r = self.records[key]
        if key not in self.features:
            self.features[key] = spectrum(np.stack([x[start:start+r['window']] for start in r['offsets']]))
        return self.features[key]
    def source_case(self, case):
        parts = {side: [] for side in ('tr', 'va')}; labels = {side: [] for side in parts}; groups = {side: [] for side in parts}; domains = {side: [] for side in parts}
        for key, sides in sorted(case['roles'].items()):
            r = self.records[key]; feature = self.record_features(key, case['task'], 'source'); positions = {offset: i for i, offset in enumerate(r['offsets'])}
            for side, offsets in sides.items():
                if not offsets: continue
                parts[side].append(feature[[positions[v] for v in offsets]])
                labels[side].extend([r['label']]*len(offsets)); groups[side].extend([key]*len(offsets)); domains[side].extend([r['domain']]*len(offsets))
        return {side: dict(x=np.concatenate(parts[side]), y=np.array(labels[side], np.int64), g=np.array(groups[side]), domain=np.array(domains[side])) for side in parts}
    def target(self, task):
        xs = []; ys = []; gs = []
        for key, r in sorted(self.records.items()):
            if r['protocol'] != task['protocol'] or r['domain'] != task['target']: continue
            feature = self.record_features(key, task, 'target'); xs.append(feature); ys.extend([r['label']]*len(feature)); gs.extend([key]*len(feature))
        return dict(x=np.concatenate(xs), y=np.array(ys, np.int64), g=np.array(gs))
