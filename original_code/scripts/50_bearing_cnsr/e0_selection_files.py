"""Authorized source-cache adapter. No Torch, raw signal, FFT or inverse fit.

Import is inert. Real NPZ I/O requires the separately approved full runtime.
Tests inject synthetic in-memory I/O; those tests never open historical NPZs.
"""
import copy
from pathlib import Path
import e0_selection_control as c


def _immutable(*args,**kwargs):raise TypeError('locked metadata snapshot is immutable')


class FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable
    def __deepcopy__(self,memo):return self


class FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable
    def __deepcopy__(self,memo):return self


def immutable(value):
    if isinstance(value,dict):return FrozenDict((k,immutable(v)) for k,v in value.items())
    if isinstance(value,list):return FrozenList(immutable(v) for v in value)
    return value


class Catalog:
    """Resolve only lock-bound metadata; immutable snapshots, not array reads."""
    def __init__(self, registry, documents):
        self.registry = immutable(copy.deepcopy(registry));self.documents = immutable(copy.deepcopy(documents))
        self.registry_fingerprint = registry['fingerprint']
        if c.canonical({k:v for k,v in registry.items() if k != 'fingerprint'}) != self.registry_fingerprint:
            raise PermissionError('registry fingerprint')
        self.cases = {v['key']:v for v in self.registry['cases']}
        self.slots = {v['slot']:v for v in self.registry['slots']}
        if len(self.cases) != len(registry['cases']) or len(self.slots) != len(registry['slots']):raise ValueError('unique catalog identities')
        ref = registry['record_reference'];self.parent = self.documents[ref['path']]
        self.records = self.parent
        for field in ref['field']:self.records = self.records[field]
        self.features = {};self.cache_bindings = {}
        for item in registry['caches']:
            b = item['binding'];path = str(Path(b['path']).resolve())
            if path in self.cache_bindings:raise ValueError('unique cache path')
            self.cache_bindings[path] = item
            if item['kind'] == 'full_feature':
                key = b['protocol'],b['domain']
                if key in self.features:raise ValueError('unique full-domain feature binding')
                self.features[key] = b;self._feature_provenance(b)
            elif item['kind'] != 'edited_bank':raise ValueError('only factual features and old edited banks')
        self.cases,self.slots,self.features,self.cache_bindings = (FrozenDict(v) for v in (self.cases,self.slots,self.features,self.cache_bindings))
        self._roots = (self.registry,self.documents,self.cases,self.slots,self.records,self.features,self.cache_bindings)
        self._snapshot = self.snapshot()

    @classmethod
    def from_context(cls, rt):
        rt.assert_authorized();r = rt.registry
        paths = {r['record_reference']['path']} | {v['case_reference']['path'] for v in r['cases']}
        paths |= {v['support_reference']['path'] for v in r['slots']}
        return cls(r,{p:rt.read_locked_json(p) for p in sorted(paths)})

    def snapshot(self):
        return c.canonical(dict(registry=self.registry,documents=self.documents,cases=self.cases,
                                 slots=self.slots,records=self.records,
                                 features=[dict(protocol=p,domain=d,binding=b) for (p,d),b in sorted(self.features.items())],
                                 bindings=self.cache_bindings))

    def check(self):
        current = (self.registry,self.documents,self.cases,self.slots,self.records,self.features,self.cache_bindings)
        if any(left is not right for left,right in zip(current,self._roots)):
            raise PermissionError('immutable metadata root replaced')

    def _feature_provenance(self, b):
        names = {k for k,r in self.records.items() if (r['protocol'],r['domain']) == (b['protocol'],b['domain'])}
        if not names or b['shape'] != [sum(len(self.records[k]['offsets']) for k in names),2048]:raise ValueError('feature metadata shape')
        if b['FFT_code_sha256'] != self.registry['expected_FFT_code_sha256']:raise ValueError('historical FFT generator fingerprint')
        provenance = b['source_input_records']
        if len(provenance) != len(names) or {v['record'] for v in provenance} != names:raise ValueError('complete original raw provenance')
        for item in provenance:
            if {k:v for k,v in item.items() if k != 'record'} != self.parent['raw_bindings'][item['record']]:
                raise ValueError('raw metadata fingerprint drift; do not reopen raw files')

    def case(self, key):
        self.check()
        if key not in self.cases:raise PermissionError('undeclared main case')
        item = self.cases[key];ref = item['case_reference'];value = self.documents[ref['path']]
        for field in ref['field']:value = value[field]
        value = value[ref['index']]
        if c.canonical(value) != item['case_sha256'] or c.canonical(value['roles']) != item['original_roles_sha256']:
            raise PermissionError('exact original case and roles')
        if value['task'] != item['task']:raise PermissionError('original task changed')
        task = value['task']
        if task['target'] in task['sources'] or not task['sources']:raise PermissionError('disjoint source/target domains')
        return item,value

    def support(self, key, slot, side):
        item,case = self.case(key)
        if slot not in self.slots or side not in ('train','probe'):raise PermissionError('known auxiliary role and side')
        role = self.slots[slot]
        if role['profile'] != item['profile'] or role['sources'] != case['task']['sources']:
            raise PermissionError('auxiliary role belongs to current source profile')
        ref = role['support_reference'];profiles = self.documents[ref['path']]['profiles']
        matches = [p for p in profiles if p['profile'] == ref['profile']]
        if len(matches) != 1:raise PermissionError('unique source geometry')
        g = matches[0]['geometry']
        if role['fold_index'] is not None:
            fold = g['folds'][role['fold_index']]
            return {name:case['roles'][name]['tr'].copy() for name in fold[side+'_records']}
        field = 'late' if side == 'train' else 'early'
        return {name:parts[field].copy() for name,parts in g['temporal_roles'].items()}


class CacheFiles:
    def __init__(self, runtime, catalog=None):
        runtime.assert_authorized()
        self.rt = runtime;self.catalog = Catalog.from_context(runtime) if catalog is None else catalog
        if self.catalog.registry_fingerprint != runtime.registry['fingerprint']:raise PermissionError('runtime/catalog identity')
        self.loaded = {};self.domain_rows = {}

    @staticmethod
    def _decode(path, keys):
        import numpy as np
        with np.load(path,allow_pickle=False) as arrays:return {k:arrays[k].copy() for k in keys}

    def _guard(self, key):
        self.rt.assert_authorized()
        if self.rt.ledger.get('stage') not in ('P0','TEACHERS','SOURCE_QUERY','SELECT','MAIN','TARGET'):
            raise PermissionError('no cache action outside active stages')
        return self.catalog.case(key)

    def _load(self, b, kind, *, key, domain, role):
        # Always check current case BEFORE cache hits or any file hash/read.
        item,case = self._guard(key);task = case['task'];stage = self.rt.ledger.get('stage')
        if role == 'source':
            if domain not in task['sources']:raise PermissionError('current target cannot become source via memory cache hit')
        elif role == 'target':
            if kind != 'full_feature' or domain != task['target'] or stage != 'TARGET' or self.rt.ledger.get('MAIN_FREEZE') is None:
                raise PermissionError('exact target only after common main freeze')
        else:raise PermissionError('explicit source or target role')
        path = Path(b['path']).resolve();cache = self.catalog.cache_bindings.get(str(path))
        if not path.is_absolute() or path.drive.upper() != 'D:' or path.suffix != '.npz':raise PermissionError('historical D NPZ only')
        if not cache or cache['kind'] != kind or cache['binding'] != b:raise PermissionError('exact registered cache')
        if kind == 'full_feature':
            if (b['protocol'],b['domain']) != (task['protocol'],domain):raise PermissionError('exact domain binding')
        elif kind == 'edited_bank':
            if role != 'source' or b != item['bank_binding']:raise PermissionError('bank belongs to current source case')
        else:raise PermissionError('known cache kind')
        hit = str(path) in self.loaded
        detail = dict(case=key,protocol=task['protocol'],domain=domain,role=role,kind=kind,path=str(path),
                      sha256=b['sha256'],stage=stage,memory_hit=hit,
                      full_source_domain_decode_includes_validation_rows=(kind == 'full_feature' and role == 'source'))
        self.rt.ledger.io(detail,hash_checks=0 if hit else 2,decodes=0 if hit else 1)
        if not hit:
            if c.sha(path) != b['sha256']:raise PermissionError('cache bytes differ from registered historical SHA')
            keys = ('x','y','g','offset') if kind == 'full_feature' else ('clean','event','joint','warp','y','g','domain')
            arrays = self._decode(path,keys)
            if c.sha(path) != b['sha256']:raise PermissionError('cache bytes changed during decode')
            for value in arrays.values():value.setflags(write=False)
            self.loaded[str(path)] = arrays
            self.rt.ledger.put('CACHE/'+c.canonical(dict(path=str(path))),detail)
        return self.loaded[str(path)]

    def domain(self, key, domain, *, role='source'):
        item,case = self._guard(key)
        b = self.catalog.features[(case['task']['protocol'],domain)]
        arrays = self._load(b,'full_feature',key=key,domain=domain,role=role)
        path = str(Path(b['path']).resolve())
        if path not in self.domain_rows:
            from e0_selection_source_arrays import DomainRows
            self.domain_rows[path] = DomainRows(arrays,self.catalog.records,case['task']['protocol'],domain)
        return self.domain_rows[path]

    def bank(self, key):
        item,case = self._guard(key);records = self.catalog.records
        import numpy as np
        for v in case['bank']:
            name = v['record']
            if name not in case['roles'] or not case['roles'][name]['tr'] or records[name]['domain'] not in case['task']['sources']:
                raise PermissionError('edited bank contains original source training only')
            if v['pair'] is not None and any(k not in case['roles'] or not case['roles'][k]['tr'] for k in case['pairs'][v['pair']]):
                raise PermissionError('inverse partners stay in current original source training')
        if not case['bank']:raise ValueError('nonempty source bank')
        domain = records[case['bank'][0]['record']]['domain']
        arrays = self._load(item['bank_binding'],'edited_bank',key=key,domain=domain,role='source')
        n = item['expected_bank_windows']
        if n != 4 * len(case['bank']):raise ValueError('exact original crop count')
        for name,expected in (('y',[records[v['record']]['label'] for v in case['bank']]),
                              ('g',[v['record'] for v in case['bank']]),
                              ('domain',[records[v['record']]['domain'] for v in case['bank']])):
            if not np.array_equal(arrays[name],np.repeat(expected,4)):raise ValueError('source bank exact row identity')
        for name in ('clean','event','joint','warp'):
            value = arrays[name];shape = (n,2048) if name == 'clean' else (n,2,2048)
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():raise ValueError('finite FP32 two-factor bank')
        return arrays

    def auxiliary(self, key, slot, *, side='train'):
        item,case = self._guard(key);stage = self.rt.ledger.get('stage')
        if side == 'train':
            if stage not in ('P0','TEACHERS'):raise PermissionError('auxiliary training phase')
        elif side == 'probe':
            if stage != 'SOURCE_QUERY' or self.rt.ledger.get('TEACHER_FREEZE') is None:raise PermissionError('all teachers freeze before formal factual probe')
        else:raise PermissionError('known auxiliary side')
        from e0_selection_source_arrays import source_pack
        support = self.catalog.support(key,slot,side)
        features = {d:self.domain(key,d) for d in case['task']['sources']}
        return source_pack(case,self.catalog.records,features,support)

    def query(self, key, slot):
        item,case = self._guard(key)
        if self.rt.ledger.get('stage') != 'SOURCE_QUERY' or self.rt.ledger.get('TEACHER_FREEZE') is None:
            raise PermissionError('common teacher freeze before edited queries')
        groups = [g for g in item['query_groups'] if g['slot'] == slot]
        if len(groups) != 1:raise PermissionError('exact case-to-helper query group')
        from e0_selection_source_arrays import query_pack
        result = query_pack(case,self.catalog.records,self.bank(key),bank_item_indices=groups[0]['bank_item_indices'],
            training_support=self.catalog.support(key,slot,'train'),probe_support=self.catalog.support(key,slot,'probe'),
            mode='temporal' if self.catalog.slots[slot]['fold_index'] is None else 'record')
        if result['row_index'].tolist() != groups[0]['bank_window_rows']:raise ValueError('query row partition drift')
        return result

    def main_source(self, key):
        item,case = self._guard(key)
        if self.rt.ledger.get('stage') not in ('P0','MAIN'):raise PermissionError('main training phase only')
        import numpy as np
        from e0_selection_source_arrays import frozen, allowed_support
        allowed_support(case,self.catalog.records,{k:v['tr'] for k,v in case['roles'].items() if v['tr']})
        features = {d:self.domain(key,d) for d in case['task']['sources']}
        result = {}
        for side in ('tr','va'):
            values = {k:[] for k in ('x','y','g','domain','offset')}
            for name,role in sorted(case['roles'].items()):
                if not role[side]:continue
                r = self.catalog.records[name];f = features[r['domain']]
                if not set(role[side]) <= set(r['offsets']):raise PermissionError('original main role outside full feature rows')
                ids = [f.positions[name][o] for o in role[side]]
                for k in ('x','y','g','offset'):values[k].append(getattr(f,k)[ids])
                values['domain'].append(np.repeat(r['domain'],len(ids)))
            if not values['y']:raise ValueError('nonempty original main train and validation')
            result[side] = {k:frozen(np.concatenate(v)) for k,v in values.items()}
        return result

    def target(self, key):
        item,case = self._guard(key);f = self.domain(key,case['task']['target'],role='target')
        return dict(x=f.x,y=f.y,g=f.g,offset=f.offset)
