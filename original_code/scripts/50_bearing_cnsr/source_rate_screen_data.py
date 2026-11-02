"""Fingerprint checked full pools; source permissions are checked on every view."""
from pathlib import Path
from source_rate_plan import require
from source_rate_screen_plan import expected


def check_vectors(arrays, vectors):
    import numpy as np
    for key in ('y', 'g', 'domain', 'offset'):
        require(np.array_equal(arrays[key], np.asarray(vectors[key])), 'exact recording/label/domain/offset: ' + key)


class Files:
    def __init__(self, rt):
        self.rt = rt; self.lock = rt.lock; self.loaded = {}; self.banks = {}; self.accesses = []
        self.bindings = {(b['protocol'], b['domain']): b for b in self.lock['feature_bindings']}

    def domain(self, case, domain, role):
        import numpy as np
        task = case['task']; protocol = task['protocol']; rt = self.rt
        require(role in ('source', 'target') and
            ((role == 'source' and domain in task['sources']) or
             (role == 'target' and rt.phase == 'TARGET' and rt.get('SOURCE_FREEZE') and domain == task['target'])),
            'current task source or jointly frozen target only, even when cached')
        self.accesses.append(dict(case_key=case['key'], domain=domain, role=role, phase=rt.phase))
        key = (protocol, domain)
        if key not in self.loaded:
            binding = self.bindings[key]
            require(rt.hash(binding['path'], 'feature_hashes') == binding['sha256'], 'full source-feature SHA')
            rt.claim('feature_decodes'); rt.permit(binding['path'])
            try:
                with np.load(binding['path'], allow_pickle=False) as z: a = {k: z[k].copy() for k in ('x', 'y', 'g', 'offset')}
            finally: rt.revoke(binding['path'])
            records = self.lock['records']; vectors = dict(y=[], g=[], offset=[], domain=[]); positions = {}
            for name, record in sorted(records.items()):
                if (record['protocol'], record['domain']) != key: continue
                positions[name] = {}
                for offset in record['offsets']:
                    positions[name][offset] = len(vectors['y']); vectors['y'].append(record['label'])
                    vectors['g'].append(name); vectors['offset'].append(offset); vectors['domain'].append(domain)
            a['domain'] = np.asarray(vectors['domain']); check_vectors(a, vectors)
            require(a['x'].shape == (len(vectors['y']), 2048) and a['x'].dtype == np.float32 and
                np.isfinite(a['x']).all() and binding['shape'] == list(a['x'].shape), 'full source features finite shape/dtype')
            require(binding['FFT_code_sha256'] == self.lock['expected_FFT_code_sha256'], 'old FFT lineage')
            require({v['record'] for v in binding['source_input_records']} == set(positions), 'complete provenance records')
            for item in binding['source_input_records']:
                require(Path(item['path']).resolve() == Path(records[item['record']]['path']).resolve(), 'provenance raw identity metadata')
            a['positions'] = positions; self.loaded[key] = a
        return self.loaded[key]

    def source(self, case):
        import numpy as np
        old = self.lock['original_cases'][case['key']]; records = self.lock['records']; result = {}
        for side in ('tr', 'va'):
            vectors = expected(records, old, side); parts = []
            for name, roles in sorted(old['roles'].items()):
                record = records[name]; a = self.domain(case, record['domain'], 'source')
                require(not set(roles['tr']) & set(roles['va']), 'original source train/validation offsets disjoint')
                indices = [a['positions'][name][offset] for offset in roles[side]]
                if indices: parts.append(a['x'][indices])
            values = {k: np.asarray(v) for k, v in vectors.items()}; values['x'] = np.concatenate(parts)
            require(len(values['x']) == len(values['y']), 'full untruncated source coverage')
            result[side] = values
        require(len(result['tr']['y']) == case['source_training_windows'] and
            len(result['va']['y']) == case['source_validation_windows'], 'original full source counts')
        return result

    def bank(self, case):
        import numpy as np
        key = case['key']
        if key not in self.banks:
            binding = self.lock['banks'][key]; rt = self.rt
            require(rt.hash(binding['path'], 'bank_hashes') == binding['sha256'], 'new complete source-bank SHA')
            rt.claim('bank_decodes'); rt.permit(binding['path'])
            try:
                with np.load(binding['path'], allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
            finally: rt.revoke(binding['path'])
            vectors = dict(y=[], g=[], domain=[], offset=[])
            for item in case['bank']:
                require(item['domain'] in case['task']['sources'], 'all bank records current source')
                record = self.lock['records'][item['record']]
                from design_e0_natural_source_probe_v2 import centered_three
                starts = centered_three(32768, case['task']['window'])
                vectors['g'].extend([item['record']] * 3); vectors['y'].extend([record['label']] * 3)
                vectors['domain'].extend([record['domain']] * 3); vectors['offset'].extend([65536 + v for v in starts])
            check_vectors(a, vectors)
            for name in ('clean', 'fixed', 'coupled', 'ring'):
                shape = (case['bank_windows'], 2048) if name == 'clean' else (case['bank_windows'], 2, 2048)
                require(a[name].shape == shape and a[name].dtype == np.float32 and np.isfinite(a[name]).all(), 'finite bank ' + name)
            if not case['source_speed_diversity']:
                require(np.array_equal(a['fixed'], a['coupled']) and np.array_equal(a['fixed'], a['ring']), 'exact same-clock bank fallback')
            self.banks[key] = a
        return self.banks[key]

    def target(self, case):
        a = self.domain(case, case['task']['target'], 'target')
        return {k: a[k] for k in ('x', 'y', 'g', 'domain', 'offset')}
