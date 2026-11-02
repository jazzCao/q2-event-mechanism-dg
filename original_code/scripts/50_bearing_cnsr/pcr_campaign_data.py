"""Role-bound native/short source features. Actual data access needs Runtime.

Importing this file does not load data, create arrays on disk or probe CUDA.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from pcr_campaign_core import STORE, Runtime, canonical, fresh_path, require_store_path, sha
import pcr_physics_adapter as adapter


def build_features(raw, fs, shaft_hz):
    """Pure computation; synthetic tests pass their own invented waveform."""
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or len(x) < adapter.RECORD_CAP or not np.isfinite(x).all():
        raise ValueError('one complete, finite common recording support required')
    x = x[:adapter.RECORD_CAP]
    native = adapter.native_angular_windows(x, fs, shaft_hz)
    spec, axis = adapter.envelope_spectrum(native)
    # All used IF/OF peaks and sidebands are below order31. The native network
    # discards everything beyond51.175; keeping the first2048 before masking
    # is exactly equivalent for these inspected fixed harmonic masks.
    factual_native = adapter.network_input(spec, axis, native=True)
    result = {'NATIVE40': dict(x=factual_native.astype(np.float32),
                               spectrum=factual_native, axis=axis[:2048].copy())}
    angular = np.concatenate([adapter.short_angular_window(x[a:a+5120], fs, shaft_hz)
                              for a in adapter.SHORT_OFFSETS], axis=0)
    spec, axis = adapter.envelope_spectrum(angular)
    result['SHORT_PHYSICAL_WIDTH'] = dict(x=adapter.network_input(spec, axis, native=False).astype(np.float32),
                                          spectrum=spec, axis=axis)
    return result


def source_cf(features, local_label, frontend):
    if local_label not in (1, 2):
        raise ValueError('CF auxiliary pool contains actual source faults only')
    x = features['spectrum']
    cf, y, idx, width = adapter.counterfactual_source(x, features['axis'], np.full(len(x), local_label, dtype=np.int64),
                                                     role='source_train', native=frontend == 'NATIVE40')
    if len(cf) != len(x) or y.any() or not np.array_equal(idx, np.arange(len(x))):
        raise AssertionError('source fault to Healthy conversion')
    return adapter.network_input(cf, features['axis'], native=frontend == 'NATIVE40').astype(np.float32), width


def write_arrays(folder, arrays):
    path = fresh_path(folder, '.npz')
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(path), sha256=sha(path),
                shapes={k: list(v.shape) for k, v in arrays.items()},
                dtypes={k: str(v.dtype) for k, v in arrays.items()})


def load_arrays(binding):
    path = require_store_path(binding['path'])
    if sha(path) != binding['sha256']:
        raise PermissionError('cache contents differ from completed binding')
    with np.load(path, allow_pickle=False) as stream:
        arrays = {k: stream[k].copy() for k in stream.files}
    if set(arrays) != set(binding['shapes']) or set(arrays) != set(binding['dtypes']):
        raise PermissionError('cache schema differs from completed binding')
    for key, a in arrays.items():
        if list(a.shape) != binding['shapes'][key] or str(a.dtype) != binding['dtypes'][key] or not np.isfinite(a).all():
            raise ValueError('cache shape/dtype/nonfinite')
    return arrays


class FeatureStore:
    def __init__(self, runtime):
        if not isinstance(runtime, Runtime):
            raise PermissionError('actual scoped campaign authority is required before real data access')
        self.rt = runtime
        self.records = runtime.proposal['records']
        self.checked_raw = set()
        self.identities = {}
        self.factual = {}
        self.cf = {}
        self.generated = {'factual': 0, 'counterfactual': 0}

    def identity(self, case, key, role):
        r = self.rt.authorize_record(case, key, role)
        old = self.rt.lock['raw_bindings'][key]
        path = Path(r['path'])
        stat = path.stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != r['stat']:
            raise PermissionError('raw stat changed')
        if key not in self.checked_raw:
            with self.rt.permit(path):
                if sha(path) != old['sha256']:
                    raise PermissionError('raw SHA differs from audited historical source')
            self.checked_raw.add(key)
        identity = canonical(dict(record=r, raw_sha256=old['sha256'],
                                  adapter_sha256=self.rt.lock['code_sha256'][str(Path(adapter.__file__).resolve())],
                                  data_code_sha256=self.rt.lock['code_sha256'][str(Path(__file__).resolve())],
                                  geometry=self.rt.proposal['nominal_geometry'], support_samples=246784))
        if key in self.identities and self.identities[key] != identity:
            raise PermissionError('record cache identity drift')
        self.identities[key] = identity
        return r, identity

    def raw(self, case, key, role):
        r, identity = self.identity(case, key, role)
        self.rt.claim('raw_record_loads')
        with self.rt.permit(r['path']):
            values = loadmat(r['path'], struct_as_record=False, squeeze_me=True)
        names = [k for k in values if not k.startswith('__')]
        if len(names) != 1:
            raise ValueError('one PU recording structure expected')
        channels = np.atleast_1d(values[names[0]].Y)
        if len(channels) <= 6:
            raise ValueError('PU vibration channel missing')
        channel = channels[6]
        name = channel.Name if isinstance(channel.Name, str) else ''.join(str(v) for v in np.asarray(channel.Name).reshape(-1))
        if name.strip().lower() != 'vibration_1':
            raise ValueError('PU Y(7) is not vibration_1')
        x = np.asarray(channel.Data, dtype=np.float64).reshape(-1)
        if len(x) < 246784 or not np.isfinite(x).all():
            raise ValueError('raw support too short/nonfinite; no padding')
        self.rt.ledger.put('RAW/' + key, dict(identity=identity, sha256=self.rt.lock['raw_bindings'][key]['sha256'],
                                             samples=len(x), channel='vibration_1'))
        return x[:246784].copy()

    def record(self, case, key, role):
        r, identity = self.identity(case, key, role)
        bindings = {}
        for frontend in ('NATIVE40', 'SHORT_PHYSICAL_WIDTH'):
            item = self.rt.ledger.get('FEATURE/' + frontend + '/' + key)
            if item is not None:
                if item['identity'] != identity or item['record'] != key or item['frontend'] != frontend:
                    raise PermissionError('cache fingerprint is not just file existence')
                bindings[frontend] = item
        if len(bindings) != 2:
            raw = self.raw(case, key, role)
            arrays = build_features(raw, r['fs'], r['shaft_hz'])
            for frontend, data in arrays.items():
                if frontend in bindings:
                    old = load_arrays(bindings[frontend]['array'])
                    if any(not np.array_equal(old[k], v) for k, v in data.items()):
                        raise PermissionError('partial cache recovery must regenerate exact same features')
                    continue
                array = write_arrays(STORE / 'features' / frontend, data)
                item = dict(identity=identity, record=key, frontend=frontend, array=array)
                self.rt.ledger.put('FEATURE/' + frontend + '/' + key, item)
                bindings[frontend] = item
                self.generated['factual'] += 1
        self.factual.update({(key, f): b for f, b in bindings.items()})
        if role == 'source_train' and r['label'] != 0:
            self.rt.authorize_record(case, key, role, counterfactual=True)
            for frontend, factual in bindings.items():
                cfid = canonical(dict(factual_identity=identity, frontend=frontend,
                                      local_fault_label=r['label'], author_output_label=0))
                item = self.rt.ledger.get('CF/' + frontend + '/' + key)
                if item is not None:
                    if item['identity'] != cfid:
                        raise PermissionError('source-CF cache label/fingerprint drift')
                else:
                    cf, width = source_cf(load_arrays(factual['array']), r['label'], frontend)
                    item = dict(identity=cfid, record=key, frontend=frontend, width=width,
                                source_role='source_train', local_source_label=r['label'], author_output_label=0,
                                array=write_arrays(STORE / 'counterfactual' / frontend, dict(x=cf)))
                    self.rt.ledger.put('CF/' + frontend + '/' + key, item)
                    self.generated['counterfactual'] += 1
                self.cf[key, frontend] = item
        return bindings

    def prepare_sources(self):
        if self.rt.phase == 'TARGET':
            return
        for case in self.rt.proposal['cases']:
            for role in ('source_train', 'source_validation'):
                for key in case['roles'][role]:
                    self.record(case, key, role)
            self.rt.ledger.put('GENERATION_PROGRESS', dict(case=[case['task']['fold'], case['seed']], generated=self.generated,
                                                          checked_raw=len(self.checked_raw)))
        receipt = dict(status='SOURCE_FEATURES_GENERATED', fingerprint=self.rt.lock['fingerprint'],
                       raw_checked=len(self.checked_raw), counters=self.rt.ledger.snapshot(),
                       actual_new_inverse_fits=0,
                       factual_bindings={key + '|' + f: b for (key, f), b in self.factual.items()},
                       cf_bindings={key + '|' + f: b for (key, f), b in self.cf.items()})
        self.rt.ledger.put('SOURCE_GENERATION', receipt)

    def case_data(self, case, frontend, *, auxiliary=True):
        if frontend not in ('NATIVE40', 'SHORT_PHYSICAL_WIDTH'):
            raise ValueError('fixed frontend')
        train, cf, validation, bindings = {}, {}, {}, {}
        for role in ('source_train', 'source_validation'):
            for key in case['roles'][role]:
                r = self.rt.authorize_record(case, key, role)
                item = self.rt.ledger.get('FEATURE/' + frontend + '/' + key)
                if item is None:
                    raise PermissionError('all source feature generation precedes training')
                _, identity = self.identity(case, key, role)
                if item['identity'] != identity:
                    raise PermissionError('source factual feature identity')
                data = load_arrays(item['array'])
                bindings[role + '/factual/' + key] = item
                (train if role == 'source_train' else validation)[key] = data['x']
                if auxiliary and role == 'source_train' and r['label'] != 0:
                    self.rt.authorize_record(case, key, role, counterfactual=True)
                    item = self.rt.ledger.get('CF/' + frontend + '/' + key)
                    expected = canonical(dict(factual_identity=identity, frontend=frontend,
                                              local_fault_label=r['label'], author_output_label=0))
                    if item is None or item['identity'] != expected:
                        raise PermissionError('source counterfactual feature identity')
                    cf[key] = load_arrays(item['array'])['x']
                    bindings[role + '/counterfactual/' + key] = item
        if set(train) != set(case['roles']['source_train']) or set(validation) != set(case['roles']['source_validation']):
            raise ValueError('complete source datasets')
        return dict(train=train, cf=cf, validation=validation, source_fingerprint=canonical(bindings), bindings=bindings)

    def target_data(self, case, frontend):
        target = {}
        for key in case['roles']['target']:
            self.rt.authorize_record(case, key, 'target')
            item = self.record(case, key, 'target')[frontend]
            target[key] = load_arrays(item['array'])['x']
        return target
