"""Source payload adapter and exact old physics/frontend bridge.

This module has NO production runner or authorization factory. A future locked
runtime must grant each read and account counters BEFORE supplying this adapter.
Only synthetic fixtures may use it until a new actual source-side grant exists.
No Torch, models, targets, fitting, label-based edit selection or optimizer.
"""
import ast
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import protected_response_core as core

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = frozenset(('K006', 'KA08', 'KA09'))
CONTEXT_START, CONTEXT_STOP = 65536, 98304
CONTEXT_LENGTH = CONTEXT_STOP-CONTEXT_START


def require(value, message):
    if not value:
        raise ValueError(message)


def source_ok(case, record, record_id):
    """Current-case roles, not historical first_role of a shared record."""
    if record_id not in case['roles'] or record['protocol'] != case['task']['protocol'] or record['domain'] not in case['task']['sources'] or record.get('bearing') in PROTECTED:
        raise PermissionError('not current-case unprotected source')
    role, window = case['roles'][record_id], record['window']
    end, begins = None, None
    for start in sorted(role['tr']):
        if end is not None and start <= end:
            end = max(end, start+window)
        else:
            begins, end = start, start+window
        if begins <= CONTEXT_START and end >= CONTEXT_STOP:
            break
    else:
        raise PermissionError('full context is not covered by source TRAIN')
    if any(start < CONTEXT_STOP and start+window > CONTEXT_START for start in role['va']):
        raise PermissionError('context touches source validation')


def measured_context(normalized, mean, scale, *, rows):
    x, mu, sigma = [np.asarray(v, dtype=np.float64) for v in (normalized, mean, scale)]
    require(x.shape == (rows, CONTEXT_LENGTH) and mu.shape == (rows,) and sigma.shape == (rows,), 'exact normalized context and scalar-per-row affine schema')
    require(np.isfinite(x).all() and np.isfinite(mu).all() and np.isfinite(sigma).all() and np.all(sigma > 0), 'finite measurement and positive recorded scales')
    observed = x*sigma[:, None]+mu[:, None]
    require(np.isfinite(observed).all(), 'finite reconstructed physical-unit observation')
    return dict(normalized=x, mean=mu, scale=sigma, observed=observed)


class SourcePayloads:
    """A runtime supplies exact whitelist, permits, counters and byte binding.

    runtime.source_read(binding,kind,case,records) is a context manager. It must
    enforce a NEW real grant and immutable allowlist before any open, and meter
    the hash/decode request. No stub runtime is a production authorization.
    """
    def __init__(self, runtime, records):
        self.runtime, self.records = runtime, records

    @contextmanager
    def checked_stream(self, binding, kind, case, record_ids):
        require(len(set(record_ids)) == len(record_ids) and record_ids, 'explicit unique input record set')
        for key in record_ids:
            source_ok(case, self.records[key], key)
        with self.runtime.source_read(binding, kind, case, list(record_ids)):
            path = Path(binding['path'])
            with path.open('rb') as stream:
                before = os.fstat(stream.fileno())
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1 << 20), b''):
                    h.update(block)
                require(h.hexdigest() == binding['sha256'], 'source bytes differ from locked binding')
                if 'stat' in binding:
                    require(dict(size=before.st_size, mtime_ns=before.st_mtime_ns) == binding['stat'], 'raw source stat differs from locked metadata')
                stream.seek(0)
                yield stream
                after = os.fstat(stream.fileno())
                require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'input changed while decoding same hashed handle')

    def inverse(self, case, binding):
        pair = binding['pair']
        require(len(pair) == 2 and len(binding['raw_sha256']) == 2, 'exact dual-source inverse identity')
        for record, expected in zip(pair, binding['raw_sha256']):
            require(self.runtime.raw_identity(record) == expected, 'inverse raw input SHA differs from current source identity')
        with self.checked_stream(binding, 'inverse', case, pair) as stream:
            with np.load(stream, allow_pickle=False) as archive:
                require({'s', 'params'} <= set(archive.files), 'inverse must store original events and kernels')
                s, params = archive['s'].copy(), archive['params'].copy()
                require(s.dtype.kind == 'f' and params.dtype.kind == 'f' and s.shape == (2, 48, CONTEXT_LENGTH) and params.shape == (2, 48, 3), 'original inverse axes/dtypes')
                require(np.isfinite(s).all() and np.isfinite(params).all() and np.all(s >= 0), 'finite nonnegative original events')
                require(np.all((params[..., 0] > 0) & (params[..., 0] <= .45)) and np.all((params[..., 1] > 0) & (params[..., 1] < 1)), 'valid recorded carrier/decay, not clipped or refitted')
                measurement = None
                if binding['declared_affine_context'] is True:
                    require({'x', 'source_mean', 'source_scale', 'kernel_length'} <= set(archive.files), 'declared self-contained inverse is missing measurement fields')
                    measurement = measured_context(archive['x'], archive['source_mean'], archive['source_scale'], rows=2)
                    size = np.asarray(archive['kernel_length'])
                    require(size.ndim == 0 and int(size) == size and int(size) == math.ceil(.004*self.records[pair[0]]['fs']), 'unchanged exact original kernel length')
        return dict(s=s, params=params, measurement=measurement, pair=pair,
            kernel_length=math.ceil(.004*self.records[pair[0]]['fs']))

    def context(self, case, record, binding):
        require(binding['record'] == record and binding['input_sha256'] == self.runtime.raw_identity(record), 'single-record cache measured identity')
        with self.checked_stream(binding, 'measurement_context', case, [record]) as stream:
            with np.load(stream, allow_pickle=False) as archive:
                require({'x', 'source_mean', 'source_scale'} <= set(archive.files), 'measured context fields')
                x, mu, scale = archive['x'], np.asarray(archive['source_mean']), np.asarray(archive['source_scale'])
                require(mu.ndim == scale.ndim == 0, 'one scalar affine per single-record cache')
                return measured_context(np.asarray(x)[None], mu.reshape(1), scale.reshape(1), rows=1)

    def raw_pu_context(self, case, record, binding):
        require(self.records[record]['protocol'] == 'pu_clock4' and binding['record'] == record and binding['sha256'] == self.runtime.raw_identity(record), 'only exact explicitly missing PU source context')
        # Production runtime must additionally require membership of the fixed
        # 36-record missing-context list BEFORE the stream is opened.
        with self.checked_stream(binding, 'raw_pu_missing_context', case, [record]) as stream:
            from scipy.io import loadmat
            data = loadmat(stream, struct_as_record=False, squeeze_me=True)
            names = [key for key in data if not key.startswith('__')]
            require(len(names) == 1 and hasattr(data[names[0]], 'Y'), 'one PU root with original channels')
            channels = np.atleast_1d(data[names[0]].Y)
            require(len(channels) >= 7, 'PU seventh channel exists')
            channel = channels[6]
            name = channel.Name if isinstance(channel.Name, str) else ''.join(str(v) for v in np.asarray(channel.Name).reshape(-1))
            require(name.strip().lower() == 'vibration_1', 'PU seventh channel is not vibration_1')
            signal = np.asarray(channel.Data, dtype=np.float64).reshape(-1)
            r = self.records[record]
            require(np.isfinite(signal).all() and len(signal) >= CONTEXT_STOP and max(r['offsets'])+r['window'] <= len(signal), 'finite complete original source recording, no padding')
            require(r.get('expected_samples') is None or len(signal) == r['expected_samples'], 'known original recording length')
            x = signal[CONTEXT_START:CONTEXT_STOP].copy()
            mu = float(x.mean()); scale = max(float(np.sqrt(np.mean((x-mu)**2))), 1e-12)
        return measured_context(((x-mu)/scale)[None], np.array([mu]), np.array([scale]), rows=1)

    def legacy_bank(self, case, binding):
        expected_g = np.repeat([v['record'] for v in case['bank']], 4)
        expected_y = np.repeat([self.records[v['record']]['label'] for v in case['bank']], 4)
        expected_d = np.repeat([self.records[v['record']]['domain'] for v in case['bank']], 4)
        with self.checked_stream(binding, 'legacy_source_bank', case, sorted(set(expected_g))) as stream:
            with np.load(stream, allow_pickle=False) as archive:
                require({'clean', 'y', 'g', 'domain'} <= set(archive.files), 'old bank schema')
                result = {k: archive[k].copy() for k in ('clean', 'y', 'g', 'domain')}
        require(result['clean'].dtype == np.float32 and result['clean'].shape == (len(expected_y), 2048) and np.isfinite(result['clean']).all(), 'unchanged finite factual source bank')
        require(all(np.array_equal(result[k], v) for k, v in (('y', expected_y), ('g', expected_g), ('domain', expected_d))), 'old source bank label/record/domain order')
        return result


class CountedFFT:
    def __init__(self, charge):
        self.charge = charge

    def operation(self, name):
        def transform(value, n=None, axis=-1, **kwargs):
            x = np.asarray(value); axis = axis % x.ndim
            count = x.size//x.shape[axis]
            size = n if n is not None else 2*(x.shape[axis]-1) if name == 'irfft' else x.shape[axis]
            self.charge('fft_calls', 1); self.charge('fft_1d_transforms', int(count)); self.charge('fft_points', int(count*size))
            return getattr(np.fft, name)(value, n=n, axis=axis, **kwargs)
        return transform

    def __getattr__(self, name):
        if name not in ('rfft', 'irfft'):
            raise PermissionError('only the locked real FFT operations')
        return self.operation(name)


class CountedNumpy:
    def __init__(self, charge):
        self.fft = CountedFFT(charge)

    def __getattr__(self, name):
        return getattr(np, name)


def extract(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    chosen = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in chosen} == set(names) and all(not n.decorator_list for n in chosen), 'exact undecorated old pure functions')
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), 'exec'), namespace)
    return {key: namespace[key] for key in names}


class PhysicsFrontend:
    """Exact old pure functions; caller binds the three source-code hashes.

    Every FFT is metered through a local NumPy proxy; global np.fft is untouched.
    Production must charge these as REAL transforms, synthetic tests as fixtures.
    """
    def __init__(self, charge):
        proxy = CountedNumpy(charge)
        base = ROOT/'scripts/50_bearing_cnsr'
        self.clock = extract(base/'clock_real_core.py', ('kernel', 'render_phase', 'render_direct'), dict(np=proxy, Fraction=Fraction))
        self.channel = extract(base/'stable_channel_core.py', ('bases', 'coefficients'), dict(np=proxy, MODE_COUNT=6, DURATION=.004, RHO=1/3))
        self.spectrum = extract(base/'clock_real_data.py', ('spectrum',), dict(np=proxy))['spectrum']

    def component(self, inverse, source):
        return self.clock['render_phase'](inverse['s'][source], inverse['params'][source], 1., inverse['kernel_length'])

    def difference_kernel(self, fs, raw_input_sha256, sign):
        require(type(sign) is int and sign in (-1, 1), 'fixed two source-only views')
        require(len(raw_input_sha256) == 64, 'record byte identity for deterministic mode')
        tag = hashlib.sha256(('PROTECTED_RESPONSE_V1|'+raw_input_sha256).encode()).digest()
        mode = int.from_bytes(tag[:4], 'little') % 6
        theta = np.zeros(6); theta[mode] = sign
        _, d = self.channel['coefficients'](theta, self.channel['bases'](fs))
        return d, mode

    def views(self, *, observed, component_normalized, source_mean, source_scale, fs, starts, window, raw_input_sha256, factual_features):
        """Unattenuated primary candidate PLUS four separately matched controls.

        A source-byte-keyed mode, then fixed +/- views; no seed, label, target,
        teacher or performance score determines the response direction.
        """
        require(factual_features.dtype == np.float32 and factual_features.shape == (4, 2048), 'exact unchanged four-row factual bank anchor')
        require(starts == [5792+j*window for j in range(4)], 'unchanged four query crops')
        primary, matched, receipts = [], {k: [] for k in core.ARMS}, []
        for sign in (-1, 1):
            d, mode = self.difference_kernel(fs, raw_input_sha256, sign)
            waves, deltas = core.path_deltas(observed, component_normalized, source_mean=source_mean, source_scale=source_scale,
                difference_kernel=d, starts=starts, window=window, gain_sign=sign)
            base = self.spectrum(waves)
            require(float(np.max(abs(base.astype(float)-factual_features.astype(float)))) <= 1e-5, 'raw recovery/frontend differs from original factual bank')
            value = factual_features+(self.spectrum(waves+deltas['COMPONENT_RESPONSE']).astype(np.float64)-base).astype(np.float32)
            require(np.isfinite(value).all(), 'finite unattenuated primary view')
            primary.append(value)
            comparison, detail = core.feature_match(waves, deltas, encode=self.spectrum, factual_features=factual_features,
                max_iterations=40, relative_tolerance=1e-4)
            for key in core.ARMS:
                matched[key].append(comparison[key])
            receipts.append(dict(sign=sign, mode=mode, primary_alpha=1., primary_input_edit_norm=core.norm(value.astype(float)-factual_features.astype(float)), **detail))
        return dict(primary=np.stack(primary, axis=1), **{k: np.stack(v, axis=1) for k, v in matched.items()}), receipts
