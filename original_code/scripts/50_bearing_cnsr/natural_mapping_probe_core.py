"""Numerical operations only. No fitting, torch, I/O or target adaptation."""
from fractions import Fraction
import io
import hashlib
import numpy as np
from protected_response_payloads import extract, CountedNumpy
from natural_mapping_probe_plan import ROOT, require


class Physics:
    def __init__(self, charge):
        proxy = CountedNumpy(charge)
        ns = dict(np=proxy, Fraction=Fraction)
        self.functions = extract(ROOT / 'scripts/50_bearing_cnsr/clock_real_core.py',
            ('kernel', 'render_phase', 'render_direct', 'support'), ns)
        self.spectrum = extract(ROOT / 'scripts/50_bearing_cnsr/clock_real_data.py',
                                ('spectrum',), dict(np=proxy))['spectrum']
        self.charge = charge

    def encode(self, waves):
        self.charge('encoded_windows', len(waves))
        return self.spectrum(waves)


def checked_archive(raw, expected):
    require(hashlib.sha256(raw).hexdigest() == expected, 'locked payload SHA mismatch')
    return np.load(io.BytesIO(raw), allow_pickle=False)


def references(archive, record, offsets, label):
    x, g, off, y = [archive[k] for k in ('x', 'g', 'offset', 'y')]
    require(x.ndim == 2 and x.shape[1] == 2048 and x.dtype == np.float32 and
        len(g) == len(off) == len(y) == len(x), 'old full-feature arrays and axes')
    idx = []
    for offset in offsets:
        match = np.flatnonzero((g == record) & (off == offset))
        require(len(match) == 1, 'unique exact natural-reference record/offset')
        idx.append(int(match[0]))
    require(np.all(y[idx] == label) and np.isfinite(x[idx]).all(), 'natural reference labels and finite features')
    return x[idx].copy()


def distance(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1] and len(a) and len(b), 'nonempty equal-dimensional samples')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'finite energy-distance inputs')
    # Direct differences avoid cancellation in squared norm expansion.
    ab = np.sqrt(np.sum((a[:, None] - b[None]) ** 2, axis=-1)).mean()
    aa = np.sqrt(np.sum((a[:, None] - a[None]) ** 2, axis=-1)).mean()
    bb = np.sqrt(np.sum((b[:, None] - b[None]) ** 2, axis=-1)).mean()
    result = float(2 * ab - aa - bb)
    floor = float(128 * np.finfo(np.float64).eps * max(1., 2 * ab + aa + bb))
    require(result >= -floor, 'negative distance beyond declared roundoff')
    return max(result, 0.), floor


def metrics(features, ref, waves):
    values, bounds = {}, []
    for name in ('F', 'C', 'N', 'W'):
        values[name], bound = distance(features[name], ref)
        bounds.append(bound)
    informative = values['F'] > max(bounds)
    return dict(distances=values, roundoff_floor=max(bounds), informative=bool(informative),
        skill_C=(values['F'] - values['C']) / values['F'] if informative else None,
        skill_N=(values['F'] - values['N']) / values['F'] if informative else None,
        skill_W=(values['F'] - values['W']) / values['F'] if informative else None,
        specificity=(values['N'] - values['C']) / values['F'] if informative else None,
        time_l2={k: float(np.linalg.norm(waves[k] - waves['F'])) for k in waves},
        frontend_l2={k: float(np.linalg.norm(features[k].astype(float) - features['F'].astype(float))) for k in features},
        time_mean={k: float(v.mean()) for k, v in waves.items()},
        time_rms={k: float(np.sqrt(np.mean(v * v))) for k, v in waves.items()})


def generate(unit, inv, qualified, physics):
    source, n, size = unit['origin_source_index'], 32768, unit['kernel_length']
    s, params = inv['s'], inv['params']
    require(s.shape == (2, 48, n) and params.shape == (2, 48, 3) and
            np.isfinite(s).all() and np.isfinite(params).all() and np.all(s >= 0), 'old inverse shapes/finite coefficients')
    require(qualified['observed'].shape == qualified['u_normalized'].shape == (2, n) and
            qualified['mean'].shape == qualified['scale'].shape == (2,) and
            np.all(qualified['scale'] > 0), 'qualified two-source physical-unit packet')
    observed = qualified['observed'][source]
    scale = float(qualified['scale'][source])
    require(np.isfinite(observed).all() and np.isfinite(scale), 'finite measured signal and recorded scale')
    q = float(Fraction(unit['correct_ratio'])); starts, window = unit['origin_query_starts'], unit['window']
    for ratio in (q, 1 / q):
        for start in starts:
            physics.functions['support'](n, size, ratio, start, start + window)
    rendered = []
    for ratio in (1., q, 1 / q):
        physics.charge('formal_context_renders', 1)
        rendered.append(physics.functions['render_phase'](s[source], params[source], ratio, size))
    original, correct, reverse = rendered
    error = float(np.max(abs(original - qualified['u_normalized'][source])))
    require(error <= 1e-8 * max(1., float(np.max(abs(original)))), 'u1 agrees with frozen qualified component')
    center = (n - 1) / 2
    waves = {k: [] for k in ('F', 'C', 'N', 'W')}
    for start in starts:
        t = np.arange(start, start + window)
        factual = observed[t]
        waves['F'].append(factual)
        waves['C'].append(factual + scale * (correct[t] - original[t]))
        waves['N'].append(factual + scale * (reverse[t] - original[t]))
        waves['W'].append(np.interp(center + q * (t - center), np.arange(n), observed))
    waves = {k: np.stack(v) for k, v in waves.items()}
    require(all(np.isfinite(v).all() for v in waves.values()), 'finite F/C/N/W without padding')
    features = {k: physics.encode(v) for k, v in waves.items()}
    return waves, features, error
