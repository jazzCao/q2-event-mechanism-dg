"""Batched original-kernel commutators, fixed1% L1 family, no fitting/I/O."""
from fractions import Fraction
import math
import numpy as np

CANONICAL = tuple(i for i in range(48) if i % 4 in (0, 1))
PAIRS = tuple((i, j) for i in range(24) for j in range(i+1, 24))
SPOTS = ((0, 1), (0, 6), (0, 23), (1, 22), (10, 11), (22, 23))


def require(value, message):
    if not value:
        raise ValueError(message)


def prepare(s, params, size, kernel):
    require(s.ndim == 2 and s.shape[0] == 48 and params.shape == (48, 3), 'original48-kernel axes')
    require(np.isfinite(s).all() and np.all(s >= 0), 'finite original nonnegative coefficients')
    n = s.shape[-1]
    selected = params[list(CANONICAL)].copy()
    h = np.stack([kernel(np.arange(size), p, size) for p in selected])
    negative = np.stack([kernel(np.arange(size), params[i+2], size) for i in CANONICAL])
    require(float(np.max(abs(h+negative))) < 1e-12, 'unchanged phase-negative atoms')
    seed = s.sum(axis=0).copy(); seed[:n//4] = 0.; seed[3*n//4-2*size:] = 0.
    require(float(seed.sum()) > 0, 'fixed existing central activation')
    b = np.stack([np.convolve(atom, seed)[:n] for atom in h])
    cost = np.abs(b).sum(axis=1)
    scales = np.asarray([.01*float(s.sum())/(cost[i]+cost[j]) for i, j in PAIRS])
    return dict(params=selected, B=b, scales=scales, seed=seed,
                original_l1=float(s.sum()), size=size)


def batched(family, ratio, kernel, fft):
    b, params = family['B'], family['params']; n = b.shape[-1]; size = family['size']
    q = Fraction(str(ratio)); a, denominator = q.numerator, q.denominator
    center = (n-1)/2; nfft = 1 << (n+size-1).bit_length()
    spectrum = np.zeros((len(PAIRS), nfft//2+1), complex)
    ii, jj = np.asarray(PAIRS).T
    for residue in range(a):
        ids = np.arange(residue, n, a)
        positions = center+(ids-center)*denominator/a
        floor = np.floor(positions+1e-10).astype(int)
        fractional = float(positions[0]-floor[0])
        require(np.allclose(positions-floor, fractional, atol=1e-9, rtol=0), 'same original rational phase grouping')
        valid = (floor >= 0) & (floor < n)
        grid = np.zeros_like(b); grid[:, floor[valid]] = b[:, ids[valid]]
        h = np.stack([kernel(np.arange(size+1)-fractional, p, size) for p in params])
        sb, sh = fft('rfft', grid, n=nfft), fft('rfft', h, n=nfft)
        spectrum += sh[ii]*sb[jj]-sh[jj]*sb[ii]
    return fft('irfft', spectrum, n=nfft)[:, :n]*family['scales'][:, None]


def old_positive_direction(family, full_params, pair, sign=1):
    i, j = pair; factor = family['scales'][PAIRS.index(pair)]
    d1, d2 = sign*family['B'][j], -sign*family['B'][i]
    values = np.stack((np.maximum(d1, 0), np.maximum(-d1, 0), np.maximum(d2, 0), np.maximum(-d2, 0)))*factor
    ci, cj = CANONICAL[i], CANONICAL[j]
    return values, full_params[[ci, ci+2, cj, cj+2]]


def interpolation_kappa(window, output=2048):
    original = window//2+1
    column = np.zeros(original)
    for k in range(output):
        position = Fraction(k*(original-1), output-1)
        low = position.numerator//position.denominator; high = min(low+1, original-1)
        right = float(position-low)
        column[low] += 1-right; column[high] += right
    return float(column.max())


def frontend_bound(query_l2, window):
    # rFFT norm <=sqrt(W), magnitude/log1p are1-Lipschitz, interpolation
    # ||P||2 <=sqrt(||P||1||P||infty)=sqrt(kappa), row sums exactly1.
    return query_l2*math.sqrt(window*interpolation_kappa(window))


def query_crops(waves, starts, window):
    values = np.asarray(waves)
    require(values.ndim in (1, 2), 'one wave or rows of waves')
    if values.ndim == 1:
        return np.stack([values[t:t+window] for t in starts])
    return np.stack([values[:, t:t+window] for t in starts], axis=1).reshape(-1, window)


def all_metrics(deltas, old, factual_features, old_features, scale, group, spectrum):
    n = len(PAIRS); starts, window = group['starts'], group['window']
    query = query_crops(scale*deltas, starts, window).reshape(n, 3, window)
    old_crops = query_crops(old, starts, window)
    positive = spectrum((old_crops[None]+query).reshape(-1, window)).reshape(n, 3, 2048).astype(float)
    negative = spectrum((old_crops[None]-query).reshape(-1, window)).reshape(n, 3, 2048).astype(float)
    numerator = np.stack((np.linalg.norm(positive-old_features[None], axis=(1, 2)),
                          np.linalg.norm(negative-old_features[None], axis=(1, 2))), axis=1)
    base_norm = float(np.linalg.norm(old_features-factual_features))
    full_norm = np.linalg.norm(scale*deltas, axis=1)
    query_norm = np.linalg.norm(query, axis=(1, 2))
    bound = frontend_bound(query_norm, window)
    require(np.all(numerator <= bound[:, None]+1e-5), 'actual finite-vertex logFFT norms below valid Lipschitz bounds')
    require(base_norm > 0, 'nonidentity original E0 perturbation')
    return dict(time_l2=full_norm, query_l2=query_norm, frontend_l2=numerator,
        frontend_relative=numerator/base_norm, frontend_lipschitz_bound=bound,
        original_E0_frontend_l2=base_norm)
