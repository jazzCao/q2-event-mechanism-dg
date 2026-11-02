"""Exact sparse convolution algebra and bounded old-dictionary null directions.

These operations do not estimate a new inverse or identify a physical mechanism.
"""
from fractions import Fraction as Q
import cmath
import math
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def clean(values):
    return {Q(k): Q(v) for k, v in values.items() if v}


def add(*items):
    out = {}
    for item in items:
        for k, v in item.items():
            out[k] = out.get(k, Q(0))+v
    return clean(out)


def multiply(item, value):
    return clean({k: v*value for k, v in item.items()})


def convolution(left, right):
    out = {}
    for a, x in left.items():
        for b, y in right.items():
            out[a+b] = out.get(a+b, Q(0))+x*y
    return clean(out)


def retime(item, ratio):
    # Center0 is sufficient for identities; actual implementation has fixed center.
    return clean({k/Q(ratio): v for k, v in item.items()})


def shift(item, amount):
    return clean({k+amount: v for k, v in item.items()})


def synthesize(kernels, events):
    return add(*(convolution(h, s) for h, s in zip(kernels, events)))


def magnitude(item, angular_frequency):
    return abs(sum(float(v)*cmath.exp(-1j*angular_frequency*float(t)) for t, v in item.items()))


def algebra(ratio):
    q = Q(ratio)
    h1, h2 = clean({0: 1, 1: Q(1, 2)}), clean({0: Q(1, 3), 2: 1})
    s1, s2 = clean({0: 1, 6: Q(1, 2), 12: 1}), clean({3: Q(2, 3), 9: 1})
    # H'=H R and S'=R^-1 S, R=[[1,1/3],[1/4,1]].
    determinant = Q(11, 12)
    new_h = [add(h1, multiply(h2, Q(1, 4))), add(multiply(h1, Q(1, 3)), h2)]
    new_s = [multiply(add(s1, multiply(s2, -Q(1, 3))), 1/determinant),
             multiply(add(multiply(s1, -Q(1, 4)), s2), 1/determinant)]
    basis_equal = synthesize(new_h, [retime(s, q) for s in new_s]) == synthesize([h1, h2], [retime(s1, q), retime(s2, q)])
    delay = Q(3)
    delayed_h, delayed_s = shift(h1, -delay), shift(s1, delay)
    delay_original = convolution(h1, s1)
    delay_original_alt = convolution(delayed_h, delayed_s)
    delayed_cf = convolution(delayed_h, retime(delayed_s, q))
    plain_cf = convolution(h1, retime(s1, q))
    shift_amount = delay/q-delay
    delay_shift_equal = delayed_cf == shift(plain_cf, shift_amount)
    delay_spectral_error = max(abs(magnitude(delayed_cf, w)-magnitude(plain_cf, w))
                               for w in (.17, .41, .93, 1.37, 2.13))
    d1, d2 = convolution(h2, s1), multiply(convolution(h1, s1), -1)
    factual_null = add(convolution(h1, d1), convolution(h2, d2))
    changed_null = add(convolution(h1, retime(d1, q)), convolution(h2, retime(d2, q)))
    added_l1 = sum(abs(v) for v in d1.values())+sum(abs(v) for v in d2.values())
    alpha = Q(1, 100)*sum(s1.values())/added_l1
    base_cf = convolution(h1, retime(s1, q))
    alternative_cf = add(base_cf, multiply(changed_null, alpha))
    spectral_gap = max(abs(magnitude(alternative_cf, w)-magnitude(base_cf, w))
                       for w in (.17, .41, .93, 1.37, 2.13))
    return dict(ratio=str(q), basis_change_same_counterfactual=basis_equal,
                delay_same_factual=delay_original == delay_original_alt,
                delay_only_global_translation=delay_shift_equal,
                delay_translation=str(shift_amount), delay_magnitude_spectrum_max_error=delay_spectral_error,
                convolution_commutator_same_factual=not factual_null,
                convolution_commutator_changed_outcome=bool(changed_null),
                coefficient_l1_increase_fraction=str(alpha*added_l1/sum(s1.values())),
                full_support_magnitude_gap_at_one_percent=spectral_gap)


def null_family(s, params, size, kernel, fraction=.01):
    require(s.ndim == 2 and s.shape[0] == 48 and params.shape == (48, 3), 'original48-kernel source')
    require(np.all(s >= 0) and np.isfinite(s).all() and np.isfinite(params).all(), 'finite nonnegative old coefficients')
    n = s.shape[1]
    indices = (0, 2, 12, 14)
    h = [kernel(np.arange(size), params[k], size) for k in indices]
    require(max(np.max(abs(h[0]+h[1])), np.max(abs(h[2]+h[3]))) < 1e-12, 'original phase0/pi kernel pairs')
    require(params[0, 0] != params[12, 0], 'two existing distinct carriers, no new kernels')
    seed = s.sum(axis=0).copy()
    seed[:n//4] = 0.; seed[3*n//4-2*size:] = 0.
    require(float(seed.sum()) > 0 and float(s.sum()) > 0, 'nonzero old central source activation; no replacement case')
    d1 = np.convolve(h[2], seed)[:n]
    d2 = -np.convolve(h[0], seed)[:n]
    delta = np.stack((np.maximum(d1, 0), np.maximum(-d1, 0),
                      np.maximum(d2, 0), np.maximum(-d2, 0)))
    before = float(s.sum()); unscaled = float(delta.sum())
    require(unscaled > 0 and 0 < fraction <= .01, 'fixed at most1% source L1 budget')
    delta *= fraction*before/unscaled
    require(np.all(delta >= 0), 'additions preserve original nonnegativity')
    trivial = np.stack((seed.copy(), seed.copy()))
    trivial *= fraction*before/float(trivial.sum())
    return dict(delta=delta, params=params[list(indices)].copy(), indices=indices,
                trivial_delta=trivial, trivial_params=params[[0, 2]].copy(),
                original_coefficient_l1=before, added_coefficient_l1=float(delta.sum()),
                addition_fraction=float(delta.sum()/before),
                seed_interval=[n//4, 3*n//4-2*size])
