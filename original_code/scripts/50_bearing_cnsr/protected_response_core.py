"""Array-only response localization and attenuation-only matched controls.

No loaders, model, FFT implementation, RNG, optimizer, labels or selection.
A future authorized caller supplies waveforms and its exact feature encoder.
Matching uses ONE scalar per arm/context/view, shared by all fixed crops.
"""
import math
import numpy as np

ARMS = ('COMPONENT_RESPONSE', 'WHOLE_RESPONSE', 'REMAINDER_RESPONSE', 'COMPONENT_GAIN')
RHO = 1/3


def norm(value):
    value = np.asarray(value, dtype=np.float64)
    if not value.size or not np.isfinite(value).all():
        raise ValueError('finite nonempty array for edit norm')
    scale = float(np.max(abs(value)))
    result = 0. if scale == 0 else scale*float(np.sqrt(np.sum((value/scale)**2)))
    if not math.isfinite(result):
        raise ValueError('finite edit norm')
    return result


def path_deltas(observed, component_normalized, *, source_mean, source_scale, difference_kernel, starts, window, gain_sign):
    """Raw-unit edits using full known source-context history, never zero prefix.

    The difference kernel is D in A=I+D. Its family and selection policy are
    supplied/frozen elsewhere. No event time or inverse coefficient is edited.
    mean is validated for provenance; observed is already in raw units.
    """
    x = np.asarray(observed, dtype=np.float64)
    u = np.asarray(component_normalized, dtype=np.float64)
    d = np.asarray(difference_kernel, dtype=np.float64)
    if x.ndim != 1 or u.shape != x.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError('same finite one-dimensional measured context and estimated component')
    if any(np.ndim(v) != 0 or not np.isfinite(v) for v in (source_mean, source_scale)) or float(source_scale) <= 0:
        raise ValueError('recorded finite scalar affine transform with positive scale')
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all() or d[0] != 0 or abs(float(d.sum())) > 1e-12 or float(abs(d).sum()) > RHO+1e-12:
        raise ValueError('strictly causal DC-zero difference kernel with L1 <=1/3')
    if type(window) is not int or window <= 0 or not starts or any(type(t) is not int for t in starts) or list(starts) != sorted(set(starts)):
        raise ValueError('fixed unique ordered integer crops')
    if starts[0] < len(d)-1 or starts[-1]+window > len(x):
        raise ValueError('complete known causal history and crop required; no padding/clamping')
    if type(gain_sign) is not int or gain_sign not in (-1, 1):
        raise ValueError('explicit fixed gain sign, not learned from labels')
    u_raw = u*float(source_scale)
    if not np.isfinite(u_raw).all():
        raise ValueError('finite physical-unit component')
    crop = lambda v: np.stack([v[t:t+window] for t in starts])
    causal = lambda v: np.convolve(v, d)[:len(v)]
    factual = crop(x)
    delta = dict(COMPONENT_RESPONSE=crop(causal(u_raw)), WHOLE_RESPONSE=crop(causal(x)),
        REMAINDER_RESPONSE=crop(causal(x-u_raw)), COMPONENT_GAIN=gain_sign*RHO*crop(u_raw))
    if any(not np.isfinite(v).all() for v in delta.values()):
        raise ValueError('finite raw-unit edit, no silent overflow')
    return factual, delta


def validate_deltas(deltas):
    if set(deltas) != set(ARMS):
        raise ValueError('exact four matched arms, not a selected subset')
    arrays = {key: np.asarray(deltas[key], dtype=np.float64) for key in ARMS}
    shape = arrays[ARMS[0]].shape
    if len(shape) != 2 or min(shape) == 0 or any(v.shape != shape or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('same finite nonempty crop matrix for every arm')
    return arrays


def waveform_match(deltas):
    """Exact pooled-waveform L2 matching, attenuation only, no dropped rows.

    A zero endpoint in ANY arm makes ALL arms identity for this context/view.
    This avoids impossible gain-on-zero or upward scaling past channel bounds.
    It does not also claim feature-space equality.
    """
    arrays = validate_deltas(deltas)
    endpoint = {key: norm(v) for key, v in arrays.items()}
    target = min(endpoint.values())
    alpha = {key: target/endpoint[key] if target > 0 else 0. for key in ARMS}
    changed = {key: arrays[key]*alpha[key] for key in ARMS}
    return changed, dict(space='pooled_raw_waveform_L2', target_norm=target, alpha=alpha,
        endpoint_norm=endpoint, actual_norm={k: norm(v) for k, v in changed.items()},
        status='ALL_IDENTITY_ZERO_COMMON_ENDPOINT' if target == 0 else 'MATCHED',
        all_original_crops_retained=True, channel_bound_not_increased=True)


def feature_match(factual_waves, deltas, *, encode, factual_features, max_iterations=40, relative_tolerance=1e-4):
    """Bounded matching in ACTUAL FP32 anchored classifier input, not logits.

    Evaluates z_a(alpha)=FP32(z_factual+FP32(F(x+alpha*delta_a)-F(x))).
    No monotonicity is assumed: a sign-bracketed bisection retains endpoints
    surrounding the common target. Quantization is covered by an explicit
    FP32 roundoff floor; failure to meet tolerance raises, never discards rows.
    This mode matches input-feature L2, NOT simultaneous raw-waveform L2.
    """
    arrays = validate_deltas(deltas)
    waves = np.asarray(factual_waves, dtype=np.float64)
    legacy = np.asarray(factual_features)
    if waves.shape != arrays[ARMS[0]].shape or not np.isfinite(waves).all():
        raise ValueError('same finite factual crop matrix')
    if legacy.dtype != np.float32 or legacy.ndim != 2 or legacy.shape[0] != len(waves) or not legacy.size or not np.isfinite(legacy).all():
        raise ValueError('unchanged finite FP32 factual features with same crop rows')
    if type(max_iterations) is not int or max_iterations < 1 or not math.isfinite(relative_tolerance) or not 0 < relative_tolerance <= 1e-3:
        raise ValueError('finite fixed matching iteration/tolerance policy')
    calls = 0

    def checked_encode(value):
        nonlocal calls
        calls += 1
        result = np.asarray(encode(value), dtype=np.float64)
        if result.shape != legacy.shape or not np.isfinite(result).all():
            raise ValueError('finite exact original feature-encoder shape')
        return result

    base = checked_encode(waves)

    def view(key, alpha):
        if alpha == 0:
            return legacy.copy(), 0.
        with np.errstate(over='raise', invalid='raise'):
            value = legacy + (checked_encode(waves+alpha*arrays[key])-base).astype(np.float32)
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('finite actual FP32 classifier input')
        return value, norm(value.astype(np.float64)-legacy.astype(np.float64))

    endpoint = {key: view(key, 1.) for key in ARMS}
    target = min(v[1] for v in endpoint.values())
    floor = 8*np.finfo(np.float32).eps*math.sqrt(legacy.size)*max(1., float(np.max(abs(legacy))))
    tolerance = max(relative_tolerance*target, floor)
    if target <= floor:
        return {k: legacy.copy() for k in ARMS}, dict(space='pooled_actual_FP32_anchored_input_L2',
            status='ALL_IDENTITY_ZERO_OR_ROUNDOFF_COMMON_ENDPOINT', target_norm=0., requested_common_endpoint=target,
            actual_norm={k: 0. for k in ARMS}, alpha={k: 0. for k in ARMS},
            endpoint_norm={k: endpoint[k][1] for k in ARMS}, encoder_calls=calls, matching_iterations={k: 0 for k in ARMS},
            absolute_roundoff_floor=floor, tolerance=tolerance, all_original_crops_retained=True, channel_bound_not_increased=True)
    output, alpha, iterations, actual = {}, {}, {}, {}
    for key in ARMS:
        best, best_alpha, best_norm = endpoint[key][0], 1., endpoint[key][1]
        low, high, used = 0., 1., 0
        while abs(best_norm-target) > tolerance and used < max_iterations:
            mid = (low+high)/2
            candidate, value = view(key, mid)
            used += 1
            if abs(value-target) < abs(best_norm-target):
                best, best_alpha, best_norm = candidate, mid, value
            if value >= target:
                high = mid
            else:
                low = mid
        if abs(best_norm-target) > tolerance:
            raise ValueError('feature matching infeasible within fixed tolerance/iterations; stop, do not drop or tune the context')
        output[key], alpha[key], iterations[key], actual[key] = best, best_alpha, used, best_norm
    return output, dict(space='pooled_actual_FP32_anchored_input_L2', status='MATCHED', target_norm=target,
        alpha=alpha, actual_norm=actual, endpoint_norm={k: endpoint[k][1] for k in ARMS},
        matching_iterations=iterations, encoder_calls=calls, absolute_roundoff_floor=floor, tolerance=tolerance,
        all_original_crops_retained=True, channel_bound_not_increased=True,
        waveform_edit_norm={k: norm(arrays[k]*alpha[k]) for k in ARMS})
