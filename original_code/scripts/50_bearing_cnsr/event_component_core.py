"""Pure synthetic-testable controls for fixed Q2; no data access or neural imports."""
import numpy as np


def need(value, message):
    if not value: raise ValueError(message)


def impulse_only(factual, original_response, ratio):
    f, u = np.asarray(factual,dtype=np.float64), np.asarray(original_response,dtype=np.float64)
    need(f.ndim == 2 and f.shape == u.shape and f.shape[1] > 0, 'same query/response axes')
    need(np.isfinite(f).all() and np.isfinite(u).all() and np.isfinite(ratio) and ratio > 0, 'finite waveform and positive ratio')
    if ratio == 1.: return f.copy()
    out = f+(ratio-1.)*u
    need(np.isfinite(out).all(), 'finite impulse-only output')
    return out


def global_affine_moments(factual, coupled):
    f, q = np.asarray(factual,dtype=np.float64), np.asarray(coupled,dtype=np.float64)
    need(f.ndim == 2 and f.shape == q.shape and f.shape[1] > 0, 'same factual/coupled query axes')
    need(np.isfinite(f).all() and np.isfinite(q).all(), 'finite global moment inputs')
    mean_f, mean_q = f.mean(axis=1,keepdims=True), q.mean(axis=1,keepdims=True)
    centered_f, centered_q = f-mean_f, q-mean_q
    rms_f = np.sqrt(np.mean(centered_f**2,axis=1,keepdims=True))
    rms_q = np.sqrt(np.mean(centered_q**2,axis=1,keepdims=True))
    need(np.isfinite(rms_f).all() and np.isfinite(rms_q).all(), 'finite source moments')
    need(not np.any((rms_f == 0) & (rms_q > 0)), 'SOURCE_MOMENT_DEGENERATE')
    gain = np.divide(rms_q,rms_f,out=np.ones_like(rms_f),where=rms_f > 0)
    result = mean_q+gain*centered_f
    # Preserve exact factual identity for q=1 or any unchanged query.
    identity = np.all(f == q,axis=1)
    result[identity] = f[identity]
    need(np.isfinite(result).all(), 'finite affine output')
    mean_error = np.abs(result.mean(axis=1,keepdims=True)-mean_q)
    scale_error = np.abs(np.sqrt(np.mean((result-result.mean(axis=1,keepdims=True))**2,axis=1,keepdims=True))-rms_q)
    tol = 2e-12*np.maximum(1.,np.maximum(np.max(np.abs(q),axis=1,keepdims=True),np.max(np.abs(result),axis=1,keepdims=True)))
    need(np.all(mean_error <= tol) and np.all(scale_error <= tol), 'float64 per-query mean/RMS match')
    return result,dict(mean=mean_q[:,0],centered_rms=rms_q[:,0],gain=gain[:,0],
        mean_error=mean_error[:,0],centered_rms_error=scale_error[:,0],identity=identity)


def factorial(factual, timing, amplitude, coupled):
    return dict(timing_given_amplitude=coupled-amplitude,amplitude_given_timing=coupled-timing,
                timing_alone=timing-factual,amplitude_alone=amplitude-factual,
                interaction=coupled-timing-amplitude+factual)
