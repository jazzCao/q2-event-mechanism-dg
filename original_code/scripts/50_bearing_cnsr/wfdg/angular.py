"""Order tracking as a computable do(speed=rpm_ref).

Modes:
- linear_zero: linear interp, zero-pad (legacy; can write pad artifacts)
- resample_crop: FFT resample, crop if longer, edge-repeat if shorter
- resample_reflect: FFT resample, crop if longer, reflect-pad if shorter
"""
from __future__ import annotations

import numpy as np

OT_MODES = ("linear_zero", "resample_crop", "resample_reflect")


def order_track_windows(x, rpm, rpm_ref, mode="linear_zero"):
    """Resample windows so apparent speed matches rpm_ref; length stays L."""
    if mode not in OT_MODES:
        raise ValueError(mode)
    x = np.asarray(x, dtype=np.float32)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None]
    if x.ndim != 2:
        raise ValueError(f"expected (N,L) or (L,), got {x.shape}")
    rpm = float(rpm)
    rpm_ref = float(rpm_ref)
    if abs(rpm - rpm_ref) < 1e-3:
        out = np.array(x, copy=True)
        return out[0] if squeeze else out
    if rpm <= 0 or rpm_ref <= 0:
        raise ValueError(f"rpm and rpm_ref must be positive, got {rpm}, {rpm_ref}")
    if mode == "linear_zero":
        out = _linear_zero(x, rpm, rpm_ref)
    else:
        out = _fft_resample(x, rpm, rpm_ref, reflect=(mode == "resample_reflect"))
    return out[0] if squeeze else out


def _linear_zero(x, rpm, rpm_ref):
    _n, L = x.shape
    src = np.arange(L, dtype=np.float64) * (rpm_ref / rpm)
    valid = src < (L - 1e-12)
    src_c = np.clip(src, 0.0, L - 1 - 1e-9)
    i0 = np.floor(src_c).astype(np.int64)
    i1 = np.minimum(i0 + 1, L - 1)
    w = (src_c - i0).astype(np.float32)
    out = x[:, i0] * (1.0 - w) + x[:, i1] * w
    return np.where(valid[None, :], out, 0.0).astype(np.float32, copy=False)


def _fft_resample(x, rpm, rpm_ref, reflect):
    from scipy.signal import resample
    _n, L = x.shape
    n_mid = max(16, int(round(L * rpm / rpm_ref)))
    y = resample(x, n_mid, axis=1).astype(np.float32, copy=False)
    if y.shape[1] == L:
        return y
    if y.shape[1] > L:
        return y[:, :L]
    pad = L - y.shape[1]
    if reflect:
        return np.pad(y, ((0, 0), (0, pad)), mode="reflect")
    return np.pad(y, ((0, 0), (0, pad)), mode="edge")
