"""Observation-anchored edit using the frozen analytic renderer.

The inputs are a legal source context and its estimated event coefficients;
this module never chooses recordings or reads a target domain.
"""
import math
import numpy as np
from .renderer import render
from .frontend import spectrum

Q2_RATES = (math.exp(-math.log(1.5)/math.sqrt(3)),
            math.exp(math.log(1.5)/math.sqrt(3)))


def edit_context(observed, events, parameters, scale, ratio, kernel_length):
    """Return x + scale * (ratio*u_ratio - u_1), in observed signal units."""
    x = np.asarray(observed, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('one finite measured context required')
    if np.asarray(events).ndim != 2 or np.asarray(events).shape[1] != len(x):
        raise ValueError('events and observation must cover the same context')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive measured source scale required')
    original = render(events, parameters, 1., kernel_length)
    changed = render(events, parameters, ratio, kernel_length)
    return x + float(scale)*(ratio*changed-original)


def query_features(observed, events, parameters, scale, starts, window, kernel_length):
    """Clean and two Q2 log-FFT views, restricted to fully supported crops.

    Arrays: clean [queries,2048], coupled [queries,2,2048]. Unlike the old
    bank containing four rates, this compact interface contains Q2 rates only.
    """
    x = np.asarray(observed, dtype=np.float64)
    if type(window) is not int or window <= 0 or not starts:
        raise ValueError('positive window and nonempty integer crop list')
    if any(type(s) is not int or s < 0 or s+window > len(x) for s in starts):
        raise ValueError('all crops must be wholly observed; no padding')
    center = (len(x)-1)/2
    for ratio in (1., *Q2_RATES):
        lo = center+ratio*(min(starts)-center-(kernel_length-1))
        hi = center+ratio*(max(starts)+window-1-center)
        if lo < 0 or hi > len(x)-1:
            raise ValueError('query lacks complete event/kernel history at a locked rate')
    crop = lambda v: np.stack([v[s:s+window] for s in starts])
    return {'clean':spectrum(crop(x)), 'coupled':np.stack([
        spectrum(crop(edit_context(x,events,parameters,scale,q,kernel_length)))
        for q in Q2_RATES],axis=1)}
