"""The experiment log-FFT frontend, not a different normalization."""
import numpy as np

def spectrum(x):
    magnitude = np.log1p(abs(np.fft.rfft(np.asarray(x, dtype=np.float64), axis=-1)))
    old = np.linspace(0., 1., magnitude.shape[-1]); new = np.linspace(0., 1., 2048)
    return np.stack([np.interp(new, old, row) for row in magnitude]).astype(np.float32)
