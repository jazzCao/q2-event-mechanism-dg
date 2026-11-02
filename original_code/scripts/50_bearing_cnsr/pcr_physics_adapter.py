"""PCR author-core preparation. No Torch import, file-cache I/O or data loading.

Native is the author's 40-revolution angular envelope spectrum. Short is an
explicit physical-bandwidth adaptation, not a reproduction of the paper.
Inputs to these pure routines are supplied by their caller. Real source data
access and neural execution require a separate, finite approved campaign.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert

AUTHOR_ROOT = Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_cf_redesign_20261019/prior_art/PCR-OIPN')
AUTHOR_SHA = {
    'generate_bearing_fault_angular.py': '4b8049f91165be9ca3e659cdafeea4b249ce90e5182cfee3d4c9737e57eab530',
    'train_dg.py': '9827986415652d8c24da80c12e6382287196661ae3f155e7a02ceb1503551f84',
    'model.py': '94891bb98a15e35491e1583d89f414120e2539886bf772f1fee6eb539afd5c88',
    'data_set.py': '03ee52e2fecb1b1b94178460a67e5919fd7c08c9f5ff56c9c66161fd213abeba',
}
SAMPLES_PER_REV = 1024
NATIVE_REVS = 40
NATIVE_WIDTH_BINS = 24
MASK_WIDTH_ORDERS = NATIVE_WIDTH_BINS / NATIVE_REVS
NETWORK_BINS = 2048
RECORD_CAP = 246784  # same continuous observation support as old 60 windows
SHORT_SAMPLES = 5120
SHORT_OFFSETS = tuple(i * 4096 for i in range(60))
LOCAL_TO_AUTHOR = np.array([0, 2, 1], dtype=np.int64)  # H/OF/IF -> H/IF/OF
HARMONICS = {'BPFI': 6, 'BPFO': 3, 'BSF': 3}


@dataclass(frozen=True)
class NominalGeometry:
    """Official PU paper Table 1 example, NOT per-bearing measurements."""
    pitch_mm: float = 28.55
    ball_mm: float = 6.75
    balls: int = 8
    contact_degrees: float = 0.0

    def orders(self):
        if not (np.isfinite(self.pitch_mm) and np.isfinite(self.ball_mm)
                and 0 < self.ball_mm < self.pitch_mm and self.balls > 0
                and np.isfinite(self.contact_degrees)):
            raise ValueError('valid nominal bearing geometry required')
        q = self.ball_mm / self.pitch_mm * np.cos(np.deg2rad(self.contact_degrees))
        return {'normal': 0., 'BPFI': self.balls / 2 * (1 + q),
                'BPFO': self.balls / 2 * (1 - q),
                'BSF': self.pitch_mm / (2 * self.ball_mm) * (1 - q * q)}


def author_source(name):
    if name not in AUTHOR_SHA:
        raise PermissionError('unbound author source')
    data = (AUTHOR_ROOT / name).read_bytes()
    if sha256(data).hexdigest() != AUTHOR_SHA[name]:
        raise PermissionError('author source drift: ' + name)
    return data.decode('utf-8-sig')


def reference_mask_functions():
    """Execute ONLY the three inspected pure NumPy functions, no author imports."""
    wanted = {'create_rectangular_window', 'generate_adaptive_masks', 'apply_advanced_masking'}
    tree = ast.parse(author_source('generate_bearing_fault_angular.py'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('author pure-mask interface changed')
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<bound-author-pure-masks>', 'exec'), namespace)
    return namespace


def _signal(raw, fs, shaft_hz):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('one finite raw waveform, no implicit padding')
    if not (np.isfinite(fs) and np.isfinite(shaft_hz) and fs > 0 and 0 < shaft_hz < fs / 2):
        raise ValueError('positive sampling and shaft frequencies required')
    return x


def native_layout(n_samples, fs, shaft_hz):
    if type(n_samples) is not int or n_samples < 2 or not (fs > 0 and shaft_hz > 0):
        raise ValueError('valid sample support')
    # Preserve author's floating-point order of operations and last-sample rule.
    last_angle = shaft_hz * 360 * ((n_samples - 1) / fs)
    turns = int(last_angle // 360)
    if turns < NATIVE_REVS:
        raise ValueError('native frontend needs at least 40 complete turns')
    return {'whole_turns': turns, 'angular_samples': turns * SAMPLES_PER_REV,
            'windows': turns - NATIVE_REVS + 1,
            'window_samples': NATIVE_REVS * SAMPLES_PER_REV,
            'step_samples': SAMPLES_PER_REV,
            'last_observed_seconds': (n_samples - 1) / fs}


def native_angular_windows(raw, fs, shaft_hz):
    x = _signal(raw, fs, shaft_hz)
    layout = native_layout(len(x), fs, shaft_hz)
    turns = layout['whole_turns']
    target = np.linspace(0, turns * 360, turns * SAMPLES_PER_REV, endpoint=False)
    # Native source computes through RPM, not a direct shaft-Hz approximation.
    rpm = shaft_hz * 60
    end_index = int(((turns * 60) / rpm) * fs)
    original = (rpm / 60) * 360 * (np.arange(end_index) / fs)
    if target[-1] > original[-1]:
        raise ValueError('native interpolation would extrapolate; do not fill')
    # Preserve 2D interpolation too: SciPy uses a slightly different arithmetic
    # path for 1D y, giving sub-ulp differences to the original [channel,time].
    angular = interp1d(original, x[None, :end_index], axis=1,
                       kind='linear', bounds_error=True)(target)[0]
    windows = np.lib.stride_tricks.sliding_window_view(angular, layout['window_samples'])[::SAMPLES_PER_REV]
    if windows.shape != (layout['windows'], layout['window_samples']) or not np.isfinite(windows).all():
        raise AssertionError('native support/window mismatch')
    return windows


def short_angular_window(raw, fs, shaft_hz):
    x = _signal(raw, fs, shaft_hz)
    if len(x) != SHORT_SAMPLES:
        raise ValueError('short adaptation is exactly 5120 original samples')
    n = int(np.floor(((len(x) - 1) / fs * shaft_hz) * SAMPLES_PER_REV)) + 1
    positions = np.arange(n, dtype=np.float64) / SAMPLES_PER_REV
    original = np.arange(len(x), dtype=np.float64) / fs * shaft_hz
    if positions[-1] > original[-1]:
        raise AssertionError('fractional-turn interpolation outside observed support')
    return interp1d(original, x, kind='linear', bounds_error=True)(positions)[None, :]


def envelope_spectrum(angular_windows):
    x = np.asarray(angular_windows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError('finite angular windows required')
    # Native author uses the NON-normalized angular waveform, not minmax output.
    spec = np.abs(np.fft.rfft(np.abs(hilbert(x, axis=1)), axis=1) / (x.shape[1] / 2))
    spec[:, 0] = 0
    axis = np.fft.rfftfreq(x.shape[1]) * SAMPLES_PER_REV
    return spec, axis


def fault_masks(axis, orders=None, *, native):
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
        raise ValueError('finite uniformly-spaced order axis')
    delta = float(axis[1] - axis[0])
    if delta <= 0 or axis[0] != 0 or not np.allclose(np.diff(axis), delta, rtol=1e-10, atol=1e-12):
        raise ValueError('zero-origin uniform positive order spacing')
    if native and not np.isclose(delta, 1 / NATIVE_REVS, rtol=0, atol=1e-12):
        raise ValueError('native width only valid on the 40-turn order grid')
    orders = NominalGeometry().orders() if orders is None else orders
    if list(orders) != ['normal', 'BPFI', 'BPFO', 'BSF'] or not all(np.isfinite(v) for v in orders.values()):
        raise ValueError('explicit author class/order semantics required')
    width = NATIVE_WIDTH_BINS if native else max(1, int(np.rint(MASK_WIDTH_ORDERS / delta)))
    masks, metadata = {}, {}
    for kind, order in orders.items():
        mask = np.zeros_like(axis)
        windows = []
        for h in range(1, HARMONICS.get(kind, 0) + 1):
            center = order * h
            if center < axis[0] or center > axis[-1]:
                continue
            centers = [('main', center)]
            if kind in ('BPFI', 'BSF'):
                centers.extend([('sideband', center - 1), ('sideband', center + 1)])
            for peak_type, value in centers:
                idx = int(np.argmin(np.abs(axis - value)))
                start = max(0, idx - width // 2)
                end = min(len(axis), start + width)
                if start >= end:
                    continue
                mask[start:end] = 1
                windows.append(dict(type=peak_type, harmonic=h, center=value, start=start, end=end, width=width))
        masks[kind], metadata[kind] = mask, windows
    return masks, metadata, dict(width_bins=width, order_spacing=delta,
                                 requested_width_orders=MASK_WIDTH_ORDERS,
                                 realized_width_orders=width * delta)


def local_labels_to_author(labels):
    y = np.asarray(labels)
    if y.ndim != 1 or y.dtype.kind not in 'iu' or (y < 0).any() or (y > 2).any():
        raise ValueError('verified local H=0, OF=1, IF=2 integer labels required')
    return LOCAL_TO_AUTHOR[y].copy()


def counterfactual_source(spec, axis, local_labels, *, role, native, orders=None):
    """Only source-training labels may select a mask; fault CF labels become H."""
    if role != 'source_train':
        raise PermissionError('counterfactual labels are source-train only')
    x = np.asarray(spec, dtype=np.float64)
    y = local_labels_to_author(local_labels)
    if x.ndim != 2 or x.shape != (len(y), len(axis)) or not np.isfinite(x).all():
        raise ValueError('aligned finite spectrum/labels')
    masks, _, info = fault_masks(axis, orders, native=native)
    out = x.copy()
    for i, label in enumerate(y):
        out[i, masks[('normal', 'BPFI', 'BPFO')[label]] > 0] = 0
    fault = np.flatnonzero(y != 0)
    return out[fault], np.zeros(len(fault), dtype=np.int64), fault, info


def network_input(spec, axis, *, native):
    x = np.asarray(spec, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(axis) or not np.isfinite(x).all():
        raise ValueError('finite matching spectra')
    grid = np.arange(NETWORK_BINS, dtype=np.float64) / NATIVE_REVS
    if native:
        if len(axis) < NETWORK_BINS or not np.allclose(axis[:NETWORK_BINS], grid, rtol=0, atol=1e-12):
            raise ValueError('native network takes the first 2048 author bins')
        return x[:, :NETWORK_BINS].copy()
    if len(axis) < 2 or axis[0] != 0 or (np.diff(axis) <= 0).any() or grid[-1] > axis[-1]:
        raise ValueError('short interpolation must stay inside actual FFT support')
    # Mask BEFORE this interpolation. This adds no physical spectral resolution.
    return np.stack([np.interp(grid, axis, row) for row in x])


class _LoggingOnlyStrip(ast.NodeTransformer):
    TARGETS = {'self.qk_matrix', 'self.attn_weights',
               'self.attention_outputs[idx]', 'self.ffn_outputs[idx]'}

    def __init__(self):
        self.removed = []

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and ast.unparse(node.targets[0]) in self.TARGETS
                and ast.unparse(node.value).endswith('.detach().cpu().numpy()')):
            self.removed.append(ast.unparse(node))
            return None
        return self.generic_visit(node)


def author_model_definition_source(*, strip_logging):
    """Return syntax-checked class source, NEVER execute it or construct a NN.

    Deliberately exclude top-level author CUDA detection. Only four unused
    host-copy assignments can be removed; all neural math stays verbatim AST.
    Actual numerical/gradient equivalence is a future approved CUDA P0 test.
    """
    original = ast.parse(author_source('model.py'))
    classes = [n for n in original.body if isinstance(n, ast.ClassDef)]
    if [n.name for n in classes] != ['FeedForward', 'Attention', 'Transformer', 'FeatureExtractor', 'BaseModel']:
        raise ValueError('author model interface drift')
    tree = ast.Module(body=classes, type_ignores=[])
    transform = _LoggingOnlyStrip()
    if strip_logging:
        tree = transform.visit(tree)
        if len(transform.removed) != 4:
            raise ValueError('logging-only surgery must remove exactly four assignments')
    tree = ast.fix_missing_locations(tree)
    text = 'import torch\nimport torch.nn as nn\nfrom einops import rearrange\n\n' + ast.unparse(tree) + '\n'
    compile(text, '<prepared-not-executed-author-model>', 'exec')
    return text, transform.removed
