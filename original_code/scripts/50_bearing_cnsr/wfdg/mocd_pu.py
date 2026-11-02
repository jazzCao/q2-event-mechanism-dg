"""Song et al. MOCD-DG PU protocol: 4 conditions, 13 natural classes, Table 5 A–L."""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RAW_PU = os.path.join(ROOT, "Paderborn")
OUT_DIR = os.path.join(ROOT, "data", "processed", "pu_mocd")

T = 256000
WIN = 2048
STRIDE = 2048
N_RUNS = 20
N_TRAIN = 60
N_TEST = 30
MATLAB_VIB = 7

# Table 1. Domain id is a fold index, not stored as a class in tensors.
CONDITIONS = {
    "0": ("N15_M07_F10", 1500.0, 0.7, 1000.0),
    "1": ("N09_M07_F10", 900.0, 0.7, 1000.0),
    "2": ("N15_M01_F10", 1500.0, 0.1, 1000.0),
    "3": ("N15_M07_F04", 1500.0, 0.7, 400.0),
}

# 13 natural-damage bearings. KI04 omitted (often treated as KI14 duplicate).
CLASSES = [
    "KA04", "KA15", "KA16", "KA22", "KA30",
    "KB23", "KB24", "KB27",
    "KI14", "KI16", "KI17", "KI18", "KI21",
]

# Table 5 PU tasks A–L.
TASKS = {
    "A": (["0", "1"], ["2"]),
    "B": (["0", "1"], ["3"]),
    "C": (["0", "2"], ["1"]),
    "D": (["0", "2"], ["3"]),
    "E": (["0", "3"], ["1"]),
    "F": (["0", "3"], ["2"]),
    "G": (["1", "2"], ["0"]),
    "H": (["1", "2"], ["3"]),
    "I": (["1", "3"], ["0"]),
    "J": (["1", "3"], ["2"]),
    "K": (["2", "3"], ["0"]),
    "L": (["2", "3"], ["1"]),
}

PAPER_MOCD = {
    "A": 98.77, "B": 65.69, "C": 53.54, "D": 80.92, "E": 49.59, "F": 97.54,
    "G": 99.39, "H": 75.64, "I": 90.15, "J": 90.20, "K": 98.15, "L": 54.05,
}
PAPER_ERM = {
    "A": 97.59, "B": 55.49, "C": 33.49, "D": 76.98, "E": 34.05, "F": 94.56,
    "G": 98.00, "H": 68.15, "I": 84.57, "J": 85.90, "K": 97.95, "L": 36.00,
}
PAPER_CCDG = {
    "A": 88.61, "B": 51.67, "C": 30.44, "D": 54.55, "E": 32.11, "F": 93.61,
    "G": 90.61, "H": 54.39, "I": 49.83, "J": 59.61, "K": 91.33, "L": 29.22,
}


def mocd_cvec(dom):
    _tag, rpm, torque, radial = CONDITIONS[str(dom)]
    return np.array([torque / 0.7, radial / 1000.0, rpm / 1500.0], dtype=np.float32)


def mocd_rpm(dom):
    return float(CONDITIONS[str(dom)][1])


def _channel_name(ch):
    name = getattr(ch, "Name", "")
    arr = np.asarray(name).squeeze()
    if arr.ndim == 0:
        return str(arr)
    return "".join(str(x) for x in arr.tolist())


def load_vibration_1(path):
    from scipy.io import loadmat

    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    keys = [k for k in mat if not k.startswith("__")]
    obj = mat[keys[0]]
    y = np.atleast_1d(obj.Y)
    ch = y[MATLAB_VIB - 1]
    name = _channel_name(ch).strip()
    if name.lower() != "vibration_1":
        raise RuntimeError(f"{path}: Y(7) is {name!r}")
    sig = np.asarray(ch.Data, dtype=np.float64).reshape(-1)
    if sig.size < T:
        out = np.zeros(T, dtype=np.float64)
        out[: sig.size] = sig
        return out
    return sig[:T].copy()


def mat_path(bearing, cond, run):
    fn = f"{cond}_{bearing}_{run}.mat"
    return os.path.join(RAW_PU, bearing, bearing, fn)


def window_pool(bearing, cond):
    starts = list(range(0, T - WIN + 1, STRIDE))
    xs, gs = [], []
    for run in range(1, N_RUNS + 1):
        path = mat_path(bearing, cond, run)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        sig = load_vibration_1(path)
        idx = np.asarray(starts)[:, None] + np.arange(WIN)[None, :]
        xs.append(sig[idx].astype(np.float32))
        gs.extend([f"{bearing}_{run}"] * len(starts))
    return np.concatenate(xs, axis=0), np.array(gs)


def build_domain(dom):
    cond = CONDITIONS[dom][0]
    chunks_x, chunks_y, chunks_g = [], [], []
    for lab, bearing in enumerate(CLASSES):
        x, g = window_pool(bearing, cond)
        chunks_x.append(x)
        chunks_y.append(np.full(len(x), lab, dtype=np.int64))
        chunks_g.append(g)
    return (
        np.concatenate(chunks_x),
        np.concatenate(chunks_y),
        np.concatenate(chunks_g),
    )


def build_all(force=False):
    import torch

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "classes": CLASSES, "conditions": {k: v[0] for k, v in CONDITIONS.items()},
        "win": WIN, "n_train": N_TRAIN, "n_test": N_TEST, "tasks": TASKS,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    for dom in CONDITIONS:
        path = os.path.join(OUT_DIR, f"{dom}.pt")
        if os.path.isfile(path) and not force:
            print(f"[mocd] skip {path}", flush=True)
            continue
        print(f"[mocd] build domain {dom}", flush=True)
        x, y, g = build_domain(dom)
        torch.save({"samples": x, "labels": y, "groups": g}, path)
        print(f"[mocd] {dom} {x.shape} -> {path}", flush=True)


def mocd_60_30_indices(y, seed):
    """Per class: 60 train, 30 test, no overlap. Remainder unused."""
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        if len(idx) < N_TRAIN + N_TEST:
            raise RuntimeError(f"class {c} has {len(idx)} windows")
        pick = rng.choice(idx, N_TRAIN + N_TEST, replace=False)
        tr.append(pick[:N_TRAIN])
        te.append(pick[N_TRAIN:])
    return np.concatenate(tr), np.concatenate(te)


def source_tr_va(tr_idx, seed, val_frac=0.2):
    rng = np.random.RandomState(seed + 17)
    tr_idx = np.asarray(tr_idx)
    perm = rng.permutation(len(tr_idx))
    n_val = max(1, int(round(len(tr_idx) * val_frac)))
    n_val = min(n_val, len(tr_idx) - 1)
    va = tr_idx[perm[:n_val]]
    tr = tr_idx[perm[n_val:]]
    return tr, va


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build_all(force=args.force)
