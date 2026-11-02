"""author_pu_6domain — isolated CCDG Table-V Paderborn data config.

Reads ONLY data/processed/pu_ccdg/{I..N}. Never reads data/processed/pu/.

Author label map (domain_generalization_8_domains.ipynb):
    0 = Healthy, 1 = OF, 2 = IF
Exploratory 50e map was 0=Healthy, 1=IF, 2=OF. This loader does NOT remap.
Callers that still think 1=IF must convert explicitly via swap_if_of_to_exploratory().
"""
from __future__ import annotations

import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PU_CCDG_DIR = os.path.join(ROOT, "data", "processed", "pu_ccdg")
EXPLORATORY_PU_DIR = os.path.join(ROOT, "data", "processed", "pu")
CWRU_AUTHOR_DIR = os.path.join(ROOT, "ccdg数据预处理", "CWRU")

DOMAINS = ("I", "J", "K", "L", "M", "N")
INPUT_LEN = 5120
NUM_CLASSES = 3
HEALTHY_LABEL = 0

# 0=Healthy, 1=OF, 2=IF  (author notebook)
AUTHOR_LABEL_MAP = {0: "Healthy", 1: "OF", 2: "IF"}
# Historical exploratory map — DO NOT apply when loading pu_ccdg
EXPLORATORY_LABEL_MAP = {0: "Healthy", 1: "IF", 2: "OF"}

EXPECTED = {
    "I": {"n": 12340, "counts": {0: 4900, 1: 3720, 2: 3720}, "kind": "artificial"},
    "J": {"n": 12340, "counts": {0: 4900, 1: 3720, 2: 3720}, "kind": "artificial"},
    "K": {"n": 12340, "counts": {0: 4900, 1: 3720, 2: 3720}, "kind": "artificial"},
    "L": {"n": 13640, "counts": {0: 1240, 1: 6200, 2: 6200}, "kind": "real"},
    "M": {"n": 13640, "counts": {0: 1240, 1: 6200, 2: 6200}, "kind": "real"},
    "N": {"n": 13640, "counts": {0: 1240, 1: 6200, 2: 6200}, "kind": "real"},
}

# rpm, torque_Nm, radial_N  — I=L, J=M, K=N
PU_COND = {
    "I": (1500, 0.1, 1000.0), "J": (1500, 0.7, 400.0), "K": (1500, 0.7, 1000.0),
    "L": (1500, 0.1, 1000.0), "M": (1500, 0.7, 400.0), "N": (1500, 0.7, 1000.0),
}


def author_cvec(domain: str) -> np.ndarray:
    """c = [torque/0.7, radial/1000, rpm/1500]. No domain id, no damage flag."""
    if domain not in PU_COND:
        raise KeyError(f"unknown domain {domain}; N09/A-H are not in author_pu_6domain")
    rpm, torque, radial = PU_COND[domain]
    return np.array([torque / 0.7, radial / 1000.0, rpm / 1500.0], dtype=np.float32)


def paired_condition_domains():
    return (("I", "L"), ("J", "M"), ("K", "N"))


def swap_if_of_to_exploratory(y: np.ndarray) -> np.ndarray:
    """Explicit 1<->2 swap: author (1=OF,2=IF) -> exploratory (1=IF,2=OF).

    Never called by load_author_pu_domain. Provided so a caller cannot 'accidentally'
    invert labels without naming the conversion.
    """
    y = np.asarray(y, dtype=np.int64).copy()
    out = y.copy()
    out[y == 1] = 2
    out[y == 2] = 1
    return out


def _forbid_exploratory(path: str) -> None:
    exp = os.path.normpath(EXPLORATORY_PU_DIR)
    if os.path.normpath(path).startswith(exp + os.sep) or os.path.normpath(path) == exp:
        raise RuntimeError(
            f"refusing to load exploratory PU path {path}; "
            f"author_pu_6domain must use {PU_CCDG_DIR}"
        )


def load_author_pu_domain(dom: str, *, remap_to_exploratory: bool = False):
    """Load one I-N tensor. Labels stay 0=H,1=OF,2=IF unless remap is explicit."""
    import torch
    if dom not in DOMAINS:
        raise ValueError(f"domain {dom} not in {DOMAINS}")
    path = os.path.join(PU_CCDG_DIR, dom)
    _forbid_exploratory(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    blob = torch.load(path, weights_only=False)
    x = np.asarray(blob["samples"], dtype=np.float32)
    y = np.asarray(blob["labels"], dtype=np.int64)
    if remap_to_exploratory:
        y = swap_if_of_to_exploratory(y)
    assert_domain_tensor(dom, x, y, remapped=remap_to_exploratory)
    return x, y


def assert_domain_tensor(dom: str, x: np.ndarray, y: np.ndarray, remapped: bool = False) -> None:
    exp = EXPECTED[dom]
    if x.ndim != 2 or x.shape[1] != INPUT_LEN:
        raise AssertionError(f"{dom} shape {x.shape}, expected (N,{INPUT_LEN})")
    if x.dtype != np.float32:
        raise AssertionError(f"{dom} samples dtype {x.dtype}")
    if y.dtype != np.int64:
        raise AssertionError(f"{dom} labels dtype {y.dtype}")
    if x.shape[0] != exp["n"] or y.shape[0] != exp["n"]:
        raise AssertionError(f"{dom} n={x.shape[0]} expected {exp['n']}")
    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    if remapped:
        want = {0: exp["counts"][0], 1: exp["counts"][2], 2: exp["counts"][1]}
    else:
        want = exp["counts"]
    if counts != want:
        raise AssertionError(
            f"{dom} counts {counts} != {want} "
            f"(author map 0=H,1=OF,2=IF; remapped={remapped})"
        )
    if set(np.unique(y).tolist()) - {0, 1, 2}:
        raise AssertionError(f"{dom} unexpected labels {np.unique(y)}")


def load_all_author_pu():
    return {d: load_author_pu_domain(d) for d in DOMAINS}


def source_target_split(cache: dict, test_dom: str, seed: int, val_frac: float = 0.2):
    """80/20 within each SOURCE domain. Target is returned full and unused until test."""
    if test_dom not in cache:
        raise KeyError(test_dom)
    src_doms = [d for d in DOMAINS if d != test_dom]
    assert test_dom not in src_doms
    src_train, src_val = [], []
    for d in src_doms:
        x, y = cache[d]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(x))
        nval = int(len(x) * val_frac)
        va, tr = idx[:nval], idx[nval:]
        src_train.append((x[tr], y[tr], d))
        src_val.append((x[va], y[va], d))
        assert len(set(tr) & set(va)) == 0
    x_t, y_t = cache[test_dom]
    return src_doms, src_train, src_val, (x_t, y_t, test_dom)


def assert_no_target_leak(src_train, src_val, test_dom: str) -> None:
    for split_name, split in (("train", src_train), ("val", src_val)):
        for _x, _y, d in split:
            if d == test_dom:
                raise AssertionError(f"target {test_dom} leaked into {split_name}")


def load_cwru_author_domain(dom: str):
    """Load CCDG author CWRU tensor as-is. No re-windowing, no standardization."""
    import torch
    if dom not in list("ABCDEFGH"):
        raise ValueError(f"CWRU domain {dom} not in A-H")
    path = os.path.join(CWRU_AUTHOR_DIR, f"{dom}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    blob = torch.load(path, weights_only=False)
    if isinstance(blob, dict) and "samples" in blob:
        x = np.asarray(blob["samples"], dtype=np.float32)
        y = np.asarray(blob["labels"], dtype=np.int64)
    else:
        raise TypeError(f"{path} expected dict with samples/labels, got {type(blob)}")
    if x.ndim != 2:
        raise AssertionError(f"{dom} samples ndim {x.ndim}")
    # Author CWRU windows are 4096; refuse to silently reshape.
    if x.shape[1] != 4096:
        raise AssertionError(f"{dom} window {x.shape[1]} != 4096; refusing to recut")
    return x, y
