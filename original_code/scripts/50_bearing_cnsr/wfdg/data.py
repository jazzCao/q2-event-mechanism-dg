"""Load pu_ccdg / exploratory PU / CWRU. Grouped source split. No explicit-c inputs."""
from __future__ import annotations

import os
import json
from itertools import combinations

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PU_CCDG = os.path.join(ROOT, "data", "processed", "pu_ccdg")
PU_OLD = os.path.join(ROOT, "data", "processed", "pu")
CWRU_AUTHOR = os.path.join(ROOT, "ccdg数据预处理", "CWRU")
CWRU_RAW = os.path.join(ROOT, "data", "raw", "ccdg_cwru")
CWRU_OURS = os.path.join(ROOT, "data", "processed", "cwru")
JNU_SPEED = os.path.join(ROOT, "data", "processed", "jnu_speed")
HUST_SPEED = os.path.join(ROOT, "data", "processed", "hust_speed")
PHM09_SPEED = os.path.join(ROOT, "data", "processed", "phm09_speed")

PROTOCOLS = {
    "pu_ccdg": {
        "dataset": "pu", "protocol": "pu_ccdg", "dir": PU_CCDG,
        "domains": list("IJKLMN"), "input_len": 5120, "num_classes": 3,
        "fs": 64000.0, "kind": "pu_ccdg",
        "labels": {0: "Healthy", 1: "OF", 2: "IF"},
    },
    "pu_old": {
        "dataset": "pu", "protocol": "pu_old", "dir": PU_OLD,
        "domains": list("IJKLMN"), "input_len": 5120, "num_classes": 3,
        "fs": 64000.0, "kind": "pu_old",
        "labels": {0: "Healthy", 1: "IF", 2: "OF"},
    },
    "cwru_author": {
        "dataset": "cwru", "protocol": "cwru_author",
        "dir": CWRU_AUTHOR if os.path.isdir(CWRU_AUTHOR) else CWRU_RAW,
        "domains": list("ABCDEFGH"), "input_len": 4096, "num_classes": 10,
        "fs": 12000.0, "kind": "cwru_pt",
        "labels": {i: str(i) for i in range(10)},
        "note": "CCDG author 4000x4096, no z-score; healthy DE/FE may duplicate",
    },
    "cwru_ours": {
        "dataset": "cwru", "protocol": "cwru_ours",
        "dir": CWRU_OURS,
        "domains": list("ABCDEFGH"), "input_len": 4096, "num_classes": 10,
        "fs": 12000.0, "kind": "cwru_dir",
        "labels": {i: str(i) for i in range(10)},
        "note": "50b local preprocess, window 4096 stride 290, per-window z-score",
    },
    "pu_mocd": {
        "dataset": "pu", "protocol": "pu_mocd",
        "dir": os.path.join(ROOT, "data", "processed", "pu_mocd"),
        "domains": list("0123"), "input_len": 2048, "num_classes": 13,
        "fs": 64000.0, "kind": "pu_mocd_pt",
        "labels": {i: n for i, n in enumerate(
            ["KA04", "KA15", "KA16", "KA22", "KA30", "KB23", "KB24", "KB27",
             "KI14", "KI16", "KI17", "KI18", "KI21"])},
        "note": "Song et al. MOCD-DG PU: 4 conditions, 13 natural classes, 2048",
        "fold_tasks": {
            "A": (["0", "1"], ["2"]), "B": (["0", "1"], ["3"]),
            "C": (["0", "2"], ["1"]), "D": (["0", "2"], ["3"]),
            "E": (["0", "3"], ["1"]), "F": (["0", "3"], ["2"]),
            "G": (["1", "2"], ["0"]), "H": (["1", "2"], ["3"]),
            "I": (["1", "3"], ["0"]), "J": (["1", "3"], ["2"]),
            "K": (["2", "3"], ["0"]), "L": (["2", "3"], ["1"]),
        },
    },
    "jnu_speed": {
        "dataset": "jnu", "protocol": "jnu_speed", "dir": JNU_SPEED,
        "domains": list("ABC"), "input_len": 4000, "num_classes": 4,
        "fs": 50000.0, "kind": "jnu_pt",
        "labels": {0: "Healthy", 1: "IF", 2: "OF", 3: "BF"},
        "domain_rpm": {"A": 600.0, "B": 800.0, "C": 1000.0},
        "split_mode": "temporal_block",
        "selection_label": "temporal_block_source_val",
        "uses_physical_c": False,
        "cond_dim": 0,
        "win": 4000, "stride": 2000,
        "note": (
            "JNU feasibility leave-one-speed-out. One file per class×speed. "
            "Temporal-block source-val is NOT recording-grouped DG. "
            "No physical c; no PU cvec fallback."
        ),
        "fold_tasks": {
            "A": (["B", "C"], ["A"]),
            "B": (["A", "C"], ["B"]),
            "C": (["A", "B"], ["C"]),
            "E8": (["B", "C"], ["A"]),
            "E9": (["A", "B"], ["C"]),
            "I3": (["A", "C"], ["B"]),
        },
    },
    "hust_speed": {
        "dataset": "hust", "protocol": "hust_speed", "dir": HUST_SPEED,
        "domains": list("LMH"), "input_len": 2048, "num_classes": 9,
        "fs": 25600.0, "kind": "hust_pt",
        "labels": {
            0: "H", 1: "0.5X_I", 2: "I", 3: "0.5X_O", 4: "O",
            5: "0.5X_B", 6: "B", 7: "0.5X_C", 8: "C",
        },
        "domain_rpm": {"L": 20.0, "M": 40.0, "H": 80.0},  # Hz stored under rpm key
        "domain_hz": {"L": 20.0, "M": 40.0, "H": 80.0},
        "split_mode": "temporal_block",
        "selection_label": "temporal_block_source_val",
        "uses_physical_c": False,
        "cond_dim": 0,
        "win": 2048, "stride": 1024,
        "note": (
            "HUST feasibility leave-one-speed-out (20/40/80 Hz). "
            "One Excel per class×speed. Temporal-block source-val is NOT "
            "recording-grouped DG. No physical c; no PU cvec fallback."
        ),
        "fold_tasks": {
            "L": (["M", "H"], ["L"]),
            "M": (["L", "H"], ["M"]),
            "H": (["L", "M"], ["H"]),
        },
    },
    # W3 five-domain HUST (20/25/40/70/80 Hz). Does not overwrite hust_speed.
    "hust_w3": {
        "dataset": "hust", "protocol": "hust_w3",
        "dir": os.path.join(ROOT, "data", "processed", "hust_w3"),
        "domains": list("ABCDE"), "input_len": 2048, "num_classes": 9,
        "fs": 25600.0, "kind": "hust_pt",
        "labels": {
            0: "H", 1: "0.5X_I", 2: "I", 3: "0.5X_O", 4: "O",
            5: "0.5X_B", 6: "B", 7: "0.5X_C", 8: "C",
        },
        "domain_rpm": {"A": 20.0, "B": 25.0, "C": 40.0, "D": 70.0, "E": 80.0},
        "domain_hz": {"A": 20.0, "B": 25.0, "C": 40.0, "D": 70.0, "E": 80.0},
        "split_mode": "temporal_block",
        "selection_label": "temporal_block_source_val",
        "uses_physical_c": False,
        "cond_dim": 0,
        "win": 2048, "stride": 1024,
        "note": (
            "W3 HUST five-domain leave-two-source (20/25/40/70/80 Hz). "
            "Same window lock as hust_speed. Temporal-block source-val."
        ),
        "fold_tasks": {
            # Prereg E*/I* (Hz): A=20 B=25 C=40 D=70 E=80
            "E1": (["C", "D"], ["E"]),   # 40,70 → 80
            "E2": (["B", "C"], ["E"]),   # 25,40 → 80
            "E3": (["A", "B"], ["E"]),   # 20,25 → 80
            "E4": (["D", "E"], ["A"]),   # 70,80 → 20
            "E5": (["C", "E"], ["A"]),   # 40,80 → 20 (= hust_speed L)
            "E6": (["D", "E"], ["B"]),   # 70,80 → 25
            "E7": (["A", "C"], ["E"]),   # 20,40 → 80 (= hust_speed H)
            "I1": (["A", "E"], ["C"]),   # 20,80 → 40 (= hust_speed M)
            "I2": (["B", "E"], ["D"]),   # 25,80 → 70
        },
    },
    "phm09_speed": {
        "dataset": "phm09", "protocol": "phm09_speed", "dir": PHM09_SPEED,
        "domains": list("ABC"), "input_len": 5333, "num_classes": 8,
        "fs": 66666.67, "kind": "phm09_pt",
        "labels": {
            0: "spur1", 1: "spur2", 2: "spur3", 3: "spur4",
            4: "spur5", 5: "spur6", 6: "spur7", 7: "spur8",
        },
        "domain_rpm": {"A": 30.0, "B": 40.0, "C": 50.0},
        "domain_hz": {"A": 30.0, "B": 40.0, "C": 50.0},
        "split_mode": "grouped",
        "selection_label": "grouped_source_val",
        "uses_physical_c": False,
        "cond_dim": 0,
        "win": 5333, "stride": 2666,
        "note": (
            "PHM09 gearbox leave-one-speed-out (30/40/50 Hz), spur 1–8. "
            "Recording-grouped source-val (2 reps × High/Low). "
            "No physical c; no PU cvec fallback. Main-table eligible (W2 §5)."
        ),
        "fold_tasks": {
            "A": (["B", "C"], ["A"]),
            "B": (["A", "C"], ["B"]),
            "C": (["A", "B"], ["C"]),
        },
    },
    "phm09_w3": {
        "dataset": "phm09", "protocol": "phm09_w3",
        "dir": os.path.join(ROOT, "data", "processed", "phm09_w3"),
        "domains": list("ABCDEFGHIJ"), "input_len": 5333, "num_classes": 8,
        "fs": 66666.67, "kind": "phm09_pt",
        "labels": {
            0: "spur1", 1: "spur2", 2: "spur3", 3: "spur4",
            4: "spur5", 5: "spur6", 6: "spur7", 7: "spur8",
        },
        # rpm key stores Hz (same convention as phm09_speed)
        "domain_rpm": {
            "A": 30.0, "B": 30.0, "C": 35.0, "D": 35.0, "E": 40.0,
            "F": 40.0, "G": 45.0, "H": 45.0, "I": 50.0, "J": 50.0,
        },
        "domain_hz": {
            "A": 30.0, "B": 30.0, "C": 35.0, "D": 35.0, "E": 40.0,
            "F": 40.0, "G": 45.0, "H": 45.0, "I": 50.0, "J": 50.0,
        },
        "domain_load": {
            "A": "High", "B": "Low", "C": "High", "D": "Low", "E": "High",
            "F": "Low", "G": "High", "H": "Low", "I": "High", "J": "Low",
        },
        "split_mode": "grouped_class_guard",
        "selection_label": "grouped_source_val",
        "uses_physical_c": False,
        "cond_dim": 0,
        "win": 5333, "stride": 2666,
        "note": (
            "W3 Phase F / Phase 2: PHM09 10 OC = 5 speeds x 2 loads, spur 1-8. "
            "Recording-grouped with per-class train guard (2 reps/OC). "
            "Does not replace phm09_speed."
        ),
        "fold_tasks": {},  # filled below after PROTOCOLS literal
    },
}


def _phm09_w3_fold_tasks():
    """20 folds: per target adj (nearest two) + ext (farthest two on OC axis).
    Plus W6 far_* double-shift folds (locked in W6_PHM09_FAR_FOLDS.md)."""
    D = list("ABCDEFGHIJ")
    n = len(D)
    tasks = {}
    for i, t in enumerate(D):
        if i == 0:
            src_a = [D[1], D[2]]
        elif i == n - 1:
            src_a = [D[n - 3], D[n - 2]]
        else:
            src_a = [D[i - 1], D[i + 1]]
        # extreme: farthest end of the OC axis
        if i < n / 2:
            src_b = [D[n - 2], D[n - 1]]
        else:
            src_b = [D[0], D[1]]
        src_b = [d for d in src_b if d != t]
        if len(src_b) < 2:
            # should not happen for n=10
            cand = [d for d in D if d != t]
            src_b = [cand[0], cand[-1]]
        src_a = sorted(src_a, key=D.index)
        src_b = sorted(src_b[:2], key=D.index)
        tasks[f"adj_{''.join(src_a)}_{t}"] = (src_a, [t])
        tasks[f"ext_{''.join(src_b)}_{t}"] = (src_b, [t])
    # W6 double-shift (speed+load): opposite-load, max |Hz| distance (tie: letter)
    hz = {
        "A": 30.0, "B": 30.0, "C": 35.0, "D": 35.0, "E": 40.0,
        "F": 40.0, "G": 45.0, "H": 45.0, "I": 50.0, "J": 50.0,
    }
    load = {
        "A": "High", "B": "Low", "C": "High", "D": "Low", "E": "High",
        "F": "Low", "G": "High", "H": "Low", "I": "High", "J": "Low",
    }
    for t in D:
        s_t, l_t = hz[t], load[t]
        opp = [d for d in D if d != t and load[d] != l_t]
        same = [d for d in D if d != t and load[d] == l_t]

        def _rank(cands, _t=t):
            return sorted(cands, key=lambda d: (-abs(hz[d] - hz[_t]), d))

        src = _rank(opp)[:2]
        if len(src) < 2:
            for d in _rank(same):
                if d not in src:
                    src.append(d)
                if len(src) == 2:
                    break
        src = sorted(src, key=D.index)
        tasks[f"far_{''.join(src)}_{t}"] = (src, [t])
    return tasks


PROTOCOLS["phm09_w3"]["fold_tasks"] = _phm09_w3_fold_tasks()


def _load_xy(path):
    import torch
    blob = torch.load(path, weights_only=False)
    x = np.asarray(blob["samples"], dtype=np.float32)
    y = np.asarray(blob["labels"], dtype=np.int64)
    return x, y


def _forbid_c_keys(blob_keys):
    banned = {"c", "cvec", "torque", "rpm", "damage_type", "domain_id"}
    hit = banned & set(blob_keys)
    if hit:
        raise RuntimeError(f"refusing metadata keys in tensor: {hit}")


def load_domain(spec, dom):
    import torch
    if spec["kind"] in ("cwru_pt", "pu_mocd_pt", "jnu_pt", "hust_pt", "phm09_pt"):
        path = os.path.join(spec["dir"], f"{dom}.pt")
    else:
        path = os.path.join(spec["dir"], dom)
    blob = torch.load(path, weights_only=False)
    if isinstance(blob, dict):
        _forbid_c_keys(blob.keys())
    x, y = np.asarray(blob["samples"], dtype=np.float32), np.asarray(blob["labels"], dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != spec["input_len"]:
        raise AssertionError(f"{dom} shape {x.shape}")
    if isinstance(blob, dict) and "groups" in blob:
        groups = np.asarray(blob["groups"])
    else:
        groups = _groups_for(spec, dom, x, y)
    return x, y, groups


def load_domain_offsets(spec, dom):
    """Window start offsets + per-window file length (temporal_block protocols)."""
    import torch
    if spec["kind"] not in ("jnu_pt", "hust_pt"):
        raise ValueError(f"offsets only for jnu_pt/hust_pt, got {spec['kind']}")
    path = os.path.join(spec["dir"], f"{dom}.pt")
    blob = torch.load(path, weights_only=False)
    if "offsets" not in blob or "file_n_samples" not in blob:
        raise RuntimeError(f"{dom} missing offsets/file_n_samples")
    return (np.asarray(blob["offsets"], dtype=np.int64),
            np.asarray(blob["file_n_samples"], dtype=np.int64))


def temporal_block_indices(groups, offsets, file_n, win, train_frac=0.7, val_frac=0.2):
    """Seed-independent in-file temporal block. Not recording-grouped."""
    groups = np.asarray(groups)
    offsets = np.asarray(offsets, dtype=np.int64)
    file_n = np.asarray(file_n, dtype=np.int64)
    win = int(win)
    tr, va = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        L = int(file_n[idx[0]])
        train_end = int(np.floor(train_frac * L))
        val_start = int(np.floor((1.0 - val_frac) * L))
        if val_start < train_end + win:
            raise RuntimeError(
                f"temporal gap too small in {g}: train_end={train_end} "
                f"val_start={val_start} win={win} L={L}")
        for i in idx:
            o = int(offsets[i])
            if o < 0 or o + win > L:
                raise RuntimeError(f"window OOB in {g}: offset={o} L={L}")
            if o + win <= train_end:
                tr.append(int(i))
            elif o >= val_start:
                va.append(int(i))
    tr = np.asarray(tr, dtype=np.int64)
    va = np.asarray(va, dtype=np.int64)
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError("temporal_block emptied a side")
    return tr, va, {
        "mode": "temporal_block",
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "win": win,
        "train_windows": int(len(tr)),
        "val_windows": int(len(va)),
        "note": "same-session temporal block; NOT recording-grouped",
    }


def _groups_for(spec, dom, x, y):
    n = len(x)
    if spec["kind"] == "pu_ccdg":
        man_path = os.path.join(spec["dir"], "manifest.csv")
        man = pd.read_csv(man_path)
        sub = man[man["domain"].astype(str) == str(dom)].reset_index(drop=True)
        if len(sub) != n:
            raise AssertionError(f"{dom} manifest {len(sub)} != N {n}")
        labs = sub["label"].to_numpy(np.int64)
        if not np.array_equal(labs, y):
            raise AssertionError(f"{dom} manifest label order != tensor labels")
        g = np.array([f"{b}_{int(r)}" for b, r in zip(sub["bearing_id"], sub["run_id"])])
        return g
    # No manifest: block adjacent windows from the same recording as far as possible.
    block = 62 if spec["dataset"] == "pu" else 20
    return np.array([f"{dom}_{i // block}" for i in range(n)])


def grouped_indices(groups, seed, val_frac=0.2):
    """Split by group = bearing_id_run_id. Same group never on both sides."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    n_val = min(n_val, len(uniq) - 1)
    val_set = set(uniq[:n_val].tolist())
    va = np.array([g in val_set for g in groups])
    tr = ~va
    if tr.sum() == 0 or va.sum() == 0:
        raise RuntimeError("grouped split emptied a side")
    return np.where(tr)[0], np.where(va)[0], {
        "n_groups": int(len(uniq)),
        "n_val_groups": n_val,
        "train_windows": int(tr.sum()),
        "val_windows": int(va.sum()),
        "train_frac": float(tr.mean()),
        "val_frac": float(va.mean()),
    }


def grouped_indices_class_guard(groups, y, seed, val_frac=0.2):
    """Recording-grouped split that keeps >=1 train recording per class.

    Needed for PHM09 10-OC (2 reps/class/OC): naive group shuffle can put both
    reps of a class into val, emptying train cells and breaking MOCD matching.
    """
    rng = np.random.RandomState(seed)
    groups = np.asarray(groups)
    y = np.asarray(y)
    uniq = np.unique(groups)
    from collections import defaultdict
    c_to_g = defaultdict(set)
    for g, yi in zip(groups.tolist(), y.tolist()):
        c_to_g[int(yi)].add(g)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    n_val = min(n_val, len(uniq) - 1)
    val_set = []
    remaining = set(uniq.tolist())
    for g in uniq.tolist():
        if len(val_set) >= n_val:
            break
        trial = remaining - {g}
        ok = True
        for _c, gs in c_to_g.items():
            if len(gs & trial) < 1:
                ok = False
                break
        if ok:
            val_set.append(g)
            remaining.remove(g)
    if len(val_set) == 0:
        raise RuntimeError("class-guard grouped split could not place any val group")
    val_set = set(val_set)
    va = np.array([g in val_set for g in groups])
    tr = ~va
    # final guard
    for c in sorted(c_to_g):
        if int((y[tr] == c).sum()) == 0:
            raise RuntimeError(f"class-guard failed: class {c} empty in train")
    return np.where(tr)[0], np.where(va)[0], {
        "mode": "grouped_class_guard",
        "n_groups": int(len(uniq)),
        "n_val_groups": int(len(val_set)),
        "train_windows": int(tr.sum()),
        "val_windows": int(va.sum()),
        "train_frac": float(tr.mean()),
        "val_frac": float(va.mean()),
    }


def random_window_indices(n, seed, val_frac=0.2):
    """Paper-style split: independently permute windows within each source domain."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    va, tr = idx[:n_val], idx[n_val:]
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError("random split emptied a side")
    return tr, va, {
        "mode": "random_window",
        "n": int(n),
        "train_windows": int(len(tr)),
        "val_windows": int(len(va)),
        "train_frac": float(len(tr) / n),
        "val_frac": float(len(va) / n),
    }


def parse_fold(spec, fold):
    """Map a fold token to source and target domain lists.

    ``I``: LOCO 5→1 (existing). ``k2_IJ_K``: 2→1. ``t2_IJ``: 4→2
    (targets I,J; sources the rest). ``t3_IJK``: 3→3, listed not launched.
    """
    domains = list(spec["domains"])
    dset = set(domains)
    tasks = spec.get("fold_tasks") or {}
    if fold in tasks:
        src, tgts = list(tasks[fold][0]), list(tasks[fold][1])
        if set(src) & set(tgts):
            raise ValueError(f"overlap in {fold}")
        return src, tgts
    if fold in dset:
        src = [d for d in domains if d != fold]
        return src, [fold]
    if fold.startswith("k2_"):
        parts = fold[3:].split("_")
        if len(parts) != 2 or len(parts[0]) != 2 or len(parts[1]) != 1:
            raise ValueError(f"bad k2 fold {fold}")
        src, tgts = list(parts[0]), [parts[1]]
    elif fold.startswith("t2_"):
        tgts = list(fold[3:])
        if len(tgts) != 2:
            raise ValueError(f"bad t2 fold {fold}")
        src = [d for d in domains if d not in tgts]
    elif fold.startswith("t3_"):
        tgts = list(fold[3:])
        if len(tgts) != 3:
            raise ValueError(f"bad t3 fold {fold}")
        src = [d for d in domains if d not in tgts]
    else:
        raise ValueError(f"unknown fold {fold}")
    if len(set(src)) != len(src) or len(set(tgts)) != len(tgts):
        raise ValueError(f"duplicate domain in {fold}")
    unknown = (set(src) | set(tgts)) - dset
    if unknown:
        raise ValueError(f"fold {fold} not in {domains}: {sorted(unknown)}")
    if set(src) & set(tgts):
        raise ValueError(f"overlap in {fold}")
    if sorted(src, key=domains.index) != src:
        raise ValueError(f"sources not protocol-ordered in {fold}")
    if sorted(tgts, key=domains.index) != tgts:
        raise ValueError(f"targets not protocol-ordered in {fold}")
    return src, tgts


def folds_k2(spec):
    """All 2-source 1-target tasks. Unused domains stay out. Count = C(n,2)*(n-2)."""
    D = list(spec["domains"])
    out = []
    for src in combinations(D, 2):
        for t in D:
            if t in src:
                continue
            out.append(f"k2_{''.join(src)}_{t}")
    return out


def folds_t2(spec):
    """All 4-source 2-target tasks. Targets choose the split. Count = C(n,2)."""
    D = list(spec["domains"])
    return [f"t2_{''.join(tg)}" for tg in combinations(D, 2)]


def folds_t3(spec):
    """All 3-source 3-target tasks. Count = C(n,3). F4 cardinality curve."""
    D = list(spec["domains"])
    return [f"t3_{''.join(tg)}" for tg in combinations(D, 3)]


def save_split(path, spec_name, fold, seed, src_doms, splits, te_idx=None,
               meta_extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        return
    payload = {"spec": spec_name, "fold": fold, "seed": int(seed), "src_doms": src_doms}
    if meta_extra:
        payload.update(meta_extra)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    kw = {f"{d}_tr": splits[d][0] for d in src_doms}
    kw.update({f"{d}_va": splits[d][1] for d in src_doms})
    if te_idx:
        for d, idx in te_idx.items():
            kw[f"{d}_te"] = np.asarray(idx)
    np.savez_compressed(tmp, **kw, meta=np.array([json.dumps(payload)]))
    try:
        os.replace(tmp, path)
    except OSError:
        if not os.path.isfile(path):
            raise
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _split_meta_from_npz(z):
    if "meta" not in z.files:
        return None
    raw = z["meta"]
    try:
        s = raw.item() if getattr(raw, "shape", ()) == () else raw[0]
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return json.loads(str(s))
    except Exception:
        return None


def _protocol_split_dir(split_dir, protocol):
    """Formal temporal-block splits live under splits/{protocol}/ — never share PU."""
    return os.path.join(split_dir, protocol)


def _is_temporal_block_proto(spec):
    return spec.get("protocol") in ("jnu_speed", "hust_speed", "hust_w3")


def load_or_make_split(cache, spec, fold, seed, split_dir, split_mode="grouped"):
    src, tgts = parse_fold(spec, fold)
    if spec.get("kind") == "pu_mocd_pt":
        split_mode = "mocd_60_30"
    elif spec.get("split_mode"):
        split_mode = spec["split_mode"]
    proto = spec.get("protocol")
    if _is_temporal_block_proto(spec):
        if split_mode != "temporal_block":
            raise RuntimeError(
                f"{proto} requires temporal_block, got {split_mode}")
        split_dir = _protocol_split_dir(split_dir, proto)
    path = os.path.join(split_dir, f"{spec['protocol']}_{split_mode}_{fold}_seed{seed}.npz")
    splits, reports = {}, {}
    if os.path.isfile(path):
        try:
            z = np.load(path, allow_pickle=True)
            meta = _split_meta_from_npz(z)
            if _is_temporal_block_proto(spec):
                if not meta:
                    raise RuntimeError(f"{proto} split cache missing meta: {path}")
                if meta.get("spec") != proto:
                    raise RuntimeError(f"{proto} split meta.spec={meta.get('spec')}")
                if meta.get("fold") != fold:
                    raise RuntimeError(f"{proto} split meta.fold={meta.get('fold')}")
                if int(meta.get("seed", -1)) != int(seed):
                    raise RuntimeError(f"{proto} split meta.seed={meta.get('seed')}")
                if meta.get("split_mode") != "temporal_block":
                    raise RuntimeError(
                        f"{proto} split meta.split_mode={meta.get('split_mode')}")
                if list(meta.get("src_doms") or []) != list(src):
                    raise RuntimeError(f"{proto} split meta.src_doms={meta.get('src_doms')}")
                if list(meta.get("tgt_doms") or []) != list(tgts):
                    raise RuntimeError(f"{proto} split meta.tgt_doms={meta.get('tgt_doms')}")
            for d in src:
                splits[d] = (z[f"{d}_tr"], z[f"{d}_va"])
            te_idx = {k[:-3]: np.asarray(z[k]).tolist() for k in z.files if k.endswith("_te")}
            return src, tgts, splits, {
                "reused": True, "path": path, "tgt_doms": tgts, "te_idx": te_idx,
                "split_mode": split_mode, "meta": meta,
            }
        except (EOFError, OSError, ValueError):
            splits = {}
    te_idx = {}
    if split_mode == "mocd_60_30":
        from wfdg.mocd_pu import mocd_60_30_indices, source_tr_va
        for d in spec["domains"]:
            _x, y, _g = cache[d]
            tr60, te30 = mocd_60_30_indices(y, seed * 8 + int(d))
            te_idx[d] = te30
            if d in src:
                tr, va = source_tr_va(tr60, seed)
                splits[d] = (tr, va)
                reports[d] = {"mode": "mocd_60_30", "train": int(len(tr)), "val": int(len(va)),
                              "test": int(len(te30))}
    else:
        for d in src:
            _x, _y, g = cache[d]
            if split_mode == "grouped":
                tr, va, rep = grouped_indices(g, seed)
            elif split_mode == "grouped_class_guard":
                _x, y, g = cache[d]
                tr, va, rep = grouped_indices_class_guard(g, y, seed)
            elif split_mode == "random_window":
                tr, va, rep = random_window_indices(len(g), seed)
            elif split_mode == "temporal_block":
                off, flen = load_domain_offsets(spec, d)
                tr, va, rep = temporal_block_indices(
                    g, off, flen, win=int(spec.get("win", spec["input_len"])))
            else:
                raise ValueError(f"unknown split_mode: {split_mode}")
            splits[d] = (tr, va)
            reports[d] = rep
    meta_extra = None
    if _is_temporal_block_proto(spec):
        meta_extra = {
            "split_mode": "temporal_block",
            "tgt_doms": list(tgts),
            "protocol": proto,
            "win": int(spec.get("win", spec["input_len"])),
            "stride": int(spec.get("stride", 0)),
            "note": "temporal_block; NOT recording-grouped; not _cpu_smoke",
        }
    save_split(path, spec["protocol"], fold, seed, src, splits, te_idx=te_idx or None,
               meta_extra=meta_extra)
    te_list = {k: np.asarray(v).tolist() for k, v in te_idx.items()}
    return src, tgts, splits, {
        "reused": False, "path": path, "reports": reports, "tgt_doms": tgts,
        "te_idx": te_list, "split_mode": split_mode, "meta": meta_extra,
    }


def assert_no_target(src_doms, fold, tgt_doms=None):
    tgts = list(tgt_doms) if tgt_doms is not None else [fold]
    hit = set(src_doms) & set(tgts)
    if hit:
        raise AssertionError(f"target {sorted(hit)} in sources")


def assert_no_c_in_batch(batch_keys):
    if any(k in batch_keys for k in ("c", "cvec", "rpm", "torque")):
        raise AssertionError("explicit c leaked into batch")


def target_xy(cache, t, split_meta):
    x, y, _g = cache[t]
    te = (split_meta or {}).get("te_idx") or {}
    if t in te:
        idx = np.asarray(te[t], dtype=np.int64)
        return x[idx], y[idx]
    return x, y


def pu_cvec(dom):
    """Physical c for R1 only. Not stored in tensors; not a domain id."""
    from author_pu_6domain import author_cvec
    return author_cvec(dom)


def jnu_cvec(spec, dom):
    """Locked JNU condition representation (speed only). Not a PU vector.

    Formal JNU methods (R3t3 / R0t3 / erm_t3fft / mocd) set uses_physical_c=False
    and must not call this. Kept for an explicit future path only.
    """
    if spec.get("protocol") != "jnu_speed" and spec.get("dataset") != "jnu":
        raise RuntimeError("jnu_cvec only for jnu_speed")
    rpm = float(spec["domain_rpm"][dom])
    # Single scalar speed in [0,1] vs 1000 rpm. Not author_cvec / pu_cvec.
    return np.asarray([rpm / 1000.0], dtype=np.float32)


def hust_cvec(spec, dom):
    """Locked HUST condition representation (speed Hz only). Not a PU vector.

    Formal HUST methods set uses_physical_c=False and must not call this.
    """
    if spec.get("protocol") != "hust_speed" and spec.get("dataset") != "hust":
        raise RuntimeError("hust_cvec only for hust_speed")
    hz = float(spec.get("domain_hz", spec["domain_rpm"])[dom])
    return np.asarray([hz / 80.0], dtype=np.float32)


def domain_cvec(spec, dom):
    """Physical c for residual levels. Fail-closed per dataset — no silent PU fallback."""
    if spec.get("protocol") == "jnu_speed" or spec.get("dataset") == "jnu":
        if not spec.get("uses_physical_c", False):
            raise RuntimeError(
                "jnu_speed declares uses_physical_c=False; methods must not "
                "request a condition vector. No PU pu_cvec fallback.")
        return jnu_cvec(spec, dom)
    if spec.get("protocol") == "phm09_speed" or spec.get("dataset") == "phm09":
        raise RuntimeError(
            "phm09_speed declares uses_physical_c=False; methods must not "
            "request a condition vector. No PU pu_cvec fallback.")
    if spec.get("protocol") == "hust_speed" or spec.get("dataset") == "hust":
        if not spec.get("uses_physical_c", False):
            raise RuntimeError(
                "hust_speed declares uses_physical_c=False; methods must not "
                "request a condition vector. No PU pu_cvec fallback.")
        return hust_cvec(spec, dom)
    if spec["protocol"] == "pu_mocd":
        from wfdg.mocd_pu import mocd_cvec
        return mocd_cvec(dom)
    if spec["dataset"] == "cwru":
        from common.spectral import cwru_state_cvec
        return cwru_state_cvec(dom)
    if spec["dataset"] == "pu":
        return pu_cvec(dom)
    raise RuntimeError(
        f"domain_cvec: unknown dataset={spec.get('dataset')!r} "
        f"protocol={spec.get('protocol')!r}; refusing PU fallback")


def domain_sensor(spec, dom):
    """Sensor slot m. CWRU DE=0 FE=1. PU/JNU/HUST have no sensor factor → 0."""
    if spec["dataset"] == "cwru":
        from common.spectral import cwru_sensor_id
        return int(cwru_sensor_id(dom))
    if spec.get("dataset") in ("pu", "jnu", "hust"):
        return 0
    raise RuntimeError(f"domain_sensor: unknown dataset {spec.get('dataset')!r}")


def domain_rpm(spec, dom):
    if "domain_rpm" in spec:
        return float(spec["domain_rpm"][dom])
    if spec["protocol"] == "pu_mocd":
        from wfdg.mocd_pu import mocd_rpm
        return mocd_rpm(dom)
    if spec["dataset"] == "cwru":
        from common.spectral import CWRU_COND
        return float(CWRU_COND[dom][1])
    if spec["dataset"] == "pu":
        from common.spectral import PU_COND
        return float(PU_COND[dom][0])
    raise RuntimeError(
        f"domain_rpm: unknown dataset={spec.get('dataset')!r}; refusing PU fallback")


def protocol_rpm_ref(spec, mode="max"):
    """Reference speed over domains in this protocol."""
    rpms = [domain_rpm(spec, d) for d in spec["domains"]]
    if mode == "max":
        return max(rpms)
    if mode == "min":
        return min(rpms)
    if mode == "median":
        return float(np.median(rpms))
    raise ValueError(mode)


def cond_dim_for(spec):
    if (spec.get("protocol") in ("jnu_speed", "hust_speed", "hust_w3", "phm09_speed", "phm09_w3")
            or spec.get("dataset") in ("jnu", "hust", "phm09")):
        return int(spec.get("cond_dim", 0))
    if spec["dataset"] == "cwru":
        return 2
    if spec["dataset"] == "pu":
        return 3
    raise RuntimeError(
        f"cond_dim_for: unknown dataset={spec.get('dataset')!r}; refusing PU default")


def rewrite_c_after_do_speed(c, spec, rpm_ref):
    """After order-tracking to rpm_ref, overwrite the rpm slot of c."""
    if (spec.get("protocol") in ("jnu_speed", "hust_speed", "hust_w3", "phm09_speed", "phm09_w3")
            or spec.get("dataset") in ("jnu", "hust", "phm09")):
        raise RuntimeError(f"{spec.get('protocol')} does not order-track physical c")
    c = np.array(c, dtype=np.float32, copy=True)
    if spec["dataset"] == "cwru":
        c[..., 1] = (float(rpm_ref) - 1720.0) / (1797.0 - 1720.0)
    elif spec["dataset"] == "pu":
        c[..., 2] = float(rpm_ref) / 1500.0
    else:
        raise RuntimeError(f"rewrite_c_after_do_speed: unknown {spec.get('dataset')!r}")
    return c


class BalancedDomainClassSampler:
    """(domain, class) cells; skip empty cells. Does not downsample majority class."""

    def __init__(self, domains_xy, n_cls, per_cell, seed):
        self.n_cls = int(n_cls)
        self.per_cell = int(per_cell)
        self.rng = np.random.RandomState(seed)
        self.cells = []
        for di, (x, y) in enumerate(domains_xy):
            for c in range(self.n_cls):
                idx = np.where(y == c)[0]
                self.cells.append((di, c, idx, x, y))
        nonzero = sum(1 for _di, _c, idx, _x, _y in self.cells if len(idx) > 0)
        self.meta = {
            "type": "BalancedDomainClassSampler",
            "n_cls": self.n_cls,
            "per_cell": self.per_cell,
            "n_domains": len(domains_xy),
            "n_cells_nonzero": int(nonzero),
            "batch_size": int(nonzero * self.per_cell),
            "source_only": True,
            "target_in_sampler": False,
            "downsample_majority": False,
        }

    def sample(self):
        import torch
        xs, ys, ds = [], [], []
        for di, c, idx, x, y in self.cells:
            if len(idx) == 0:
                continue
            pick = self.rng.randint(0, len(idx), self.per_cell)
            ii = idx[pick]
            xs.append(x[ii])
            ys.append(y[ii])
            ds.append(np.full(self.per_cell, di, dtype=np.int64))
        x = torch.from_numpy(np.concatenate(xs)).float()
        y = torch.from_numpy(np.concatenate(ys)).long()
        d = torch.from_numpy(np.concatenate(ds)).long()
        return x, y, d


def jnu_balanced_per_cell(batch_per_domain, n_cls):
    """Keep ~batch_per_domain draws per domain via per_cell = bpd // n_cls."""
    return max(1, int(batch_per_domain) // int(n_cls))


def build_jnu_source_class_balanced_sampler(
        src_tr, n_cls, batch_per_domain, seed, src_doms=None, tgt_doms=None,
        require_all_classes=True):
    """Temporal-block formal train sampler: source-train only, class×domain balanced.

    Used by jnu_speed and hust_speed. Does not downsample majority class.
    Target domains must not appear in src_tr.
    If require_all_classes=False (phm09_w3 2-rep OC), empty cells are skipped.
    """
    if tgt_doms is not None and src_doms is not None:
        overlap = set(src_doms) & set(tgt_doms)
        if overlap:
            raise RuntimeError(f"target domains in sampler src: {overlap}")
    per_cell = jnu_balanced_per_cell(batch_per_domain, n_cls)
    if require_all_classes:
        for di, (_x, y) in enumerate(src_tr):
            for c in range(int(n_cls)):
                if int((y == c).sum()) == 0:
                    label = src_doms[di] if src_doms is not None else di
                    raise RuntimeError(
                        f"source domain {label!r} missing class {c} in train split")
    samp = BalancedDomainClassSampler(src_tr, n_cls, per_cell, seed)
    meta = dict(samp.meta)
    meta["batch_per_domain_nominal"] = int(batch_per_domain)
    meta["src_doms"] = list(src_doms) if src_doms is not None else None
    meta["tgt_doms_excluded"] = list(tgt_doms) if tgt_doms is not None else None
    meta["require_all_classes"] = bool(require_all_classes)
    samp.meta = meta
    return samp, meta


# Alias for HUST / shared temporal-block protocols.
build_source_class_balanced_sampler = build_jnu_source_class_balanced_sampler


class DomainSampler:
    def __init__(self, domains_xy, batch, seed, domains_c=None):
        self.src = domains_xy
        self.batch = batch
        self.rng = np.random.RandomState(seed)
        self.c_src = domains_c

    def sample(self):
        import torch
        xs, ys, ds, cs = [], [], [], []
        for di, (x, y) in enumerate(self.src):
            idx = self.rng.randint(0, len(x), self.batch)
            xs.append(x[idx])
            ys.append(y[idx])
            ds.append(np.full(self.batch, di, dtype=np.int64))
            if self.c_src is not None:
                cs.append(self.c_src[di][idx])
        x = torch.from_numpy(np.concatenate(xs)).float()
        y = torch.from_numpy(np.concatenate(ys)).long()
        d = torch.from_numpy(np.concatenate(ds)).long()
        if self.c_src is not None:
            return x, y, d, torch.from_numpy(np.concatenate(cs)).float()
        return x, y, d
