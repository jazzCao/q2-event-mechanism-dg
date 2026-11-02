"""Lock loader and task inventory. No tensors."""
from __future__ import annotations

import hashlib
import itertools
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_LOCK = os.path.join(ROOT, "results", "wfdg", "ress_source_dg_plan", "LOCK_SPEC.json")
PLAN_MD = os.path.join(ROOT, "results", "wfdg", "RESS_SOURCE_DG_EXECUTION_PLAN.md")
RESULTS = os.path.join(ROOT, "results", "wfdg", "ress_source_dg")

INTERNAL = ["E0_ERM", "E1_MIXUP", "E2_IN_MIXUP", "G0_RANDOM", "G1_GLOBAL", "C0_CSDR", "N0_WRONG_CLASS"]
DIRECTION = {"G0_RANDOM": "random", "G1_GLOBAL": "global", "C0_CSDR": "conditional", "N0_WRONG_CLASS": "wrong"}


def load_lock(path=None):
    path = path or DEFAULT_LOCK
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("version") != "RESS_SOURCE_DG_V1":
        raise RuntimeError("SPEC_CONFLICT lock version")
    return spec


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256():
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            h.update(name.encode())
            h.update(file_sha256(os.path.join(here, name)).encode())
    return h.hexdigest()


def seed_int(parts):
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def all_main_tasks(spec):
    domains = spec["data"]["domains"]
    cohort = spec["data"]["cohorts"]
    salt = spec["tasks"]["development_hash_salt"]
    tasks = []
    for sources in itertools.combinations(domains, 2):
        for target in domains:
            if target in sources:
                continue
            if cohort[sources[0]] != cohort[sources[1]]:
                stratum = "mixed_source"
            elif cohort[sources[0]] == cohort[target]:
                stratum = "same_origin"
            else:
                stratum = "opposite_origin"
            task_id = f"k2_{''.join(sources)}_{target}"
            tasks.append({
                "task": task_id,
                "sources": list(sources),
                "target": target,
                "stratum": stratum,
                "n_sources": 2,
                "hash": hashlib.sha256(f"{salt}|pu_ccdg|{task_id}".encode()).hexdigest(),
            })
    tasks.sort(key=lambda t: t["task"])
    return tasks


def supplement_tasks(spec):
    domains = spec["data"]["domains"]
    out = []
    for target in domains:
        sources = [d for d in domains if d != target]
        task_id = f"k5_{''.join(sources)}_{target}"
        out.append({
            "task": task_id,
            "sources": sources,
            "target": target,
            "stratum": "supplement_5to1",
            "n_sources": 5,
            "hash": hashlib.sha256(f"RESS_SOURCE_DG_V1|pu_ccdg|{task_id}".encode()).hexdigest(),
        })
    return out


def task_lookup(spec):
    rows = all_main_tasks(spec) + supplement_tasks(spec)
    return {t["task"]: t for t in rows}
