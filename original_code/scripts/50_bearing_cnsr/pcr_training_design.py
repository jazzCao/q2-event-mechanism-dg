"""Pure source sampler, schedule and metric conventions for a future campaign.

No neural imports, optimizer objects, actual array reads or approval bypass.
"""
from __future__ import annotations
import copy
from hashlib import sha256
import json
import math
import numpy as np

PROFILES = ('AUTHOR_CE', 'CE_PLUS_REAL_HEALTHY', 'PCR_FAULT_TO_HEALTHY')
FRONTENDS = ('NATIVE40', 'SHORT_PHYSICAL_WIDTH')


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def model_initialization_seed(split_seed):
    if split_seed not in (0, 1, 2):
        raise ValueError('fixed three native initialization seeds')
    # Author Config uses base_seed=0, seed_increment=1000, run_times=3.
    return split_seed * 1000


def source_stream_seed(fold, split_seed, stream):
    if fold not in ('BCDtoA', 'ACDtoB', 'ABDtoC', 'ABCtoD') or split_seed not in (0, 1, 2) or stream not in ('factual', 'auxiliary'):
        raise ValueError('fixed source stream identity')
    return int(sha256(f'PCR_PU_MDG_V1|{fold}|split{split_seed}|stream{stream}'.encode()).hexdigest()[:8], 16)


def learning_rate_after_updates(completed):
    if type(completed) is not int or not 0 <= completed <= 2000:
        raise ValueError('fixed 2000-update schedule')
    # Mirrors AdamW + CosineAnnealingLR, stepped AFTER each actual update.
    # Fixed 4000 horizon is an explicit full-source MDG schedule adaptation.
    return 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * completed / 4000)) / 2


class SourceSampler:
    """Domain -> class -> recording -> legal window, independent RNG streams.

    Returns only (record-key, window-index) metadata. The future data-access
    boundary must independently enforce actual raw/feature role membership.
    A caller-supplied string 'source_train' is NOT sufficient provenance.
    """
    def __init__(self, case, records, frontend, profile, state=None):
        if frontend not in FRONTENDS or profile not in PROFILES:
            raise ValueError('fixed frontend/profile')
        self.case = case
        self.records = records
        self.frontend = frontend
        self.profile = profile
        self.domains = tuple(case['task']['sources'])
        self.allowed = frozenset(case['roles']['source_train'])
        if len(self.domains) != 3 or case['task']['target'] in self.domains:
            raise ValueError('fixed existing PU 3-to-1 task')
        self.pools = {}
        for d in self.domains:
            for cls in range(3):
                keys = tuple(sorted(k for k in self.allowed if records[k]['domain'] == d and records[k]['label'] == cls))
                if not keys:
                    raise ValueError('source class/domain missing')
                self.pools[d, cls] = keys
        self.window_counts = {key: 60 if frontend == 'SHORT_PHYSICAL_WIDTH' else
                              int(((246784 - 1) * records[key]['shaft_hz']) // records[key]['fs']) - 40 + 1
                              for key in self.allowed}
        if any(value <= 0 for value in self.window_counts.values()):
            raise ValueError('native window support')
        self.identity = fingerprint(dict(role_fingerprint=case['role_fingerprint'],
                                         frontend=frontend, profile=profile,
                                         domains=self.domains,
                                         pools={f'{d}/{c}': v for (d, c), v in self.pools.items()},
                                         legal_window_counts=self.window_counts))
        self.factual = np.random.RandomState(source_stream_seed(case['task']['fold'], case['seed'], 'factual'))
        self.auxiliary = np.random.RandomState(source_stream_seed(case['task']['fold'], case['seed'], 'auxiliary'))
        self.drawn = {'factual': 0, 'auxiliary': 0}
        if state is not None:
            self.restore(state)

    def _one(self, rng, mode):
        d = self.domains[rng.randint(3)]
        if mode == 'factual':
            cls = int(rng.randint(3))
        elif mode == 'healthy':
            cls = 0
        elif mode == 'fault_cf':
            cls = 1 + int(rng.randint(2))
        else:
            raise ValueError('source sampler mode')
        keys = self.pools[d, cls]
        key = keys[rng.randint(len(keys))]
        if key not in self.allowed or self.records[key]['domain'] not in self.domains:
            raise PermissionError('source training membership')
        count = self.window_counts[key]
        return key, int(rng.randint(count))

    def batch(self, n=200):
        if type(n) is not int or n <= 0:
            raise ValueError('positive batch size')
        fact = [self._one(self.factual, 'factual') for _ in range(n)]
        self.drawn['factual'] += n
        auxiliary = []
        if self.profile != 'AUTHOR_CE':
            mode = 'healthy' if self.profile == 'CE_PLUS_REAL_HEALTHY' else 'fault_cf'
            auxiliary = [self._one(self.auxiliary, mode) for _ in range(n)]
            self.drawn['auxiliary'] += n
        return fact, auxiliary

    def packet(self):
        return dict(identity=self.identity, factual=copy.deepcopy(self.factual.get_state()),
                    auxiliary=copy.deepcopy(self.auxiliary.get_state()), drawn=copy.deepcopy(self.drawn))

    def restore(self, state):
        if set(state) != {'identity', 'factual', 'auxiliary', 'drawn'} or state['identity'] != self.identity:
            raise PermissionError('sampler checkpoint belongs to another case/frontend/profile')
        if set(state['drawn']) != {'factual', 'auxiliary'} or any(type(v) is not int or v < 0 for v in state['drawn'].values()):
            raise ValueError('complete nonnegative sampler cursor')
        self.factual.set_state(copy.deepcopy(state['factual']))
        self.auxiliary.set_state(copy.deepcopy(state['auxiliary']))
        self.drawn = copy.deepcopy(state['drawn'])


def source_selection_score(probability_local, local_labels, domains, source_domains):
    p, y, d = np.asarray(probability_local), np.asarray(local_labels), np.asarray(domains)
    if p.shape != (len(y), 3) or y.ndim != 1 or d.shape != y.shape or not np.isfinite(p).all():
        raise ValueError('complete local-order source probabilities')
    if set(d.tolist()) != set(source_domains) or len(source_domains) != 3:
        raise PermissionError('only the three existing source-validation domains')
    # Exact old E0 convention: window accuracy within each source, then domain mean.
    return float(np.mean([(p[d == domain].argmax(1) == y[d == domain]).mean() for domain in source_domains]))


def author_probability_to_local(probability):
    p = np.asarray(probability)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('normalized author H/IF/OF probabilities required')
    return p[:, [0, 2, 1]].copy()


def recording_predictions(probability_local, labels, recording_keys):
    p, y, g = np.asarray(probability_local), np.asarray(labels), np.asarray(recording_keys)
    if p.shape != (len(y), 3) or y.ndim != 1 or g.shape != y.shape or not np.isfinite(p).all():
        raise ValueError('aligned recording/window probabilities')
    output = []
    for key in sorted(set(g.tolist())):
        side = g == key
        if len(set(y[side].tolist())) != 1:
            raise ValueError('a recording cannot change class between windows')
        output.append(dict(record=key, label=int(y[side][0]),
                           probability=p[side].mean(0).tolist(), windows=int(side.sum())))
    return output
