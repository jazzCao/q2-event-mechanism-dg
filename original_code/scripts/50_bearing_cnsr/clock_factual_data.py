"""Verified frozen features; full factual supervision and separate auxiliary labels."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
import clock_real_data as old
from clock_real_model import Net as OriginalNet
from clock_uniform_data import UniformData

ROOT = old.ROOT
OUT = ROOT/'results/wfdg/ress_clock_factual'
STORE = Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_clock_factual')
ARMS = ('S0_INTACT', 'N0_WARP', 'C0_CLOCK', 'E0_EVENTS')


class Runtime(old.Runtime):
    def clock(self):
        now = time.time()
        if now >= self.spec['deadline_unix'] or self.gpu_start is not None and now >= self.gpu_start+1800:
            raise TimeoutError('factual original90min/30min clocks')
    def gpu(self):
        self.clock()
        if self.gpu_start is None:
            self.gpu_start = time.time()
            old.save(STORE/'GPU_START.json', dict(at_unix=self.gpu_start, deadline=min(self.spec['deadline_unix'], self.gpu_start+1800)))


class Net(OriginalNet):
    def losses(self, out, labels, c, permutation, lam):
        factual_y, auxiliary_y = labels
        n = len(factual_y); logits = out['logits']
        if len(auxiliary_y) != n or len(logits) != 2*n: raise ValueError('two explicitly labelled training views')
        clean = F.cross_entropy(logits[:n], factual_y)
        mixed = lam*F.cross_entropy(logits[n:], auxiliary_y)+(1-lam)*F.cross_entropy(logits[n:], auxiliary_y[permutation])
        return dict(total=.5*(clean+mixed), clean_ce=clean, mixup_ce=mixed)


def verify_file(binding, rt):
    rt.claim('cache_hash_checks')
    rt.permit(binding['path'])
    try:
        if old.sha(binding['path']) != binding['sha256']: raise ValueError('cache SHA mismatch')
    finally: rt.revoke(binding['path'])


class CachedStore:
    def __init__(self, rt):
        self.rt = rt; self.records = rt.spec['data']['records']; self.loaded = {}; self.accesses = []
        self.binding = {(b['protocol'], b['domain']): b for b in rt.spec['feature_bindings']}
    def domain(self, protocol, domain, task, role):
        rt = self.rt
        if protocol != task['protocol'] or role not in ('source', 'target'): raise PermissionError('cache protocol/role')
        if role == 'source' and domain not in task['sources'] or role == 'target' and (rt.phase != 'TARGET' or domain != task['target']):
            raise PermissionError('cache target before full freeze or wrong source')
        self.accesses.append(dict(protocol=protocol, domain=domain, fold=task['fold'], role=role, phase=rt.phase, at_unix=time.time()))
        key = (protocol, domain)
        if key not in self.loaded:
            binding = self.binding[key]; verify_file(binding, rt)
            rt.claim('source_feature_NPZ_loads' if role == 'source' else 'target_feature_NPZ_loads')
            rt.permit(binding['path'])
            with np.load(binding['path'], allow_pickle=False) as z: arrays = {k: z[k].copy() for k in ('x', 'y', 'g', 'offset')}
            rt.revoke(binding['path']); ey = []; eg = []; eo = []; positions = {}
            for record_key, record in sorted(self.records.items()):
                if (record['protocol'], record['domain']) != key: continue
                positions[record_key] = {offset: len(eo)+i for i, offset in enumerate(record['offsets'])}
                n = len(record['offsets']); ey.extend([record['label']]*n); eg.extend([record_key]*n); eo.extend(record['offsets'])
            if arrays['x'].shape != (len(ey), 2048) or arrays['x'].dtype != np.float32 or not np.isfinite(arrays['x']).all(): raise ValueError('full feature shape/type')
            if not all(np.array_equal(arrays[k], np.asarray(v)) for k, v in (('y', ey), ('g', eg), ('offset', eo))): raise ValueError('full feature labels/recordings/offsets')
            if binding['shape'] != list(arrays['x'].shape) or binding['FFT_code_sha256'] != rt.spec['FFT_code_sha256']: raise ValueError('FFT provenance')
            expected = {k for k, r in self.records.items() if (r['protocol'], r['domain']) == key}
            if {r['record'] for r in binding['source_input_records']} != expected: raise ValueError('raw provenance coverage')
            for item in binding['source_input_records']:
                raw = rt.spec['raw_bindings'][item['record']]
                if {k: v for k, v in item.items() if k != 'record'} != raw: raise ValueError('raw provenance SHA')
            arrays['positions'] = positions; self.loaded[key] = arrays
        return self.loaded[key]
    def source_case(self, case):
        values = {side: {k: [] for k in ('x', 'y', 'g', 'domain')} for side in ('tr', 'va')}
        for record_key, sides in sorted(case['roles'].items()):
            record = self.records[record_key]; arrays = self.domain(record['protocol'], record['domain'], case['task'], 'source')
            for side, offsets in sides.items():
                if not offsets: continue
                positions = [arrays['positions'][record_key][offset] for offset in offsets]
                values[side]['x'].append(arrays['x'][positions]); values[side]['y'].append(arrays['y'][positions]); values[side]['g'].append(arrays['g'][positions])
                values[side]['domain'].append(np.repeat(record['domain'], len(positions)))
        return {side: {k: np.concatenate(v) for k, v in part.items()} for side, part in values.items()}
    def target(self, task):
        arrays = self.domain(task['protocol'], task['target'], task, 'target')
        return {k: arrays[k] for k in ('x', 'y', 'g')}
    def bank(self, case):
        key = f'{case["task"]["protocol"]}/{case["task"]["fold"]}/s{case["seed"]}'
        binding = self.rt.spec['bank_bindings'][key]; verify_file(binding, self.rt)
        self.rt.claim('source_bank_NPZ_loads'); self.rt.permit(binding['path'])
        with np.load(binding['path'], allow_pickle=False) as z: arrays = {k: z[k].copy() for k in ('clean', 'joint', 'event', 'warp', 'y', 'domain', 'g')}
        self.rt.revoke(binding['path']); ey = []; eg = []; ed = []
        for item in case['bank']:
            record_key = item['record']; record = self.records[record_key]
            if record_key not in case['roles'] or not case['roles'][record_key]['tr'] or record['domain'] not in case['task']['sources']: raise PermissionError('bank source training only')
            ey.extend([record['label']]*4); eg.extend([record_key]*4); ed.extend([record['domain']]*4)
            if item['pair'] is not None:
                inverse = self.rt.spec['inverse_references'][item['pair']]
                if inverse['pair'] != case['pairs'][item['pair']] or inverse['input_sha256'] != [self.rt.spec['raw_bindings'][k]['sha256'] for k in inverse['pair']]: raise ValueError('parent inverse reference identity')
        if not all(np.array_equal(arrays[k], np.asarray(v)) for k, v in (('y', ey), ('g', eg), ('domain', ed))): raise ValueError('bank label/record/order')
        for name in ('clean', 'joint', 'event', 'warp'):
            shape = (len(ey), 2048) if name == 'clean' else (len(ey), 2, 2048)
            if arrays[name].shape != shape or arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all(): raise ValueError('bank shape/finites')
        return arrays


class Data(UniformData):
    def __init__(self, case, source, bank, rt):
        super().__init__(case, source, rt)
        self.case = case; self.source_arrays = source; self.eligible = case['eligible']
        self.bx = torch.from_numpy(bank['clean']).cuda(); self.by = torch.from_numpy(bank['y']).cuda()
        self.transforms = {name: torch.from_numpy(bank[name]).cuda() for name in ('joint', 'event', 'warp')}
    def batch(self, arm, sampler, augmentation):
        if arm not in ARMS+('B0_ID_MIX', 'B1_IN_MIX'): raise ValueError('fixed factual interface arms')
        full = np.concatenate([sampler.choice(pool, self.per_cell, replace=True) for pool in self.pools])
        bank = np.concatenate([sampler.choice(pool, self.per_cell, replace=True) for pool in self.bpools])
        mask = torch.as_tensor(sampler.uniform(size=self.batch_size) < .5, device='cuda')
        factors = torch.as_tensor(augmentation.randint(0, 2, size=self.batch_size), device='cuda')
        fi = torch.as_tensor(full, device='cuda'); bi = torch.as_tensor(bank, device='cuda')
        factual = self.x[fi]; factual_y = self.y[fi]; bank_y = self.by[bi]
        if not torch.equal(factual_y, bank_y): raise AssertionError('stratum factual/auxiliary labels')
        use_bank = arm not in ('B0_ID_MIX', 'B1_IN_MIX') and (self.eligible or arm == 'N0_WARP')
        if use_bank:
            kind = dict(N0_WARP='warp', C0_CLOCK='joint', E0_EVENTS='event').get(arm)
            changed = self.bx[bi] if kind is None else self.transforms[kind][bi, factors]
            auxiliary = torch.where(mask[:, None], changed, factual)
            auxiliary_y = torch.where(mask, bank_y, factual_y)
        else:
            auxiliary = factual; auxiliary_y = factual_y.clone()
        lam = float(augmentation.beta(.2, .2)); permutation = torch.as_tensor(augmentation.permutation(len(factual_y)), device='cuda')
        mixed = lam*auxiliary+(1-lam)*auxiliary[permutation]
        return torch.cat((factual, mixed)), (factual_y, auxiliary_y), None, permutation, lam
