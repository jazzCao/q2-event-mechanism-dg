"""Matched two-view classifier. Import is inert until libraries(rt) is called."""
import copy
import hashlib
import os
import random
import time
from source_rate_plan import require


def libraries(rt):
    rt.before_neural()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from clock_real_model import Net
    from clock_factual_data import Net as FactualNet
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'real CUDA required, no silent CPU fallback')
    return np, torch, Net, FactualNet


def cpu(value, torch, np):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, dict): return {k: cpu(v, torch, np) for k, v in value.items()}
    if isinstance(value, list): return [cpu(v, torch, np) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v, torch, np) for v in value)
    return copy.deepcopy(value)


def exact(a, b, torch, np, where='state'):
    require(type(a) is type(b), where + ' type')
    if isinstance(a, torch.Tensor):
        require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), where + ' tensor')
    elif isinstance(a, np.ndarray):
        require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), where + ' array')
    elif isinstance(a, dict):
        require(a.keys() == b.keys(), where + ' keys')
        for k in a: exact(a[k], b[k], torch, np, where + '/' + str(k))
    elif isinstance(a, (tuple, list)):
        require(len(a) == len(b), where + ' length')
        for i, (x, y) in enumerate(zip(a, b)): exact(x, y, torch, np, where + '/' + str(i))
    else:
        require(a == b, where + ' value')


def tensor_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def initial_seed(task, seed):
    token = f'CONTEXT_EVIDENCE|{task["protocol"]}|{task["fold"]}|{seed}'
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)


class Data:
    def __init__(self, case, source, bank, rt):
        np, torch, _, _ = libraries(rt)
        self.rt, self.case, self.source, self.task = rt, case, source, case['task']
        self.per_cell = 120 // (len(self.task['sources']) * self.task['classes'])
        self.batch_size = self.per_cell * len(self.task['sources']) * self.task['classes']
        self.x, self.y = torch.as_tensor(source['tr']['x'], device='cuda'), torch.as_tensor(source['tr']['y'], device='cuda')
        self.bx = {k: torch.as_tensor(bank[k], device='cuda') for k in ('clean', 'fixed', 'coupled', 'ring')}
        self.by = torch.as_tensor(bank['y'], device='cuda')
        self.pools, self.bpools = [], []
        for domain in self.task['sources']:
            for label in range(self.task['classes']):
                self.pools.append(np.flatnonzero((source['tr']['domain'] == domain) & (source['tr']['y'] == label)))
                self.bpools.append(np.flatnonzero((bank['domain'] == domain) & (bank['y'] == label)))
        require(all(len(v) for v in self.pools + self.bpools), 'complete full-source and bank domain/class pools')

    def batch(self, arm, sampler, augmentation):
        import numpy as np
        import torch
        from source_rate_core import shared_edit_mask, batch_rates
        full = np.concatenate([sampler.choice(v, self.per_cell, replace=True) for v in self.pools])
        bank = np.concatenate([sampler.choice(v, self.per_cell, replace=True) for v in self.bpools])
        uniforms = sampler.uniform(size=self.batch_size)
        mask = shared_edit_mask(uniforms, len(self.task['sources']), self.task['classes'], self.per_cell)
        factors = augmentation.randint(0, 2, size=self.batch_size)
        if self.case['source_speed_diversity']:
            batch_rates(self.task['sources'], self.case['rates'], self.task['classes'], self.per_cell, mask)
        fi, bi, mi, factor = [torch.as_tensor(v, device='cuda') for v in (full, bank, mask, factors)]
        factual, yf, ya = self.x[fi], self.y[fi], self.by[bi]
        require(torch.equal(yf, ya), 'balanced full/auxiliary class order')
        name = {'F0_MATCHED_FACTUAL_BANK': 'clean', 'E0_MATCHED_FIXED_RATE': 'fixed',
                'C0_SOURCE_COUPLED_RATE': 'coupled', 'N0_SOURCE_RING_RATE': 'ring'}[arm]
        changed = self.bx[name][bi] if name == 'clean' else self.bx[name][bi, factor]
        auxiliary = torch.where(mi[:, None], changed, factual)
        auxiliary_y = torch.where(mi, ya, yf)
        lam = float(augmentation.beta(.2, .2))
        permutation = torch.as_tensor(augmentation.permutation(self.batch_size), device='cuda')
        mixed = lam * auxiliary + (1 - lam) * auxiliary[permutation]
        return torch.cat((factual, mixed)), (yf, auxiliary_y), permutation, lam


class Chain:
    def __init__(self, rt, case, arm, name, data=None, packet=None):
        self.rt, self.case, self.arm, self.name, self.data = rt, case, arm, name, data
        definition = rt.lock['chains'].get(name)
        require(definition is not None and definition['case_key'] == case['key'] and definition['arm'] == arm,
                'only registered fixed model identities')
        np, torch, Net, self.FactualNet = libraries(rt); self.np, self.torch = np, torch
        seed = initial_seed(case['task'], case['seed'])
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
        rt.claim('neural_model_builds')
        self.model = Net(case['task']['classes'], 'E0_EVENTS').cuda()
        self.initial_encoder = tensor_sha(self.model.enc.state_dict())
        require(self.initial_encoder == rt.lock['initial_encoder_sha256'][case['key']], 'same archived Table3 encoder initialization')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                          weight_decay=0, foreach=False)
        self.sampler = np.random.RandomState((seed + 11) % 2**32)
        self.augmentation = np.random.RandomState((seed + 29) % 2**32)
        self.auxiliary = np.random.RandomState((seed + 101) % 2**32)
        self.step_n = 0; self.losses = []; self.curve = []; self.best = self.selected = self.gradient_audit = None
        if packet is not None:
            require(packet['fingerprint'] == rt.lock['fingerprint'] and packet['case_key'] == case['key'] and
                    packet['arm'] == arm and packet['chain'] == name, 'full resume identity')
            self.model.load_state_dict(packet['model']); self.optimizer.load_state_dict(packet['optimizer'])
            self.step_n = packet['step']; self.losses = copy.deepcopy(packet['losses']); self.curve = copy.deepcopy(packet['curve'])
            self.best, self.selected, self.gradient_audit = packet['best'], packet['selected'], packet['gradient_audit']
            for module_name, module in self.model.named_modules(): module.training = packet['model_modes'][module_name]
            self.sampler.set_state(packet['sampler']); self.augmentation.set_state(packet['augmentation']); self.auxiliary.set_state(packet['auxiliary'])
            torch.set_rng_state(packet['torch_rng']); torch.cuda.set_rng_state_all(packet['cuda_rng'])
            np.random.set_state(packet['numpy_rng']); random.setstate(packet['python_rng'])

    def step(self, inspect=False):
        torch = self.torch
        started = time.perf_counter()
        self.rt.attempt(self.name, self.step_n + 1)
        self.model.train()
        x, labels, permutation, lam = self.data.batch(self.arm, self.sampler, self.augmentation)
        self.rt.claim('training_forward_windows', len(x))
        loss = self.FactualNet.losses(None, self.model(x), labels, None, permutation, lam)
        require(all(torch.isfinite(v) for v in loss.values()), 'finite matched two-view loss')
        self.optimizer.zero_grad(set_to_none=True); loss['total'].backward()
        if inspect:
            grads = {k: v.grad for k, v in self.model.named_parameters()}
            require(all(v is not None and torch.isfinite(v).all() for v in grads.values()), 'actual finite gradients on all parameters')
            norm = float(sum(v.square().sum() for v in grads.values()))
            require(norm > 1e-24, 'actual nonzero gradient norm')
            self.gradient_audit = dict(squared_norm=norm, sha256=tensor_sha(grads))
        self.optimizer.step(); self.step_n += 1; self.rt.completed(self.name, self.step_n)
        values = {k: float(v.detach()) for k, v in loss.items()}
        if self.step_n % 100 == 0: self.losses.append(dict(step=self.step_n, **values))
        if self.name.startswith('P0/'):
            durations = self.rt.get('P0_UPDATE_TIMES') or []
            durations.append(time.perf_counter() - started)
            with self.rt.db: self.rt.set('P0_UPDATE_TIMES', durations)
        return values

    def packet(self):
        torch, np = self.torch, self.np
        return cpu(dict(fingerprint=self.rt.lock['fingerprint'], case_key=self.case['key'], chain=self.name, arm=self.arm,
            step=self.step_n, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            model_modes={k: v.training for k, v in self.model.named_modules()},
            sampler=self.sampler.get_state(), augmentation=self.augmentation.get_state(), auxiliary=self.auxiliary.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), losses=self.losses, curve=self.curve, best=self.best, selected=self.selected,
            gradient_audit=self.gradient_audit, initial_encoder=self.initial_encoder,
            AMP_GradScaler='not_applicable_AMP_disabled', scheduler=None), torch, np)
