"""Actual neural implementation. Import ONLY after new scoped authorization.

Preparation compiles this source but never imports it or constructs any model.
"""
from __future__ import annotations
import copy
import os
import random
import time
from hashlib import sha256

import numpy as np
import torch
from torch import nn

from pcr_campaign_core import Runtime, STORE, canonical, fresh_path, require_store_path, sha
from pcr_physics_adapter import author_model_definition_source, local_labels_to_author
from pcr_training_design import (SourceSampler, model_initialization_seed,
                                author_probability_to_local, source_selection_score)


def setup(runtime):
    if not isinstance(runtime, Runtime):
        raise PermissionError('actual scoped Runtime first')
    runtime.neural()
    if torch.__version__ != runtime.lock['torch_version'] or np.__version__ != runtime.lock['numpy_version']:
        raise PermissionError('locked neural runtime version')
    if not torch.cuda.is_available():
        raise RuntimeError('locked actual CUDA qualification; no silent CPU fallback')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def exact(a, b, name='complete learning state'):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError(name)
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or not np.array_equal(a, b):
            raise AssertionError(name)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError(name)
        for k in a:
            exact(a[k], b[k], name + '/' + str(k))
    elif isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(name)
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, name + '/' + str(i))
    elif type(a) is not type(b) or a != b:
        raise AssertionError(name)


def rng_packet():
    return cpu(dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))


def restore_rng(packet):
    random.setstate(packet['python'])
    np.random.set_state(packet['numpy'])
    torch.set_rng_state(packet['torch'])
    torch.cuda.set_rng_state_all(packet['cuda'])


def tensor_digest(mapping):
    h = sha256()
    for name, x in mapping.items():
        x = x.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def build_model(rt, *, stripped):
    rt.neural()
    rt.claim('neural_model_constructions')
    text, _ = author_model_definition_source(strip_logging=stripped)
    namespace = {}
    # Only bound classes/imports; no original top-level CUDA detection or data module.
    exec(compile(text, '<bound-pcr-author-model>', 'exec'), namespace)
    return namespace['BaseModel'](num_classes=3).to('cuda')


def write_checkpoint(rt, packet, folder):
    rt.clock()
    path = fresh_path(folder, '.pt')
    with path.open('xb') as stream:
        torch.save(packet, stream)
        stream.flush(); os.fsync(stream.fileno())
    return dict(path=str(path), sha256=sha(path), step=packet['step'], chain=packet['chain'],
                fingerprint=rt.lock['fingerprint'])


def load_checkpoint(rt, binding):
    path = require_store_path(binding['path'])
    if binding['fingerprint'] != rt.lock['fingerprint'] or sha(path) != binding['sha256']:
        raise PermissionError('own complete checkpoint binding')
    packet = torch.load(path, map_location='cpu', weights_only=False)
    if packet['fingerprint'] != binding['fingerprint'] or packet['chain'] != binding['chain'] or packet['step'] != binding['step']:
        raise PermissionError('checkpoint payload/descriptor identity')
    return packet


class Chain:
    def __init__(self, rt, case, frontend, profile, data, name, *, stripped=True, packet=None):
        if not isinstance(rt, Runtime):
            raise PermissionError('actual scoped Runtime')
        self.rt, self.case, self.frontend, self.profile, self.data, self.name = rt, case, frontend, profile, data, name
        self.stripped = stripped
        self.case_fingerprint = canonical(case)
        rt.neural()
        seed = model_initialization_seed(case['seed'])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.model = build_model(rt, stripped=stripped)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4,
                                          weight_decay=1e-5, betas=(.9, .999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=4000, eta_min=1e-6)
        self.sampler = SourceSampler(case, rt.proposal['records'], frontend, profile)
        self.step_n = 0
        self.best = None
        self.selected_step = None
        self.curve = []
        self.last_loss = None
        self.gradient_audit = None
        self.model.train()
        if packet is not None:
            self.restore(packet)

    def tensors(self):
        fact, aux = self.sampler.batch(200)
        x = np.stack([self.data['train'][key][index] for key, index in fact])
        y = local_labels_to_author(np.array([self.rt.proposal['records'][key]['label'] for key, _ in fact], dtype=np.int64))
        if aux:
            pool = self.data['cf'] if self.profile == 'PCR_FAULT_TO_HEALTHY' else self.data['train']
            a = np.stack([pool[key][index] for key, index in aux])
        else:
            a = None
        return (torch.as_tensor(x, dtype=torch.float32, device='cuda'),
                torch.as_tensor(y, dtype=torch.long, device='cuda'),
                None if a is None else torch.as_tensor(a, dtype=torch.float32, device='cuda'))

    def step(self, *, inspect=False):
        self.rt.attempt(self.name, self.step_n + 1, self.profile)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        x, y, auxiliary = self.tensors()
        _, logits = self.model(x, anti=False)
        factual = nn.functional.cross_entropy(logits, y)
        if auxiliary is None:
            auxiliary_loss = factual.new_zeros(())
        else:
            _, auxiliary_logits = self.model(auxiliary, anti=True)
            auxiliary_loss = nn.functional.cross_entropy(auxiliary_logits, torch.zeros(len(auxiliary), dtype=torch.long, device='cuda'))
        loss = factual + auxiliary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite factual/auxiliary loss')
        loss.backward()
        if inspect:
            grads = {k: v.grad for k, v in self.model.named_parameters() if v.grad is not None}
            if not grads or not all(torch.isfinite(g).all() for g in grads.values()) or not any(torch.count_nonzero(g) for g in grads.values()):
                raise FloatingPointError('genuine finite nonzero gradient check failed')
            self.gradient_audit = dict(step=self.step_n+1, sha256=tensor_digest(grads), parameters_with_gradients=len(grads),
                                       factual_loss=float(factual.detach()), auxiliary_loss=float(auxiliary_loss.detach()))
        self.optimizer.step()
        self.scheduler.step()
        self.step_n += 1
        self.last_loss = dict(factual=float(factual.detach()), auxiliary=float(auxiliary_loss.detach()),
                              total=float(loss.detach()), learning_rate=self.optimizer.param_groups[0]['lr'])
        self.rt.ledger.completed(self.name, self.step_n)
        return self.last_loss

    def packet(self):
        return cpu(dict(fingerprint=self.rt.lock['fingerprint'], chain=self.name,
                        frontend=self.frontend, profile=self.profile, case_fingerprint=self.case_fingerprint,
                        source_fingerprint=self.data['source_fingerprint'], step=self.step_n,
                        model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        scheduler=self.scheduler.state_dict(), sampler=self.sampler.packet(), rng=rng_packet(),
                        modes={name: m.training for name, m in self.model.named_modules()},
                        transformer_hooked=self.model.feature_extractor.transformer_hooked,
                        best=self.best, selected_step=self.selected_step, curve=self.curve,
                        last_loss=self.last_loss, gradient_audit=self.gradient_audit))

    def restore(self, packet):
        expected = dict(fingerprint=self.rt.lock['fingerprint'], chain=self.name,
                        frontend=self.frontend, profile=self.profile, case_fingerprint=self.case_fingerprint,
                        source_fingerprint=self.data['source_fingerprint'])
        if any(packet[k] != v for k, v in expected.items()):
            raise PermissionError('complete learning state identity')
        self.model.load_state_dict(packet['model'], strict=True)
        self.optimizer.load_state_dict(packet['optimizer'])
        self.scheduler.load_state_dict(packet['scheduler'])
        self.sampler.restore(packet['sampler'])
        self.step_n = packet['step']
        if self.rt.ledger.cursor(self.name) != (self.step_n,) * 3:
            raise PermissionError('checkpoint does not match exact durable paid cursor')
        self.best, self.selected_step, self.curve = packet['best'], packet['selected_step'], copy.deepcopy(packet['curve'])
        self.last_loss = copy.deepcopy(packet['last_loss'])
        self.gradient_audit = copy.deepcopy(packet['gradient_audit'])
        modules = dict(self.model.named_modules())
        if set(modules) != set(packet['modes']):
            raise ValueError('module-mode state coverage')
        for name, mode in packet['modes'].items():
            modules[name].training = mode
        self.model.feature_extractor.transformer_hooked = packet['transformer_hooked']
        # MUST be last: constructing/loading models and optimizers consumes state.
        restore_rng(packet['rng'])


def predict(model, records, arrays, rt, role, *, limit=None):
    if role == 'target' and rt.phase != 'TARGET':
        raise PermissionError('target prediction before common freeze')
    if role not in ('source', 'target'):
        raise ValueError('prediction role')
    modes = {name: m.training for name, m in model.named_modules()}
    model.eval()
    chunks, labels, groups, domains = [], [], [], []
    taken = 0
    try:
        with torch.no_grad():
            for key in sorted(arrays):
                x = arrays[key]
                for start in range(0, len(x), 128):
                    size = min(128, len(x) - start)
                    if limit is not None:
                        size = min(size, limit - taken)
                    if size <= 0:
                        break
                    rt.claim('source_prediction_windows' if role == 'source' else 'target_prediction_windows', size)
                    _, logits = model(torch.as_tensor(x[start:start+size], dtype=torch.float32, device='cuda'), anti=False)
                    z = logits.detach().cpu().numpy().astype(np.float64)
                    z -= z.max(axis=1, keepdims=True)
                    probability = np.exp(z); probability /= probability.sum(axis=1, keepdims=True)
                    chunks.append(author_probability_to_local(probability))
                    labels.extend([records[key]['label']] * size)
                    groups.extend([key] * size)
                    domains.extend([records[key]['domain']] * size)
                    taken += size
                if limit is not None and taken >= limit:
                    break
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    if not chunks or limit is not None and taken != limit:
        raise ValueError('complete requested prediction support')
    return dict(prob=np.concatenate(chunks), y=np.asarray(labels, dtype=np.int64),
                g=np.asarray(groups), domain=np.asarray(domains))


def save_probability(rt, values, folder, metadata):
    path = fresh_path(folder, '.npz')
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)
        stream.flush(); os.fsync(stream.fileno())
    return dict(path=str(path), sha256=sha(path), windows=len(values['y']), **metadata)


def source_score(chain, folder):
    values = predict(chain.model, chain.rt.proposal['records'], chain.data['validation'], chain.rt, 'source')
    score = source_selection_score(values['prob'], values['y'], values['domain'], chain.case['task']['sources'])
    binding = save_probability(chain.rt, values, folder, dict(role='source_validation', step=chain.step_n, chain=chain.name))
    row = dict(step=chain.step_n, score=score, probability=binding, target_used_for_selection=False)
    if chain.best is None or score > chain.best:
        chain.best = score
        chain.selected_step = chain.step_n
    chain.curve.append(row)
    return row
