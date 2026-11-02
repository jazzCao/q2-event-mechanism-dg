"""Strict indexed matching and checkpoint interfaces; no training entrypoint.

Synthetic tests exercise these interfaces without optimizer steps. CPU packet
restoration is not proof of actual CPU/CUDA training-trajectory equivalence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random

import numpy as np
import torch

from mocd_paper_components import strict_matching_indices


def label_fingerprint(labels):
    digest = hashlib.sha256()
    for domain, values in enumerate(labels):
        y = np.asarray(values)
        if y.ndim != 1 or y.dtype.kind not in "iu":
            raise ValueError("integer one-dimensional source labels required")
        canonical = y.astype("<i8", copy=False)
        digest.update(json.dumps([domain, len(y)]).encode())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


class IndexedMatching:
    def __init__(self, labels_by_domain, seed=0):
        self.labels = [np.array(y, dtype=np.int64, copy=True) for y in labels_by_domain]
        self.label_hash = label_fingerprint(labels_by_domain)
        self.state = strict_matching_indices(self.labels, seed)
        self.rng = np.random.RandomState()
        self.rng.set_state(self.state["rng_state"])
        self.refreshes = 0
        self.validate()

    def validate(self):
        classes = sorted(set(np.concatenate(self.labels).tolist()))
        if set(self.state["fault"]) != set(classes) or set(self.state["domain"]) != set(range(len(self.labels))):
            raise ValueError("matching matrix key coverage")
        for c, rows in self.state["fault"].items():
            self._validate_rows(rows, len(self.labels))
            if any(self.labels[d][i] != c for d, i in rows.reshape(-1, 2)):
                raise ValueError("fault matrix class provenance")
            if any(len(set(row[:, 0].tolist())) != len(self.labels) for row in rows):
                raise ValueError("fault matrix repeated domains")
            ref = self.state["fault_reference"][c]
            expected_ref = max(range(len(self.labels)), key=lambda d: (np.sum(self.labels[d] == c), -d))
            expected_indices = np.flatnonzero(self.labels[ref] == c)
            if ref != expected_ref or not np.all(rows[:, 0, 0] == ref) or not np.array_equal(rows[:, 0, 1], expected_indices):
                raise ValueError("fault reference anchors altered")
        for d, rows in self.state["domain"].items():
            self._validate_rows(rows, len(classes))
            if not np.all(rows[:, :, 0] == d):
                raise ValueError("domain matrix domain provenance")
            if any(set(self.labels[d][row[:, 1]].tolist()) != set(classes) for row in rows):
                raise ValueError("domain matrix class coverage")
            ref = self.state["domain_reference"][d]
            expected_ref = max(classes, key=lambda c: (np.sum(self.labels[d] == c), -c))
            if ref != expected_ref or not np.array_equal(rows[:, 0, 1], np.flatnonzero(self.labels[d] == ref)):
                raise ValueError("domain reference anchors altered")

    def _validate_rows(self, rows, columns):
        if not isinstance(rows, np.ndarray) or rows.dtype.kind not in "iu" or rows.ndim != 3 or rows.shape[1:] != (columns, 2) or not len(rows):
            raise ValueError("invalid index matrix shape/type")
        for d, i in rows.reshape(-1, 2):
            if not 0 <= d < len(self.labels) or not 0 <= i < len(self.labels[d]):
                raise ValueError("index outside true source domain")

    def sample(self, branch, k):
        if branch not in ("f", "d") or not isinstance(k, int) or k <= 0:
            raise ValueError("matching sample arguments")
        matrices = self.state["fault" if branch == "f" else "domain"]
        selected = []
        for key in sorted(matrices):
            rows = matrices[key]
            indices = self.rng.choice(len(rows), min(k, len(rows)), replace=False)
            selected.append(rows[indices])
        rows = np.concatenate(selected)
        pairs = rows.reshape(-1, 2)
        y = np.asarray([self.labels[d][i] for d, i in pairs], dtype=np.int64)
        d = pairs[:, 0].copy()
        anchor_mask = np.zeros(len(pairs), dtype=bool)
        anchor_mask[::rows.shape[1]] = True
        return {"pairs": pairs.copy(), "y": y, "d": d, "anchor_mask": anchor_mask}

    def refresh(self, embeddings_by_domain, branch, space):
        """Full-candidate L2 maximum, stable lowest-index ties; anchors fixed.

        Caller declares projection (StageA) or encoder (StageB) space. No model
        is called here and no cap/borrowed domain is silently introduced.
        """
        if branch not in ("f", "d") or space not in ("projection", "encoder"):
            raise ValueError("refresh branch/space must be explicit")
        embeddings = [np.asarray(z, dtype=np.float64) for z in embeddings_by_domain]
        if len(embeddings) != len(self.labels):
            raise ValueError("embedding domain count")
        if any(z.ndim != 2 or len(z) != len(y) or not np.isfinite(z).all() for z, y in zip(embeddings, self.labels)) or len({z.shape[1] for z in embeddings}) != 1:
            raise ValueError("embedding shape/finite/provenance")
        self.validate()
        matrices = self.state["fault" if branch == "f" else "domain"]
        updated = {key: rows.copy() for key, rows in matrices.items()}
        for rows in updated.values():
            for row in rows:
                anchor_dom, anchor_i = row[0]
                reference = embeddings[anchor_dom][anchor_i]
                for column in range(1, len(row)):
                    target_dom, old_i = row[column]
                    target_class = self.labels[target_dom][old_i]
                    candidates = np.flatnonzero(self.labels[target_dom] == target_class)
                    distances = ((embeddings[target_dom][candidates] - reference) ** 2).sum(1)
                    winner = candidates[int(np.argmax(distances))]
                    row[column] = (target_dom, winner)
        self.state["fault" if branch == "f" else "domain"] = updated
        self.refreshes += 1
        self.state["last_refresh_space"] = space
        self.validate()

    def state_dict(self):
        self.validate()
        return {"schema": "MOCD_INDEXED_MATCHING_V1", "label_hash": self.label_hash,
                "state": copy.deepcopy(self.state), "rng": self.rng.get_state(),
                "refreshes": self.refreshes}

    def load_state_dict(self, packet):
        if packet.get("schema") != "MOCD_INDEXED_MATCHING_V1" or packet.get("label_hash") != self.label_hash:
            raise ValueError("matching schema/source label fingerprint mismatch")
        old = self.state_dict()
        try:
            self.state = copy.deepcopy(packet["state"])
            self.rng.set_state(packet["rng"])
            self.refreshes = int(packet["refreshes"])
            self.validate()
        except Exception:
            self.state = old["state"]
            self.rng.set_state(old["rng"])
            self.refreshes = old["refreshes"]
            raise


PACKET_KEYS = {"schema", "fingerprint", "model", "model_behavior", "optimizers", "scaler", "matching", "sampler",
               "rng", "progress", "selection", "profile"}


def capture_packet(model, optimizers, scaler_state, matching, sampler_state,
                   fingerprint, progress, selection, profile):
    if not fingerprint or not optimizers:
        raise ValueError("state must bind data/code/profile and optimizers")
    return copy.deepcopy({
        "schema": "MOCD_RESUME_INTERFACE_V1", "fingerprint": fingerprint,
        "model": model.state_dict(),
        "model_behavior": {"module_training": {name: module.training for name, module in model.named_modules()},
                           "parameter_requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters()}},
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "scaler": scaler_state, "matching": matching.state_dict(), "sampler": sampler_state,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None},
        "progress": progress, "selection": selection, "profile": profile,
    })


def validate_packet(packet, expected_fingerprint, expected_profile, optimizer_names):
    if not isinstance(packet, dict) or set(packet) != PACKET_KEYS:
        raise ValueError("resume packet missing or unexpected fields")
    if packet["schema"] != "MOCD_RESUME_INTERFACE_V1" or packet["fingerprint"] != expected_fingerprint or packet["profile"] != expected_profile:
        raise ValueError("resume code/data/interpretation identity mismatch")
    if set(packet["optimizers"]) != set(optimizer_names):
        raise ValueError("optimizer stage keys mismatch")
    if set(packet["rng"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("incomplete RNG packet")
    if not {"stage", "outer", "inner", "updates"} <= set(packet["progress"]):
        raise ValueError("incomplete progress")
    if packet["progress"]["stage"] not in ("A", "B") or any(not isinstance(packet["progress"][name], int) or packet["progress"][name] < 0 for name in ("outer", "inner", "updates")):
        raise ValueError("invalid stage/progress counters")
    if not {"best_step", "best_score", "best_state", "target_used_for_selection"} <= set(packet["selection"]):
        raise ValueError("incomplete source selection")
    if packet["selection"]["target_used_for_selection"] is not False:
        raise ValueError("source-only selection required")


def restore_packet(packet, model, optimizers, matching, expected_fingerprint,
                   expected_profile, scaler=None):
    validate_packet(packet, expected_fingerprint, expected_profile, optimizers)
    if packet["rng"]["torch_cuda"] is not None and not torch.cuda.is_available():
        raise ValueError("CUDA state cannot be silently restored on CPU")
    if packet["scaler"] and scaler is None:
        raise ValueError("nonempty GradScaler state requires scaler")
    # Stage into isolated objects: bad packets must not partially mutate caller.
    staged_model = copy.deepcopy(model)
    staged_model.load_state_dict(packet["model"], strict=True)
    behavior = packet["model_behavior"]
    if not isinstance(behavior, dict) or set(behavior) != {"module_training", "parameter_requires_grad"}:
        raise ValueError("model execution behavior missing")
    if set(behavior["module_training"]) != set(dict(model.named_modules())) or set(behavior["parameter_requires_grad"]) != set(dict(model.named_parameters())):
        raise ValueError("model behavior keys mismatch")
    if any(type(flag) is not bool for fields in behavior.values() for flag in fields.values()):
        raise ValueError("model behavior flags must be boolean")
    if packet["selection"]["best_state"] is not None:
        copy.deepcopy(model).load_state_dict(packet["selection"]["best_state"], strict=True)
    staged_matching = copy.deepcopy(matching)
    staged_matching.load_state_dict(packet["matching"])
    staged_optimizers = copy.deepcopy(optimizers)
    for name, opt in staged_optimizers.items():
        saved = packet["optimizers"][name]
        if len(opt.param_groups) != len(saved["param_groups"]):
            raise ValueError("optimizer parameter groups mismatch")
        for group, saved_group in zip(opt.param_groups, saved["param_groups"]):
            if len(group["params"]) != len(saved_group["params"]):
                raise ValueError("optimizer parameter count mismatch")
            for parameter, key in zip(group["params"], saved_group["params"]):
                entry = saved["state"].get(key, {})
                for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
                    if field in entry and entry[field].shape != parameter.shape:
                        raise ValueError("optimizer moment shape mismatch")
        opt.load_state_dict(packet["optimizers"][name])
    # Validate auxiliary restoration before changing caller or global RNG.
    probe_python = random.Random()
    probe_python.setstate(packet["rng"]["python"])
    probe_numpy = np.random.RandomState()
    probe_numpy.set_state(packet["rng"]["numpy"])
    probe_torch = torch.Generator(device="cpu")
    probe_torch.set_state(packet["rng"]["torch_cpu"])
    if packet["rng"]["torch_cuda"] is not None:
        if len(packet["rng"]["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA generator/device count mismatch")
        for device, state in enumerate(packet["rng"]["torch_cuda"]):
            torch.Generator(device=f"cuda:{device}").set_state(state)
    if scaler is not None:
        staged_scaler = copy.deepcopy(scaler)
        staged_scaler.load_state_dict(packet["scaler"])
    for name, opt in optimizers.items():
        opt.load_state_dict(packet["optimizers"][name])
    model.load_state_dict(packet["model"], strict=True)
    for name, module in model.named_modules():
        module.training = behavior["module_training"][name]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(behavior["parameter_requires_grad"][name])
    matching.load_state_dict(packet["matching"])
    if scaler is not None:
        scaler.load_state_dict(packet["scaler"])
    random.setstate(packet["rng"]["python"])
    np.random.set_state(packet["rng"]["numpy"])
    torch.set_rng_state(packet["rng"]["torch_cpu"])
    if packet["rng"]["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(packet["rng"]["torch_cuda"])
    return copy.deepcopy({name: packet[name] for name in ("sampler", "progress", "selection")})
