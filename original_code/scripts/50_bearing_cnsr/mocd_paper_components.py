"""Isolated MOCD equation components; no training runner or legacy replacement.

Eq. 5/6 anchor selection remains explicit because the paper's row average and
historical all-window symmetric average are different interpretations.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def supervised_contrast(z, labels, tau=.2):
    """Eq.13: average log probability over positives, then valid anchors."""
    if z.ndim != 2 or labels.shape != (len(z),) or tau <= 0:
        raise ValueError("invalid contrast shape/temperature")
    sim = F.normalize(z, dim=1) @ F.normalize(z, dim=1).T / tau
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    count = positive.sum(1)
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError("Eq13 has no valid positive anchors")
    denominator = torch.logsumexp(sim.masked_fill(diagonal, -torch.inf), dim=1)
    logp = sim - denominator[:, None]
    positive_logp = torch.where(positive, logp, torch.zeros_like(logp)).sum(1)
    return -(positive_logp[valid] / count[valid]).mean()


def causal_consistency(zf, zd, y, d, tau=.2, alpha=.5):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha")
    return alpha * supervised_contrast(zf, y, tau) + (1-alpha) * supervised_contrast(zd, d, tau)


def redundancy(zf, zd, diagonal_weight=1.):
    """Eq14-17 literally: raw Gram, off-diagonal mean, diagonal sum."""
    if zf.ndim != 2 or zf.shape != zd.shape or len(zf) == 0 or zf.shape[1] < 2:
        raise ValueError("invalid redundancy dimensions")
    b, dim = zf.shape

    def intra(z):
        gram = z.T @ z / b
        off = gram - torch.diag(gram.diag())
        return off.square().sum() / (dim * (dim-1)) + diagonal_weight * (gram.diag()-1).square().sum()

    return .1 * intra(zf) + .1 * intra(zd) + .8 * (zf.T @ zd / b).square().sum()


def matching_contrast(z, y, d, branch, anchor_mask, tau=.2):
    """Eq5/6 sum-positive form; caller declares row vs symmetric anchors."""
    if branch not in ("f", "d") or tau <= 0 or anchor_mask.shape != (len(z),):
        raise ValueError("matching arguments")
    same_y = y[:, None].eq(y[None, :])
    same_d = d[:, None].eq(d[None, :])
    pos = same_y & ~same_d if branch == "f" else same_d & ~same_y
    neg = ~same_y if branch == "f" else ~same_d
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos, neg = pos & ~diagonal, neg & ~diagonal
    valid = anchor_mask.bool() & pos.any(1) & neg.any(1)
    if not bool(valid.any()):
        raise ValueError("matching has no valid anchors")
    q = F.normalize(z, dim=1)
    sim = q @ q.T / tau
    numerator = torch.logsumexp(sim.masked_fill(~pos, -torch.inf)[valid], dim=1)
    denominator = torch.logsumexp(sim.masked_fill(~(pos | neg), -torch.inf)[valid], dim=1)
    return (denominator - numerator).mean()


def weighted_total(losses, detach_weights=False):
    """Eq20 literal gradient by default; stop-gradient is explicit sensitivity."""
    if not losses:
        raise ValueError("empty losses")
    values = torch.stack(tuple(losses))
    weights = torch.softmax(-(values.detach() if detach_weights else values), dim=0)
    return weights, (weights * values).sum()


def strict_matching_indices(labels_by_domain, seed=0):
    """Sec3.2.1 full reference rows, real (domain, local index) provenance.

    Never invent a missing domain/class cell. Row arrays are (rows, columns, 2)
    and store actual domain then local sample index; reference column is first.
    """
    labels = [np.asarray(y, dtype=np.int64) for y in labels_by_domain]
    if len(labels) < 2 or any(y.ndim != 1 or not len(y) for y in labels):
        raise ValueError("need nonempty source label arrays")
    classes = sorted({int(c) for y in labels for c in y})
    by = [{c: np.flatnonzero(y == c) for c in classes} for y in labels]
    absent = [(d, c) for d in range(len(labels)) for c in classes if not len(by[d][c])]
    if absent:
        raise ValueError(f"missing real domain/class cells: {absent}")
    rng = np.random.RandomState(seed)
    fault, domain, fault_reference, domain_reference = {}, {}, {}, {}
    for c in classes:
        ref = max(range(len(labels)), key=lambda d: (len(by[d][c]), -d))
        order = [ref] + [d for d in range(len(labels)) if d != ref]
        rows = []
        for i in by[ref][c]:
            rows.append([(ref, int(i))] + [(d, int(rng.choice(by[d][c]))) for d in order[1:]])
        fault[c] = np.asarray(rows, dtype=np.int64)
        fault_reference[c] = ref
    for d in range(len(labels)):
        ref = max(classes, key=lambda c: (len(by[d][c]), -c))
        order = [ref] + [c for c in classes if c != ref]
        rows = []
        for i in by[d][ref]:
            rows.append([(d, int(i))] + [(d, int(rng.choice(by[d][c]))) for c in order[1:]])
        domain[d] = np.asarray(rows, dtype=np.int64)
        domain_reference[d] = ref
    return {"fault": fault, "domain": domain, "fault_reference": fault_reference,
            "domain_reference": domain_reference, "rng_state": rng.get_state()}
