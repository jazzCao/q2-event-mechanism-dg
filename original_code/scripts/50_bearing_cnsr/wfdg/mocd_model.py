"""MOCD-DG dual CNN. Test path is G_F then H_F.

Table 3's 8-16-32 CNN does not transfer on this 13-way split (ERM task A ~57%
vs printed 97.59). EncoderG is the R0 SharedResNet mapped to the paper's 256-d
interface. CE-only R0 on the same 60/30 cells hits 97.69% on task A.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wfdg.model import InputProj, SharedResNet


class EncoderGTable3(nn.Module):
    """Song Table 3 G: Conv 8-16-32, flatten 4d, two linears to 256."""

    def __init__(self, d=2048):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 33, stride=1, padding=16),
            nn.BatchNorm1d(8), nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 16, 17, stride=1, padding=8),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (d // 8), 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.fc(self.net(x))


class EncoderG(nn.Module):
    def __init__(self, d=2048):
        super().__init__()
        self.d = d
        self.proj = InputProj(1, 32)
        self.body = SharedResNet(32, 64, 3)
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.body(self.proj(x)).mean(-1)
        return self.fc(h)


def make_encoder(d=2048, arch="resnet"):
    if arch == "table3":
        return EncoderGTable3(d)
    if arch != "resnet":
        raise ValueError(arch)
    return EncoderG(d)


class ProjHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128), nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, z):
        return self.net(z)


class ClfHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, n_out),
        )

    def forward(self, z):
        return self.net(z)


class DecoderD(nn.Module):
    def __init__(self, d=2048):
        super().__init__()
        self.d = d
        self.fc = nn.Sequential(
            nn.Linear(512, 32 * (d // 8)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, 8, 17, padding=8),
            nn.BatchNorm1d(8), nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(8, 1, 33, padding=16),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, zf, zd):
        h = self.fc(torch.cat([zf, zd], dim=1)).view(zf.size(0), 32, self.d // 8)
        y = self.up(h)
        return y.squeeze(1)


class MOCD(nn.Module):
    def __init__(self, n_cls, n_dom, d=2048, arch="resnet"):
        super().__init__()
        self.gf = make_encoder(d, arch)
        self.gd = make_encoder(d, arch)
        self.pf = ProjHead()
        self.pd = ProjHead()
        self.hf = ClfHead(n_cls)
        self.hd = ClfHead(n_dom)
        self.dec = DecoderD(d)

    def logits_fault(self, x):
        return self.hf(self.gf(x))


def infonce_multi(q, pos, neg, tau=0.2):
    """q (B,D), pos (B,P,D), neg (B,N,D) -> mean InfoNCE."""
    q = F.normalize(q, dim=-1)
    pos = F.normalize(pos, dim=-1)
    neg = F.normalize(neg, dim=-1)
    sp = (q.unsqueeze(1) * pos).sum(-1) / tau
    sn = (q.unsqueeze(1) * neg).sum(-1) / tau
    num = torch.logsumexp(sp, dim=1)
    den = torch.logsumexp(torch.cat([sp, sn], dim=1), dim=1)
    return -(num - den).mean()


def barlow_pair(z, diag=1.0):
    z = z - z.mean(0, keepdim=True)
    std = z.std(0, keepdim=True).clamp_min(1e-6)
    z = z / std
    b = z.size(0)
    c = (z.T @ z) / max(b, 1)
    on = (c.diag() - 1.0).pow(2).sum()
    off = (c - torch.diag(c.diag())).pow(2).sum()
    return off + float(diag) * on


def softmax_weights(losses):
    stacked = torch.stack(losses)
    w = torch.softmax(-stacked.detach(), dim=0)
    return w, (w * stacked).sum()
