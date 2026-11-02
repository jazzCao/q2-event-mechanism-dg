"""Shared ResNet body + CWFDG. L1b and L2 share the ResNet body class.

Raw:  x -> proj_raw(1->C0) -> SharedResNet
Wave: x -> Morlet(K) -> proj_wav(K->C0) -> SharedResNet  (same SharedResNet class)

z_d comes from stats(H), never from H_f.
encode_from_wave / encode_from_A: counterfactual maps skip Morlet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .morlet import LearnableComplexMorlet1d


class GNAct(nn.Module):
    def __init__(self, ch):
        super().__init__()
        g = max(1, min(8, ch // 4 if ch >= 8 else 1))
        while ch % g != 0:
            g -= 1
        self.gn = nn.GroupNorm(g, ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.gn(x))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.n1 = GNAct(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(max(1, min(8, ch // 4 if ch >= 8 else 1)), ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h = self.n1(self.conv1(x))
        h = self.n2(self.conv2(h))
        return self.act(x + h)


class SharedResNet(nn.Module):
    """Body after C0 projection. Identical for L1b (raw) and L2 (wavelet)."""

    def __init__(self, c0=32, hidden=64, n_blocks=3):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(c0, hidden, 5, stride=2, padding=2),
            GNAct(hidden),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.out_ch = hidden

    def forward(self, x):
        return self.blocks(self.down(x))


class InputProj(nn.Module):
    def __init__(self, in_ch, c0=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, c0, 7, stride=2, padding=3),
            GNAct(c0),
        )

    def forward(self, x):
        return self.net(x)


class FiLMDecoder(nn.Module):
    """Reconstruct log-energy A from H_f + FiLM(z_d). Does not take H_f into style."""

    def __init__(self, h_ch, zd_dim, out_ch):
        super().__init__()
        self.to_gb = nn.Linear(zd_dim, h_ch * 2)
        self.up = nn.Sequential(
            nn.Conv1d(h_ch, h_ch, 3, padding=1),
            GNAct(h_ch),
            nn.Conv1d(h_ch, out_ch, 3, padding=1),
        )

    def forward(self, hf, zd, t_out):
        gb = self.to_gb(zd)
        gamma, beta = gb.chunk(2, dim=1)
        x = hf * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        x = F.interpolate(x, size=t_out, mode="linear", align_corners=False)
        return self.up(x)


class CNNDeepWrapper(nn.Module):
    """CNN_Deep with either the historical linear head or CCDG's MLP_Deep."""

    def __init__(self, input_len, num_classes, hidden=32, classifier="linear"):
        super().__init__()
        from common.backbone import CNN_Deep, MLP_Deep
        self.feat = CNN_Deep(in_channels=1, input_len=input_len, hidden_dim=hidden)
        if classifier == "mlp":
            self.clf = MLP_Deep(hidden, num_classes)
        elif classifier == "linear":
            self.clf = nn.Linear(hidden, num_classes)
        else:
            raise ValueError(f"unknown classifier: {classifier}")
        self.zf_dim = hidden
        self.uses_wavelet = False

    def encode_from_wave(self, x):
        zf = self.feat(x)
        return {"zf": zf, "zd": None, "logits": self.clf(zf), "A": None, "hf": None, "H": None}

    def forward(self, x):
        return self.encode_from_wave(x)["logits"]


class CWFDG(nn.Module):
    """Wavelet-optional factorized encoder. Test: forward(x) -> logits from z_f only."""

    def __init__(self, input_len, num_classes, n_src, *, use_wavelet=True,
                 use_style=False, fs=64000.0, n_kernels=16, kernel_size=127,
                 stride=8, c0=32, hidden=64, n_blocks=3, zf_dim=64, zd_dim=16):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_style = use_style
        self.num_classes = num_classes
        self.n_src = n_src
        self.zf_dim = zf_dim
        self.zd_dim = zd_dim
        self.n_kernels = n_kernels
        if use_wavelet:
            self.wave = LearnableComplexMorlet1d(
                n_kernels=n_kernels, kernel_size=kernel_size, stride=stride, fs=fs)
            self.proj = InputProj(n_kernels, c0)
        else:
            self.wave = None
            self.proj = InputProj(1, c0)
        self.body = SharedResNet(c0=c0, hidden=hidden, n_blocks=n_blocks)
        self.content = nn.Conv1d(hidden, hidden, 1)
        self.to_zf = nn.Linear(hidden, zf_dim)
        self.clf = nn.Linear(zf_dim, num_classes)
        if use_style:
            self.to_zd = nn.Linear(hidden * 2, zd_dim)
            self.domain_head = nn.Linear(zd_dim, n_src)
            self.decoder = FiLMDecoder(hidden, zd_dim, n_kernels if use_wavelet else 1)
        else:
            self.to_zd = None
            self.domain_head = None
            self.decoder = None

    def _stats(self, h):
        mu = h.mean(dim=-1)
        sd = h.std(dim=-1, unbiased=False) + 1e-6
        return torch.cat([mu, sd], dim=1)

    def _from_H(self, h):
        hf = self.content(h)
        zf = self.to_zf(hf.mean(dim=-1))
        out = {"H": h, "hf": hf, "zf": zf, "logits": self.clf(zf), "zd": None,
               "domain_logits": None, "A_hat": None}
        if self.use_style:
            zd = self.to_zd(self._stats(h))
            out["zd"] = zd
            out["domain_logits"] = self.domain_head(zd)
        return out

    def encode_from_A(self, a):
        """Re-encode in the same space as pack['A']. Skips Morlet.
        Wavelet: a is log-energy (B,K,T). Raw L4b/L5b: a is waveform (B,1,L).
        """
        if a.dim() == 2:
            a = a.unsqueeze(1)
        h = self.body(self.proj(a))
        out = self._from_H(h)
        out["A"] = a
        if self.use_style:
            out["A_hat"] = self.decoder(out["hf"], out["zd"], a.size(-1))
        return out

    def encode_from_wave(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.use_wavelet:
            a = self.wave(x)
            h = self.body(self.proj(a))
            out = self._from_H(h)
            out["A"] = a
        else:
            h = self.body(self.proj(x))
            out = self._from_H(h)
            out["A"] = x
        if self.use_style:
            out["A_hat"] = self.decoder(out["hf"], out["zd"], out["A"].size(-1))
        return out

    def forward(self, x):
        """Test-time API: waveform only, no domain/c/metadata."""
        return self.encode_from_wave(x)["logits"]

    def decode_swap(self, hf_i, zd_j, t_out):
        if not self.use_style:
            raise RuntimeError("style swap requires use_style")
        return self.decoder(hf_i, zd_j, t_out)


def build_model(level, input_len, num_classes, n_src, fs, cfg):
    """Build one ablation. ``ccdg_exact`` uses the original MLP_Deep head."""
    w = cfg["wavelet"]
    r = cfg["resnet"]
    kw = dict(fs=fs, n_kernels=w["n_kernels"], kernel_size=w["kernel_size"],
              stride=w["stride"], c0=r["c0"], hidden=r["hidden"],
              n_blocks=r["n_blocks"], zf_dim=r["zf_dim"], zd_dim=r["zd_dim"])
    if level in ("L1", "ccdg"):
        return CNNDeepWrapper(input_len, num_classes, hidden=32, classifier="linear")
    if level in ("erm_exact", "ccdg_exact"):
        return CNNDeepWrapper(input_len, num_classes, hidden=32, classifier="mlp")
    if level == "L1b":
        return CWFDG(input_len, num_classes, n_src, use_wavelet=False, use_style=False, **kw)
    if level in ("L2", "L3a", "L3b"):
        return CWFDG(input_len, num_classes, n_src, use_wavelet=True, use_style=False, **kw)
    if level in ("L4", "L5"):
        return CWFDG(input_len, num_classes, n_src, use_wavelet=True, use_style=True, **kw)
    if level in ("L4b", "L5b"):
        return CWFDG(input_len, num_classes, n_src, use_wavelet=False, use_style=True, **kw)
    raise ValueError(level)
