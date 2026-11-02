"""Channel-wise residual de-conditioning on Raw-ResNet.

R0: L1b ERM. R1: explicit physical c. R1i: R1 + same-y different-c invariance.
R1a: order-track then R1. R2/R3: continuous inferred z_d.
z_d is never a domain id / one-hot / domain-logits. Domain aux (R3) is train-only.
H_f is the only classification path when residualizing. No domain classifier.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import InputProj, SharedResNet

# R1* share explicit-c affine residualize. R2c infers z_d but is trained with c.
RESID_ARCH = {
    "R0": "R0",
    "R1": "R1", "R1i": "R1", "R1a": "R1", "R1ai": "R1",
    "R1c": "R1", "R1m": "R1", "R1h": "R1",
    "R1x": "R1", "R1d": "R1", "R1e": "R1",
    "R1r": "R1", "R1p": "R1", "R1q": "R1", "R1t": "R1",
    "R1s": "R1", "R1k": "R1", "R1f": "R1", "R1l": "R1", "R1g": "R1",
    "R1o": "R1", "R1z": "R1", "R1w": "R1", "R1y": "R1",
    "R1u": "R1", "R1v": "R1", "R1j": "R1",
    "R2": "R2", "R2c": "R2", "R3": "R3",
    # R3 module ablations (same destyle net; training schedule / losses differ).
    "R3n": "R3", "R3h": "R3", "R3a": "R3", "R3j": "R3",
}


def residual_invariance_loss(feat, y, c, c_eps=0.05, normalize=True):
    """Align R for same fault y and distinct physical c. Skip same-c pairs.

    Pairwise MSE on (optionally L2-normalized) features. Zero if no valid pair.
    Not a domain classifier / GRL term.
    """
    z = F.normalize(feat, dim=-1, eps=1e-6) if normalize else feat
    same_y = y.unsqueeze(0).eq(y.unsqueeze(1))
    dc = torch.cdist(c.float(), c.float(), p=2)
    mask = same_y & (dc > float(c_eps))
    mask.fill_diagonal_(False)
    if not bool(mask.any()):
        return z.new_zeros(())
    mse = (z.unsqueeze(0) - z.unsqueeze(1)).pow(2).mean(dim=-1)
    return mse[mask].mean()


def same_c_diff_domain_align(feat, y, c, d, c_eps=0.05, normalize=True):
    """Align R for same y and same physical c across source domains.

    Intended for sensor U (CWRU DE/FE at the same load). Skip same-domain pairs.
    Not a domain classifier / GRL term.
    """
    z = F.normalize(feat, dim=-1, eps=1e-6) if normalize else feat
    same_y = y.unsqueeze(0).eq(y.unsqueeze(1))
    dc = torch.cdist(c.float(), c.float(), p=2)
    same_c = dc <= float(c_eps)
    diff_d = d.unsqueeze(0).ne(d.unsqueeze(1))
    mask = same_y & same_c & diff_d
    mask.fill_diagonal_(False)
    if not bool(mask.any()):
        return z.new_zeros(())
    mse = (z.unsqueeze(0) - z.unsqueeze(1)).pow(2).mean(dim=-1)
    return mse[mask].mean()


def complement_uniform_loss(logits):
    """CE to uniform: baseline / non-fault path must not classify Y."""
    logp = F.log_softmax(logits, dim=-1)
    return -logp.mean()


def hsic_linear(z, c):
    """Linear-kernel HSIC. Zero iff centered z and c are uncorrelated."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    zc = z - z.mean(dim=0, keepdim=True)
    cc = c.float() - c.float().mean(dim=0, keepdim=True)
    n = float(z.shape[0])
    return (zc.T @ cc).pow(2).sum() / (n * n)


def mmd_linear_same_y(z, y, c, c_eps=0.05):
    """Class-conditional linear MMD of z across a median split of c[:, 0]."""
    if z.shape[0] < 4 or c is None:
        return z.new_zeros(())
    losses = []
    key = c.float()[:, 0]
    for cls in y.unique():
        m = y.eq(cls)
        if int(m.sum()) < 4:
            continue
        zk, ck = z[m], key[m]
        med = ck.median()
        a, b = zk[ck <= med], zk[ck > med]
        if a.shape[0] < 1 or b.shape[0] < 1:
            continue
        if float((ck.max() - ck.min())) <= float(c_eps):
            continue
        losses.append((a.mean(dim=0) - b.mean(dim=0)).pow(2).sum())
    if not losses:
        return z.new_zeros(())
    return torch.stack(losses).mean()


def irm_penalty(logits, y, d):
    """IRM-v1 dummy-classifier penalty across source domains. Train-only."""
    if logits.shape[0] < 2:
        return logits.new_zeros(())
    logits_f = logits.float()
    scale = torch.ones((), device=logits.device, dtype=logits_f.dtype, requires_grad=True)
    pens = []
    for e in d.unique():
        m = d.eq(e)
        if int(m.sum()) < 2:
            continue
        ce = F.cross_entropy(logits_f[m] * scale, y[m])
        g, = torch.autograd.grad(ce, scale, create_graph=True)
        pens.append(g.pow(2))
    if not pens:
        return logits.new_zeros(())
    return torch.stack(pens).mean().to(dtype=logits.dtype)


def c_balance_weights(c):
    """Inverse kernel density of c. Mean weight is 1. Train-only; not a mask."""
    if c is None or c.shape[0] < 2:
        return None
    x = c.float()
    d = torch.cdist(x, x, p=2)
    med = d.median().clamp_min(1e-3)
    dens = torch.exp(-(d / med).pow(2)).mean(dim=1).clamp_min(1e-4)
    w = 1.0 / dens
    return w * (float(w.numel()) / w.sum().clamp_min(1e-8))


def pinball(pred, target, tau):
    """Quantile (pinball) loss. tau in (0, 1)."""
    u = target - pred
    t = float(tau)
    return torch.maximum(u * t, u * (t - 1.0)).mean()


def mixstyle_channel(h, c=None, p=0.5, alpha=0.1, c_eps=0.05):
    """Train-time MixStyle on (B,C,T). Prefer shuffled pairs with different c."""
    if h.shape[0] < 2:
        return h
    if float(torch.rand((), device=h.device)) > float(p):
        return h
    mu = h.mean(dim=-1, keepdim=True)
    sd = h.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    hn = (h - mu) / sd
    b = h.shape[0]
    idx = torch.randperm(b, device=h.device)
    if c is not None:
        dc = (c.float() - c.float()[idx]).norm(dim=-1)
        same = dc <= float(c_eps)
        if bool(same.any()):
            idx2 = torch.randperm(b, device=h.device)
            idx = torch.where(same, idx2, idx)
    lam = torch.distributions.Beta(float(alpha), float(alpha)).sample((b, 1, 1)).to(h.device)
    mu_m = lam * mu + (1.0 - lam) * mu[idx]
    sd_m = lam * sd + (1.0 - lam) * sd[idx]
    return hn * sd_m + mu_m


def rpm_from_c(c):
    """Decode rpm from protocol c. CWRU 2-d rpm_norm; PU 3-d last slot."""
    if c.shape[-1] == 2:
        return 1720.0 + c[:, 1].float() * (1797.0 - 1720.0)
    return c[:, 2].float() * 1500.0


def shaft_comb_notch(x, rpm, fs, n_harm=12):
    """Notch shaft harmonics k*rpm/60. Dictionary, not a learned Hz mask."""
    wave = x if x.dim() == 2 else x.squeeze(1)
    n = wave.shape[-1]
    spec = torch.fft.rfft(wave, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / float(fs), device=wave.device)
    f0 = (rpm.float() / 60.0).clamp_min(1.0).unsqueeze(-1)
    df = float(fs) / float(n)
    width = max(2.0 * df, 2.0)
    mask = torch.ones(spec.shape, device=wave.device, dtype=wave.dtype)
    for k in range(1, int(n_harm) + 1):
        dist = (freqs.unsqueeze(0) - float(k) * f0).abs()
        mask = mask * (dist > width).to(dtype=wave.dtype)
    out = torch.fft.irfft(spec * mask, n=n, dim=-1).to(dtype=wave.dtype)
    return out if x.dim() == 2 else out.unsqueeze(1)


def shaft_cepstrum_lifter(x, rpm, fs, n_rah=4, half_win=12):
    """Zero shaft rahmonics in the real cepstrum; keep original phase."""
    wave = x if x.dim() == 2 else x.squeeze(1)
    n = wave.shape[-1]
    spec = torch.fft.rfft(wave, dim=-1)
    mag = spec.abs().clamp_min(1e-8)
    phase = torch.angle(spec)
    ceps = torch.fft.irfft(mag.log(), n=n, dim=-1)
    tau = (float(fs) * 60.0 / rpm.float().clamp_min(1.0)).unsqueeze(-1)
    idx = torch.arange(n, device=wave.device, dtype=ceps.dtype).unsqueeze(0)
    keep = torch.ones_like(ceps)
    hw = float(half_win)
    for k in range(1, int(n_rah) + 1):
        center = float(k) * tau
        dist = torch.minimum((idx - center).abs(), (idx - (n - center)).abs())
        keep = keep * (dist > hw).to(dtype=ceps.dtype)
    logmag = torch.fft.rfft(ceps * keep, n=n, dim=-1).real
    spec_hat = torch.exp(logmag.to(dtype=mag.dtype)) * torch.exp(1j * phase)
    out = torch.fft.irfft(spec_hat, n=n, dim=-1).to(dtype=wave.dtype)
    return out if x.dim() == 2 else out.unsqueeze(1)


def morphological_baseline_residual(x, rpm, fs):
    """Subtract a Gaussian-smoothed shaft-scale baseline. Not a learned Hz mask."""
    wave = x if x.dim() == 2 else x.squeeze(1)
    n = wave.shape[-1]
    spec = torch.fft.rfft(wave, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / float(fs), device=wave.device)
    sig_t = (0.25 * 60.0 / rpm.float().clamp_min(1.0)).unsqueeze(-1)
    g = torch.exp(-2.0 * (math.pi ** 2) * sig_t.pow(2) * freqs.unsqueeze(0).pow(2))
    low = torch.fft.irfft(spec * g.to(dtype=spec.real.dtype), n=n, dim=-1)
    out = (wave - low).to(dtype=wave.dtype)
    return out if x.dim() == 2 else out.unsqueeze(1)


def hilbert_envelope(x):
    """|analytic(x)| along time. Not an FFT-band CMI mask."""
    wave = x if x.dim() == 2 else x.squeeze(1)
    n = wave.shape[-1]
    spec = torch.fft.fft(wave, dim=-1)
    h = torch.zeros_like(spec)
    h[..., 0] = spec[..., 0]
    if n % 2 == 0:
        h[..., 1:n // 2] = 2 * spec[..., 1:n // 2]
        h[..., n // 2] = spec[..., n // 2]
    else:
        h[..., 1:(n + 1) // 2] = 2 * spec[..., 1:(n + 1) // 2]
    env = torch.fft.ifft(h, dim=-1).abs().to(dtype=wave.dtype)
    return env if x.dim() == 2 else env.unsqueeze(1)


class AffineFromCond(nn.Module):
    """mu, sigma from a continuous vector; residualize H at (B,C,1)."""

    def __init__(self, cond_dim, ch, s_clamp=2.0, eps=1e-6, mlp_hidden=0):
        super().__init__()
        hid = int(mlp_hidden)
        if hid > 0:
            self.to_mu = nn.Sequential(
                nn.Linear(cond_dim, hid), nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(hid, ch),
            )
            self.to_s = nn.Sequential(
                nn.Linear(cond_dim, hid), nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(hid, ch),
            )
        else:
            self.to_mu = nn.Linear(cond_dim, ch)
            self.to_s = nn.Linear(cond_dim, ch)
        self.s_clamp = float(s_clamp)
        self.eps = eps
        self.ch = ch

    def mu_sigma(self, z):
        mu = self.to_mu(z)
        s = self.to_s(z).clamp(-self.s_clamp, self.s_clamp)
        sigma = F.softplus(s) + self.eps
        return mu, sigma, s

    def residualize(self, h, z):
        mu, sigma, s = self.mu_sigma(z)
        hf = (h - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        return hf, mu, sigma, s

    def reconstruct(self, hf, mu, sigma):
        return hf * sigma.unsqueeze(-1) + mu.unsqueeze(-1)


class StyleEncoder(nn.Module):
    """H (B,C,T) -> continuous z_d. Not a domain classifier."""

    def __init__(self, ch, zd_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ch * 2, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, zd_dim),
        )
        self.zd_dim = zd_dim

    def forward(self, h):
        mu = h.mean(dim=-1)
        sd = h.std(dim=-1, unbiased=False) + 1e-6
        return self.net(torch.cat([mu, sd], dim=1))


class CondCoupling(nn.Module):
    """Affine coupling on H: scale/shift of the second channel half from (first-half mean, c)."""

    def __init__(self, ch, cond_dim, hid=64):
        super().__init__()
        self.n1 = ch // 2
        self.n2 = ch - self.n1
        self.net = nn.Sequential(
            nn.Linear(self.n1 + cond_dim, hid),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hid, self.n2 * 2),
        )

    def forward(self, h, c):
        x1, x2 = h[:, :self.n1], h[:, self.n1:]
        st = self.net(torch.cat([x1.mean(dim=-1), c.float()], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = 0.5 * torch.tanh(s)
        hf2 = (x2 - t.unsqueeze(-1)) * torch.exp(-s).unsqueeze(-1)
        hf = torch.cat([x1, hf2], dim=1)
        y = torch.cat([x1.mean(dim=-1), hf2.mean(dim=-1)], dim=-1)
        logdet = -s.sum(dim=-1)
        return hf, y, logdet


class CondFlow(nn.Module):
    """Two couplings with a channel flip. Not a CXDG frequency mask."""

    def __init__(self, ch, cond_dim):
        super().__init__()
        self.a = CondCoupling(ch, cond_dim)
        self.b = CondCoupling(ch, cond_dim)

    def residualize(self, h, c):
        hf, _, ld1 = self.a(h, c)
        hf = torch.flip(hf, dims=[1])
        hf, y, ld2 = self.b(hf, c)
        hf = torch.flip(hf, dims=[1])
        return hf, y, ld1 + ld2


class ResidualResNet(nn.Module):
    """Raw-ResNet + optional channel residual. Test: forward(x[, c]) -> logits from H_f."""

    def __init__(self, mode, num_classes, n_src, *, c0=32, hidden=64, n_blocks=3,
                 zf_dim=64, zd_dim=16, cond_dim=3, s_clamp=2.0, affine_mlp=0,
                 fs=64000.0, backbone="resnet", input_len=None, resid_at="conv"):
        super().__init__()
        if mode not in RESID_ARCH:
            raise ValueError(mode)
        self.level = mode
        self.mode = RESID_ARCH[mode]
        self.num_classes = num_classes
        self.n_src = n_src
        self.zf_dim = zf_dim
        self.zd_dim = zd_dim
        self.hidden = hidden
        self.cond_dim = int(cond_dim)
        self.use_wavelet = False
        self.use_mixstyle = mode == "R1x"
        self.use_envelope = mode == "R1e"
        self.use_proto = mode == "R1p"
        self.use_quantile = mode == "R1q"
        self.use_twoscale = mode == "R1t"
        self.use_split = mode in ("R1s", "R1k")
        self.use_split_concat = mode == "R1s"
        self.use_split_add = mode == "R1k"
        self.use_flow = mode == "R1f"
        self.use_loadcal = mode == "R1l"
        self.use_cf = mode == "R1g"
        self.use_comb = mode == "R1o"
        self.use_ceps = mode == "R1z"
        self.use_cate = mode == "R1y"
        self.use_morph = mode == "R1u"
        self.fs = float(fs)
        self.uses_c = self.mode == "R1"
        self.backbone_kind = backbone
        self.resid_at = resid_at or "conv"
        if self.resid_at not in ("conv", "fc"):
            raise ValueError(self.resid_at)
        if self.resid_at == "fc" and backbone != "table3":
            raise ValueError("resid_at=fc needs table3 backbone")
        if backbone == "table3":
            from wfdg.mocd_model import EncoderGTable3
            if input_len is None:
                raise ValueError("table3 backbone needs input_len")
            self.t3 = EncoderGTable3(d=int(input_len))
            self.proj = None
            self.body = None
            hidden = 256 if self.resid_at == "fc" else 32
            self.hidden = hidden
        else:
            self.t3 = None
            self.proj = InputProj(1, c0)
            self.body = SharedResNet(c0=c0, hidden=hidden, n_blocks=n_blocks)
        if backbone == "table3":
            zf_in = 256
        else:
            zf_in = hidden * 2 if self.use_twoscale else hidden
        self.to_zf = nn.Linear(zf_in, zf_dim)
        self.clf = nn.Linear(zf_dim, num_classes)
        self.to_cate = nn.Linear(self.cond_dim, zf_dim) if self.use_cate else None
        if self.use_proto:
            self.register_buffer("healthy_proto", torch.zeros(zf_dim))
            self.register_buffer("healthy_proto_ready", torch.zeros(()))
        if self.use_cf:
            self.register_buffer("c_ref", torch.zeros(self.cond_dim))
            self.register_buffer("c_ref_ready", torch.zeros(()))
        mlp = int(affine_mlp) if mode == "R1m" else 0
        if self.mode == "R1":
            self.style = None
            self.affine = None if self.use_flow else AffineFromCond(
                cond_dim, hidden, s_clamp=s_clamp, mlp_hidden=mlp)
            self.flow = CondFlow(hidden, cond_dim) if self.use_flow else None
            self.domain_aux = None
            self.to_c = None
        elif self.mode in ("R2", "R3"):
            self.style = StyleEncoder(hidden, zd_dim)
            self.affine = AffineFromCond(zd_dim, hidden, s_clamp=s_clamp)
            self.flow = None
            self.domain_aux = nn.Linear(zd_dim, n_src) if self.mode == "R3" else None
            self.to_c = nn.Linear(zd_dim, self.cond_dim) if mode == "R2c" else None
        else:
            self.style = None
            self.affine = None
            self.flow = None
            self.domain_aux = None
            self.to_c = None
        if self.use_quantile:
            self.to_qlo = nn.Linear(self.cond_dim, hidden)
            self.to_qhi = nn.Linear(self.cond_dim, hidden)
        else:
            self.to_qlo = None
            self.to_qhi = None
        if self.use_split:
            self.to_zn = nn.Linear(self.cond_dim, self.zd_dim)
            self.from_zn = nn.Linear(self.zd_dim, self.cond_dim)
            if self.use_split_concat:
                self.dec_both = nn.Linear(zf_dim + self.zd_dim, hidden)
                self.dec_n = None
                self.dec_y = None
            else:
                self.dec_both = None
                self.dec_n = nn.Linear(self.zd_dim, hidden)
                self.dec_y = nn.Linear(zf_dim, hidden)
        else:
            self.to_zn = None
            self.from_zn = None
            self.dec_both = None
            self.dec_n = None
            self.dec_y = None

    def freeze_residual_operator(self):
        """Stage 2: keep μ,σ (and style) from healthy / stage-1 fit."""
        for mod in (self.style, self.affine, getattr(self, "flow", None),
                    getattr(self, "to_qlo", None),
                    getattr(self, "to_qhi", None), getattr(self, "to_zn", None),
                    getattr(self, "from_zn", None), getattr(self, "dec_n", None),
                    getattr(self, "to_cate", None)):
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False
            mod.eval()

    def note_healthy_zf(self, zf, momentum=0.9):
        """EMA of healthy z_f. Stage-1 only; not a domain id."""
        if not self.use_proto:
            return
        with torch.no_grad():
            m = zf.mean(dim=0)
            if float(self.healthy_proto_ready) < 0.5:
                self.healthy_proto.copy_(m)
                self.healthy_proto_ready.fill_(1.0)
            else:
                self.healthy_proto.mul_(float(momentum)).add_(m, alpha=1.0 - float(momentum))

    def note_c_ref(self, c, momentum=0.9):
        """EMA of source c. Stage-1 only; reference for do(c=c_ref)."""
        if not self.use_cf or c is None:
            return
        with torch.no_grad():
            m = c.float().mean(dim=0)
            if float(self.c_ref_ready) < 0.5:
                self.c_ref.copy_(m)
                self.c_ref_ready.fill_(1.0)
            else:
                self.c_ref.mul_(float(momentum)).add_(m, alpha=1.0 - float(momentum))

    def _backbone(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.t3 is not None:
            if self.resid_at == "fc":
                return self.t3.fc(self.t3.net(x)).unsqueeze(-1)
            return self.t3.net(x)
        return self.body(self.proj(x))

    def _pool_zf(self, h, mu=None):
        """h: (B,C,T). mu: (B,C) for logits_mu when two-scale."""
        if self.t3 is not None:
            if self.resid_at == "fc":
                return self.to_zf(h.squeeze(-1))
            return self.to_zf(self.t3.fc(h))
        g = h.mean(dim=-1)
        if not self.use_twoscale:
            return self.to_zf(g)
        loc = h[..., h.shape[-1] // 2:].mean(dim=-1)
        return self.to_zf(torch.cat([g, loc], dim=-1))

    def encode_from_wave(self, x, c=None):
        if self.use_envelope:
            x = hilbert_envelope(x)
        if self.use_comb or self.use_ceps or self.use_morph:
            if c is None:
                raise RuntimeError("shaft dictionary requires c for rpm")
            rpm = rpm_from_c(c)
            if self.use_comb:
                x = shaft_comb_notch(x, rpm, self.fs)
            elif self.use_ceps:
                x = shaft_cepstrum_lifter(x, rpm, self.fs)
            else:
                x = morphological_baseline_residual(x, rpm, self.fs)
        if self.use_loadcal:
            if c is None:
                raise RuntimeError("R1l requires c to scale energy by load")
            wave = x if x.dim() == 2 else x.squeeze(1)
            scale = (c[:, 0].abs() + 1.0).sqrt().unsqueeze(-1)
            wave = wave / scale
            x = wave if x.dim() == 2 else wave.unsqueeze(1)
        h = self._backbone(x)
        if self.use_mixstyle and self.training:
            h = mixstyle_channel(h, c=c if self.uses_c else None)
        out = {"H": h, "hf": None, "zd": None, "mu": None, "sigma": None,
               "s_pre": None, "H_hat": None, "domain_logits": None}
        if self.mode == "R0":
            zf = self._pool_zf(h)
            out.update(zf=zf, logits=self.clf(zf), hf=h)
            return out
        if self.mode == "R1":
            if c is None:
                raise RuntimeError("R1 requires explicit c at train and test")
            z = c
        else:
            if c is not None:
                raise RuntimeError(f"{self.mode} must not take c; z_d is inferred")
            z = self.style(h)
            if z.shape[-1] == self.n_src:
                raise RuntimeError("z_d collapsed to domain logits")
        if self.use_flow:
            hf, y, logdet = self.flow.residualize(h, z)
            zf = self._pool_zf(hf)
            out.update(hf=hf, zf=zf, logits=self.clf(zf), zd=z,
                       flow_z=y, logdet=logdet)
            return out
        hf, mu, sigma, s = self.affine.residualize(h, z)
        feat = hf
        if self.use_cf:
            cref = self.c_ref if float(self.c_ref_ready) > 0.5 else z.float().mean(dim=0)
            mu_r, sigma_r, _ = self.affine.mu_sigma(cref.expand(h.shape[0], -1))
            feat = self.affine.reconstruct(hf, mu_r, sigma_r)
        zf = self._pool_zf(feat)
        zf_c = None
        if self.use_cate:
            zf_c = self.to_cate(z.float())
            zf = zf - zf_c
        if self.t3 is not None:
            logits_mu = None
        else:
            mu_in = torch.cat([mu, mu], dim=-1) if self.use_twoscale else mu
            logits_mu = self.clf(self.to_zf(mu_in))
        out.update(
            hf=hf, zf=zf, logits=self.clf(zf), zd=z,
            mu=mu, sigma=sigma, s_pre=s, logits_mu=logits_mu,
            H_hat=self.affine.reconstruct(hf, mu, sigma),
            c_hat=self.to_c(z) if self.to_c is not None else None,
            q_lo=self.to_qlo(z) if self.to_qlo is not None else None,
            q_hi=self.to_qhi(z) if self.to_qhi is not None else None,
            zn=None, zn_c=None, h_dec=None, zf_c=zf_c,
        )
        if self.use_split:
            zn = self.to_zn(z)
            if self.use_split_concat:
                h_dec = self.dec_both(torch.cat([zf, zn], dim=-1))
            else:
                h_dec = self.dec_n(zn) + self.dec_y(zf)
            out["zn"] = zn
            out["zn_c"] = self.from_zn(zn)
            out["h_dec"] = h_dec
        if self.domain_aux is not None:
            out["domain_logits"] = self.domain_aux(z)
        return out

    def forward(self, x, c=None):
        return self.encode_from_wave(x, c)["logits"]


def build_resid(level, num_classes, n_src, cfg, cond_dim=None,
                backbone="resnet", input_len=None, resid_at="conv"):
    r = cfg["resnet"]
    rc = cfg["residual"]
    cd = int(cond_dim if cond_dim is not None else rc["cond_dim"])
    mlp = int(rc.get("affine_mlp", 0) or 0)
    if level == "R1m" and mlp <= 0:
        mlp = 64
    fs = float(rc.get("fs") or 64000.0)
    return ResidualResNet(
        level, num_classes, n_src,
        c0=r["c0"], hidden=r["hidden"], n_blocks=r["n_blocks"],
        zf_dim=r["zf_dim"], zd_dim=rc["zd_dim"], cond_dim=cd,
        s_clamp=rc["s_clamp"], affine_mlp=mlp, fs=fs,
        backbone=backbone, input_len=input_len, resid_at=resid_at,
    )
