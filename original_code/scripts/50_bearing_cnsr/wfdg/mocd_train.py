"""Train MOCD-DG / ERM on pu_mocd Table 5 tasks."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.utils import set_seed
from wfdg.angular import order_track_windows
from wfdg.data import (
    PROTOCOLS, load_domain, load_or_make_split, assert_no_target,
    build_jnu_source_class_balanced_sampler,
)
from wfdg.mocd_model import (
    MOCD, ClfHead, make_encoder, infonce_multi, barlow_pair, softmax_weights,
)
from wfdg.mocd_pu import mocd_rpm, mocd_60_30_indices, source_tr_va, TASKS, WIN as MOCD_WIN

RPM_REF = 1500.0

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.environ.get("WFDG_RESULTS_DIR", os.path.join(ROOT, "results", "wfdg"))
SPLIT_DIR = os.path.join(RESULTS, "splits")

SOURCE_VAL_SELECTIONS = frozenset({
    "grouped_source_val",
    "temporal_block_source_val",
    "grouped_source_val_acc",
})


def _selects_on_source_val(selection: str) -> bool:
    return str(selection) in SOURCE_VAL_SELECTIONS


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _xy(cache, splits, d, which):
    x, y, _g = cache[d]
    idx = splits[d][0 if which == "tr" else 1]
    return x[idx], y[idx]


def _xy_all60(cache, splits, d):
    """Paper trains on all 60 source windows. The 80/20 split is only for R3 val."""
    x, y, _g = cache[d]
    idx = np.concatenate([np.asarray(splits[d][0]), np.asarray(splits[d][1])])
    return x[idx], y[idx]


def _np_fft(x, length=2048):
    spec = np.log1p(np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64), axis=-1)))
    n = spec.shape[-1]
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, int(length))
    return np.stack([np.interp(dst, src, row) for row in spec]).astype(np.float32)


def _ot_rewindow(x, y, g, rpm, rpm_ref, win=MOCD_WIN):
    """Resample each recording then recut 2048 windows. Extra vs Table 4."""
    from scipy.signal import resample
    xs, ys, gs = [], [], []
    x = np.asarray(x)
    y = np.asarray(y)
    g = np.asarray(g)
    for grp in np.unique(g):
        idx = np.where(g == grp)[0]
        sig = x[idx].reshape(-1)
        lab = int(y[idx][0])
        if abs(float(rpm) - float(rpm_ref)) >= 1e-3:
            n_mid = max(int(win), int(round(len(sig) * float(rpm) / float(rpm_ref))))
            sig = resample(sig.astype(np.float64), n_mid).astype(np.float32)
        else:
            sig = sig.astype(np.float32, copy=False)
        n_win = int(len(sig) // int(win))
        if n_win < 1:
            continue
        w = sig[: n_win * int(win)].reshape(n_win, int(win))
        xs.append(w)
        ys.append(np.full(n_win, lab, dtype=np.int64))
        gs.append(np.array([grp] * n_win))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)


def _np_envelope(x):
    from scipy.signal import hilbert
    return np.abs(hilbert(np.asarray(x, dtype=np.float64), axis=-1)).astype(np.float32)


def _np_envspec(x, length=2048):
    from scipy.signal import hilbert
    env = np.abs(hilbert(np.asarray(x, dtype=np.float64), axis=-1))
    env = env - env.mean(axis=-1, keepdims=True)
    spec = np.log1p(np.abs(np.fft.rfft(env, axis=-1)))
    n = spec.shape[-1]
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, int(length))
    return np.stack([np.interp(dst, src, row) for row in spec]).astype(np.float32)


def _np_envspec_ot(x, dom, length=2048):
    from scipy.signal import hilbert
    env = np.abs(hilbert(np.asarray(x, dtype=np.float64), axis=-1))
    env = env - env.mean(axis=-1, keepdims=True)
    spec = np.log1p(np.abs(np.fft.rfft(env, axis=-1)))
    n = spec.shape[-1]
    scale = float(mocd_rpm(dom)) / RPM_REF
    src = np.arange(n, dtype=np.float64) * scale
    dst = np.arange(n, dtype=np.float64)
    aligned = np.stack([np.interp(dst, src, row, left=0.0, right=0.0) for row in spec])
    src2 = np.linspace(0.0, 1.0, n)
    dst2 = np.linspace(0.0, 1.0, int(length))
    return np.stack([np.interp(dst2, src2, row) for row in aligned]).astype(np.float32)


def _ot_spec(x, dom):
    """Map rFFT bins onto the 1500 rpm grid, then irFFT. Avoids time-OT pad junk."""
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.rfft(x, axis=-1)
    n = spec.shape[-1]
    scale = float(mocd_rpm(dom)) / RPM_REF
    src = np.arange(n, dtype=np.float64) * scale
    dst = np.arange(n, dtype=np.float64)
    mag = np.abs(spec)
    ang = np.angle(spec)
    mag_i = np.stack([np.interp(dst, src, row, left=0.0, right=0.0) for row in mag])
    ang_i = np.stack([np.interp(dst, src, row, left=0.0, right=0.0) for row in ang])
    y = np.fft.irfft(mag_i * np.exp(1j * ang_i), n=x.shape[-1], axis=-1)
    return y.astype(np.float32)


def _preproc_windows(x, dom, kind):
    """Diagnostic speed-robust features. Not in Song Table 4; extra vs printed protocol."""
    if not kind:
        return np.asarray(x, dtype=np.float32)
    if kind not in ("env", "ot", "env_ot", "ot_env", "envspec", "envspec_ot", "ot_spec", "fft"):
        raise ValueError(kind)
    x = np.asarray(x, dtype=np.float32)
    if kind == "fft":
        return _np_fft(x)
    if kind == "envspec":
        return _np_envspec(x)
    if kind == "envspec_ot":
        return _np_envspec_ot(x, dom)
    if kind == "ot_spec":
        return _ot_spec(x, dom)
    if kind in ("env", "env_ot"):
        x = _np_envelope(x)
    if kind in ("ot", "env_ot", "ot_env"):
        x = order_track_windows(x, mocd_rpm(dom), RPM_REF, mode="resample_reflect")
    if kind == "ot_env":
        x = _np_envelope(x)
    return x


def _te(cache, split_meta, d):
    x, y, _g = cache[d]
    te = (split_meta or {}).get("te_idx", {}).get(d)
    if te is None:
        return x, y
    te = np.asarray(te)
    return x[te], y[te]


class Pool:
    def __init__(self, xs, ys, seed):
        self.xs = [np.asarray(x) for x in xs]
        self.ys = [np.asarray(y) for y in ys]
        self.rng = np.random.RandomState(seed)
        self.by = []
        for di, y in enumerate(self.ys):
            m = {}
            for c in np.unique(y):
                m[int(c)] = np.where(y == c)[0]
            self.by.append(m)

    def sample_xy(self, n):
        xs, ys, ds = [], [], []
        for _ in range(n):
            di = int(self.rng.randint(0, len(self.xs)))
            j = int(self.rng.randint(0, len(self.xs[di])))
            xs.append(self.xs[di][j])
            ys.append(self.ys[di][j])
            ds.append(di)
        x = torch.from_numpy(np.stack(xs)).float()
        y = torch.from_numpy(np.array(ys, np.int64))
        d = torch.from_numpy(np.array(ds, np.int64))
        return x, y, d

    def candidates(self, di, c):
        """Windows and indices for class `c`. Borrow another source domain if (di,c) is empty."""
        c = int(c)
        if c in self.by[di] and len(self.by[di][c]) > 0:
            return self.xs[di], self.by[di][c]
        for dj in range(len(self.by)):
            if c in self.by[dj] and len(self.by[dj][c]) > 0:
                return self.xs[dj], self.by[dj][c]
        raise KeyError(f"class {c} missing in all source domains")

    def pick(self, di, c):
        c = int(c)
        xs, idx = self.candidates(di, c)
        j = int(self.rng.randint(0, len(idx)))
        return xs[idx[j]]


def init_mf(pool, n_cls, n_dom, copies):
    """M_F[copy, class, dom] = window."""
    rows = []
    for _ in range(copies):
        grid = []
        for c in range(n_cls):
            grid.append([pool.pick(di, c) for di in range(n_dom)])
        rows.append(grid)
    return rows


def init_md(pool, n_cls, n_dom, copies):
    """M_D[copy, dom, class] = window."""
    rows = []
    for _ in range(copies):
        grid = []
        for di in range(n_dom):
            grid.append([pool.pick(di, c) for c in range(n_cls)])
        rows.append(grid)
    return rows


def mf_batch(mf, k, rng):
    n_copy, n_cls, n_dom = len(mf), len(mf[0]), len(mf[0][0])
    xs, ys, ds = [], [], []
    for c in range(n_cls):
        picks = rng.choice(n_copy, min(k, n_copy), replace=False)
        for p in picks:
            for di in range(n_dom):
                xs.append(mf[int(p)][c][di])
                ys.append(c)
                ds.append(di)
    x = torch.from_numpy(np.stack(xs)).float()
    y = torch.from_numpy(np.array(ys, np.int64))
    d = torch.from_numpy(np.array(ds, np.int64))
    return x, y, d


def md_batch(md, k, rng):
    n_copy, n_dom, n_cls = len(md), len(md[0]), len(md[0][0])
    xs, ys, ds = [], [], []
    for di in range(n_dom):
        picks = rng.choice(n_copy, min(k, n_copy), replace=False)
        for p in picks:
            for c in range(n_cls):
                xs.append(md[int(p)][di][c])
                ys.append(c)
                ds.append(di)
    x = torch.from_numpy(np.stack(xs)).float()
    y = torch.from_numpy(np.array(ys, np.int64))
    d = torch.from_numpy(np.array(ds, np.int64))
    return x, y, d


def _masked_infonce(z, pos_mask, neg_mask, tau):
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / float(tau)
    neg_inf = torch.finfo(sim.dtype).min
    pos_m = pos_mask.float()
    both = (pos_mask | neg_mask).float()
    sp = sim.masked_fill(~pos_mask, neg_inf)
    sb = sim.masked_fill(both < 0.5, neg_inf)
    ok = (pos_m.sum(1) > 0) & (both.sum(1) > pos_m.sum(1))
    if not bool(ok.any()):
        return z.new_zeros(())
    num = torch.logsumexp(sp[ok], dim=1)
    den = torch.logsumexp(sb[ok], dim=1)
    return -(num - den).mean()


def match_loss_fault(z, y, d, tau):
    same = y.unsqueeze(0).eq(y.unsqueeze(1))
    diff_d = d.unsqueeze(0).ne(d.unsqueeze(1))
    pos = same & diff_d
    neg = y.unsqueeze(0).ne(y.unsqueeze(1))
    pos.fill_diagonal_(False)
    neg.fill_diagonal_(False)
    return _masked_infonce(z, pos, neg, tau)


def match_loss_domain(z, y, d, tau):
    same = d.unsqueeze(0).eq(d.unsqueeze(1))
    diff_y = y.unsqueeze(0).ne(y.unsqueeze(1))
    pos = same & diff_y
    neg = d.unsqueeze(0).ne(d.unsqueeze(1))
    pos.fill_diagonal_(False)
    neg.fill_diagonal_(False)
    return _masked_infonce(z, pos, neg, tau)


def _cap_idx(idx, max_cand, rng):
    idx = np.asarray(idx)
    if max_cand is None or len(idx) <= int(max_cand):
        return idx
    return rng.choice(idx, int(max_cand), replace=False)


def hard_update_mf(mf, pool, model, device, max_cand=256, rng=None):
    model.eval()
    rng = rng or np.random.RandomState(0)
    n_copy, n_cls, n_dom = len(mf), len(mf[0]), len(mf[0][0])
    with torch.no_grad():
        for p in range(n_copy):
            for c in range(n_cls):
                ref = torch.from_numpy(mf[p][c][0]).float().unsqueeze(0).to(device)
                zref = model.pf(model.gf(ref))
                for di in range(1, n_dom):
                    xs_d, raw_idx = pool.candidates(di, c)
                    cand_idx = _cap_idx(raw_idx, max_cand, rng)
                    cx = torch.from_numpy(xs_d[cand_idx]).float().to(device)
                    zc = model.pf(model.gf(cx))
                    dist = (zc - zref).pow(2).sum(-1)
                    j = int(dist.argmax().item())
                    mf[p][c][di] = xs_d[cand_idx[j]]
    model.train()


def hard_update_md(md, pool, model, device, max_cand=256, rng=None):
    model.eval()
    rng = rng or np.random.RandomState(0)
    n_copy, n_dom, n_cls = len(md), len(md[0]), len(md[0][0])
    with torch.no_grad():
        for p in range(n_copy):
            for di in range(n_dom):
                ref = torch.from_numpy(md[p][di][0]).float().unsqueeze(0).to(device)
                zref = model.pd(model.gd(ref))
                for c in range(1, n_cls):
                    xs_d, raw_idx = pool.candidates(di, c)
                    cand_idx = _cap_idx(raw_idx, max_cand, rng)
                    cx = torch.from_numpy(xs_d[cand_idx]).float().to(device)
                    zc = model.pd(model.gd(cx))
                    dist = (zc - zref).pow(2).sum(-1)
                    j = int(dist.argmax().item())
                    md[p][di][c] = xs_d[cand_idx[j]]
    model.train()


def vat_pair(model, x, y, d, eps, which="f"):
    """Miyato-style feature VAT: FGSM on CE, then MSE on encoder features (Eq. 8–11)."""
    x_in = x.detach().requires_grad_(True)
    if which == "f":
        ce = F.cross_entropy(model.hf(model.gf(x_in)), y)
        enc = model.gf
    else:
        ce = F.cross_entropy(model.hd(model.gd(x_in)), d)
        enc = model.gd
    g = torch.autograd.grad(ce, x_in, retain_graph=False, create_graph=False)[0]
    x_adv = (x.detach() + float(eps) * g.sign()).detach()
    z = enc(x.detach())
    return F.mse_loss(enc(x_adv), z.detach())


@torch.no_grad()
def eval_acc(model, x, y, device, bs=256):
    model.eval()
    n = len(x)
    pred = []
    for i in range(0, n, bs):
        xb = torch.from_numpy(x[i:i + bs]).float().to(device)
        pred.append(model.logits_fault(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    return float((pred == y).mean())


def train_mocd(protocol, fold, seed, steps_a=None, steps_b=None, save_as="mocd",
               preproc="fft", arch="table3", lr=1e-4, b_mode="novat",
               skip_a=False, vat_eps=0.1, selection="last_step", steps=None,
               split_mode=None):
    spec = PROTOCOLS[protocol]
    device = _device()
    set_seed(seed)
    cache = {d: load_domain(spec, d) for d in spec["domains"]}
    split_mode_eff = split_mode or "grouped"
    src_doms, tgt_doms, splits, split_meta = load_or_make_split(
        cache, spec, fold, seed, SPLIT_DIR, split_mode=split_mode_eff)
    assert_no_target(src_doms, fold, tgt_doms)
    n_cls, n_dom = spec["num_classes"], len(src_doms)
    if spec.get("kind") == "pu_mocd_pt":
        xs = [_preproc_windows(_xy_all60(cache, splits, d)[0], d, preproc) for d in src_doms]
        ys = [_xy_all60(cache, splits, d)[1] for d in src_doms]
    else:
        xs = [_preproc_windows(_xy(cache, splits, d, "tr")[0], d, preproc) for d in src_doms]
        ys = [_xy(cache, splits, d, "tr")[1] for d in src_doms]
    xv = np.concatenate([
        _preproc_windows(_xy(cache, splits, d, "va")[0], d, preproc) for d in src_doms])
    yv = np.concatenate([_xy(cache, splits, d, "va")[1] for d in src_doms])
    pool = Pool(xs, ys, seed)
    ce_sampler = None
    sampler_meta = {
        "type": "Pool.sample_xy",
        "batch": 32,
        "source_only": True,
        "target_in_sampler": False,
        "downsample_majority": False,
        "src_doms": list(src_doms),
        "tgt_doms_excluded": list(tgt_doms),
        "matching": "mf_md_class_domain_grids",
    }
    if protocol in ("jnu_speed", "hust_speed", "hust_w3", "phm09_speed", "phm09_w3"):
        src_tr = list(zip(xs, ys))
        ce_sampler, sampler_meta = build_jnu_source_class_balanced_sampler(
            src_tr, n_cls, 32, seed, src_doms=src_doms, tgt_doms=tgt_doms,
            require_all_classes=(protocol != "phm09_w3"))
        sampler_meta = dict(sampler_meta)
        sampler_meta["matching"] = "mf_md_class_domain_grids"
        sampler_meta["ce_path"] = "BalancedDomainClassSampler"
    feat_d = 2048 if preproc == "fft" else int(spec["input_len"])
    model = MOCD(n_cls, n_dom, feat_d, arch=arch).to(device)
    opt_a = torch.optim.Adam(
        list(model.gf.parameters()) + list(model.gd.parameters())
        + list(model.pf.parameters()) + list(model.pd.parameters()),
        lr=float(lr))
    copies = 5
    mf = init_mf(pool, n_cls, n_dom, copies)
    md = init_md(pool, n_cls, n_dom, copies)
    rng = np.random.RandomState(seed + 3)
    a_outer, a_inner = 20, 20
    if skip_a:
        a_outer = 0
    b_outer, b_inner = 20, 80
    if steps is not None and int(steps) < 400:
        a_outer = 0 if skip_a else 1
        a_inner = 2
        b_outer, b_inner = 1, 2
    t0 = time.time()
    model.train()
    for _a in range(a_outer):
        for _t in range(a_inner):
            xf, yf, df = mf_batch(mf, 5, rng)
            xd, yd, dd = md_batch(md, 5, rng)
            xf, yf, df = xf.to(device), yf.to(device), df.to(device)
            xd, yd, dd = xd.to(device), yd.to(device), dd.to(device)
            zf = model.pf(model.gf(xf))
            zd = model.pd(model.gd(xd))
            loss = match_loss_fault(zf, yf, df, 0.2) + match_loss_domain(zd, yd, dd, 0.2)
            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            opt_a.step()
        hard_update_mf(mf, pool, model, device, rng=rng)
        hard_update_md(md, pool, model, device, rng=rng)
        print(f"[mocd] stageA {_a+1}/{a_outer}", flush=True)

    opt_b = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_sva, best_state = -1.0, None
    for _b in range(b_outer):
        for _t in range(b_inner):
            x, y, d = (
                ce_sampler.sample() if ce_sampler is not None else pool.sample_xy(32)
            )
            x, y, d = x.to(device), y.to(device), d.to(device)
            xf, yf, df = mf_batch(mf, 2, rng)
            xd, yd, dd = md_batch(md, 2, rng)
            xf, yf, df = xf.to(device), yf.to(device), df.to(device)
            xd, yd, dd = xd.to(device), yd.to(device), dd.to(device)
            zf = model.gf(x)
            zd = model.gd(x)
            lf = F.cross_entropy(model.hf(zf), y)
            ld = F.cross_entropy(model.hd(zd), d)
            lmf = match_loss_fault(model.pf(model.gf(xf)), yf, df, 0.2)
            lmd = match_loss_domain(model.pd(model.gd(xd)), yd, dd, 0.2)
            la = vat_pair(model, x, y, d, float(vat_eps), "f") + vat_pair(model, x, y, d, float(vat_eps), "d")
            zfn = F.normalize(zf, dim=-1)
            zdn = F.normalize(zd, dim=-1)
            lcc = match_loss_fault(0.5 * zfn + 0.5 * zdn, y, d, 0.2)
            lrs = 0.1 * barlow_pair(zf) + 0.1 * barlow_pair(zd) + 0.8 * (zf.T @ zd).div(zf.size(0)).pow(2).sum()
            rec = model.dec(zf, zd)
            lrc = F.mse_loss(rec, x)
            if b_mode == "ce":
                loss = lf
            elif b_mode == "ce_match":
                loss = lf + lmf + lmd
            elif b_mode == "novat":
                _w, loss = softmax_weights([lf, ld, lmf, lmd, lcc, lrs, lrc])
            elif b_mode == "norecon":
                _w, loss = softmax_weights([lf, ld, lmf, lmd, la, lcc, lrs])
            else:
                _w, loss = softmax_weights([lf, ld, lmf, lmd, la, lcc, lrs, lrc])
            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            opt_b.step()
        hard_update_mf(mf, pool, model, device, rng=rng)
        hard_update_md(md, pool, model, device, rng=rng)
        if _selects_on_source_val(selection):
            sva_ck = eval_acc(model, xv, yv, device)
            if sva_ck > best_sva:
                best_sva = sva_ck
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
        print(f"[mocd] stageB {_b+1}/{b_outer}", flush=True)

    if _selects_on_source_val(selection) and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    sva = eval_acc(model, xv, yv, device)
    by_t = {}
    for t in tgt_doms:
        xt, yt = _te(cache, split_meta, t)
        xt = _preproc_windows(xt, t, preproc)
        by_t[t] = eval_acc(model, xt, yt, device)
    ta = float(np.mean(list(by_t.values())))
    elapsed = time.time() - t0
    out_dir = os.path.join(RESULTS, protocol, save_as, f"fold{fold}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict()}, os.path.join(out_dir, "last.pt"))
    metrics = {
        "protocol": protocol, "level": save_as, "fold": fold, "seed": seed,
        "n_steps": a_outer * a_inner + b_outer * b_inner + 4,
        "source_val_acc": sva, "target_acc": ta,
        "target_acc_by_dom": {k: float(v) for k, v in by_t.items()},
        "selection": selection,
        "split_mode": split_mode_eff,
        "target_used_for_selection": False,
        "n_src": n_dom, "src_doms": src_doms, "tgt_doms": tgt_doms,
        "elapsed_s": elapsed,
        "preproc": preproc, "arch": arch, "lr": float(lr), "b_mode": b_mode,
        "skip_a": bool(skip_a), "vat_eps": float(vat_eps),
        "sampler": sampler_meta,
        "note": "Table 3 CNN + log-FFT; novat default; hard-positive scan capped at 256. FFT not named in Table 4.",
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[mocd] {fold} seed{seed} sva={sva*100:.2f} tgt={ta*100:.2f} {elapsed:.1f}s", flush=True)
    return metrics


def train_erm(protocol, fold, seed, steps=4000, save_as="erm", preproc=None,
              oracle_tgt=False, use_all_src=False, src_only=None,
              arch="table3", lr=1e-4, ot_tgt=False, selection="last_step",
              irm=None, split_mode=None, ot_mode="resample_reflect",
              rpm_ref_mode="max"):
    spec = PROTOCOLS[protocol]
    device = _device()
    set_seed(seed)
    cache = {d: load_domain(spec, d) for d in spec["domains"]}
    split_mode_eff = split_mode or "grouped"
    from wfdg.data import domain_rpm, parse_fold
    from wfdg.angular import order_track_windows
    src_peek, tgt_peek = parse_fold(spec, fold)
    rpm_ref = None
    ot_spec = str(preproc or "") == "ot_spec"
    if ot_spec:
        rpms = [domain_rpm(spec, d) for d in src_peek]
        if rpm_ref_mode == "max":
            rpm_ref = max(rpms)
        elif rpm_ref_mode == "min":
            rpm_ref = min(rpms)
        elif rpm_ref_mode == "median":
            rpm_ref = float(np.median(rpms))
        else:
            raise ValueError(rpm_ref_mode)
        # Time-OT every domain (incl. target nominal rpm) — W3 R5 / erm_ot.
        cache = {
            d: (order_track_windows(x, domain_rpm(spec, d), rpm_ref, mode=ot_mode), y, g)
            for d, (x, y, g) in cache.items()
        }
        feat_kind = "fft"
    else:
        feat_kind = preproc
    src_doms, tgt_doms, splits, split_meta = load_or_make_split(
        cache, spec, fold, seed, SPLIT_DIR, split_mode=split_mode_eff)
    assert_no_target(src_doms, fold, tgt_doms)
    if oracle_tgt:
        src_doms = list(tgt_doms)
        for d in src_doms:
            y = cache[d][1]
            tr60, te30 = mocd_60_30_indices(y, seed * 8 + int(d))
            tr, va = source_tr_va(tr60, seed)
            splits[d] = (tr, va)
            split_meta.setdefault("te_idx", {})[d] = np.asarray(te30)
    if ot_tgt and not ot_spec:
        # Legacy PU path (mocd_rpm). Not used for W3 hust/jnu/phm09.
        for t in tgt_doms:
            cache[t] = _ot_rewindow(*cache[t], mocd_rpm(t), RPM_REF)
            _tr60, te30 = mocd_60_30_indices(cache[t][1], seed * 8 + int(t))
            split_meta.setdefault("te_idx", {})[t] = np.asarray(te30)
    if src_only is not None:
        src_doms = [str(src_only)]
    n_cls = spec["num_classes"]
    if use_all_src:
        xs = [_preproc_windows(cache[d][0], d, feat_kind) for d in src_doms]
        ys = [cache[d][1] for d in src_doms]
    elif spec.get("kind") == "pu_mocd_pt":
        xs = [_preproc_windows(_xy_all60(cache, splits, d)[0], d, feat_kind) for d in src_doms]
        ys = [_xy_all60(cache, splits, d)[1] for d in src_doms]
    else:
        xs = [_preproc_windows(_xy(cache, splits, d, "tr")[0], d, feat_kind) for d in src_doms]
        ys = [_xy(cache, splits, d, "tr")[1] for d in src_doms]
    xv = np.concatenate([
        _preproc_windows(_xy(cache, splits, d, "va")[0], d, feat_kind) for d in src_doms])
    yv = np.concatenate([_xy(cache, splits, d, "va")[1] for d in src_doms])
    pool = Pool(xs, ys, seed)
    ce_sampler = None
    sampler_meta = {
        "type": "Pool.sample_xy",
        "batch": 32,
        "source_only": True,
        "target_in_sampler": False,
        "downsample_majority": False,
        "src_doms": list(src_doms),
        "tgt_doms_excluded": list(tgt_doms),
    }
    if protocol in ("jnu_speed", "hust_speed", "hust_w3", "phm09_speed", "phm09_w3"):
        src_tr = list(zip(xs, ys))
        ce_sampler, sampler_meta = build_jnu_source_class_balanced_sampler(
            src_tr, n_cls, 32, seed, src_doms=src_doms, tgt_doms=tgt_doms,
            require_all_classes=(protocol != "phm09_w3"))
    feat_d = 2048 if feat_kind == "fft" else int(spec["input_len"])
    gf = make_encoder(feat_d, arch).to(device)
    hf = ClfHead(n_cls).to(device)
    opt = torch.optim.Adam(list(gf.parameters()) + list(hf.parameters()), lr=float(lr))
    t0 = time.time()
    n_loop = 2500 if int(steps) >= 10000 else int(steps)
    gf.train(); hf.train()
    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gf = gf
            self.hf = hf

        def logits_fault(self, x):
            return self.hf(self.gf(x))
    wrap = Wrap()
    best_sva, best_g, best_h = -1.0, None, None
    ckpt_every = max(1, int(n_loop) // 20)
    irm_w = None if irm is None else float(irm)
    if irm_w is not None:
        from wfdg.residual import irm_penalty
    for step in range(int(n_loop)):
        x, y, d = (
            ce_sampler.sample() if ce_sampler is not None else pool.sample_xy(32)
        )
        x, y, d = x.to(device), y.to(device), d.to(device)
        logits = hf(gf(x))
        loss = F.cross_entropy(logits, y)
        if irm_w is not None and irm_w > 0:
            with torch.amp.autocast("cuda", enabled=False):
                ir = irm_penalty(logits, y, d)
            loss = loss + irm_w * ir
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if _selects_on_source_val(selection) and ((step + 1) % ckpt_every == 0 or step + 1 == int(n_loop)):
            sva_ck = eval_acc(wrap, xv, yv, device)
            if sva_ck > best_sva:
                best_sva = sva_ck
                best_g = {k: v.detach().cpu().clone() for k, v in gf.state_dict().items()}
                best_h = {k: v.detach().cpu().clone() for k, v in hf.state_dict().items()}
            gf.train(); hf.train()
    if _selects_on_source_val(selection) and best_g is not None:
        gf.load_state_dict(best_g)
        hf.load_state_dict(best_h)
        gf.to(device); hf.to(device)
    sva = eval_acc(wrap, xv, yv, device)
    by_t = {}
    for t in tgt_doms:
        xt, yt = _te(cache, split_meta, t)
        xt = _preproc_windows(xt, t, feat_kind)
        by_t[t] = eval_acc(wrap, xt, yt, device)
    ta = float(np.mean(list(by_t.values())))
    elapsed = time.time() - t0
    out_dir = os.path.join(RESULTS, protocol, save_as, f"fold{fold}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "protocol": protocol, "level": save_as, "fold": fold, "seed": seed,
        "n_steps": max(int(steps), int(n_loop)), "n_train": int(n_loop),
        "source_val_acc": sva, "target_acc": ta,
        "target_acc_by_dom": {k: float(v) for k, v in by_t.items()},
        "selection": selection, "target_used_for_selection": False,
        "split_mode": split_mode_eff,
        "n_src": len(src_doms), "src_doms": src_doms, "tgt_doms": tgt_doms,
        "elapsed_s": elapsed,
        "preproc": preproc,
        "oracle_tgt": bool(oracle_tgt),
        "use_all_src": bool(use_all_src),
        "src_only": src_only,
        "arch": arch,
        "lr": float(lr),
        "ot_tgt": bool(ot_tgt or ot_spec),
        "order_track": bool(ot_spec),
        "ot_mode": ot_mode if ot_spec else None,
        "rpm_ref": float(rpm_ref) if rpm_ref is not None else None,
        "rpm_ref_mode": rpm_ref_mode if ot_spec else None,
        "irm_weight": float(irm_w) if irm_w is not None else 0.0,
        "sampler": sampler_meta,
        "note": "Table 3 CNN + log-FFT + Adam 1e-4. Fair-protocol cells train on source tr only and pick grouped source-val. pu_mocd still uses all 60 + last step. Optional IRMv1 via --irm. preproc=ot_spec = W3 time-OT then FFT (domain_rpm).",
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[erm] {fold} seed{seed} sva={sva*100:.2f} tgt={ta*100:.2f} {elapsed:.1f}s", flush=True)
    return metrics
