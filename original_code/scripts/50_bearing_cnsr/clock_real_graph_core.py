"""Fixed-shape, ten-step CUDA-graph FISTA; same mathematical source inverse."""
import numpy as np
import torch


class GraphInverse:
    def __init__(self, x, d, tick):
        self.n = x.shape[-1]; self.nfft = int(d['nfft']); self.tick = tick
        zeros = lambda *shape: torch.zeros(shape, device='cuda', dtype=torch.float32)
        self.x = zeros(2, self.n); self.H = torch.zeros((2, 48, self.nfft//2+1), device='cuda', dtype=torch.complex64)
        self.A = zeros(2, self.n, 8); self.F = zeros(self.n, 16)
        self.L = zeros(1); self.ps = zeros(2, 1, 1); self.pb = zeros(1); self.pc = zeros(1); self.ratios = zeros(10)
        self.s = zeros(2, 48, self.n); self.b = zeros(2, 8); self.c = zeros(2, 16)
        self.zs = zeros(2, 48, self.n); self.zb = zeros(2, 8); self.zc = zeros(2, 16)
        self.momenta = [1.]; self.coefficients = []
        for _ in range(500):
            old = self.momenta[-1]; new = (1+np.sqrt(1+4*old**2))/2
            self.coefficients.append((old-1)/new); self.momenta.append(new)
        self.coefficient_tensor = torch.tensor(self.coefficients, device='cuda', dtype=torch.float32)
        self.load(x, d); self.ratios.copy_(self.coefficient_tensor[:10])
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                tick('gpu_inverse_iterations', 10); tick('graph_warmup_iterations', 10)
                self.block()
        torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize(); self.reset()
        self.graph = torch.cuda.CUDAGraph(); tick('graph_capture_blueprint_steps', 10)
        # Per PyTorch2.12 stream-capture semantics, capture records but does not execute kernels.
        with torch.cuda.graph(self.graph): self.block()
        self.reset(); torch.cuda.synchronize()

    @torch.no_grad()
    def reset(self):
        for value in (self.s, self.b, self.c, self.zs, self.zb, self.zc): value.zero_()

    @torch.no_grad()
    def load(self, x, d, state=None):
        if x.shape != (2, self.n) or int(d['nfft']) != self.nfft or d['F'].shape[1] > 16: raise ValueError('fixed graph shape')
        convert = lambda v: torch.as_tensor(v, device='cuda', dtype=torch.float32)
        self.x.copy_(convert(x)); self.H.copy_(torch.fft.rfft(convert(d['h']), self.nfft, dim=-1)); self.A.copy_(convert(d['A']))
        self.F.zero_(); self.columns = d['F'].shape[1]
        if self.columns: self.F[:, :self.columns].copy_(convert(d['F']))
        self.L.fill_(float(d['L'])); self.ps.copy_(convert(d['penalty_s'])[:, None, None]/d['L'])
        self.pb.fill_(float(d['penalty_b']/d['L'])); self.pc.fill_(float(d['penalty_n']/d['L']))
        self.reset(); self.begin = 0
        if state is not None:
            self.begin = int(state['step'])
            if float(state['momentum']) != self.momenta[self.begin]: raise AssertionError('same FISTA momentum schedule')
            for name in ('s', 'b', 'zs', 'zb'): getattr(self, name).copy_(convert(state[name]))
            for name in ('c', 'zc'):
                if self.columns: getattr(self, name)[:, :self.columns].copy_(convert(state[name]))

    @staticmethod
    def prox(value, threshold):
        shaped = value.reshape(2, -1, 2); norm = torch.sqrt((shaped*shaped).sum((0, 2)))
        return (shaped*(1-threshold/norm.clamp_min(1e-30)).clamp_min(0)[None, :, None]).reshape(value.shape)

    @torch.no_grad()
    def block(self):
        for i in range(10):
            pred = torch.fft.irfft((self.H*torch.fft.rfft(self.zs, self.nfft, dim=-1)).sum(1), self.nfft, dim=-1)[:, :self.n]
            residual = pred+torch.einsum('dlk,dk->dl', self.A, self.zb)+self.zc@self.F.T-self.x
            gs = torch.fft.irfft(self.H.conj()*torch.fft.rfft(residual, self.nfft, dim=-1)[:, None], self.nfft, dim=-1)[:, :, :self.n]
            ns = (self.zs-gs/self.L-self.ps).clamp_min(0)
            nb = self.prox(self.zb-torch.einsum('dlk,dl->dk', self.A, residual)/self.L, self.pb)
            nc = self.prox(self.zc-residual@self.F/self.L, self.pc)
            zs = ns+self.ratios[i]*(ns-self.s); zb = nb+self.ratios[i]*(nb-self.b); zc = nc+self.ratios[i]*(nc-self.c)
            self.s.copy_(ns); self.b.copy_(nb); self.c.copy_(nc); self.zs.copy_(zs); self.zb.copy_(zb); self.zc.copy_(zc)

    @torch.no_grad()
    def fit(self, x, d, steps, state=None, tick=None):
        self.load(x, d, state)
        if self.begin % 10 or steps % 10 or steps > 500: raise ValueError('qualified graph block schedule')
        claim = tick or self.tick
        for begin in range(self.begin, steps, 10):
            claim('gpu_inverse_iterations', 10); self.ratios.copy_(self.coefficient_tensor[begin:begin+10]); self.graph.replay()
        torch.cuda.synchronize()
        result = {name: getattr(self, name).cpu().numpy().copy() for name in ('s', 'b', 'c', 'zs', 'zb', 'zc')}
        result['c'] = result['c'][:, :self.columns]; result['zc'] = result['zc'][:, :self.columns]
        result.update(momentum=self.momenta[steps], step=steps)
        return result
