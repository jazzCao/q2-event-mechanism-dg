"""Experiment classifier; extracted definitions, original parameter names."""
import torch
from torch import nn
from torch.nn import functional as F

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


class ClfHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, n_out),
        )

    def forward(self, z):
        return self.net(z)

class Net(nn.Module):
    def __init__(self, classes, arm):
        super().__init__(); self.arm = arm; self.enc = EncoderGTable3(2048); self.clf = ClfHead(classes)
    def forward(self, x):
        h = self.enc.net(x[:, None] if x.ndim == 2 else x)
        if self.arm != 'B0_ID_MIX':
            h = (h-h.mean(-1, keepdim=True))/torch.sqrt(h.var(-1, unbiased=False, keepdim=True)+1e-5)
        logits = self.clf(self.enc.fc(h))
        return dict(logits=logits, r=None, z=None)
    def losses(self, out, y, c, permutation, lam):
        n = len(y); q = out['logits']; clean = F.cross_entropy(q[:n], y)
        mixed = lam*F.cross_entropy(q[n:], y)+(1-lam)*F.cross_entropy(q[n:], y[permutation])
        return dict(total=.5*(clean+mixed), clean_ce=clean, mixup_ce=mixed)
