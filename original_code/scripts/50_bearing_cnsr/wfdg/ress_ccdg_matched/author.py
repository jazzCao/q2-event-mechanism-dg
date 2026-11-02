"""Author CNN_Deep and contrast, with the declared 2048-input adaptation.

Source: related projects/CCDG-main. The class definitions stay in that tree.
"""
from __future__ import annotations

import os
import sys

import torch.nn as nn

from wfdg.ress_source_dg.spec import ROOT

CCDG_ROOT = os.path.join(ROOT, "related projects", "CCDG-main")
if CCDG_ROOT not in sys.path:
    sys.path.insert(0, CCDG_ROOT)

from domainbed.TSnetworks import CNN_Deep, MLP_Deep  # noqa: E402
from domainbed.lib.misc import domain_contrastive_loss  # noqa: E402


def contrast(features, labels, temperature=0.7):
    return domain_contrastive_loss(features, labels, float(temperature), features.device)


def author_net(last_layer_dim, num_classes=3, hidden=32, with_in=False):
    featurizer = CNN_Deep(1, {"cnn_hidden_dim": hidden, "last_layer_dim": int(last_layer_dim)})
    classifier = MLP_Deep(hidden, num_classes)
    if with_in:
        layers = list(featurizer.encoder.children())
        idx = max(i for i, mod in enumerate(layers) if isinstance(mod, nn.Conv1d))
        layers.insert(idx + 1, nn.InstanceNorm1d(8, affine=False, eps=1e-5, track_running_stats=False))
        featurizer.encoder = nn.Sequential(*layers)
    return featurizer, classifier


def flatten_dim(length):
    """Closed form of the author conv stack. Used only as a test oracle."""
    length = int(length)
    specs = [(64, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1), (8, 1, 1)]
    for kernel, stride, pad in specs:
        length = (length + 2 * pad - kernel) // stride + 1
    return 8 * length
