"""Type-preserving CPU view for a future audit revision; not wired into locked V1."""
from __future__ import annotations

import copy
import torch


def cpu_view_preserving_type(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        out = copy.copy(value)
        for key, child in value.items():
            out[key] = cpu_view_preserving_type(child)
        if hasattr(value, "_metadata"):
            out._metadata = cpu_view_preserving_type(value._metadata)
        return out
    if isinstance(value, list):
        out = copy.copy(value)
        for i, child in enumerate(value):
            out[i] = cpu_view_preserving_type(child)
        return out
    if isinstance(value, tuple):
        items = [cpu_view_preserving_type(child) for child in value]
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    return value
