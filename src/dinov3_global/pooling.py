"""Geometry and local multiple-instance evidence aggregation."""

import torch
from torch.nn import functional as F

RR_STATES = (2, 5, 3)  # red, yellow, red_yellow
GREEN_STATE = 0


def token_geo(h, w):
    """Normalized token centers and bearing: (x, y, 2*x-1, 1-y)."""
    yy, xx = torch.meshgrid(
        (torch.arange(h, dtype=torch.float32) + 0.5) / h,
        (torch.arange(w, dtype=torch.float32) + 0.5) / w,
        indexing="ij",
    )
    x, y = xx.reshape(-1), yy.reshape(-1)
    return torch.stack([x, y, (x - 0.5) * 2, 1 - y], dim=1)


def local_evidence_pool(evidence, lamp, grid_hw, k, content_mask=None):
    """Lamp-weighted 5x5 local mean followed by a top-k mean."""
    batch, count = evidence.shape
    h, w = grid_hw
    eg, lg = evidence.reshape(batch, 1, h, w), lamp.reshape(batch, 1, h, w)
    numerator = F.avg_pool2d(eg * lg, 5, stride=1, padding=2)
    denominator = F.avg_pool2d(lg, 5, stride=1, padding=2).clamp_min(1e-4)
    local = (numerator / denominator).reshape(batch, count)
    if content_mask is not None:
        local = local * content_mask.reshape(batch, count)
    return local.topk(min(k, count), dim=1).values.mean(1)
