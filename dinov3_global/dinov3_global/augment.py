"""Batched GPU photometric augmentation for domain generalization.

Design rule (M3): every transform must be LABEL-PRESERVING for traffic-light
state semantics. White-balance gains, gamma, contrast, sensor noise, JPEG-ish
down/up-sampling, mild blur and fog veils model ISP/weather/framing shift
across cameras and datasets without ever rotating hue (which would turn red
lamps green and corrupt the RR/RG labels).

Applied in engine.train() right after H2D transfer, on [0,1] float tensors.
Whole-batch shares draw parameters (cheap + fine for our batch sizes).
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn.functional as F


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def photometric(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """images: (B,3,H,W) float in [0,1]. Returns augmented copy. ~ms on GPU."""
    if random.random() < 0.7:  # brightness (exposure)
        x = x * random.uniform(1 - 0.3 * strength, 1 + 0.3 * strength)
    if random.random() < 0.5:  # per-channel white balance (camera colour response)
        gains = torch.tensor([random.uniform(1 - 0.1 * strength, 1 + 0.1 * strength)
                              for _ in range(3)], device=x.device).view(1, 3, 1, 1)
        x = x * gains
    if random.random() < 0.4:  # gamma (tone curve)
        x = x.clamp(1e-4, 1.0).pow(random.uniform(1 - 0.3 * strength, 1 + 0.4 * strength))
    if random.random() < 0.4:  # contrast around scene mean
        m = x.mean()
        x = (x - m) * random.uniform(1 - 0.2 * strength, 1 + 0.2 * strength) + m
    if random.random() < 0.3:  # saturation (dull ISP / overcast)
        g = _rgb_to_gray(x)
        x = x + (g - x) * random.uniform(0.0, 0.5 * strength)
    return x.clamp_(0.0, 1.0)


def degradations(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Sensor/compression/weather degradations. Separate so consistency views
    can draw photometric twice but degrade less often."""
    if random.random() < 0.3:  # sensor noise
        x = x + torch.randn_like(x) * random.uniform(0.005, 0.02 * strength)
    if random.random() < 0.2:  # JPEG-ish: down/up-sample + slight quantisation feel
        f = random.uniform(0.5, 0.85)
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(int(h * f), int(w * f)), mode="bilinear",
                          align_corners=False)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    if random.random() < 0.2:  # defocus / motion blur
        k = random.choice([3, 5])
        pad = k // 2
        x = F.avg_pool2d(F.pad(x, (pad,) * 4, mode="reflect"), k, stride=1)
    if random.random() < 0.15:  # fog / veil
        a = random.uniform(0.05, 0.2 * strength)
        x = x * (1 - a) + a * random.uniform(0.6, 0.9)
    return x.clamp_(0.0, 1.0)


def random_framing(x: torch.Tensor, min_scale: float = 0.85) -> torch.Tensor:
    """Mild random scale+shift crop (camera framing variation) then resize back.

    NOT label-preserving for existential RR/RG labels: cropping can remove the
    only relevant signal. Legacy ablation only; do not use for consistency
    without transforming boxes and recomputing the global label."""
    if random.random() >= 0.3:
        return x
    B, _, H, W = x.shape
    s = random.uniform(min_scale, 1.0)
    ch, cw = int(H * s), int(W * s)
    top = random.randint(0, H - ch)
    left = random.randint(0, W - cw)
    x = x[:, :, top:top + ch, left:left + cw]
    return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)


def erase_lamps(x: torch.Tensor, lamp_grid: torch.Tensor,
                margin_tokens: int = 1, iters: int = 12) -> torch.Tensor:
    """Paint over every lamp region of the selected frames (hole filling).

    ``lamp_grid`` [B,45,80] in {0,1} (the rasterised GT boxes) marks what to
    remove. The hole is filled by iterative diffusion of the surrounding
    texture (normalized convolution repeated ``iters`` times at 1/4 resolution,
    propagating ~16 px/iteration), so no lamp colour/glow survives and no
    black-box artifact appears for the model to latch onto. A single-pass fill
    is NOT enough: a fill window smaller than the hole averages over nothing
    and degenerates to black in the hole's centre.
    """
    B, _, H, W = x.shape
    m = lamp_grid.float().unsqueeze(1)                       # [B,1,45,80]
    k = 2 * margin_tokens + 1
    m = F.max_pool2d(m, k, stride=1, padding=k // 2)         # dilate ~16px/token
    m = F.interpolate(m, size=(H, W), mode="nearest")
    inv = 1.0 - m
    inv4 = F.avg_pool2d(inv, 4)                              # 1/4 res is plenty
    known = (inv4 > 0.9999).to(x.dtype)                      # qpx fully outside
    cur = F.avg_pool2d(x * inv, 4) / inv4.clamp_min(1e-3)
    cur = cur * known
    for _ in range(iters):
        num = F.avg_pool2d(cur, 9, stride=1, padding=4)
        den = F.avg_pool2d(known, 9, stride=1, padding=4)
        est = num / den.clamp_min(1e-6)
        cur = torch.where(known > 0.5, cur, est)
        known = torch.maximum(known, (den > 1e-6).to(x.dtype))
    fill = F.interpolate(cur, size=(H, W), mode="bilinear", align_corners=False)
    return (x * inv + fill * m).clamp_(0.0, 1.0)


def apply_lamp_erasure(x: torch.Tensor, lamp_grid: torch.Tensor,
                       y: torch.Tensor, w: torch.Tensor,
                       lamp_tgt: torch.Tensor, valid: torch.Tensor,
                       rel_tgt: torch.Tensor, p_erase: float
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor, torch.Tensor]:
    """v3 "manufactured NoR": with prob ``p_erase`` per frame, erase EVERY lamp
    of the frame and relabel it NoR.

    DTLD contains zero lamp-free frames, so "no relevant lamp present" is
    structurally unlearned - yet it is the common case on foreign datasets and
    the frame the user inspects by hand. A frame with all lamps erased is NoR by
    the label definition (``2 = everything else``), so this is label-honest.
    Targets follow the pixels: lamp/rel targets go to 0 (no lamp left) and the
    ignore band is released (those tokens are now clean background).

    Call AFTER the per-sample batch transfer and BEFORE ``train_view`` so both
    consistency views see the same frame/label pair.
    """
    if p_erase <= 0:
        return x, y, w, lamp_tgt, valid, rel_tgt
    has_lamp = lamp_grid.reshape(x.shape[0], -1).sum(1) > 0
    sel = (torch.rand(x.shape[0], device=x.device) < p_erase) & has_lamp
    if not bool(sel.any()):
        return x, y, w, lamp_tgt, valid, rel_tgt
    x = x.clone()
    x[sel] = erase_lamps(x[sel], lamp_grid[sel])
    y = y.clone(); y[sel] = 2
    w = w.clone(); w[sel] = 1.0
    lamp_tgt = lamp_tgt.clone(); lamp_tgt[sel] = 0.0
    rel_tgt = rel_tgt.clone(); rel_tgt[sel] = 0.0
    valid = valid.clone(); valid[sel] = True
    return x, y, w, lamp_tgt, valid, rel_tgt


def train_view(x: torch.Tensor, strength: float = 1.0, crop: bool = False,
               photometric_enabled: bool = True,
               degradations_enabled: bool = True,
               per_image: bool = True, clean_prob: float = 0.0,
               blend_min: float = 0.0) -> torch.Tensor:
    """One augmented training view: framing -> photometric -> degradations.

    ``crop=False`` skips ``random_framing``. REQUIRED for the view that carries
    the token-level aux losses: ``lamp_tgt``/``state_tgt``/``rel_tgt`` are
    rasterised on the uncropped frame, so cropping the input supervises the
    wrong tokens (measured on exp2: top-1 lamp token inside a GT box 0.780 on
    clean input vs 0.005 after one framing crop, applied with p=0.3).
    """
    if strength < 0 or not 0 <= clean_prob <= 1 or not 0 <= blend_min <= 1:
        raise ValueError("invalid augmentation strength/probability/blend")
    # Independent camera draws per image; retain some original pixels so tiny
    # signal colours are not erased by a long chain of degradations. Inspired
    # by AugMix, not a reproduction of its three-view objective.
    if per_image and x.shape[0] > 1:
        return torch.cat([train_view(im.unsqueeze(0), strength, crop,
                                     photometric_enabled, degradations_enabled,
                                     False, clean_prob, blend_min) for im in x])
    if strength == 0 or random.random() < clean_prob:
        return x.clone()
    original = x
    x = x.clone()
    if crop:
        x = random_framing(x)
    if photometric_enabled:
        x = photometric(x, strength)
    if degradations_enabled:
        x = degradations(x, strength)
    if blend_min > 0:
        keep = random.uniform(blend_min, 1.0)
        x = keep * original + (1 - keep) * x
    return x.clamp(0.0, 1.0)
