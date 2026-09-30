"""Losses for DinoGlobal-MIL.

- ``global_loss``: label-smoothed CE with LOGIT ADJUSTMENT (Menon et al.,
  ICLR 2021): training adds ``tau_la * log(pi_c)`` to the logits. This is
  Fisher-consistent for the *balanced* error - exactly the ``acc_bal`` metric
  and the per-class AP we optimise - and replaces the naive 1/freq class
  weights of exp1 (which are not consistent for balanced error).
  Optional per-sample ``weight`` implements the label-policy knob for frames
  whose only relevant lamps are off/unknown (pseudo-NoR).
- ``token_losses``: auxiliary supervision from DTLD lamp boxes (training-only;
  inference is image-in -> image-out so it transfers to unlabeled datasets).
  lampness = BCE over all non-ignored tokens; state = CE and relevance = BCE
  over lamp-positive tokens only. This is what teaches the MIL pooling WHERE
  the lamps are - critical for 1-2-token-sized lamps.
- ``consistency_kl``: symmetric KL between two photometric views of the same
  frame (domain-generalisation regulariser).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def global_loss(logits: torch.Tensor, targets: torch.Tensor,
                log_pi: Optional[torch.Tensor] = None,
                tau_la: float = 1.0,
                label_smoothing: float = 0.05,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if log_pi is not None and tau_la > 0:
        logits = logits + tau_la * log_pi.to(logits.device).view(1, -1)
    per = F.cross_entropy(logits, targets, reduction="none",
                          label_smoothing=label_smoothing)
    if weight is not None:
        per = per * weight.to(per.device)
    return per.mean()


def _bce(logit: torch.Tensor, y: torch.Tensor,
         pos_weight: Optional[torch.Tensor] = None,
         focal_gamma: float = 0.0) -> torch.Tensor:
    """BCE with logits; ``focal_gamma > 0`` switches to focal modulation.

    Focal (Lin et al. 2017) multiplies each term by ``(1 - p_t)^gamma`` so
    well-classified tokens stop dominating the gradient. For the relevance head
    this is the difference between learning from ~3,500 trivial background
    tokens per frame and learning from the hard negatives - the irrelevant
    lamps whose red/green state looks exactly like a relevant one (measured
    DTLD train split: 27% of irrelevant lamps are red, 23% green).
    """
    if focal_gamma and focal_gamma > 0:
        p = torch.sigmoid(logit)
        p_t = torch.where(y > 0.5, p, 1.0 - p)
        ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none",
                                                pos_weight=pos_weight)
        return ((1.0 - p_t) ** focal_gamma * ce).mean()
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)


def token_losses(maps: Dict[str, torch.Tensor],
                 lamp_tgt: torch.Tensor, valid: torch.Tensor,
                 state_tgt: torch.Tensor, rel_tgt: torch.Tensor,
                 w_lamp: float = 0.3, w_state: float = 0.2,
                 w_rel: float = 0.5, lamp_pos_weight: float = 10.0,
                 rel_pos_weight: float = 100.0,
                 rel_focal_gamma: float = 0.0,
                 w_dir: float = 0.0, dir_tgt: Optional[torch.Tensor] = None,
                 state_weights: Optional[torch.Tensor] = None,
                 state_rel_boost: float = 0.0
                 ) -> Dict[str, torch.Tensor]:
    """maps: lamp_logit [B,N], rel_logit [B,N], state_logit [B,N,6],
    optionally dir_logit [B,N,4]. Targets on the patch grid: lamp_tgt/rel_tgt
    float, state_tgt/dir_tgt long, valid bool (ignore band around boxes).
    Returns weighted total + parts.

    v3 change: relevance is supervised on **all valid tokens**, not only on
    lamp-positive ones. v2 masked state/rel to `valid & lamp_tgt>0.5`, so the
    head was never shown the other 99.7% of the grid as an explicit negative -
    which is exactly the measured exp2 failure (confident "relevant red lamp"
    evidence on 23.5% of NoR frames). `rel_pos_weight` restores a sane
    positive/negative mass now that the background is included (relevant tokens
    are ~0.13% of valid tokens).

    v4 changes:
    - `rel_focal_gamma` focal-modulates the rel BCE so the trivial background
      stops swamping the hard negatives (irrelevant-but-lit lamps).
    - `state_weights` (per class) and `state_rel_boost` (per token, for
      relevance-positive lamp tokens) counter the state confusion that
      manufactures false evidence on pseudo-NoR frames (relevant lamp
      off/unknown read as red).
    - `w_dir`/`dir_tgt`: the housing-direction aux head (see decoder.py).
    """
    if maps is None or w_lamp + w_state + w_rel + w_dir <= 0:
        z = torch.tensor(0.0)
        return {"total": z, "lamp": z, "state": z, "rel": z, "dir": z}

    B = lamp_tgt.shape[0]
    lamp_logit = maps["lamp_logit"].reshape(B, -1)
    rel_logit = maps["rel_logit"].reshape(B, -1)
    state_logit = maps["state_logit"].reshape(B, -1, maps["state_logit"].shape[-1])
    lamp_y = lamp_tgt.reshape(B, -1)
    rel_y = rel_tgt.reshape(B, -1)
    state_y = state_tgt.reshape(B, -1)
    v = valid.reshape(B, -1)

    zero = lamp_logit.new_zeros(())
    # --- lampness: BCE on every non-ignored token -------------------------
    if v.any() and w_lamp > 0:
        sel = v
        pw = torch.tensor(lamp_pos_weight, device=lamp_logit.device)
        l_lamp = F.binary_cross_entropy_with_logits(
            lamp_logit[sel], lamp_y[sel], pos_weight=pw)
    else:
        l_lamp = zero

    # --- state: only tokens inside a lamp box (no state label elsewhere) -----
    pos = v & (lamp_y > 0.5)
    n_pos = int(pos.sum())
    if n_pos > 0 and w_state > 0:
        y_s = state_y[pos]
        per = F.cross_entropy(state_logit[pos], y_s, reduction="none",
                              weight=state_weights.to(state_logit.device)
                              if state_weights is not None else None)
        if state_rel_boost > 0:
            per = per * (1.0 + state_rel_boost * rel_y[pos])
        l_state = per.mean()
    else:
        l_state = zero
    # --- relevance: ALL valid tokens (lamp tokens carry their GT flag, the
    # background is an explicit negative). v2 supervised lamp tokens only. -----
    if v.any() and w_rel > 0:
        rpw = torch.tensor(rel_pos_weight, device=rel_logit.device)
        l_rel = _bce(rel_logit[v], rel_y[v], pos_weight=rpw,
                     focal_gamma=rel_focal_gamma)
    else:
        l_rel = zero
    # --- direction aux (lamp tokens only; -1 = no lamp) ---------------------
    d_logit = maps.get("dir_logit")
    if d_logit is not None and dir_tgt is not None and w_dir > 0:
        d_logit = d_logit.reshape(B, -1, d_logit.shape[-1])
        d_y = dir_tgt.reshape(B, -1)
        sel = pos & (d_y >= 0)
        if int(sel.sum()) > 0:
            l_dir = F.cross_entropy(d_logit[sel], d_y[sel])
        else:
            l_dir = zero
    else:
        l_dir = zero

    total = w_lamp * l_lamp + w_state * l_state + w_rel * l_rel + w_dir * l_dir
    return {"total": total, "lamp": l_lamp.detach(),
            "state": l_state.detach(), "rel": l_rel.detach(),
            "dir": l_dir.detach() if torch.is_tensor(l_dir) else zero}


def consistency_kl(logits_a: torch.Tensor, logits_b: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
    """Symmetric KL between two augmented views (mean over batch)."""
    t = max(temperature, 1e-3)
    la = F.log_softmax(logits_a / t, dim=-1)
    lb = F.log_softmax(logits_b / t, dim=-1)
    pa, pb = la.exp(), lb.exp()
    kl = (F.kl_div(la, pb, reduction="batchmean")
          + F.kl_div(lb, pa, reduction="batchmean")) * 0.5
    return kl * (t * t)  # temperature scaling (Hinton distillation)
