"""Global classification, attribute supervision, and two-view consistency.

Global CE uses label smoothing and training-time logit adjustment from natural
class frequencies. Lamp and relevance losses cover valid tokens; state and
direction CE cover annotated lamp tokens. Relevance focal loss balances lamp
tokens against background separately. Symmetric KL compares two photometric
views. See configs/default.yaml for the retained v5 coefficients.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def global_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    log_pi: Optional[torch.Tensor] = None,
    tau_la: float = 1.0,
    label_smoothing: float = 0.05,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if log_pi is not None and tau_la > 0:
        logits = logits + tau_la * log_pi.to(logits.device).view(1, -1)
    per = F.cross_entropy(logits, targets, reduction="none", label_smoothing=label_smoothing)
    if weight is not None:
        per = per * weight.to(per.device)
    return per.mean()


def _bce(
    logit: torch.Tensor, y: torch.Tensor, pos_weight: Optional[torch.Tensor] = None, focal_gamma: float = 0.0
) -> torch.Tensor:
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
        ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=pos_weight)
        return ((1.0 - p_t) ** focal_gamma * ce).mean()
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)


def token_losses(
    maps: Dict[str, torch.Tensor],
    lamp_tgt: torch.Tensor,
    valid: torch.Tensor,
    state_tgt: torch.Tensor,
    rel_tgt: torch.Tensor,
    w_lamp: float = 0.3,
    w_state: float = 0.2,
    w_rel: float = 0.5,
    lamp_pos_weight: float = 10.0,
    rel_pos_weight: float = 100.0,
    rel_focal_gamma: float = 0.0,
    w_dir: float = 0.0,
    dir_tgt: Optional[torch.Tensor] = None,
    state_weights: Optional[torch.Tensor] = None,
    state_rel_boost: float = 0.0,
    rel_lamp_weight: Optional[float] = None,
    valid_masks=None,
    instance_id=None,
    reduction="token",
    pictogram_tgt=None,
    pictogram_weights=None,
    w_pictogram=0.0,
    instance_weights=None,
) -> Dict[str, torch.Tensor]:
    """Weighted auxiliary losses on rasterized lamp targets.

    Lamp and relevance use all valid tokens. State and direction use annotated
    lamps. Relevance combines separately normalized lamp/background focal BCE;
    state CE uses training-frequency weights and a relevant-lamp multiplier.
    Return the differentiable weighted total and detached component diagnostics.
    """
    if valid_masks is not None or reduction == "instance" or w_pictogram:
        return attribute_losses(
            maps,
            lamp_tgt,
            valid,
            state_tgt,
            rel_tgt,
            dir_tgt,
            pictogram_tgt,
            valid_masks or {},
            instance_id,
            reduction,
            state_weights,
            pictogram_weights,
            w_lamp,
            w_state,
            w_rel,
            w_dir,
            w_pictogram,
            lamp_pos_weight,
            rel_pos_weight,
            rel_focal_gamma,
            state_rel_boost,
            rel_lamp_weight,
            instance_weights,
        )
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
        l_lamp = F.binary_cross_entropy_with_logits(lamp_logit[sel], lamp_y[sel], pos_weight=pw)
    else:
        l_lamp = zero

    # --- state: only tokens inside a lamp box (no state label elsewhere) -----
    pos = v & (lamp_y > 0.5)
    n_pos = int(pos.sum())
    if n_pos > 0 and w_state > 0:
        y_s = state_y[pos]
        per = F.cross_entropy(
            state_logit[pos],
            y_s,
            reduction="none",
            weight=state_weights.to(state_logit.device) if state_weights is not None else None,
        )
        if state_rel_boost > 0:
            per = per * (1.0 + state_rel_boost * rel_y[pos])
        l_state = per.mean()
    else:
        l_state = zero
    # --- relevance: ALL valid tokens (lamp tokens carry their GT flag, the
    # background is an explicit negative). -----------------------------------
    if v.any() and w_rel > 0:
        rpw = torch.tensor(rel_pos_weight, device=rel_logit.device)
        if rel_lamp_weight is None:
            l_rel = _bce(rel_logit[v], rel_y[v], pos_weight=rpw, focal_gamma=rel_focal_gamma)
        else:
            if not 0 <= rel_lamp_weight <= 1:
                raise ValueError("rel_lamp_weight must be in [0,1]")
            # Separate actual lamps (including lit, irrelevant hard negatives)
            # from background. Loss mass no longer shrinks with resolution.
            bg = v & ~pos
            lamp_loss = (
                _bce(rel_logit[pos], rel_y[pos], pos_weight=rpw, focal_gamma=rel_focal_gamma)
                if pos.any()
                else zero
            )
            bg_loss = _bce(rel_logit[bg], rel_y[bg], focal_gamma=rel_focal_gamma) if bg.any() else zero
            a = float(rel_lamp_weight) if pos.any() else 0.0
            b = 1 - float(rel_lamp_weight) if bg.any() else 0.0
            l_rel = (a * lamp_loss + b * bg_loss) / max(a + b, 1e-8)
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
    return {
        "total": total,
        "lamp": l_lamp.detach(),
        "state": l_state.detach(),
        "rel": l_rel.detach(),
        "dir": l_dir.detach() if torch.is_tensor(l_dir) else zero,
    }


def masked_mean(values, mask, instances=None, per_image=False):
    """Equal instance means inside each image, then equal nonempty image means."""
    zero = values.sum() * 0
    if not mask.any():
        return zero
    if instances is None:
        if not per_image:
            return values[mask].mean()
        count = mask.sum(1)
        image_means = (values * mask).sum(1) / count.clamp_min(1)
        return image_means[count > 0].mean()
    mask = mask & (instances >= 0)
    capacity = max(1, int(instances.max().item()) + 1)
    indices = instances.clamp_min(0)
    sums = values.new_zeros((len(values), capacity)).scatter_add(1, indices, values * mask)
    counts = values.new_zeros((len(values), capacity)).scatter_add(1, indices, mask.to(values.dtype))
    present = counts > 0
    image_count = present.sum(1)
    means = ((sums / counts.clamp_min(1)) * present).sum(1) / image_count.clamp_min(1)
    return means[image_count > 0].mean() if (image_count > 0).any() else zero


def attribute_losses(
    maps,
    lamp,
    valid,
    state,
    rel,
    direction,
    pictogram,
    masks,
    instances,
    reduction,
    state_weights,
    pictogram_weights,
    w_lamp,
    w_state,
    w_rel,
    w_dir,
    w_pictogram,
    lamp_pw,
    rel_pw,
    gamma,
    state_boost,
    rel_lamp_weight,
    instance_weights=None,
):
    batch = lamp.shape[0]
    flatten = lambda x: x.reshape(batch, -1)
    lamp, valid, state, rel = map(flatten, (lamp, valid, state, rel))
    ids = flatten(instances) if instances is not None and reduction == "instance" else None
    if reduction == "instance" and ids is None:
        raise ValueError("Instance reduction requires instance_id")
    image_reduction = reduction == "instance"
    pos = valid & (lamp > 0.5)
    zero = maps["lamp_logit"].sum() * 0

    def attr_mask(name):
        return valid & flatten(masks[name + "_valid"]) if name + "_valid" in masks else valid

    def instance_mean(values, mask, name):
        if image_reduction and instance_weights and name in instance_weights:
            weights = flatten(instance_weights[name]) * mask
            count = weights.sum(1)
            means = (values * weights).sum(1) / count.clamp_min(1e-8)
            return means[count > 0].mean() if (count > 0).any() else zero
        return masked_mean(values, mask, ids, image_reduction)

    def binary(logits, labels, positive_weight, focal=0):
        logits = flatten(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=logits.new_tensor(positive_weight)
        )
        if focal:
            p = logits.sigmoid()
            ce = ce * (1 - torch.where(labels > 0.5, p, 1 - p)).pow(focal)
        return ce

    lamp_loss = (
        masked_mean(binary(maps["lamp_logit"], lamp, lamp_pw), valid, per_image=image_reduction)
        if w_lamp
        else zero
    )
    rel_mask = attr_mask("rel")
    rel_values = binary(maps["rel_logit"], rel, rel_pw, gamma)
    if rel_lamp_weight is None:
        rel_loss = masked_mean(rel_values, rel_mask, per_image=image_reduction)
    else:
        if not 0 <= rel_lamp_weight <= 1:
            raise ValueError("rel_lamp_weight must be in [0,1]")
        lp, bg = pos & rel_mask, valid & ~pos
        a = rel_lamp_weight if lp.any() else 0
        b = 1 - rel_lamp_weight if bg.any() else 0
        rel_loss = (
            a * instance_mean(rel_values, lp, "rel")
            + b * masked_mean(binary(maps["rel_logit"], rel, 1, gamma), bg, per_image=image_reduction)
        ) / max(a + b, 1e-8)

    def categorical(key, targets, weights=None, boost=False):
        logits = maps.get(key + "_logit")
        if logits is None or targets is None:
            return zero
        targets = flatten(targets).long()
        mask = pos & attr_mask(key) & (targets >= 0)
        per = F.cross_entropy(
            logits.transpose(1, 2).float(),
            targets.clamp_min(0),
            reduction="none",
            weight=weights.to(logits.device) if weights is not None else None,
        )
        if boost:
            per = per * (1 + state_boost * rel * rel_mask)
        return instance_mean(per, mask, key)

    state_loss = categorical("state", state, state_weights, True) if w_state else zero
    dir_loss = categorical("dir", direction) if w_dir else zero
    pic_loss = categorical("pictogram", pictogram, pictogram_weights) if w_pictogram else zero
    components = dict(lamp=lamp_loss, state=state_loss, rel=rel_loss, dir=dir_loss, pictogram=pic_loss)
    total = sum(
        weight * value
        for weight, value in zip((w_lamp, w_state, w_rel, w_dir, w_pictogram), components.values())
    )
    return dict(total=total, **{key: value.detach() for key, value in components.items()})


def consistency_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Symmetric KL between two augmented views (mean over batch)."""
    t = max(temperature, 1e-3)
    la = F.log_softmax(logits_a / t, dim=-1)
    lb = F.log_softmax(logits_b / t, dim=-1)
    pa, pb = la.exp(), lb.exp()
    kl = (F.kl_div(la, pb, reduction="batchmean") + F.kl_div(lb, pa, reduction="batchmean")) * 0.5
    return kl * (t * t)  # temperature scaling (Hinton distillation)
