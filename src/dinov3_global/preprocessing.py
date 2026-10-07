"""Shared full-frame image/box transforms; legacy resize remains opt-in by config."""

from dataclasses import dataclass, asdict
import random

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class LetterboxTransform:
    source_hw: tuple
    target_hw: tuple
    resized_hw: tuple
    offset_yx: tuple

    def boxes(self, boxes):
        result = np.asarray(boxes, dtype=np.float32).copy()
        sh, sw = self.source_hw
        rh, rw = self.resized_hw
        oy, ox = self.offset_yx
        result[:, [0, 2]] = result[:, [0, 2]] * (rw / sw) + ox
        result[:, [1, 3]] = result[:, [1, 3]] * (rh / sh) + oy
        return result

    def patch_metadata(self, patch=16):
        h, w = self.target_hw
        rh, rw = self.resized_hw
        oy, ox = self.offset_yx
        yy, xx = np.meshgrid(np.arange(h // patch), np.arange(w // patch), indexing="ij")
        # A border patch remains usable if it intersects the image at all.
        mask = (
            (xx * patch < ox + rw)
            & ((xx + 1) * patch > ox)
            & (yy * patch < oy + rh)
            & ((yy + 1) * patch > oy)
        )
        x = np.clip(((xx + 0.5) * patch - ox) / rw, 0, 1).astype(np.float32)
        y = np.clip(((yy + 0.5) * patch - oy) / rh, 0, 1).astype(np.float32)
        geometry = np.stack([x, y, 2 * x - 1, 1 - y], -1).reshape(-1, 4)
        return mask, geometry

    def record(self):
        return asdict(self)


def letterbox(image, target_hw=(720, 1280), zoom=1.0, random_placement=False):
    if not 0 < zoom <= 1:
        raise ValueError("Letterbox zoom must be in (0,1]")
    h, w = target_hw
    if h < 16 or w < 16 or h % 16 or w % 16:
        raise ValueError("Canvas dimensions must be positive multiples of 16")
    sw, sh = image.size
    scale = min(w / sw, h / sh) * zoom
    rh, rw = max(1, round(sh * scale)), max(1, round(sw * scale))
    oy = random.randint(0, h - rh) if random_placement else (h - rh) // 2
    ox = random.randint(0, w - rw) if random_placement else (w - rw) // 2
    canvas = Image.new("RGB", (w, h), (124, 116, 104))
    canvas.paste(image.convert("RGB").resize((rw, rh), Image.Resampling.BICUBIC), (ox, oy))
    return canvas, LetterboxTransform((sh, sw), (h, w), (rh, rw), (oy, ox))


def rasterize_attributes(
    boxes, relevance, state, direction, pictogram, target_hw, content_mask, ignore_band_px=16.0
):
    """Union lampness, nearest instance ownership, independent conflict masks."""
    h, w = target_hw
    gh, gw = h // 16, w // 16
    yy, xx = np.meshgrid((np.arange(gh) + 0.5) * 16, (np.arange(gw) + 0.5) * 16, indexing="ij")
    lamp = np.zeros((gh, gw), np.float32)
    owner = np.full((gh, gw), -1, np.int64)
    best = np.full((gh, gw), np.inf)
    band = np.zeros((gh, gw), bool)
    values = dict(
        state=np.zeros((gh, gw), np.int64),
        rel=np.zeros((gh, gw), np.float32),
        dir=np.full((gh, gw), -1, np.int64),
        pictogram=np.full((gh, gw), -1, np.int64),
    )
    first = {key: np.full((gh, gw), -1, np.int64) for key in values}
    conflict = {key: np.zeros((gh, gw), bool) for key in values}
    attributes = dict(state=state, rel=relevance, dir=direction, pictogram=pictogram)
    instance_masks = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h or x2 <= x1 or y2 <= y1:
            continue
        inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2) & content_mask
        if not inside.any():
            ix, iy = int(np.clip((x1 + x2) / 32, 0, gw - 1)), int(np.clip((y1 + y2) / 32, 0, gh - 1))
            inside[iy, ix] = bool(content_mask[iy, ix])
        instance_masks.append(inside)
        previous = lamp > 0.5
        distance = (xx - (x1 + x2) / 2) ** 2 + (yy - (y1 + y2) / 2) ** 2
        take = inside & (distance < best)
        for key, targets in attributes.items():
            value = int(targets[i])
            conflict[key] |= inside & previous & (first[key] != value)
            first[key][inside & ~previous] = value
            values[key][take] = value
        best[take], owner[take] = distance[take], i
        lamp[inside] = 1
        band |= (
            (xx >= x1 - ignore_band_px)
            & (xx <= x2 + ignore_band_px)
            & (yy >= y1 - ignore_band_px)
            & (yy <= y2 + ignore_band_px)
        )
    valid = content_mask & (~band | (lamp > 0.5))
    result = dict(lamp=lamp, valid=valid, instance_id=owner, **values)
    result.update({key + "_valid": valid & ~conflict[key] for key in values})
    # A shared, agreeing patch belongs to every annotated instance covering it.
    # Its weight is the sum of their normalized contributions, rather than
    # silently dropping an instance whose nearest-center ownership is empty.
    for key in values:
        weight = np.zeros((gh, gw), np.float32)
        n_instances = 0
        for inside in instance_masks:
            selected = inside & result[key + "_valid"] & (values[key] >= 0)
            count = int(selected.sum())
            if count:
                weight[selected] += 1 / count
                n_instances += 1
        result[key + "_instance_weight"] = weight / max(n_instances, 1)
    return result


def batch_model_kwargs(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True) for key in ("content_mask", "geometry") if key in batch
    }


def preprocess_for_config(path, cfg, crop_sides=0):
    """Return the uint8 tensor, visible canvas, and optional head metadata."""
    from .inference import preprocess

    target_hw = tuple(cfg["data"]["target_hw"])
    if cfg["data"].get("preprocessing", "legacy") != "letterbox":
        tensor, array = preprocess(path, crop_sides, target_hw)
        return tensor, array, {}
    if crop_sides:
        raise ValueError("Full-frame letterboxing requires crop_sides=0")
    with Image.open(path) as source:
        image, transform = letterbox(source, target_hw)
    array = np.array(image, dtype=np.uint8)
    mask, geometry = transform.patch_metadata()
    return (
        torch.from_numpy(array).permute(2, 0, 1),
        array,
        {"content_mask": torch.from_numpy(mask), "geometry": torch.from_numpy(geometry)},
    )
