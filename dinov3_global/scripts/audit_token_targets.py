"""Audit: how much lamp supervision did the per-box ignore band delete?

Compares the v2 rasteriser (per-box band) with the fixed one (band minus the
union of all box interiors) on every parseable DTLD train frame. No image
decode needed - runs over the label JSON only.

  python dinov3_global/scripts/audit_token_targets.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dinov3_global.dinov3_global import data_global as dg  # noqa: E402
from dinov3_global.dinov3_global.dtld_native import load_split, parse_frame  # noqa: E402


def token_targets_v2(boxes, relevance, state, label_crop_sides=114,
                     target_hw=(720, 1280), ignore_band_px=16.0):
    """The old per-box-band behaviour, for comparison only."""
    out = dg.token_targets(boxes, relevance, state, label_crop_sides,
                           target_hw, ignore_band_px)
    # recompute the OLD valid mask: band of each box applied per-box
    th, tw = target_hw
    valid = np.ones((dg.GRID_H, dg.GRID_W), dtype=bool)
    if len(boxes) == 0:
        return out, valid
    sx = tw / float(dg.SRC_W - 2 * label_crop_sides)
    sy = th / float(dg.SRC_H)
    b = np.asarray(boxes, dtype=np.float32)
    cx, cy = dg._grid_centers(target_hw)
    gx, gy = cx[None, :], cy[:, None]
    for i in range(len(b)):
        X1, X2 = (b[i, 0] - label_crop_sides) * sx, (b[i, 2] - label_crop_sides) * sx
        Y1, Y2 = b[i, 1] * sy, b[i, 3] * sy
        if X2 <= 0 or X1 >= tw or Y2 <= 0 or Y1 >= th:
            continue
        inside = (gx >= X1) & (gx <= X2) & (gy >= Y1) & (gy <= Y2)
        if not inside.any():
            ix = int(np.clip((X1 + X2) / 2 / (tw / dg.GRID_W), 0, dg.GRID_W - 1))
            iy = int(np.clip((Y1 + Y2) / 2 / (th / dg.GRID_H), 0, dg.GRID_H - 1))
            inside = np.zeros_like(inside)
            inside[iy, ix] = True
        band = ((gx >= X1 - ignore_band_px) & (gx <= X2 + ignore_band_px)
                & (gy >= Y1 - ignore_band_px) & (gy <= Y2 + ignore_band_px))
        valid[band & ~inside] = False
    return out, valid


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    label_dir = os.path.join(root, "datasets", "DTLD", "v2.0")
    entries = load_split(label_dir, "train")
    n = frames_affected = lamp_tok = lamp_tok_lost = 0
    state_lost = rel_lost = 0
    for e in entries:
        f = parse_frame(e)
        if f is None:
            continue
        new = dg.token_targets(f.boxes, f.relevance, f.state)
        _, valid_old = token_targets_v2(f.boxes, f.relevance, f.state)
        lamp = new["lamp"] > 0.5
        n += 1
        lost = lamp & ~valid_old
        if lost.any():
            frames_affected += 1
        lamp_tok += int(lamp.sum())
        lamp_tok_lost += int(lost.sum())
        state_lost += int(lost.sum())
    print(f"frames audited            : {n}")
    print(f"frames with lost targets  : {frames_affected} "
          f"({100*frames_affected/max(n,1):.1f}%)")
    print(f"lamp tokens total         : {lamp_tok}")
    print(f"lamp tokens losing ALL supervision (v2 bug): {lamp_tok_lost} "
          f"({100*lamp_tok_lost/max(lamp_tok,1):.2f}%)")
    print(f"-> the fixed rasteriser recovers {lamp_tok_lost} state/rel/lamp "
          f"supervision targets across {frames_affected} frames")


if __name__ == "__main__":
    main()
