"""DTLD image labels and box-derived token targets.

Default labels prioritize relevant red/yellow/red-yellow (RR), then relevant
green (RG), and map all remaining scenes to NoR. Relevant off/unknown-only
scenes are recorded separately as pseudo-NoR for diagnostic slices.

Images are decoded as uint8. Boxes in the native 2048x1024 coordinate system
are mapped through the 114-pixel side crop and bicubic 1280x720 resize to a
45x80 grid. A surrounding ignore band prevents ambiguous background targets;
tokens with conflicting overlapping boxes are excluded from auxiliary losses.
Inference uses image pixels alone.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Native DTLD parsing is kept alongside the DinoV3 dataset.
from .dtld import (
    STATES,
    _seq_of,
    image_file_for,
    load_split,
    parse_frame,
)

# STATES = ["green","off","red","red_yellow","unknown","yellow"]
RR_STATES = {"red", "yellow", "red_yellow"}
GREEN_STATE = "green"

SRC_W, SRC_H = 2048, 1024
GRID_H, GRID_W = 45, 80  # default resolution: token grid at 1280x720 / patch 16


def grid_shape(target_hw: Tuple[int, int], patch: int = 16) -> Tuple[int, int]:
    """Token grid (h, w) for a target resolution. 720x1280 / 16 -> (45, 80)."""
    th, tw = target_hw
    return th // patch, tw // patch


# --------------------------------------------------------------------------- #
# label mapping
# --------------------------------------------------------------------------- #
def global_label_from_states(relevance: np.ndarray, state: np.ndarray) -> Tuple[int, Dict[str, int]]:
    """Map per-lamp arrays to global class id. Returns (label, debug)."""
    rel = np.asarray(relevance).astype(int)
    st = np.asarray(state).astype(int)
    debug = {"n_rel_rr": 0, "n_rel_green": 0, "n_rel_off_unknown": 0}
    if len(rel) == 0:
        return 2, debug
    for r, s in zip(rel.tolist(), st.tolist()):
        if r != 1:
            continue
        sname = STATES[s] if 0 <= s < len(STATES) else "unknown"
        if sname in RR_STATES:
            debug["n_rel_rr"] += 1
        elif sname == GREEN_STATE:
            debug["n_rel_green"] += 1
        else:
            debug["n_rel_off_unknown"] += 1
    if debug["n_rel_rr"] > 0:
        return 0, debug
    if debug["n_rel_green"] > 0:
        return 1, debug
    return 2, debug


def _grid_centers(target_hw: Tuple[int, int], patch: int = 16):
    th, tw = target_hw
    gh, gw = grid_shape(target_hw, patch)
    cx = (np.arange(gw) + 0.5) * (tw / gw)  # [gw] px
    cy = (np.arange(gh) + 0.5) * (th / gh)  # [gh] px
    return cx, cy


def token_targets(
    boxes_xyxy: np.ndarray,
    relevance: np.ndarray,
    state: np.ndarray,
    label_crop_sides: int = 114,
    target_hw: Tuple[int, int] = (720, 1280),
    ignore_band_px: float = 16.0,
    direction: Optional[np.ndarray] = None,
    patch: int = 16,
) -> Dict[str, np.ndarray]:
    """Rasterise lamp boxes onto the patch grid.

    Positive token = its centre falls inside a lamp box (or, for boxes smaller
    than a token, the token containing the box centre). Tokens within
    ``ignore_band_px`` of a box but outside **all** boxes are ignored
    (uncertain border tokens). Tokens inside any box always keep their targets,
    even when a neighbouring box's ignore band overlaps them (clustered lamps).

    Overlap conflict policy: when two boxes cover the same token (DTLD
    signal heads stack 2-3 lamps), the token is assigned to the *nearest* box
    centre instead of the last box in label order (the old last-writer-wins made
    the target depend on file order). ``lamp`` stays a union of all boxes.

    Returns lamp/valid/state/rel grids and, when ``direction`` is given, a
    ``dir`` grid (int64, -1 outside lamp tokens) for the direction aux head.
    """
    th, tw = target_hw
    gh, gw = grid_shape(target_hw, patch)
    lamp = np.zeros((gh, gw), dtype=np.float32)
    valid = np.ones((gh, gw), dtype=bool)
    state_t = np.zeros((gh, gw), dtype=np.int64)
    rel_t = np.zeros((gh, gw), dtype=np.float32)
    dir_t = np.full((gh, gw), -1, dtype=np.int64) if direction is not None else None
    if len(boxes_xyxy) == 0:
        out = {"lamp": lamp, "valid": valid, "state": state_t, "rel": rel_t}
        if dir_t is not None:
            out["dir"] = dir_t
        return out

    sx = tw / float(SRC_W - 2 * label_crop_sides)
    sy = th / float(SRC_H)
    b = np.asarray(boxes_xyxy, dtype=np.float32)
    x1 = (b[:, 0] - label_crop_sides) * sx
    x2 = (b[:, 2] - label_crop_sides) * sx
    y1 = b[:, 1] * sy
    y2 = b[:, 3] * sy

    cx, cy = _grid_centers(target_hw, patch)
    gx = cx[None, :]  # [1,gw]
    gy = cy[:, None]  # [gh,1]
    inside_any = np.zeros((gh, gw), dtype=bool)
    band_any = np.zeros((gh, gw), dtype=bool)
    best_d = np.full((gh, gw), np.inf, dtype=np.float64)
    for i in range(len(b)):
        X1, X2, Y1, Y2 = x1[i], x2[i], y1[i], y2[i]
        if X2 <= 0 or X1 >= tw or Y2 <= 0 or Y1 >= th:
            continue
        inside = (gx >= X1) & (gx <= X2) & (gy >= Y1) & (gy <= Y2)
        if not inside.any():  # box smaller than one token: snap to its centre
            ix = int(np.clip((X1 + X2) / 2 / (tw / gw), 0, gw - 1))
            iy = int(np.clip((Y1 + Y2) / 2 / (th / gh), 0, gh - 1))
            inside = np.zeros_like(inside)
            inside[iy, ix] = True
        lamp[inside] = 1.0
        # nearest-box-centre wins on shared tokens (see docstring)
        d = (gx - (X1 + X2) / 2) ** 2 + (gy - (Y1 + Y2) / 2) ** 2
        take = inside & (d < best_d)
        best_d[take] = d[take]
        state_t[take] = int(state[i])
        rel_t[take] = float(relevance[i])
        if dir_t is not None:
            dir_t[take] = int(direction[i])
        inside_any |= inside
        band_any |= (
            (gx >= X1 - ignore_band_px)
            & (gx <= X2 + ignore_band_px)
            & (gy >= Y1 - ignore_band_px)
            & (gy <= Y2 + ignore_band_px)
        )
    # Ignore band = uncertainty about a box's border tokens - but a token that
    # lies inside ANY lamp box carries supervision from that box and must stay
    # valid. Apply all positive boxes before the surrounding ignore bands.
    valid[band_any & ~inside_any] = False
    out = {"lamp": lamp, "valid": valid, "state": state_t, "rel": rel_t}
    if dir_t is not None:
        out["dir"] = dir_t
    return out


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class DTLDGlobalDataset(Dataset):
    """One item = one frame (uint8 CHW tensor) + global label + token targets."""

    def __init__(
        self,
        label_dir: str,
        img_root: str,
        split: str,
        train: bool = False,
        crop_sides: int = 114,
        label_crop_sides: int = 114,
        target_hw: Tuple[int, int] = (720, 1280),
        photometric: bool = False,
        blur: bool = False,  # kept for API compat (GPU aug now)
        label_policy: str = "map_to_nor",
        policy_weight: float = 0.3,
        items: Optional[List[Dict[str, Any]]] = None,
    ):
        self.split = split
        self.img_root = img_root
        self.train = train
        self.crop_sides = crop_sides
        self.label_crop_sides = label_crop_sides
        self.target_hw = tuple(target_hw)
        self.label_policy = label_policy
        self.policy_weight = policy_weight
        if items is not None:
            self.items = list(items)  # pre-parsed subset (train/val carve-up)
            self.n_rel_off_unknown = sum(it.get("pseudo_nor", 0) for it in self.items)
            return
        entries = load_split(label_dir, split)
        self.items: List[Dict[str, Any]] = []
        n_off_unknown = 0
        for e in entries:
            f = parse_frame(e)
            if f is None:
                continue
            fp = image_file_for(img_root, split, e.get("image_path", ""))
            if not os.path.isfile(fp):
                continue
            y, dbg = global_label_from_states(f.relevance, f.state)
            pseudo_nor = int(
                y == 2 and dbg["n_rel_off_unknown"] > 0 and dbg["n_rel_rr"] == 0 and dbg["n_rel_green"] == 0
            )
            n_off_unknown += dbg["n_rel_off_unknown"]
            if pseudo_nor and label_policy == "exclude":
                continue
            weight = policy_weight if (pseudo_nor and label_policy == "downweight") else 1.0
            self.items.append(
                {
                    "entry": e,
                    "frame": f,
                    "y": y,
                    "weight": weight,
                    "pseudo_nor": pseudo_nor,
                    "city": _seq_of(e.get("image_path", "")).split("/")[0],
                    "n_lamps": int(len(f.boxes)),
                    "max_lamp_h": float((f.boxes[:, 3] - f.boxes[:, 1]).max() * self.target_hw[0] / SRC_H),
                }
            )
        self.n_rel_off_unknown = n_off_unknown

    def __len__(self) -> int:
        return len(self.items)

    def _preprocess(self, img: Image.Image) -> Image.Image:
        w, h = img.size  # native 2048x1024 or the prepared 1280x720 dump
        cs = self.crop_sides
        if w == SRC_W and h == SRC_H and cs > 0:
            img = img.crop((cs, 0, w - cs, h))
        th, tw = self.target_hw
        if img.size == (tw, th):
            return img  # fast path: offline dump already at target size
        return img.resize((tw, th), Image.BICUBIC)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        it = self.items[i]
        entry = it["entry"]
        fp = image_file_for(self.img_root, self.split, entry.get("image_path", ""))
        img = Image.open(fp).convert("RGB")
        img = self._preprocess(img)
        arr = np.array(img, dtype=np.uint8)  # copied, writable (M0: no float32 on host)

        f = it.get("frame")  # parsed once at initialization
        if f is None:
            f = parse_frame(entry)  # boxes/attrs for token supervision
        if f is None:
            tt = token_targets(
                np.zeros((0, 4), np.float32),
                np.zeros(0),
                np.zeros(0),
                self.label_crop_sides,
                self.target_hw,
                direction=np.zeros(0, np.int64),
            )
        else:
            tt = token_targets(
                f.boxes, f.relevance, f.state, self.label_crop_sides, self.target_hw, direction=f.direction
            )

        return {
            "image": torch.from_numpy(arr).permute(2, 0, 1),  # uint8 CHW
            "label": torch.tensor(it["y"], dtype=torch.long),
            "weight": torch.tensor(it["weight"], dtype=torch.float32),
            "lamp_tgt": torch.from_numpy(tt["lamp"]),
            "valid": torch.from_numpy(tt["valid"]),
            "state_tgt": torch.from_numpy(tt["state"]),
            "rel_tgt": torch.from_numpy(tt["rel"]),
            "dir_tgt": torch.from_numpy(tt["dir"]),
            "city": it["city"],
            "max_lamp_h": torch.tensor(it["max_lamp_h"], dtype=torch.float32),
            "n_lamps": it["n_lamps"],
            "path": fp,
        }


def collate_global(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),  # uint8
        "label": torch.stack([b["label"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
        "lamp_tgt": torch.stack([b["lamp_tgt"] for b in batch]),
        "valid": torch.stack([b["valid"] for b in batch]),
        "state_tgt": torch.stack([b["state_tgt"] for b in batch]),
        "rel_tgt": torch.stack([b["rel_tgt"] for b in batch]),
        "dir_tgt": torch.stack([b["dir_tgt"] for b in batch]),
        "city": [b["city"] for b in batch],
        "max_lamp_h": torch.stack([b["max_lamp_h"] for b in batch]),
        "n_lamps": [b["n_lamps"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def state_histogram(items_or_ds, n_state: int = 6) -> np.ndarray:
    """Lamp-state counts over cached frames (for class-balanced state CE).
    Accepts a DTLDGlobalDataset or a plain list of item dicts. Order matches
    dtld_native.STATES."""
    items = getattr(items_or_ds, "items", items_or_ds)
    counts = np.zeros(n_state, dtype=np.float64)
    for it in items:
        f = it.get("frame")
        if f is None:
            continue
        for s in np.asarray(f.state).tolist():
            if 0 <= int(s) < n_state:
                counts[int(s)] += 1
    return counts


def class_frequencies(ds: DTLDGlobalDataset) -> np.ndarray:
    ys = np.array([it["y"] for it in ds.items], dtype=int)
    freq = np.bincount(ys, minlength=3).astype(float)
    return freq / max(freq.sum(), 1)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def _reshuffle(seqs: List[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    order = seqs[:]
    rng.shuffle(order)
    return order


def build_train_val_datasets(
    label_dir: str,
    img_root: str,
    crop_sides: int = 114,
    label_crop_sides: int = 114,
    target_hw: Tuple[int, int] = (720, 1280),
    photometric: bool = False,
    blur: bool = False,
    label_policy: str = "map_to_nor",
    policy_weight: float = 0.3,
    val_frac: float = 0.15,
    val_seed: int = 0,
    min_per_class: int = 30,
) -> Tuple[DTLDGlobalDataset, DTLDGlobalDataset, Dict[str, Any]]:
    """Carve a session-disjoint val set out of the official train split.

    Frames ~0.66s apart show the same intersection, so splitting by *sequence*
    (drive folder from image_path), not by frame, is required to avoid leakage.
    Deterministic: tries seeds val_seed..val_seed+99, keeps the first split whose
    val set holds >= min_per_class samples of every global class.

    The official test split is never touched here - it stays held-out for the
    final unbiased report in scripts/evaluate.py.
    """
    full = DTLDGlobalDataset(
        label_dir,
        img_root,
        "train",
        train=False,
        crop_sides=crop_sides,
        label_crop_sides=label_crop_sides,
        target_hw=target_hw,
        label_policy=label_policy,
        policy_weight=policy_weight,
    )
    seq_of_item = [_seq_of(it["entry"].get("image_path", "")) for it in full.items]
    seqs = sorted(set(seq_of_item))
    n_val_seq = max(1, round(len(seqs) * val_frac))
    labels = np.array([it["y"] for it in full.items], dtype=int)

    rng = random.Random(val_seed)
    order = seqs[:]
    rng.shuffle(order)
    chosen = None
    for attempt in range(100):
        cand = (
            set(order[:n_val_seq]) if attempt == 0 else set(_reshuffle(seqs, val_seed + attempt)[:n_val_seq])
        )
        mask = np.array([s in cand for s in seq_of_item])
        counts = np.bincount(labels[mask], minlength=3)
        if int(counts.min()) >= min_per_class:
            chosen = cand
            break
    if chosen is None:
        chosen = set(order[:n_val_seq])  # best effort; val will be NoR-poor
    train_items = [it for it, s in zip(full.items, seq_of_item) if s not in chosen]
    val_items = [it for it, s in zip(full.items, seq_of_item) if s in chosen]
    train_ds = DTLDGlobalDataset(
        label_dir,
        img_root,
        "train",
        train=True,
        crop_sides=crop_sides,
        label_crop_sides=label_crop_sides,
        target_hw=target_hw,
        label_policy=label_policy,
        policy_weight=policy_weight,
        items=train_items,
    )
    val_ds = DTLDGlobalDataset(
        label_dir,
        img_root,
        "train",
        train=False,
        crop_sides=crop_sides,
        label_crop_sides=label_crop_sides,
        target_hw=target_hw,
        label_policy=label_policy,
        policy_weight=policy_weight,
        items=val_items,
    )
    info = {
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_val_seq": len(chosen),
        "n_train_seq": len(seqs) - len(chosen),
        "val_hist": np.bincount([it["y"] for it in val_items], minlength=3).tolist(),
        "train_hist": np.bincount([it["y"] for it in train_items], minlength=3).tolist(),
        "cities": sorted({it["city"] for it in full.items}),
    }
    return train_ds, val_ds, info
