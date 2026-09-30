"""Native DTLD annotation parsing for DinoV3 global supervision.

Native label space in this release (measured on v2.0):
  relevance: relevant / not_relevant          (binary, NO unknown class)
  direction: front / back / left / right      (the rear-housing signal; spec's
                                              "orientation" actually lives here)
  orientation: vertical / horizontal          (not useful as supervision)
  state: red / green / yellow / red_yellow / off / unknown
  pictogram: circle / arrow_* / pedestrian / bicycle / tram / ... / unknown
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Fixed enum order (must match model aux heads and losses.py).
DIRECTIONS = ["back", "front", "left", "right"]
STATES = ["green", "off", "red", "red_yellow", "unknown", "yellow"]
PICTOGRAMS = [
    "arrow_left", "arrow_right", "arrow_straight", "arrow_straight_left",
    "bicycle", "circle", "pedestrian", "pedestrian_bicycle", "tram", "unknown",
]
REL2ID = {"not_relevant": 0, "relevant": 1}

IMG_W, IMG_H = 2048, 1024


@dataclass
class FrameSample:
    image_path: str
    boxes: np.ndarray      # (N,4) xyxy in px
    relevance: np.ndarray  # (N,) 0/1
    direction: np.ndarray
    state: np.ndarray
    pictogram: np.ndarray
    track_ids: List[str]
    timestamp: float = 0.0
    sequence: str = ""     # drive folder parsed from DTLD image_path


def _seq_of(dtld_path: str) -> str:
    # "./Berlin/Berlin1/2015-04-17_10-50-41/<file>.tiff" -> "Berlin/Berlin1/2015-04-17_10-50-41"
    p = dtld_path.replace("\\", "/").lstrip("./")
    parts = p.split("/")[:-1]
    return "/".join(parts)


def load_split(label_dir: str, split: str) -> List[Dict[str, Any]]:
    path = os.path.join(label_dir, f"DTLD_{split}.json")
    with open(path) as f:
        return json.load(f)["images"]


def image_file_for(img_root: str, split: str, dtld_path: str) -> str:
    base = os.path.basename(dtld_path).rsplit(".", 1)[0] + ".jpg"
    return os.path.join(img_root, split, base)


def parse_frame(entry: Dict[str, Any], min_box_w: float = 1.0) -> Optional[FrameSample]:
    boxes, rel, drc, sta, pic, tids = [], [], [], [], [], []
    H, W = IMG_H, IMG_W
    for lab in entry.get("labels", []):
        a = lab.get("attributes", {})
        try:
            r = REL2ID[a["relevance"]]
        except KeyError:
            continue  # skip anything outside the documented binary enum
        x1, y1 = float(lab["x"]), float(lab["y"])
        bw, bh = float(lab["w"]), float(lab["h"])
        if bw < min_box_w or bh < 1.0:
            continue
        # clip to frame
        x2, y2 = min(x1 + bw, W - 1), min(y1 + bh, H - 1)
        x1c, y1c = max(x1, 0.0), max(y1, 0.0)
        if x2 <= x1c or y2 <= y1c:
            continue
        boxes.append([x1c, y1c, x2, y2])
        rel.append(r)
        d = a.get("direction", "front")
        s = a.get("state", "unknown")
        p = a.get("pictogram", "unknown")
        drc.append(DIRECTIONS.index(d) if d in DIRECTIONS else DIRECTIONS.index("front"))
        sta.append(STATES.index(s) if s in STATES else STATES.index("unknown"))
        pic.append(PICTOGRAMS.index(p) if p in PICTOGRAMS else PICTOGRAMS.index("unknown"))
        tids.append(str(lab.get("track_id", "")))
    if not boxes:
        return None
    return FrameSample(
        image_path=entry.get("image_path", ""),
        boxes=np.asarray(boxes, dtype=np.float32),
        relevance=np.asarray(rel, dtype=np.int64),
        direction=np.asarray(drc, dtype=np.int64),
        state=np.asarray(sta, dtype=np.int64),
        pictogram=np.asarray(pic, dtype=np.int64),
        track_ids=tids,
        timestamp=float(entry.get("time_stamp", 0.0) or 0.0),
        sequence=_seq_of(entry.get("image_path", "")),
    )
