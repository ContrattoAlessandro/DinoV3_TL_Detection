"""Audited city-disjoint folds and immutable evaluation manifests."""

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
from .metrics import to_jsonable
from .dtld import _seq_of


def make_folds(items, k):
    """Greedy balanced folds over city groups (city = first path component)."""
    by_city = defaultdict(int)
    for it in items:
        by_city[it["city"]] += 1
    if not 2 <= k <= len(by_city):
        raise ValueError("fold count must be between 2 and the number of cities")
    order = sorted(by_city, key=lambda c: (-by_city[c], c))
    folds = [[] for _ in range(k)]
    sizes = [0] * k
    for c in order:
        j = int(np.argmin(sizes))
        folds[j].append(c)
        sizes[j] += by_city[c]
    return folds, by_city


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(value), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def fold_manifest(items, folds, data, annotation_sha256):
    """Record membership and reject leakage or unscorable city folds."""
    identities = [
        (it["entry"]["image_path"], it["city"], int(it["y"]), int(it.get("pseudo_nor", 0))) for it in items
    ]
    if len({row[0] for row in identities}) != len(identities):
        raise ValueError("duplicate frame identities in training split")
    cities = {it["city"] for it in items}
    assigned = [c for group in folds for c in group]
    if len(assigned) != len(set(assigned)) or set(assigned) != cities:
        raise ValueError("folds must assign every city exactly once")
    audited = []
    for fi, group in enumerate(folds):
        tr = [it for it in items if it["city"] not in group]
        va = [it for it in items if it["city"] in group]
        tr_seq = {_seq_of(it["entry"]["image_path"]) for it in tr}
        va_seq = {_seq_of(it["entry"]["image_path"]) for it in va}
        overlap = tr_seq & va_seq
        tr_hist = np.bincount([it["y"] for it in tr], minlength=3).tolist()
        va_hist = np.bincount([it["y"] for it in va], minlength=3).tolist()
        if overlap or min(tr_hist) == 0 or min(va_hist) == 0:
            raise ValueError(f"fold {fi}: session overlap or missing global class")
        audited.append(
            {
                "fold": fi,
                "val_cities": group,
                "n_train": len(tr),
                "n_val": len(va),
                "train_hist": tr_hist,
                "val_hist": va_hist,
                "n_train_sequences": len(tr_seq),
                "n_val_sequences": len(va_seq),
                "session_overlap": len(overlap),
                "val_pseudo_nor": sum(it.get("pseudo_nor", 0) for it in va),
                "val_membership_sha256": digest(sorted(it["entry"]["image_path"] for it in va)),
            }
        )
    keys = (
        "label_dir",
        "img_root",
        "crop_sides",
        "label_crop_sides",
        "target_hw",
        "label_policy",
        "policy_weight",
    )
    return {
        "schema_version": 1,
        "source_split": "DTLD_train.json",
        "annotation_sha256": annotation_sha256,
        "data": {k: data.get(k) for k in keys},
        "items_sha256": digest(sorted(identities)),
        "n_items": len(items),
        "class_names": ["RR", "RG", "NoR"],
        "folds": audited,
        "purpose": "Configuration/epoch selection inside official train; not the paper test split",
    }
