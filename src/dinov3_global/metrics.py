"""One-vs-rest AP, classification metrics, slices, and validation calibration.

Undefined AP is recorded as JSON null. Slice mAP averages defined classes;
n_classes identifies slices that cannot be compared as three-class benchmarks.
Temperature scaling uses validation logits and does not guarantee calibration
on a different image distribution.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# Slice evaluation can legitimately have y_pred classes absent from y_true
# (e.g. a city slice with no NoR frames). Metric math is unaffected - silence
# exactly this one informational warning.
import warnings

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _safe_ap(y_bin: np.ndarray, p: np.ndarray) -> float:
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(average_precision_score(y_bin, p))


def to_jsonable(obj):
    """Recursively convert numpy scalars and NaN/inf to strict-JSON values
    (NaN -> null). ``json.dump(..., allow_nan=False)`` then stays valid."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def global_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    logits = np.asarray(logits, dtype=float)
    prob = softmax_np(logits)
    pred = prob.argmax(axis=1)
    out: Dict[str, float] = {}
    for i, k in enumerate(["RR", "RG", "NoR"]):
        out[f"AP_{k}"] = _safe_ap((y_true == i).astype(int), prob[:, i])
    aps = [v for v in (out["AP_RR"], out["AP_RG"], out["AP_NoR"]) if not np.isnan(v)]
    out["mAP"] = float(np.mean(aps)) if aps else float("nan")
    out["acc_bal"] = float(balanced_accuracy_score(y_true, pred))
    out["n"] = int(len(y_true))
    # slice mAP averages over the classes present in the slice - n_classes tells
    # the reader (and the worst-city gate) whether a mAP is 3-class comparable
    out["n_classes"] = int(len(set(np.asarray(y_true).tolist())))
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    out["cm_RR_RR"] = int(cm[0, 0])
    out["cm_RG_RG"] = int(cm[1, 1])
    out["cm_NoR_NoR"] = int(cm[2, 2])
    for i, name in enumerate(("RR", "RG", "NoR")):
        out["recall_" + name] = float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan")
        out["precision_" + name] = float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else 0.0
    out["mean_signal_recall"] = float((out["recall_RR"] + out["recall_RG"]) / 2)
    return out


def sliced_metrics(
    y: np.ndarray,
    logits: np.ndarray,
    cities: Optional[List[str]] = None,
    max_lamp_h: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-city and per-lamp-size metrics. Groups with <20 samples or a single
    class are skipped (AP undefined / misleading)."""
    y = np.asarray(y, dtype=int)
    logits = np.asarray(logits, dtype=float)
    out: Dict[str, Dict[str, float]] = {}
    if cities is not None:
        by = defaultdict(list)
        for i, c in enumerate(cities):
            by[c].append(i)
        city = {}
        for c, idx in sorted(by.items()):
            if len(idx) < 20 or len(set(y[idx].tolist())) < 2:
                continue
            city[c] = global_metrics(y[idx], logits[idx])
        out["city"] = city
    if max_lamp_h is not None:
        h = np.asarray(max_lamp_h, dtype=float)
        buckets = {"small(<16px)": h < 16, "mid(16-48px)": (h >= 16) & (h < 48), "large(>=48px)": h >= 48}
        size = {}
        for k, m in buckets.items():
            if m.sum() < 20 or len(set(y[m].tolist())) < 2:
                continue
            size[k] = global_metrics(y[m], logits[m])
        out["size"] = size
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 50) -> float:
    """Temperature scaling on val logits (Guo et al. 2017). Returns T."""
    lg = torch.tensor(logits, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / logT.exp().clamp_min(1e-3), yy)
        loss.backward()
        return loss

    try:
        opt.step(_closure)
    except Exception:  # noqa: BLE001 - never let calibration kill the run
        return 1.0
    return float(logT.exp().detach().clamp_min(1e-3))


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE on the predicted class probability."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def report(
    y: np.ndarray,
    logits: np.ndarray,
    T: float = 1.0,
    cities: Optional[List[str]] = None,
    max_lamp_h: Optional[np.ndarray] = None,
    target_lamp_h: Optional[np.ndarray] = None,
) -> Dict:
    """AP, argmax decisions, probability quality, class metrics, and slices."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Temperature must be finite and positive")
    y = np.asarray(y, dtype=int)
    logits = np.asarray(logits, dtype=float) / max(T, 1e-3)
    met = global_metrics(y, logits)
    met["temperature"] = float(T)
    probabilities = softmax_np(logits)
    pred = probabilities.argmax(axis=1)
    per_class = classification_report(
        y, pred, labels=[0, 1, 2], target_names=["RR", "RG", "NoR"], output_dict=True, zero_division=0
    )
    met["ece"] = expected_calibration_error(probabilities, y)
    met["accuracy"] = float((pred == y).mean())
    met["macro_F1"] = float(per_class["macro avg"]["f1-score"])
    met["weighted_F1"] = float(per_class["weighted avg"]["f1-score"])
    met["nll"] = float(-np.log(probabilities[np.arange(len(y)), y].clip(1e-12, 1)).mean())
    met["brier"] = float(((probabilities - np.eye(3)[y]) ** 2).sum(axis=1).mean())
    rep = {
        "metrics": met,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "class_order": ["RR", "RG", "NoR"],
    }
    sl = sliced_metrics(y, logits, cities, max_lamp_h)
    if target_lamp_h is not None:
        target_sizes = np.asarray(target_lamp_h, dtype=float).copy()
        target_sizes[target_sizes <= 0] = np.nan
        sl["target_lamp_size"] = sliced_metrics(y, logits, max_lamp_h=target_sizes).get("size", {})
    if sl:
        rep["slices"] = sl
    return rep
