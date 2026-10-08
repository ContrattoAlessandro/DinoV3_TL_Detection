"""Paired operating differences and complete-session bootstrap intervals."""

from pathlib import Path

import numpy as np
from .metrics import report

METRICS = (
    "mAP",
    "acc_bal",
    "macro_F1",
    "mean_signal_recall",
    "recall_RR",
    "recall_RG",
    "recall_NoR",
    "precision_RR",
    "precision_RG",
)


def paired_comparison(labels, baseline, candidate, sessions=None, repetitions=2000, seed=0):
    labels, baseline, candidate = map(np.asarray, (labels, baseline, candidate))
    if baseline.shape != candidate.shape or baseline.shape != (len(labels), 3):
        raise ValueError("Paired comparison requires aligned three-class logits")
    old, new = report(labels, baseline), report(labels, candidate)
    deltas = {k: float(new["metrics"][k] - old["metrics"][k]) for k in METRICS}
    transitions = []
    for true in range(3):
        mask = labels == true
        transitions.append(
            np.bincount(3 * baseline[mask].argmax(1) + candidate[mask].argmax(1), minlength=9)
            .reshape(3, 3)
            .tolist()
        )
    result = dict(
        baseline=old,
        candidate=new,
        delta=deltas,
        paired_prediction_transitions=transitions,
    )
    if sessions is not None:
        sessions = np.asarray(sessions)
        if len(sessions) != len(labels):
            raise ValueError("Session membership must align with predictions")
        groups = [np.flatnonzero(sessions == s) for s in np.unique(sessions)]
        rng = np.random.default_rng(seed)
        samples = {k: [] for k in METRICS}
        for _ in range(repetitions):
            idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            if len(np.unique(labels[idx])) != 3:
                continue
            a, b = (
                report(labels[idx], baseline[idx])["metrics"],
                report(labels[idx], candidate[idx])["metrics"],
            )
            for k in METRICS:
                samples[k].append(b[k] - a[k])
        result["session_bootstrap"] = dict(
            seed=seed,
            sessions=len(groups),
            repetitions=repetitions,
            valid_repetitions=len(samples["mAP"]),
            confidence=0.95,
            intervals={k: np.quantile(v, [0.025, 0.975]).tolist() if v else None for k, v in samples.items()},
        )
    else:
        result["uncertainty_scope"] = (
            "Paired counts only; trustworthy independent scene/session groups unavailable."
        )
    return result


def aligned_predictions(dataset, reference, candidate):
    """Match exact image identities and labels before comparing scores."""

    def keys(paths):
        return [Path(p).stem if dataset == "DTLD" else str(p).replace("\\", "/") for p in paths]

    old_keys, new_keys = keys(reference["paths"]), keys(candidate["paths"])
    if len(set(old_keys)) != len(old_keys) or len(set(new_keys)) != len(new_keys):
        raise ValueError(f"Duplicate image identities in {dataset}")
    if set(old_keys) != set(new_keys):
        raise ValueError(f"Candidate {dataset} membership differs from the baseline")
    lookup = {key: index for index, key in enumerate(new_keys)}
    indices = [lookup[key] for key in old_keys]
    if not np.array_equal(reference["labels"], candidate["labels"][indices]):
        raise ValueError(f"Candidate {dataset} labels differ from the baseline")
    logits = candidate["logits"][indices]
    if logits.shape != (len(old_keys), 3) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid {dataset} predictions")
    return reference["labels"], logits


def published_vzc_predictions(candidate, snapshot):
    """Select the published test split from the evaluator's train/test output."""
    images = snapshot["images"]
    if candidate["paths"].tolist() != [entry["file"] for entry in images]:
        raise ValueError("VZC predictions differ from the frozen label snapshot")
    if candidate["labels"].tolist() != [entry["label"] for entry in images]:
        raise ValueError("VZC prediction labels differ from the frozen snapshot")
    mask = np.array([entry["split"] == "test" for entry in images])
    return {key: candidate[key][mask] for key in ("paths", "labels", "logits")}
