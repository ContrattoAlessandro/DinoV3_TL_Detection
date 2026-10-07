"""Validation-only selection gates and paired session uncertainty."""

import numpy as np
from .metrics import report


def recall_score(rep):
    return 0.5 * sum(rep["per_class"][name]["recall"] for name in ("RR", "RG"))


def baseline_gate(rep):
    return dict(
        mAP=rep["metrics"]["mAP"],
        **{name + "_recall": rep["per_class"][name]["recall"] for name in ("RR", "RG", "NoR")},
    )


def selection_key(rep, baseline, epoch):
    if (
        rep["metrics"]["n_classes"] != 3
        or rep["metrics"]["mAP"] < baseline["mAP"]
        or rep["per_class"]["NoR"]["recall"] < baseline["NoR_recall"] - 0.02 - 1e-12
    ):
        return None
    return (recall_score(rep), rep["metrics"]["mAP"], -int(epoch))


def promotion_checks(baseline, candidate, domain):
    bm, cm = baseline["metrics"], candidate["metrics"]
    delta = dict(
        mAP=cm["mAP"] - bm["mAP"],
        macro_F1=cm["macro_F1"] - bm["macro_F1"],
        signal_recall=recall_score(candidate) - recall_score(baseline),
        **{
            name + "_recall": candidate["per_class"][name]["recall"] - baseline["per_class"][name]["recall"]
            for name in ("RR", "RG", "NoR")
        },
    )
    if domain == "DTLD":
        limits = dict(mAP=0.01, signal_recall=0.02, RR_recall=0, RG_recall=0, NoR_recall=-0.02)
    else:
        limits = dict(mAP=-0.01, macro_F1=-0.01, signal_recall=0)
    checks = {key: delta[key] >= limit - 1e-12 for key, limit in limits.items()}
    return dict(delta=delta, checks=checks, passed=all(checks.values()))


def paired_session_bootstrap(labels, baseline_logits, candidate_logits, sessions, repeats=2000, seed=0):
    labels, sessions = np.asarray(labels), np.asarray(sessions)
    if len(labels) != len(sessions) or baseline_logits.shape != candidate_logits.shape:
        raise ValueError("Bootstrap membership differs")
    groups = [np.flatnonzero(sessions == group) for group in np.unique(sessions)]
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(repeats):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(labels[idx])) != 3:
            continue
        b, c = report(labels[idx], baseline_logits[idx]), report(labels[idx], candidate_logits[idx])
        deltas.append(
            [
                c["metrics"]["mAP"] - b["metrics"]["mAP"],
                recall_score(c) - recall_score(b),
                c["per_class"]["NoR"]["recall"] - b["per_class"]["NoR"]["recall"],
            ]
        )
    if not deltas:
        raise ValueError("No three-class bootstrap samples")
    bounds = np.quantile(deltas, [0.025, 0.975], axis=0)
    return dict(
        method="paired complete-session bootstrap",
        seed=seed,
        repeats=repeats,
        valid_repeats=len(deltas),
        n_sessions=len(groups),
        intervals={
            name: bounds[:, i].tolist() for i, name in enumerate(("mAP", "signal_recall", "NoR_recall"))
        },
        limitation="Sampling uncertainty conditional on these weights; not training-seed uncertainty",
    )
