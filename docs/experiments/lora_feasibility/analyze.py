"""Read saved fold-0 predictions/history; never train or tune on external tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("analysis.json")
HISTORY = ROOT / "runs/backbone_vitb_fold0/training/fold0/history.jsonl"
PREDICTIONS = ROOT / "runs/backbone_vitb_fold0/training/fold0/val_predictions.npz"
MANIFEST = ROOT / "docs/experiments/backbone_capacity/results/training/fold_manifest.json"
FOLD_REPORT = ROOT / "runs/backbone_vitb_fold0/training/fold0/fold_report.json"


def operating_metrics(labels, logits):
    predicted = logits.argmax(1)
    recalls = [float((predicted[labels == i] == i).mean()) for i in range(3)]
    return {
        "acc_bal": float(np.mean(recalls)),
        "mean_signal_recall": float(np.mean(recalls[:2])),
        "recall_RR": recalls[0],
        "recall_RG": recalls[1],
        "recall_NoR": recalls[2],
        "accuracy": float((predicted == labels).mean()),
        "class_counts": np.bincount(labels, minlength=3).tolist(),
    }


def main():
    rows = [json.loads(line) for line in HISTORY.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["folds"][0]
    fold_report = json.loads(FOLD_REPORT.read_text(encoding="utf-8"))
    selected_epoch = fold_report["best_epoch"]
    selected = next(row for row in rows if row["epoch"] == selected_epoch)
    with np.load(PREDICTIONS, allow_pickle=False) as saved:
        labels, logits, cities = saved["labels"], saved["logits"], saved["cities"]
    baseline = operating_metrics(labels, logits)
    assert np.isclose(baseline["acc_bal"], selected["acc_bal"], atol=1e-12)
    assert baseline["class_counts"] == manifest["val_hist"]
    assert set(cities.tolist()) == set(manifest["val_cities"])

    # Fixed validation-only diagnostic: one NoR-logit offset, no retraining.
    # Evaluate -2..+2 in steps of .05; prefer smallest absolute offset in ties.
    offsets = np.round(np.arange(-40, 41) * 0.05, 2)
    candidates = []
    for offset in offsets:
        adjusted = logits.astype(np.float64).copy()
        adjusted[:, 2] += offset
        metrics = operating_metrics(labels, adjusted)
        candidates.append({"nor_logit_offset": float(offset), **metrics})
    key = lambda entry: (entry["acc_bal"], -abs(entry["nor_logit_offset"]))
    optimum = max(candidates, key=key)
    preserve_signal = [
        entry
        for entry in candidates
        if entry["recall_RR"] >= baseline["recall_RR"] and entry["recall_RG"] >= baseline["recall_RG"]
    ]
    constrained = max(preserve_signal, key=key)

    trend = []
    for row in rows:
        probe = row["clean_probe"]
        trend.append(
            {
                "epoch": row["epoch"],
                "augmented_training_global_loss": row["training_losses"]["global"],
                "clean_probe_acc_bal": probe["metrics"]["acc_bal"],
                "validation_acc_bal": row["acc_bal"],
                "clean_probe_mAP": probe["metrics"]["mAP"],
                "validation_mAP": row["mAP"],
                "clean_probe_NoR_recall": probe["per_class"]["NoR"]["recall"],
                "validation_NoR_recall": row["validation"]["per_class"]["NoR"]["recall"],
                "balanced_accuracy_probe_val_gap_pp": 100 * (probe["metrics"]["acc_bal"] - row["acc_bal"]),
            }
        )

    train_counts = np.asarray(manifest["train_hist"], dtype=float)
    source_files = [
        HISTORY,
        PREDICTIONS,
        MANIFEST,
        FOLD_REPORT,
        ROOT / "src/dinov3_global/backbone.py",
        ROOT / "src/dinov3_global/model.py",
        ROOT / "src/dinov3_global/head.py",
        ROOT / "src/dinov3_global/engine.py",
        ROOT / "src/dinov3_global/config.py",
        ROOT / "configs/backbone_vitb.yaml",
        Path(__file__),
    ]
    result = {
        "purpose": "LoRA feasibility analysis; no new model training or external-test tuning",
        "selected_epoch": selected_epoch,
        "class_order": ["RR", "RG", "NoR"],
        "fold0_audit": manifest,
        "train_class_fractions": (train_counts / train_counts.sum()).tolist(),
        "NoR_vs_RG_sqrt_loss_weight_ratio": float(np.sqrt(train_counts[1] / train_counts[2])),
        "validation_baseline": baseline,
        "validation_epoch_with_max_balanced_accuracy": max(rows, key=lambda row: row["acc_bal"])["epoch"],
        "training_trend": trend,
        "validation_only_offset_diagnostic": {
            "grid": offsets.tolist(),
            "selection": "maximum validation balanced accuracy; ties prefer smallest absolute offset",
            "unconstrained": optimum,
            "both_signal_recalls_preserved": constrained,
            "all_candidates": candidates,
            "external_evaluation": "not performed; these are in-sample validation diagnostics",
            "limitation": "optimistic from fitting/evaluating on the same validation logits",
        },
        "recommended_adapter": {
            "backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "zero_based_layer_indices": [8, 9, 10, 11],
            "target_projections": ["q_proj", "v_proj"],
            "verified_module_names": [
                f"model.layer.{index}.attention.{projection}"
                for index in range(8, 12)
                for projection in ("q_proj", "v_proj")
            ],
            "verified_transformers_version": "5.6.2",
            "verified_backbone_parameters": 85660416,
            "rank": 4,
            "alpha": 4,
            "dropout": 0.05,
            "train_bias": False,
            "new_adapter_parameters": 4 * 2 * 4 * (768 + 768),
            "existing_head_parameters": 412830,
            "total_trainable_parameters": 412830 + 4 * 2 * 4 * (768 + 768),
            "status": "proposed; not implemented or trained",
        },
        "inputs_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_epoch": selected_epoch,
                "baseline": baseline,
                "offset_optimum": optimum,
                "offset_preserve_RR_RG": constrained,
                "selected_epoch_trend": next(row for row in trend if row["epoch"] == selected_epoch),
                "last_epoch_trend": trend[-1],
                "adapter": result["recommended_adapter"],
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
