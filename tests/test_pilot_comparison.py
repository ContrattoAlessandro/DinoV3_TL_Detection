"""Pilot exports require complete runs and exact image/label membership."""

import json
import os

import numpy as np
import pytest

from scripts.compare_axial_pilot import align_predictions, export, read_predictions, require_complete
from dinov3_global.evaluation import sha256_file


def predictions(paths=("a", "b", "c"), labels=None):
    return list(paths), np.array([0, 1, 2] if labels is None else labels), np.eye(3)


def test_comparison_aligns_by_identity_instead_of_row_order():
    baseline = predictions()
    candidate = [part[[2, 0, 1]] if isinstance(part, np.ndarray) else [part[i] for i in (2, 0, 1)]
                 for part in baseline]
    labels, old, new = align_predictions(baseline, candidate, 3)
    assert np.array_equal(labels, [0, 1, 2]) and np.array_equal(old, new)


def test_comparison_rejects_missing_images_and_changed_labels():
    with pytest.raises(ValueError, match="membership"):
        align_predictions(predictions(), predictions(("a", "b", "d")), 3)
    with pytest.raises(ValueError, match="labels"):
        align_predictions(predictions(), predictions(labels=[1, 0, 2]), 3)


def test_predictions_reject_duplicate_image_identities(tmp_path):
    path = tmp_path / "predictions.npz"
    np.savez(path, paths=["a", "a", "b"], labels=[0, 1, 2], logits=np.eye(3))
    with pytest.raises(ValueError, match="Duplicate"):
        read_predictions(path)


def test_incomplete_evaluation_cannot_be_exported(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(dict(status="evaluating")))
    with pytest.raises(ValueError, match="not completed"):
        require_complete(path)
    with pytest.raises(ValueError, match="not completed"):
        export(tmp_path, tmp_path, tmp_path / "export")


def test_complete_export_contains_all_datasets_metrics_and_hashes(tmp_path):
    old, new, out = [tmp_path / name for name in ("baseline", "pilot", "export")]

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def npz(path, prefix, count, scale):
        path.parent.mkdir(parents=True, exist_ok=True)
        paths = np.array([f"{prefix}/{i}" for i in range(count)])
        labels = np.arange(count) % 3
        np.savez(path, paths=paths, labels=labels, logits=np.eye(3)[labels] * scale)
        return paths, labels

    checkpoint = new / "city_cv/fold0/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixed checkpoint")
    digest = sha256_file(checkpoint)
    write(new / "selected_checkpoint.json", dict(checkpoint_sha256=digest))
    history = [dict(epoch=i, elapsed_seconds=10, peak_gpu_allocated_bytes=1000) for i in range(1, 13)]
    (new / "city_cv/fold0/history.jsonl").write_text("\n".join(json.dumps(h) for h in history))
    write(new / "city_cv/fold0/fold_report.json", {})
    write(new / "evaluations/dtld_test/plan.json", dict(checkpoints=[dict(checkpoint_sha256=digest)]))
    for root, name, scale in ((old, "v5_fold0", 2), (new, "v6_axial_fold0", 3)):
        npz(root / "city_cv/fold0/val_predictions.npz", "val", 7032, scale)
        npz(root / "evaluations/dtld_test/member0/predictions.npz", "test", 12453, scale)
        write(root / "evaluations/dtld_test/status.json", dict(status="complete"))
        atlas = root / "evaluations/atlas" / name / "predictions.csv"
        atlas.parent.mkdir(parents=True)
        lines = ["file,label,logit_RR,logit_RG,logit_NoR"]
        for i in range(528):
            label = i % 3
            lines.append(f"atlas/{i},{('RR','RG','NoR')[label]}," +
                         ",".join(str(scale if c == label else 0) for c in range(3)))
        atlas.write_text("\n".join(lines))
        paths, labels = npz(root / "evaluations/vzc" / name / "predictions.npz", "vzc", 598, scale)
        write(root / "evaluations/vzc/status.json", dict(state="complete"))
        write(root / "evaluations/vzc/labels_snapshot.json", dict(images=[
            dict(file=p, label=int(y), split="test") for p, y in zip(paths, labels)]))
    write(new / "evaluations/atlas/v6_axial_fold0/report.json",
          dict(metric_input="raw logits T=1", metrics=dict(n=528)))
    write(new / "evaluations/vzc/v6_axial_fold0/report.json", dict(checkpoint_sha256=digest))
    model_metrics = dict(head_parameters=1, batch_latency_ms=10, images_per_second=400,
                         peak_gpu_allocated_bytes=1000, checkpoint_sha256=digest)
    write(new / "benchmark.json", dict(status="complete", models={
        "v5_fold0": model_metrics, "v6_axial_fold0": model_metrics}))
    result = export(new, old, out)
    assert len(result["comparisons"]) == 4
    assert [c["n"] for c in result["comparisons"]] == [7032, 12453, 528, 598]
    assert result["efficiency"]["training"]["candidate_duration_seconds"] == 120
    assert len(result["source_sha256"]) >= 8
    assert all((out / filename).is_file() for filename in (
        "README.md", "comparison.json", "metrics.csv", "per_class_metrics.csv",
        "confusion_matrices.csv", "efficiency.csv"))
    # An explicitly recovered record must not fabricate a missing epoch timer.
    history[5]["elapsed_seconds"] = None
    history[5]["peak_gpu_allocated_bytes"] = None
    (new / "city_cv/fold0/history.jsonl").write_text("\n".join(json.dumps(h) for h in history))
    write(new / "history_recovery/epochs/epoch06.json", history[5])
    write(new / "status.json", dict(started_at_utc="1970-01-01T00:00:00+00:00"))
    os.utime(new / "selected_checkpoint.json", (120, 120))
    recovered = export(new, old, out)
    assert recovered["efficiency"]["training"]["candidate_duration_seconds"] == 120
    assert recovered["efficiency"]["training"]["epochs_with_missing_timing"] == [6]
