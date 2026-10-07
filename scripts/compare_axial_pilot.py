"""Export a membership-checked, single-fold axial-versus-v5 comparison."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.evaluation import sha256_file, write_json
from dinov3_global.metrics import report, to_jsonable

CLASSES = ("RR", "RG", "NoR")
METRICS = (
    "AP_RR",
    "AP_RG",
    "AP_NoR",
    "mAP",
    "acc_bal",
    "accuracy",
    "macro_F1",
    "weighted_F1",
    "ece",
    "nll",
    "brier",
)
LOWER = {"ece", "nll", "brier"}


def image_id(value):
    value = str(value).replace("\\", "/")
    while "//" in value:
        value = value.replace("//", "/")
    root = REPO.as_posix()
    if value.casefold().startswith(root.casefold() + "/"):
        value = value[len(root) + 1 :]
    if value.startswith("./"):
        value = value[2:]
    return value


def read_predictions(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            paths, labels, logits = data["paths"].tolist(), data["labels"].copy(), data["logits"].copy()
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        paths = [row["file"] for row in rows]
        labels = np.array([CLASSES.index(row["label"]) for row in rows])
        logits = np.array([[float(row[f"logit_{c}"]) for c in CLASSES] for row in rows])
    paths = [image_id(p) for p in paths]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate prediction identities: {path}")
    if not paths or labels.shape != (len(paths),) or logits.shape != (len(paths), 3):
        raise ValueError(f"Invalid prediction shapes: {path}")
    if not np.isin(labels, [0, 1, 2]).all() or not np.isfinite(logits).all():
        raise ValueError(f"Invalid prediction values: {path}")
    return paths, labels, logits


def align_predictions(baseline, candidate, expected_count):
    bp, by, bl = baseline
    cp, cy, cl = candidate
    if len(bp) != expected_count or set(bp) != set(cp):
        raise ValueError("Prediction membership differs or is incomplete")
    lookup = {p: i for i, p in enumerate(cp)}
    indices = [lookup[p] for p in bp]
    if not np.array_equal(by, cy[indices]):
        raise ValueError("Prediction labels differ")
    return by, bl, cl[indices]


def require_complete(path, key="status"):
    if not Path(path).is_file() or json.loads(Path(path).read_text())[key] != "complete":
        raise ValueError(f"Evaluation has not completed: {path}")


def vzc_test(directory, model):
    require_complete(directory / "status.json", "state")
    snapshot = json.loads((directory / "labels_snapshot.json").read_text())
    entries = {image_id(e["file"]): e["label"] for e in snapshot["images"] if e["split"] == "test"}
    paths, labels, logits = read_predictions(directory / model / "predictions.npz")
    mask = np.array([p in entries for p in paths])
    selected = [p for p in paths if p in entries]
    if set(selected) != set(entries) or labels[mask].tolist() != [entries[p] for p in selected]:
        raise ValueError("VZC predictions disagree with the published test snapshot")
    return selected, labels[mask], logits[mask]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(pilot, baseline, out):
    if not (pilot / "city_cv/fold0/history.jsonl").is_file():
        raise ValueError("Pilot training has not completed")
    history = [json.loads(line) for line in (pilot / "city_cv/fold0/history.jsonl").read_text().splitlines()]
    archive = pilot / "history_recovery/epochs"
    if archive.exists():
        merged = {row["epoch"]: row for row in history}
        for path in sorted(archive.glob("epoch*.json")):
            row = json.loads(path.read_text())
            if row["epoch"] in merged and merged[row["epoch"]] != row:
                raise ValueError("Live history differs from the preserved epoch archive")
            merged[row["epoch"]] = row
        history = [merged[k] for k in sorted(merged)]
    if [row["epoch"] for row in history] != list(range(1, 13)):
        raise ValueError("Pilot must finish exactly twelve epochs")
    if not (pilot / "city_cv/fold0/fold_report.json").is_file():
        raise ValueError("Pilot fold report is missing")
    require_complete(pilot / "evaluations/dtld_test/status.json")
    require_complete(baseline / "evaluations/dtld_test/status.json")
    require_complete(pilot / "benchmark.json")
    frozen = json.loads((pilot / "selected_checkpoint.json").read_text())
    if sha256_file(pilot / "city_cv/fold0/best.pt") != frozen["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    new_dtld_plan = json.loads((pilot / "evaluations/dtld_test/plan.json").read_text())
    if new_dtld_plan["checkpoints"][0]["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
        raise ValueError("DTLD evaluation used a different checkpoint")
    atlas_report = json.loads((pilot / "evaluations/atlas/v6_axial_fold0/report.json").read_text())
    if atlas_report["metric_input"] != "raw logits T=1" or atlas_report["metrics"]["n"] != 528:
        raise ValueError("ATLAS evaluation must use all 528 labels at T=1")
    vzc_report = json.loads((pilot / "evaluations/vzc/v6_axial_fold0/report.json").read_text())
    if vzc_report["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
        raise ValueError("VZC evaluation used a different checkpoint")
    datasets = [
        (
            "DTLD fold-0 validation",
            7032,
            read_predictions(baseline / "city_cv/fold0/val_predictions.npz"),
            read_predictions(pilot / "city_cv/fold0/val_predictions.npz"),
        ),
        (
            "DTLD official test",
            12453,
            read_predictions(baseline / "evaluations/dtld_test/member0/predictions.npz"),
            read_predictions(pilot / "evaluations/dtld_test/member0/predictions.npz"),
        ),
        (
            "ATLAS transfer",
            528,
            read_predictions(baseline / "evaluations/atlas/v5_fold0/predictions.csv"),
            read_predictions(pilot / "evaluations/atlas/v6_axial_fold0/predictions.csv"),
        ),
        (
            "VZC-TLD published test",
            598,
            vzc_test(baseline / "evaluations/vzc", "v5_fold0"),
            vzc_test(pilot / "evaluations/vzc", "v6_axial_fold0"),
        ),
    ]
    rows, class_rows, cm_rows, comparisons = [], [], [], []
    for domain, count, old, new in datasets:
        y, bl, cl = align_predictions(old, new, count)
        reports = {"v5_fold0": report(y, bl), "v6_axial_fold0": report(y, cl)}
        before, after = reports["v5_fold0"]["metrics"], reports["v6_axial_fold0"]["metrics"]
        differences = {k: after[k] - before[k] for k in METRICS}
        comparisons.append(
            dict(
                domain=domain,
                n=count,
                reports=reports,
                delta=differences,
                improved={k: differences[k] < 0 if k in LOWER else differences[k] > 0 for k in METRICS},
            )
        )
        for name, result in reports.items():
            rows.append(
                dict(domain=domain, model=name, n=count, **{k: result["metrics"][k] for k in METRICS})
            )
            for c in CLASSES:
                class_rows.append(dict(domain=domain, model=name, class_name=c, **result["per_class"][c]))
            for i, true in enumerate(CLASSES):
                for j, pred in enumerate(CLASSES):
                    cm_rows.append(
                        dict(
                            domain=domain,
                            model=name,
                            true=true,
                            predicted=pred,
                            count=result["confusion_matrix"][i][j],
                        )
                    )
        rows.append(dict(domain=domain, model="delta_v6_minus_v5", n=count, **differences))
    efficiency = json.loads((pilot / "benchmark.json").read_text())
    if efficiency["models"]["v6_axial_fold0"]["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
        raise ValueError("Benchmark used a different checkpoint")
    missing_timing = [row["epoch"] for row in history if row.get("elapsed_seconds") is None]
    recorded_duration = sum(
        row["elapsed_seconds"] for row in history if row.get("elapsed_seconds") is not None
    )
    duration = recorded_duration
    duration_scope = "sum of recorded training and epoch-validation durations"
    if missing_timing:
        run_status = json.loads((pilot / "status.json").read_text())
        duration = (pilot / "selected_checkpoint.json").stat().st_mtime - datetime.fromisoformat(
            run_status["started_at_utc"]
        ).timestamp()
        duration_scope = (
            "wall time from pipeline training start to checkpoint freeze, including final validation"
        )
    efficiency["training"] = dict(
        candidate_duration_seconds=duration,
        duration_scope=duration_scope,
        sum_recorded_epoch_seconds=recorded_duration,
        epochs_with_missing_timing=missing_timing,
        candidate_peak_gpu_allocated_bytes=max(
            row["peak_gpu_allocated_bytes"]
            for row in history
            if row.get("peak_gpu_allocated_bytes") is not None
        ),
        baseline_duration_seconds=None,
        baseline_peak_gpu_allocated_bytes=None,
        note="The historical v5 run did not record epoch durations or peak training memory.",
    )
    sources = [
        p
        for p in pilot.rglob("*")
        if p.is_file()
        and (
            p.name
            in (
                "predictions.npz",
                "predictions.csv",
                "plan.json",
                "effective_config.json",
                "fold_manifest.json",
                "source_manifest.json",
                "history.jsonl",
                "benchmark.json",
                "recovery.json",
            )
            or p.parent == archive
        )
    ]
    sources += [
        baseline / "city_cv/fold0/val_predictions.npz",
        baseline / "evaluations/dtld_test/member0/predictions.npz",
        baseline / "evaluations/atlas/v5_fold0/predictions.csv",
        baseline / "evaluations/vzc/v5_fold0/predictions.npz",
        Path(__file__),
        REPO / "scripts/benchmark_axial.py",
        REPO / "scripts/run_axial_pilot.py",
    ]
    result = to_jsonable(
        dict(
            status="complete",
            experiment="single-fold spatial relevance pilot",
            primary_temperature=1.0,
            checkpoint=frozen,
            comparisons=comparisons,
            efficiency=efficiency,
            source_sha256={image_id(p): sha256_file(p) for p in sources},
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "comparison.json", result)
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "per_class_metrics.csv", class_rows)
    write_csv(out / "confusion_matrices.csv", cm_rows)
    write_csv(out / "efficiency.csv", [dict(model=k, **v) for k, v in efficiency["models"].items()])
    text = [
        "# Axial spatial relevance pilot",
        "",
        "DTLD fold 0, seed 0, twelve epochs. Each model uses its best validation-selected EMA checkpoint.",
        "All metrics use raw logits at T=1. Deltas below are v6 minus v5 in percentage points.",
        "",
        "| Dataset | v5 mAP | v6 mAP | Δ mAP | v5 AP NoR | v6 AP NoR | Δ AP NoR | Δ balanced accuracy | Δ macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        a, b = [comparison["reports"][name]["metrics"] for name in ("v5_fold0", "v6_axial_fold0")]
        delta = comparison["delta"]
        text.append(
            f"| {comparison['domain']} | {100 * a['mAP']:.2f} | {100 * b['mAP']:.2f} | "
            f"{100 * delta['mAP']:+.2f} | {100 * a['AP_NoR']:.2f} | {100 * b['AP_NoR']:.2f} | "
            f"{100 * delta['AP_NoR']:+.2f} | {100 * delta['acc_bal']:+.2f} | {100 * delta['macro_F1']:+.2f} |"
        )
    text += [
        "",
        "## Full primary metrics",
        "",
        "AP, accuracy, F1, and ECE values are percentages; NLL and Brier are unitless. Lower ECE, NLL, and Brier are better.",
        "",
        "| Dataset | Model | AP RR | AP RG | AP NoR | mAP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for c in comparisons:
        for name in ("v5_fold0", "v6_axial_fold0"):
            m = c["reports"][name]["metrics"]
            text.append(
                f"| {c['domain']} | {name} | "
                + " | ".join(f"{100 * m[k]:.2f}" for k in ("AP_RR", "AP_RG", "AP_NoR", "mAP"))
                + " |"
            )
    text += [
        "",
        "| Dataset | Model | Bal. acc. | Accuracy | Macro F1 | Weighted F1 | ECE ↓ | NLL ↓ | Brier ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in comparisons:
        for name in ("v5_fold0", "v6_axial_fold0"):
            m = c["reports"][name]["metrics"]
            text.append(
                f"| {c['domain']} | {name} | "
                + " | ".join(
                    f"{100 * m[k]:.2f}" for k in ("acc_bal", "accuracy", "macro_F1", "weighted_F1", "ece")
                )
                + f" | {m['nll']:.4f} | {m['brier']:.4f} |"
            )
    text += [
        "",
        "## NoR operating performance",
        "",
        "| Dataset | v5 precision | v6 precision | v5 recall | v6 recall | v5 F1 | v6 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in comparisons:
        a, b = [c["reports"][name]["per_class"]["NoR"] for name in ("v5_fold0", "v6_axial_fold0")]
        values = [a["precision"], b["precision"], a["recall"], b["recall"], a["f1-score"], b["f1-score"]]
        text.append(f"| {c['domain']} | " + " | ".join(f"{100 * v:.2f}" for v in values) + " |")
    text += [
        "",
        "## Compute",
        "",
        "| Model | Head parameters | Batch-4 latency (ms) | Throughput (images/s) | Peak inference memory (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, values in efficiency["models"].items():
        text.append(
            f"| {model} | {values['head_parameters']:,} | {values['batch_latency_ms']:.2f} | "
            f"{values['images_per_second']:.2f} | {values['peak_gpu_allocated_bytes'] / 2**20:.1f} |"
        )
    text += [
        "",
        f"Candidate training duration: {efficiency['training']['candidate_duration_seconds'] / 3600:.2f} hours "
        f"({efficiency['training']['duration_scope']}). "
        f"Peak allocated training GPU memory: {efficiency['training']['candidate_peak_gpu_allocated_bytes'] / 2**20:.1f} MiB.",
        "Historical v5 training time and memory were not recorded; they are unavailable.",
        "",
        "## Interpretation",
        "",
    ]
    for c in comparisons:
        text.append(
            f"- {c['domain']}: mAP {c['delta']['mAP'] * 100:+.2f} pp; AP NoR {c['delta']['AP_NoR'] * 100:+.2f} pp."
        )
    text += [
        "",
        "This is one fold and one seed. Validation selected the checkpoints; ATLAS and VZC are exploratory transfer evaluations. "
        "The comparison does not establish attention to lane markers or a causal lane-association mechanism.",
        "",
        "[Complete metrics and provenance](comparison.json) · [Metrics CSV](metrics.csv) · "
        "[Per-class metrics](per_class_metrics.csv) · [Confusion matrices](confusion_matrices.csv) · [Efficiency](efficiency.csv)",
    ]
    if missing_timing:
        text += [
            "",
            "The live history was truncated during training. Missing epoch records were recovered with explicit provenance; "
            f"epoch timings remain unavailable for {missing_timing}. Duration therefore uses the recorded pipeline start and checkpoint-freeze timestamps. "
            "Model weights and final evaluation predictions were unaffected.",
        ]
    (out / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=REPO / "runs/v6_axial_pilot")
    parser.add_argument("--baseline", type=Path, default=REPO / "runs/v5")
    parser.add_argument("--out", type=Path, default=REPO / "docs/results/spatial_axial_pilot")
    args = parser.parse_args()
    export(args.pilot, args.baseline, args.out)
    print(f"Comparison exported to {args.out}")


if __name__ == "__main__":
    main()
