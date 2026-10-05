"""Rebuild portable v5 result tables from saved full-precision predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.metrics import report, to_jsonable
from dinov3_global.evaluation import sha256_file

CLASSES = ("RR", "RG", "NoR")
SCORES = ("AP_RR", "AP_RG", "AP_NoR", "mAP", "acc_bal", "accuracy", "macro_F1", "ece")


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(to_jsonable(rows))


def flatten(result, **identity):
    row = dict(**identity, **result["metrics"], head_parameters=489475)
    for cls in CLASSES:
        for key, value in result["per_class"][cls].items():
            row[f"{key}_{cls}"] = value
    return row


def mean_scores(rows, n_images):
    excluded = {"fold", "best_epoch", "seed", "n", "n_classes", "head_parameters", "domain", "model"}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in excluded]
    means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return dict(
        model="mean_4_individual_checkpoints",
        n_models=len(rows),
        n_images=n_images,
        **means,
        sd_mAP=float(np.std([row["mAP"] for row in rows], ddof=0)),
        min_mAP=min(row["mAP"] for row in rows),
        max_mAP=max(row["mAP"] for row in rows),
        head_parameters=489475,
    )


def markdown_table(rows):
    lines = [
        "| Model | AP RR | AP RG | AP NoR | mAP | Bal. acc. | Accuracy | Macro F1 | ECE ↓ |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        values = [f"{float(row[key]) * 100:.2f}" for key in SCORES]
        label = (
            row["model"].replace("v5_fold", "Fold ").replace("mean_4_individual_checkpoints", "Mean of four")
        )
        label = label.replace("v5_ensemble4", "Logit ensemble (four)")
        lines.append("| " + " | ".join([label, *values]) + " |")
    return "\n".join(lines)


def lf_source_hashes(recorded, root=REPO):
    """Verify executed sources while allowing Git's LF/CRLF conversion."""
    normalized = {}
    for path, expected in recorded.items():
        raw = (root / path).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        variants = (raw, lf, lf.replace(b"\n", b"\r\n"))
        if expected not in {hashlib.sha256(value).hexdigest() for value in variants}:
            raise ValueError(f"Evaluator source differs from the frozen test protocol: {path}")
        normalized[path] = hashlib.sha256(lf).hexdigest()
    return normalized


def export_test(directory, out):
    """Recompute every official-test score from the preserved member logits."""
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("The DTLD official test evaluation has not completed")
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if len(plan["checkpoints"]) != 4:
        raise ValueError("This four-fold result export requires four fixed checkpoints")
    rows, reports, logits_list, sources = [], {}, [], {}
    reference = None
    for index, checkpoint in enumerate(plan["checkpoints"]):
        model = f"v5_fold{index}"
        if checkpoint["seed"] != index:
            raise ValueError("Checkpoint order must match the four recorded fold seeds")
        member = directory / checkpoint["name"]
        with np.load(member / "predictions.npz", allow_pickle=False) as saved:
            predictions = {key: saved[key] for key in saved.files}
        if reference is not None:
            for key in ("labels", "paths", "cities", "max_lamp_h", "pseudo_nor"):
                np.testing.assert_array_equal(predictions[key], reference[key])
        reference = predictions
        result = report(
            predictions["labels"],
            predictions["logits"],
            cities=predictions["cities"].tolist(),
            max_lamp_h=predictions["max_lamp_h"],
        )
        clean = ~predictions["pseudo_nor"]
        result["without_pseudo_nor"] = report(predictions["labels"][clean], predictions["logits"][clean])
        recorded = json.loads((member / "report.json").read_text(encoding="utf-8"))
        for key in result:
            if to_jsonable(result[key]) != recorded[key]:
                raise ValueError(f"Recomputed {model} {key} differs from the frozen test report")
        reports[model] = result
        logits_list.append(predictions["logits"])
        rows.append(
            flatten(
                result,
                model=model,
                domain="DTLD official test",
                fold=index,
                seed=checkpoint["seed"],
                best_epoch=checkpoint["epoch"],
            )
        )
        for file in ("predictions.npz", "report.json"):
            sources[f"evaluations/dtld_test/{checkpoint['name']}/{file}"] = sha256_file(member / file)
    mean = mean_scores(rows, len(reference["labels"]))
    ensemble_logits = np.mean(logits_list, axis=0)
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        for key in ("labels", "paths", "cities", "max_lamp_h", "pseudo_nor"):
            np.testing.assert_array_equal(saved[key], reference[key])
        np.testing.assert_array_equal(saved["logits"], ensemble_logits)
    ensemble = report(
        reference["labels"],
        ensemble_logits,
        cities=reference["cities"].tolist(),
        max_lamp_h=reference["max_lamp_h"],
    )
    clean = ~reference["pseudo_nor"]
    ensemble["without_pseudo_nor"] = report(reference["labels"][clean], ensemble_logits[clean])
    recorded = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    for key in ensemble:
        if to_jsonable(ensemble[key]) != recorded[key]:
            raise ValueError(f"Recomputed ensemble {key} differs from the frozen test report")
    if recorded["protocol_sha256"] != sha256_file(directory / "plan.json"):
        raise ValueError("DTLD evaluation protocol has changed since inference")
    reports["v5_ensemble4"] = ensemble
    ensemble_row = flatten(ensemble, model="v5_ensemble4", domain="DTLD official test")
    ensemble_row["head_parameters"] = 4 * 489475
    ensemble_row["n_heads"] = 4
    write_csv(out / "dtld_test_metrics.csv", [*rows, ensemble_row])
    write_csv(out / "dtld_test_summary.csv", [mean, ensemble_row])
    class_rows, confusion_rows, slice_rows, sensitivity = [], [], [], []
    for model, result in reports.items():
        for index, cls in enumerate(CLASSES):
            class_rows.append(
                dict(
                    model=model, class_name=cls, AP=result["metrics"][f"AP_{cls}"], **result["per_class"][cls]
                )
            )
            for predicted, count in zip(CLASSES, result["confusion_matrix"][index]):
                confusion_rows.append(dict(model=model, actual=cls, predicted=predicted, count=count))
        for kind, slices in result["slices"].items():
            for group, metrics in slices.items():
                slice_rows.append(dict(model=model, slice_type=kind, group=group, **metrics))
        sensitivity.append(
            flatten(
                result["without_pseudo_nor"],
                model=model,
                domain="DTLD test excluding relevant-off/unknown-only NoR",
            )
        )
    sensitivity[4]["head_parameters"] = 4 * 489475
    sensitivity_mean = mean_scores(sensitivity[:4], int(clean.sum()))
    write_csv(out / "dtld_test_per_class_metrics.csv", class_rows)
    write_csv(out / "dtld_test_confusion_matrices.csv", confusion_rows)
    write_csv(out / "dtld_test_slices.csv", slice_rows)
    write_csv(out / "dtld_test_sensitivity.csv", [*sensitivity[:4], sensitivity_mean, sensitivity[4]])
    # Preserve the exact pre-inference protocol bytes, not a reconstructed plan.
    (out / "dtld_test_protocol.json").write_bytes((directory / "plan.json").read_bytes())
    for file in ("predictions.npz", "report.json", "plan.json", "data_audit.json", "membership.json"):
        sources[f"evaluations/dtld_test/{file}"] = sha256_file(directory / file)
    provenance = dict(
        protocol_sha256=sha256_file(directory / "plan.json"),
        completed_at_utc=status["completed_at_utc"],
        primary_predictor="v5_ensemble4",
        aggregation="Mean of four raw logit vectors, then one softmax at T=1",
        individual_model_summary="Arithmetic mean of four individual checkpoint metrics",
        full_training_split_refit=False,
        encoder_reuse_verification=recorded["verification"],
        checkpoints=plan["checkpoints"],
        source_sha256=sources,
        source_hash_format="Protocol source hashes use executed file bytes; LF-normalized hashes below support cross-platform checkout verification",
        source_sha256_lf=lf_source_hashes(plan["source_sha256"]),
        exporter=dict(path="scripts/summarize_results.py", sha256=sha256_file(Path(__file__))),
    )
    (out / "dtld_test_provenance.json").write_text(
        json.dumps(to_jsonable(provenance), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    counts = plan["data_audit"]["splits"]["test"]["class_counts"]
    class_lines = ["| Class | Images | AP | Precision | Recall | F1 |", "|:--|--:|--:|--:|--:|--:|"]
    for cls in CLASSES:
        values = ensemble["per_class"][cls]
        class_lines.append(
            f"| {cls} | {counts[cls]:,} | {ensemble['metrics'][f'AP_{cls}'] * 100:.2f} | "
            + " | ".join(f"{values[key] * 100:.2f}" for key in ("precision", "recall", "f1-score"))
            + " |"
        )
    cm_lines = ["| Actual \\ Predicted | RR | RG | NoR |", "|:--|--:|--:|--:|"]
    cm_lines.extend(
        "| " + " | ".join([cls, *(f"{count:,}" for count in row)]) + " |"
        for cls, row in zip(CLASSES, ensemble["confusion_matrix"])
    )
    additional = ["| Predictor | Weighted F1 (%) | NLL ↓ | Brier ↓ |", "|:--|--:|--:|--:|"]
    for row in [mean, ensemble_row]:
        name = "Mean of four" if row["model"] == mean["model"] else "Logit ensemble (four)"
        additional.append(
            f"| {name} | {row['weighted_F1'] * 100:.2f} | {row['nll']:.4f} | {row['brier']:.4f} |"
        )
    additional_table = "\n".join(additional)
    class_table = "\n".join(class_lines)
    confusion_table = "\n".join(cm_lines)
    section = f"""## Official DTLD test — final fixed evaluation

All **12,453** official test frames are evaluated: 4,359 RR, 7,569 RG, and 525
NoR, across 632 sessions in 11 cities. No images are missing or excluded. The
published train/test partition has zero shared image identities, shared sessions,
or exact prepared-JPEG duplicates. The primary NoR policy includes 213 frames
whose only relevant states are off/unknown.

The four checkpoints and the **four-head mean-logit ensemble** were fixed before
test inference. Epochs, temperatures, thresholds, and ensemble weights were not
selected using test scores. This is the final test of retained city-fold EMA
checkpoints; a model refitted on the full official training split is not claimed.

{markdown_table([*rows, mean, ensemble_row])}

Population standard deviation of individual checkpoint mAP: {mean["sd_mAP"] * 100:.2f}
percentage points. It measures checkpoint variability, not a confidence interval.
Each individual head has 489,475 trainable parameters. The ensemble holds four
heads (1,957,900 parameters) with one shared frozen encoder at inference. The
shared-encoder logits matched separate full-model forwards bit for bit on the
first batch for every checkpoint.

{additional_table}

### Primary ensemble per-class performance

{class_table}

### Primary ensemble confusion matrix

Rows are ground truth; columns are argmax predictions. Counts refer to the full
official test set at T=1.

{confusion_table}

The fixed off/unknown sensitivity diagnostic removes 213 pseudo-NoR frames,
leaving 12,240 images (312 NoR). It gives ensemble mAP
{ensemble["without_pseudo_nor"]["metrics"]["mAP"] * 100:.2f}% and balanced accuracy
{ensemble["without_pseudo_nor"]["metrics"]["acc_bal"] * 100:.2f}%; it does not replace
the full-split result.

The Bochum city slice has no NoR examples. Its `n_classes=2` mAP averages only
RR/RG AP and must be read as a two-class slice diagnostic. Undefined AP values
are blank in the slice CSV and `null` in JSON; the full test set has all three classes.

[All individual and ensemble metrics](dtld_test_metrics.csv) · [Summary](dtld_test_summary.csv)
· [Per-class metrics](dtld_test_per_class_metrics.csv) · [Confusion matrices](dtld_test_confusion_matrices.csv)
· [City and lamp-size slices](dtld_test_slices.csv) · [Off/unknown sensitivity](dtld_test_sensitivity.csv)
· [Frozen protocol](dtld_test_protocol.json) · [Artifact hashes and verification](dtld_test_provenance.json)

The protocol retains its exact execution bytes in Git, including line endings.
Provenance also gives LF-normalized source hashes for checkout verification
across platforms; executed-file hashes remain in the frozen protocol.

"""
    return (
        dict(checkpoints=rows, summary=mean, ensemble=ensemble_row, reports=reports, provenance=provenance),
        section,
        sources,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=REPO / "runs/v5")
    parser.add_argument("--city-cv", type=Path, help="override the city-fold prediction directory")
    parser.add_argument(
        "--atlas", type=Path, help="override the directory containing v5_fold*/predictions.csv"
    )
    parser.add_argument("--vzc", type=Path, help="override the VZC evaluator output directory")
    parser.add_argument("--dtld-test", type=Path, help="override the final official-test output directory")
    parser.add_argument("--out", type=Path, default=REPO / "docs/results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    city_directory = args.city_cv or args.runs / "city_cv"
    atlas_directory = args.atlas or args.runs / "evaluations/atlas"
    dtld, atlas = [], []
    provenance, detailed = {}, {"DTLD": {}, "ATLAS": {}}
    for fold in range(4):
        name = f"v5_fold{fold}"
        path = city_directory / f"fold{fold}/val_predictions.npz"
        with np.load(path, allow_pickle=False) as predictions:
            result = report(
                predictions["labels"], predictions["logits"], cities=predictions["cities"].tolist()
            )
        source = json.loads((path.parent / "fold_report.json").read_text(encoding="utf-8"))
        dtld.append(
            flatten(
                result,
                model=name,
                domain="DTLD city validation",
                fold=fold,
                best_epoch=source["best_epoch"],
                seed=source["seed"],
            )
        )
        detailed["DTLD"][name] = result
        provenance[f"city_cv/fold{fold}/val_predictions.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
        path = atlas_directory / name / "predictions.csv"
        rows = csv_rows(path)
        y = np.array([CLASSES.index(row["label"]) for row in rows])
        logits = np.array([[float(row[f"logit_{cls}"]) for cls in CLASSES] for row in rows])
        result = report(y, logits)
        atlas.append(flatten(result, model=name, domain="ATLAS all528", fold=fold))
        detailed["ATLAS"][name] = result
        provenance[f"evaluations/atlas/{name}/predictions.csv"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    dtld_mean = mean_scores(dtld, sum(row["n"] for row in dtld))
    atlas_mean = mean_scores(atlas, atlas[0]["n"])
    write_csv(args.out / "dtld_folds.csv", dtld)
    write_csv(args.out / "dtld_summary.csv", [dtld_mean])
    write_csv(args.out / "atlas_checkpoints.csv", atlas)
    write_csv(args.out / "atlas_summary.csv", [atlas_mean])
    vzc_directory = args.vzc or args.runs / "evaluations/vzc"
    for source, destination in (
        ("metrics.csv", "vzc_metrics.csv"),
        ("summary.csv", "vzc_summary.csv"),
        ("per_class_metrics.csv", "vzc_per_class_metrics.csv"),
        ("confusion_matrices.csv", "vzc_confusion_matrices.csv"),
    ):
        path = vzc_directory / source
        rows = csv_rows(path)
        write_csv(args.out / destination, rows)
        provenance[f"evaluations/vzc/{source}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    vzc = [
        row
        for row in csv_rows(vzc_directory / "metrics.csv")
        if row["split"] == "test" and row["policy"] == "primary"
    ]
    vzc_mean = next(
        row
        for row in csv_rows(vzc_directory / "summary.csv")
        if row["split"] == "test" and row["policy"] == "primary"
    )
    for row in [*vzc, vzc_mean]:
        row["macro_F1"] = row["macro_f1-score"]
    test_directory = args.dtld_test or args.runs / "evaluations/dtld_test"
    official = None
    official_section = ""
    test_scope = "No official DTLD test result has been produced for these local outputs."
    if test_directory.exists():
        official, official_section, test_sources = export_test(test_directory, args.out)
        provenance.update(test_sources)
        test_scope = "The final fixed official-test evaluation is reported separately above."
    export = dict(
        architecture="v5",
        primary_temperature=1.0,
        DTLD=dict(folds=dtld, summary=dtld_mean),
        ATLAS=dict(checkpoints=atlas, summary=atlas_mean),
        VZC_test=dict(checkpoints=vzc, summary=vzc_mean),
        reports=detailed,
        source_sha256=provenance,
    )
    if official is not None:
        export["DTLD_test"] = official
    (args.out / "v5_results.json").write_text(
        json.dumps(to_jsonable(export), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    readme = f"""# Recorded v5 results

AP, accuracy, F1, and ECE values are percentages at raw temperature **T=1**.
NLL and Brier score are unitless. ECE, NLL, and Brier are better when lower;
the other displayed scores are better when higher. Full
precision, class-specific precision/recall/F1/support, NLL, Brier score, and
confusion matrices appear in the linked CSV and JSON files.

{official_section}
## DTLD city-disjoint validation

Each of the 28,525 images is held out in one fold. The selected epochs are
8, 2, 8, and 11 for fold seeds 0, 1, 2, and 3. The mean gives each fold equal
weight; it is not a pooled out-of-fold metric or an ensemble prediction.

{markdown_table([*dtld, dtld_mean])}

Population standard deviation of fold mAP: {dtld_mean["sd_mAP"] * 100:.2f} percentage points.
These scores use the validation folds that selected the checkpoints.
{test_scope}

[Per-fold metrics](dtld_folds.csv) · [Summary](dtld_summary.csv)

## ATLAS transfer

All four checkpoints see the same 528 manually labeled images, including
uncertain annotations. Class counts: 245 RR, 95 RG, 188 NoR. These are the fixed
manual relevance labels defined in [the data protocol](../data.md).

{markdown_table([*atlas, atlas_mean])}

Population standard deviation of mAP across checkpoints: {atlas_mean["sd_mAP"] * 100:.2f} percentage points.

[Per-checkpoint metrics](atlas_checkpoints.csv) · [Summary](atlas_summary.csv)

## VZC published test split

All four checkpoints see the same 598 images: 331 RR, 176 RG, 91 NoR. The final
row averages raw logits before softmax and is a separate ensemble predictor.

{markdown_table([*vzc[:4], vzc_mean, *vzc[4:]])}

Primary NoR labels include 47 images whose only relevant state is unknown.
Excluding those images leaves 551 samples and reduces mean mAP to 64.65%; the
ensemble has 67.61% mAP under that sensitivity policy. No duplicates occur
within the test split. The source train split is a separate transfer diagnostic
(mean mAP 64.09%), with no training on VZC images.

The four checkpoints predict NoR for 53.18% of test images on average, versus
15.22% in the labels. Mean NoR precision/recall are 26.63%/91.48%; RR and RG
recall are 55.97% and 36.93%. This explains the gap between ranking and operating
metrics. Saved DTLD temperature scaling increases mean test ECE from 13.53% to
21.90%, so raw T=1 is the primary report.

[All metrics and sensitivity policies](vzc_metrics.csv) · [Mean-model summaries](vzc_summary.csv)
· [Class precision/recall/F1](vzc_per_class_metrics.csv) · [Confusion matrices](vzc_confusion_matrices.csv)

## Scope and provenance

ATLAS and VZC results were inspected during architecture selection. They are
exploratory transfer diagnostics, not an independent confirmatory test of that
selection. AP depends on class prevalence and annotation policy; cross-dataset
score differences do not alone measure generalization improvement. See the
[evaluation protocol](../evaluation.md).

[Complete v5 metrics and source hashes](v5_results.json). Rebuild these exports
with `python scripts/summarize_results.py` after restoring the local saved v5
predictions. Checkpoints and image datasets are excluded from Git.
"""
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    print(f"Exported v5 validation, official-test (when available), ATLAS, and VZC results to {args.out}")


if __name__ == "__main__":
    main()
