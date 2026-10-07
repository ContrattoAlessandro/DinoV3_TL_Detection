"""Export audited DTLD result tables from one or more fixed C checkpoints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.evaluation import sha256_file, write_json
from dinov3_global.metrics import report, to_jsonable

CLASSES = ("RR", "RG", "NoR")
SCORES = ("AP_RR", "AP_RG", "AP_NoR", "mAP", "acc_bal", "accuracy", "macro_F1", "ece")


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(to_jsonable(rows))


def markdown_table(rows):
    lines = [
        "| Model | AP RR | AP RG | AP NoR | mAP | Bal. acc. | Accuracy | Macro F1 | ECE ↓ |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        values = ["—" if row[key] is None else f"{row[key] * 100:.2f}" for key in SCORES]
        lines.append("| " + " | ".join([row["model"], *values]) + " |")
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


def recompute(predictions):
    result = report(
        predictions["labels"],
        predictions["logits"],
        cities=predictions["cities"].tolist(),
        max_lamp_h=predictions["max_lamp_h"],
    )
    clean = ~predictions["pseudo_nor"]
    result["without_pseudo_nor"] = report(predictions["labels"][clean], predictions["logits"][clean])
    return to_jsonable(result)


def read_predictions(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def export_test(directory, out):
    """Verify raw member logits, membership, reports and frozen protocol before export."""
    directory, out = Path(directory), Path(out)
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("The DTLD official test evaluation has not completed")
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if not plan["checkpoints"]:
        raise ValueError("The protocol must declare at least one checkpoint")
    reports, rows, logits, source_hashes = {}, [], [], {}
    reference = None
    for checkpoint in plan["checkpoints"]:
        name = checkpoint["name"]
        member = directory / name
        predictions = read_predictions(member / "predictions.npz")
        if reference is not None:
            for key in ("labels", "paths", "cities", "max_lamp_h", "pseudo_nor"):
                np.testing.assert_array_equal(predictions[key], reference[key])
        reference = predictions
        result = recompute(predictions)
        recorded = json.loads((member / "report.json").read_text(encoding="utf-8"))
        for key, value in result.items():
            if value != recorded[key]:
                raise ValueError(f"Recomputed {name} {key} differs from the frozen report")
        reports[name] = result
        logits.append(predictions["logits"])
        rows.append(
            dict(
                model=name,
                epoch=checkpoint["epoch"],
                seed=checkpoint["seed"],
                head_parameters=checkpoint.get("head_parameters", 265374),
                **result["metrics"],
            )
        )
        for file in ("report.json", "predictions.npz"):
            source_hashes[f"{name}/{file}"] = sha256_file(member / file)
    final = read_predictions(directory / "predictions.npz")
    for key in ("labels", "paths", "cities", "max_lamp_h", "pseudo_nor"):
        np.testing.assert_array_equal(final[key], reference[key])
    np.testing.assert_array_equal(final["logits"], np.mean(logits, axis=0))
    result = recompute(final)
    recorded = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    for key, value in result.items():
        if value != recorded[key]:
            raise ValueError(f"Recomputed final {key} differs from the frozen report")
    if recorded["protocol_sha256"] != sha256_file(directory / "plan.json"):
        raise ValueError("The evaluation protocol changed since inference")
    primary = "single_head" if len(logits) == 1 else f"ensemble_{len(logits)}"
    reports[primary] = result
    rows.append(
        dict(
            model=primary,
            n_heads=len(logits),
            head_parameters=sum(
                checkpoint.get("head_parameters", 265374) for checkpoint in plan["checkpoints"]
            ),
            **result["metrics"],
        )
    )
    for file in ("report.json", "predictions.npz", "plan.json", "data_audit.json", "membership.json"):
        source_hashes[file] = sha256_file(directory / file)
    provenance = dict(
        protocol_sha256=sha256_file(directory / "plan.json"),
        primary_predictor=primary,
        checkpoint_records=plan["checkpoints"],
        source_sha256=source_hashes,
        source_sha256_lf=lf_source_hashes(plan["source_sha256"]),
        exporter_sha256=sha256_file(Path(__file__)),
    )
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics.csv", rows)
    classes, confusion, slices, sensitivity = [], [], [], []
    for name, rep in reports.items():
        for index, cls in enumerate(CLASSES):
            classes.append(
                dict(model=name, class_name=cls, AP=rep["metrics"][f"AP_{cls}"], **rep["per_class"][cls])
            )
            for predicted, count in zip(CLASSES, rep["confusion_matrix"][index]):
                confusion.append(dict(model=name, actual=cls, predicted=predicted, count=count))
        for kind, groups in rep.get("slices", {}).items():
            for group, metrics in groups.items():
                slices.append(dict(model=name, slice_type=kind, group=group, **metrics))
        sensitivity.append(dict(model=name, **rep["without_pseudo_nor"]["metrics"]))
    write_csv(out / "per_class_metrics.csv", classes)
    write_csv(out / "confusion_matrices.csv", confusion)
    write_csv(out / "slices.csv", slices)
    write_csv(out / "sensitivity.csv", sensitivity)
    write_json(out / "results.json", dict(reports=reports, provenance=provenance))
    (out / "protocol.json").write_bytes((directory / "plan.json").read_bytes())
    (out / "README.md").write_text(
        "# DTLD official test results\n\nAll scores are percentages at T=1. "
        "Members and the final predictor are reported separately.\n\n" + markdown_table(rows) + "\n",
        encoding="utf-8",
    )
    return dict(reports=reports, provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtld-test", type=Path, default=REPO / "runs/evaluations/dtld_test")
    parser.add_argument("--out", type=Path, default=REPO / "runs/result_exports")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty output directory to preserve existing results")
    export_test(args.dtld_test, args.out)
    print(f"Exported verified DTLD results to {args.out}")


if __name__ == "__main__":
    main()
