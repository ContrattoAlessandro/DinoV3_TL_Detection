"""Evaluate reusable manual relevance labels against checkpoints or a CSV.

Labels are evaluation data only. AP uses raw softmax (T=1), as in DTLD reports.
Uncertain annotations are excluded unless explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.metrics import report, to_jsonable
from scripts.annotate_ood import CLASSES, sha256


def read_labels(path, images=None, include_uncertain=False):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("class_names") != list(CLASSES):
        raise ValueError("Unsupported labels file")
    root = Path(images or data["image_root"]).resolve()
    entries = []
    excluded = {"unlabeled": 0, "uncertain": 0}
    for k, r in data["images"].items():
        if r["label"] is None:
            excluded["unlabeled"] += 1
            continue
        if r["label"] not in CLASSES:
            raise ValueError(f"Invalid label for {k}")
        if r.get("uncertain") and not include_uncertain:
            excluded["uncertain"] += 1
            continue
        p = (root / k).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError(f"Missing/invalid image path: {k}")
        if sha256(p) != r.get("sha256"):
            raise ValueError(f"Image contents changed: {k}")
        entries.append((k, p, CLASSES.index(r["label"])))
    if not entries:
        raise ValueError("No eligible labeled images. Use scripts/annotate_ood.py first.")
    return entries, excluded


def csv_logits(path, entries):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {}
    for r in rows:
        k = r["file"].replace("\\", "/")
        if k in lookup:
            raise ValueError(f"Duplicate prediction path: {k}")
        lookup[k] = r
    names = [Path(k).name for k, _, _ in entries]
    output = []
    for k, _, _ in entries:
        r = lookup.get(k)
        if r is None:
            matches = [v for source, v in lookup.items() if k.endswith("/" + source)]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous prediction suffix: {k}")
            if matches:
                r = matches[0]
        if r is None and names.count(Path(k).name) == 1:
            # Existing external scripts exported unique basenames only.
            r = lookup.get(Path(k).name)
        if r is None:
            raise ValueError(f"Missing prediction: {k}")
        raw_keys = ["logit_" + c for c in CLASSES]
        if all(key in r for key in raw_keys):
            lg = np.array([float(r[key]) for key in raw_keys])
        else:
            prob = np.array([float(r.get("P_" + c, r.get("p_" + c, "nan"))) for c in CLASSES])
            if not np.isfinite(prob).all() or (prob < 0).any() or not np.isclose(prob.sum(), 1, atol=0.002):
                raise ValueError(f"Invalid probabilities: {k}")
            lg = np.log(np.clip(prob, 1e-12, 1))
        if not np.isfinite(lg).all():
            raise ValueError(f"Invalid logits: {k}")
        output.append(lg)
    return np.array(output)


def checkpoint_logits(paths, entries, device=None, crop_sides=0):
    import torch
    from dinov3_global.engine import load_model
    from dinov3_global.preprocessing import preprocess_for_config

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logits = []
    for path in paths:
        model, cfg, _ = load_model(path, str(REPO), dev)
        current = []
        with torch.inference_mode(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for _, p, _ in entries:
                x, _, metadata = preprocess_for_config(str(p), cfg, crop_sides)
                metadata = {k: v.unsqueeze(0).to(dev) for k, v in metadata.items()}
                out = model(x.unsqueeze(0).to(dev).float().div_(255), **metadata)
                current.append(out["logits"][0].float().cpu().numpy())
        logits.append(np.array(current))
        del model
    return np.mean(logits, axis=0)


def evaluate_labels(entries, logits, excluded=None):
    from sklearn.metrics import confusion_matrix

    y = np.array([r[2] for r in entries])
    result = report(y, logits)
    pred = logits.argmax(1)
    result["class_names"] = list(CLASSES)
    result["class_counts"] = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    result["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    result["accuracy"] = float((pred == y).mean())
    result["excluded"] = excluded or {}
    result["per_camera"] = {}
    for cam in sorted({Path(k).parts[-3] if len(Path(k).parts) >= 3 else "unknown" for k, _, _ in entries}):
        indices = [
            i
            for i, (k, _, _) in enumerate(entries)
            if (Path(k).parts[-3] if len(Path(k).parts) >= 3 else "unknown") == cam
        ]
        cam_metrics = report(y[indices], logits[indices])["metrics"]
        cam_metrics["class_counts"] = {c: int((y[indices] == i).sum()) for i, c in enumerate(CLASSES)}
        if cam_metrics["n_classes"] < 3:
            cam_metrics["mAP_present_classes"] = cam_metrics["mAP"]
            cam_metrics["mAP"] = None
        result["per_camera"][cam] = cam_metrics
    if len(set(y)) < 3:
        result["metrics"]["mAP_present_classes"] = result["metrics"]["mAP"]
        result["metrics"]["mAP"] = None
    result["limitations"] = [
        "Small, potentially correlated sample; use as a diagnostic benchmark.",
        "Do not train/tune on this set and call it unseen-data evaluation.",
        "DTLD calibration is not guaranteed to transfer to OOD images.",
    ]
    return to_jsonable(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=REPO / "runs/ood_labels.json")
    ap.add_argument("--images", type=Path, help="override root if images moved")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", nargs="+")
    source.add_argument("--predictions", type=Path, help="CSV with logits or probabilities")
    ap.add_argument("--include-uncertain", action="store_true")
    ap.add_argument("--crop-sides", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path, default=REPO / "runs/v5/ood")
    args = ap.parse_args()
    entries, excluded = read_labels(args.labels, args.images, args.include_uncertain)
    logits = (
        csv_logits(args.predictions, entries)
        if args.predictions
        else checkpoint_logits(args.ckpt, entries, args.device, args.crop_sides)
    )
    result = evaluate_labels(entries, logits, excluded)
    result.update(
        labels_file=str(args.labels.resolve()),
        source=str(args.predictions) if args.predictions else args.ckpt,
        crop_sides=args.crop_sides,
        metric_input="raw logits T=1" if args.ckpt else "CSV scores (may be calibrated/rounded)",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "pred", *["logit_" + c for c in CLASSES]])
        for (k, _, y), lg in zip(entries, logits):
            w.writerow([k, CLASSES[y], CLASSES[int(lg.argmax())], *lg.tolist()])
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    print(f"Report: {args.out / 'report.json'} (excluded={excluded})")


if __name__ == "__main__":
    main()
