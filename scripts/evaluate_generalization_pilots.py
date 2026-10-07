"""Frozen diagnostic tests of A–D; preserve the failed validation decision."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.config import architecture_name
from dinov3_global.data import collate_global
from dinov3_global.engine import build_test_loader, load_model
from dinov3_global.evaluation import audit_test_split, predict_heads, portable_path, sha256_file, write_json
from dinov3_global.metrics import to_jsonable
from dinov3_global.preprocessing import preprocess_for_config
from dinov3_global.runtime import loader_kwargs, validate_runtime, set_training_threads
from dinov3_global.study import recall_score
from scripts.compare_axial_pilot import read_predictions, vzc_test
from scripts.compare_generalization import aligned, mean_predictions
from scripts.evaluate import test_report
from scripts.evaluate_ood import read_labels, evaluate_labels
from scripts.evaluate_vzc import build_labels, metrics as vzc_metrics

CLASSES = ("RR", "RG", "NoR")


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze(path, value):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"Frozen inputs or sources changed: {path}")
    write_json(path, value)


def pilot_jobs(study):
    jobs = []
    for name in "ABCD":
        checkpoint = study / name / "fold0/best_unconstrained.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fold = json.loads((study / name / "fold0/fold_report.json").read_text())
        history = [
            json.loads(line) for line in (study / name / "fold0/history.jsonl").read_text().splitlines()
        ]
        best = max(history, key=lambda row: (row["mAP"], -row["epoch"]))
        if state["epoch"] != best["epoch"] or state["epoch"] != fold["best_epoch"] or "ema" not in state:
            raise ValueError("Diagnostic checkpoint does not match the frozen validation choice")
        jobs.append(
            dict(
                name=name,
                checkpoint=str(checkpoint.resolve()),
                sha256=sha256_file(checkpoint),
                epoch=state["epoch"],
                seed=state["cfg"]["optim"]["seed"],
                architecture=architecture_name(state["cfg"]),
                cfg=state["cfg"],
                selection="highest fold-0 validation mAP; diagnostic checkpoint, failed eligibility",
            )
        )
    return jobs


def inference_key(cfg):
    return json.dumps(
        dict(
            backbone=cfg["backbone"],
            preprocessing=cfg["data"].get("preprocessing", "legacy"),
            target_hw=cfg["data"]["target_hw"],
            img_root=cfg["data"]["img_root"],
            crop_sides=cfg["data"].get("crop_sides", 0),
        ),
        sort_keys=True,
    )


class RGBDataset(Dataset):
    def __init__(self, entries, cfg):
        self.entries, self.cfg = entries, cfg

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        image, _, metadata = preprocess_for_config(entry["image"], self.cfg)
        return dict(
            image=image,
            label=torch.tensor(entry["label"]),
            path=entry["image"],
            city=entry.get("camera", "unknown"),
            max_lamp_h=torch.tensor(entry.get("max_lamp_h", 0.0)),
            **metadata,
        )


def collate_rgb(batch):
    result = dict(
        image=torch.stack([item["image"] for item in batch]),
        label=torch.stack([item["label"] for item in batch]),
        path=[item["path"] for item in batch],
        city=[item["city"] for item in batch],
        max_lamp_h=torch.stack([item["max_lamp_h"] for item in batch]),
    )
    for key in ("content_mask", "geometry"):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    return result


def dataset_report(domain, entries, labels, logits, meta, atlas_entries=None):
    if domain == "DTLD":
        pseudo = np.array([entry["pseudo_nor"] for entry in entries], bool)
        return test_report(labels, logits, meta, pseudo)
    if domain == "ATLAS":
        return evaluate_labels(atlas_entries, logits)
    result = vzc_metrics(entries, logits)
    clean = np.array([not entry["pseudo_nor"] for entry in entries])
    result["exclude_unknown_relevant_sensitivity"] = vzc_metrics(
        [entry for entry, keep in zip(entries, clean) if keep], logits[clean]
    )
    return result


def reference_predictions(domain, name):
    baseline = REPO / "runs/v5/evaluations"
    if name.startswith("v5_fold"):
        fold = int(name[-1])
        if domain == "DTLD":
            return read_predictions(baseline / f"dtld_test/member{fold}/predictions.npz")
        if domain == "ATLAS":
            return read_predictions(baseline / f"atlas/{name}/predictions.csv")
        return vzc_test(baseline / "vzc", name)
    if name == "v5_ensemble4":
        return mean_predictions([reference_predictions(domain, f"v5_fold{fold}") for fold in range(4)])
    axial = REPO / "runs/v6_axial_pilot/evaluations"
    if domain == "DTLD":
        return read_predictions(axial / "dtld_test/member0/predictions.npz")
    if domain == "ATLAS":
        return read_predictions(axial / "atlas/v6_axial_fold0/predictions.csv")
    return vzc_test(axial / "vzc", "v6_axial_fold0")


def export_summary(out, results, jobs):
    metrics, classes, confusion = [], [], []
    summaries = {}
    for domain, models in results.items():
        reference = models["v5_fold0"]
        summaries[domain] = {}
        for name, current in models.items():
            row = dict(
                dataset=domain, model=name, **current["metrics"], mean_RR_RG_recall=recall_score(current)
            )
            metrics.append(row)
            summaries[domain][name] = dict(
                mAP=row["mAP"],
                macro_F1=row["macro_F1"],
                mean_RR_RG_recall=row["mean_RR_RG_recall"],
                **{c + "_recall": current["per_class"][c]["recall"] for c in CLASSES},
                delta_mAP_vs_v5_fold0=row["mAP"] - reference["metrics"]["mAP"],
            )
            classes += [
                dict(dataset=domain, model=name, class_name=c, **current["per_class"][c]) for c in CLASSES
            ]
            confusion += [
                dict(
                    dataset=domain, model=name, true=tc, predicted=pc, count=current["confusion_matrix"][i][j]
                )
                for i, tc in enumerate(CLASSES)
                for j, pc in enumerate(CLASSES)
            ]
    for filename, rows in (
        ("metrics.csv", metrics),
        ("per_class.csv", classes),
        ("confusion_matrices.csv", confusion),
    ):
        import csv

        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    write_json(out / "summary.json", to_jsonable(summaries))
    lines = [
        "# A–D fixed-checkpoint diagnostic evaluations",
        "",
        "A–D are single fold-0 EMA models fixed at the best validation-mAP epoch. All failed the original eligibility rule. These tests do not change the v5 retention decision.",
        "",
        "DTLD: all 12,453 official test frames. VZC_TLD: all 598 published test images; source training images excluded. ATLAS: the confirmed 528-image manual benchmark, not official test ground truth. All metrics use raw logits at T=1.",
        "",
        "The matched reference is v5_fold0. Other v5 folds, the four-head reference ensemble, and the prior axial pilot are shown separately. Cached references use exactly aligned image membership and labels.",
        "",
    ]
    for domain, models in summaries.items():
        lines += [
            f"## {domain}",
            "",
            "| Model | mAP | Macro F1 | Mean RR/RG recall | NoR recall | Δ mAP vs v5 fold 0 (pp) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in (
            "v5_fold0",
            "A",
            "B",
            "C",
            "D",
            "v6_axial_fold0",
            "v5_fold1",
            "v5_fold2",
            "v5_fold3",
            "v5_ensemble4",
        ):
            row = models[name]
            lines.append(
                f"| {name} | {100 * row['mAP']:.2f}% | {100 * row['macro_F1']:.2f}% | {100 * row['mean_RR_RG_recall']:.2f}% | {100 * row['NoR_recall']:.2f}% | {100 * row['delta_mAP_vs_v5_fold0']:+.2f} |"
            )
        lines.append("")
    lines += [
        "## Checkpoints",
        "",
        *[f"- {job['name']}: epoch {job['epoch']}, SHA-256 {job['sha256']}." for job in jobs],
        "",
        "Per-image raw predictions, per-class precision/recall/F1, confusion matrices, raw calibration metrics and available slices are retained alongside the frozen plan. External results are exploratory, as these datasets have influenced architecture development.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "comparison.json",
        to_jsonable(
            dict(
                models=results,
                checkpoint_jobs=jobs,
                scope="User-authorized post-study diagnostics; no selection, calibration fitting, retraining or promotion",
                summaries=summaries,
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=REPO / "runs/generalization_study")
    parser.add_argument("--out", type=Path, default=REPO / "runs/generalization_study/diagnostic_tests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("Batch must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists() and not args.resume:
        raise ValueError("Evaluation exists; use --resume")
    status = dict(state="auditing", started_at=now(), completed_models=[], stage="inputs")
    write_json(out / "status.json", status)
    runtime = validate_runtime(json.loads((REPO / "configs/runtime.json").read_text()))
    set_training_threads(runtime)
    torch.set_num_interop_threads(1)
    jobs = pilot_jobs(args.study)
    sources = [
        Path(__file__),
        *sorted((REPO / "src/dinov3_global").glob("*.py")),
        *[
            REPO / "scripts" / name
            for name in (
                "evaluate.py",
                "evaluate_ood.py",
                "evaluate_vzc.py",
                "compare_axial_pilot.py",
                "compare_generalization.py",
            )
        ],
    ]
    inputs = [
        REPO / "metadata/atlas_labels.json",
        REPO / "metadata/atlas_collection.json",
        REPO / "datasets/VZC_TLD/download_manifest.json",
        REPO / "datasets/VZC_TLD/labels/test_v1.json",
        REPO / "datasets/DTLD/v2.0/DTLD_test.json",
        args.study / "decision.json",
    ]
    base_cfg = jobs[0]["cfg"]
    _, base_ds = build_test_loader(base_cfg, str(REPO))
    audit, membership = audit_test_split(base_cfg, base_ds, REPO)
    dtld_entries = membership["test"]
    write_json(out / "dtld_audit.json", audit)
    write_json(out / "dtld_membership.json", membership)
    atlas_entries, excluded = read_labels(
        REPO / "metadata/atlas_labels.json", REPO / "datasets/atlas_relevance_expanded", True
    )
    atlas_rgb = [
        dict(file=name, image=str(path), label=y, camera=Path(name).parts[-3])
        for name, path, y in atlas_entries
    ]
    vzc_manifest = json.loads((REPO / "datasets/VZC_TLD/download_manifest.json").read_text())
    if not vzc_manifest.get("image_integrity_verified"):
        raise ValueError("VZC download has not been verified")
    vzc_all, vzc_audit, annotation_hashes = build_labels(REPO / "datasets/VZC_TLD", vzc_manifest)
    vzc_entries = [entry for entry in vzc_all if entry["split"] == "test"]
    if len(dtld_entries) != 12453 or len(atlas_entries) != 528 or len(vzc_entries) != 598:
        raise ValueError("Evaluation membership differs from the fixed benchmarks")
    write_json(
        out / "vzc_test_labels.json",
        dict(images=vzc_entries, audit=vzc_audit, annotation_sha256=annotation_hashes),
    )
    vzc_rgb = [dict(entry, image=str(REPO / "datasets/VZC_TLD" / entry["file"])) for entry in vzc_entries]
    domains = dict(DTLD=dtld_entries, ATLAS=atlas_rgb, VZC_TLD=vzc_entries)
    reference_names = [*[f"v5_fold{f}" for f in range(4)], "v5_ensemble4", "v6_axial_fold0"]
    references = {
        domain: {
            name: reference_predictions(domain if domain != "VZC_TLD" else "VZC", name)
            for name in reference_names
        }
        for domain in domains
    }
    reference_sources = [
        p
        for root in (REPO / "runs/v5/evaluations", REPO / "runs/v6_axial_pilot/evaluations")
        for p in root.rglob("*")
        if p.is_file()
        and p.name in ("predictions.npz", "predictions.csv", "plan.json", "labels_snapshot.json")
    ]
    plan = dict(
        jobs=jobs,
        primary_temperature=1.0,
        diagnostic_only=True,
        original_decision_sha256=sha256_file(args.study / "decision.json"),
        datasets=dict(
            DTLD=dict(n=12453, split="official test"),
            ATLAS=dict(n=528, split="manual benchmark confirmed by user", excluded=excluded),
            VZC_TLD=dict(n=598, split="published test only"),
        ),
        source_sha256={portable_path(p, REPO): sha256_file(p) for p in sources},
        input_sha256={portable_path(p, REPO): sha256_file(p) for p in [*inputs, *reference_sources]},
        runtime=runtime,
        batch=args.batch,
        checkpoint_selection="DTLD validation mAP only; no test selection",
        image_membership_hashes={
            "DTLD": audit["splits"]["test"]["membership_sha256"],
            "ATLAS": sha256_file(REPO / "metadata/atlas_labels.json"),
            "VZC_TLD": sha256_file(out / "vzc_test_labels.json"),
        },
    )
    freeze(out / "plan.json", plan)
    results = {}
    models = {}
    started = time.monotonic()
    try:
        device = torch.device("cuda")
        for job in jobs:
            models[job["name"]], _, _ = load_model(job["checkpoint"], str(REPO), device)
        groups = {}
        for job in jobs:
            groups.setdefault(inference_key(job["cfg"]), []).append(job)
        for domain, entries in domains.items():
            results[domain] = {}
            domain_ids = [
                Path(entry["native_path"]).stem if domain == "DTLD" else entry["file"] for entry in entries
            ]
            expected_labels = np.array(
                [CLASSES.index(entry["label"]) if domain == "DTLD" else entry["label"] for entry in entries]
            )
            for group in groups.values():
                remaining = []
                for job in group:
                    directory = out / domain / job["name"]
                    if (directory / "report.json").exists():
                        existing = json.loads((directory / "report.json").read_text())
                        pred = directory / "predictions.npz"
                        if existing["checkpoint_sha256"] != job["sha256"] or existing[
                            "predictions_sha256"
                        ] != sha256_file(pred):
                            raise ValueError("Cached checkpoint or predictions changed")
                        with np.load(pred) as data:
                            if data["paths"].tolist() != domain_ids or not np.array_equal(
                                data["labels"], expected_labels
                            ):
                                raise ValueError("Cached prediction membership changed")
                        results[domain][job["name"]] = existing
                    else:
                        remaining.append(job)
                if not remaining:
                    continue
                cfg = remaining[0]["cfg"]
                if domain == "DTLD":
                    _, dataset = build_test_loader(cfg, str(REPO))
                    loader = DataLoader(
                        dataset,
                        batch_size=args.batch,
                        collate_fn=collate_global,
                        shuffle=False,
                        **loader_kwargs(cfg, runtime),
                    )
                    expected_paths = [entry["image"] for entry in entries]
                else:
                    rgb_entries = atlas_rgb if domain == "ATLAS" else vzc_rgb
                    dataset = RGBDataset(rgb_entries, cfg)
                    loader = DataLoader(
                        dataset,
                        batch_size=args.batch,
                        collate_fn=collate_rgb,
                        shuffle=False,
                        **loader_kwargs(cfg, runtime),
                    )
                    expected_paths = [portable_path(entry["image"], REPO) for entry in rgb_entries]
                names = [job["name"] for job in remaining]
                status.update(state="running", stage=domain + ":" + ",".join(names), updated_at=now())
                write_json(out / "status.json", status)
                print(f"{now()} Evaluating {domain}: {names}, {len(dataset)} images", flush=True)

                def progress(value):
                    status.update(
                        progress=value, updated_at=now(), elapsed_seconds=time.monotonic() - started
                    )
                    write_json(out / "status.json", status)

                labels, logits, meta, verification = predict_heads(
                    models[names[0]], [models[name].head for name in names], loader, device, REPO, progress
                )
                if not np.array_equal(labels, expected_labels) or meta["paths"] != expected_paths:
                    raise ValueError("Inference membership differs from frozen labels")
                for job, current in zip(remaining, logits):
                    if current.shape != (len(entries), 3) or not np.isfinite(current).all():
                        raise ValueError("Invalid model logits")
                    directory = out / domain / job["name"]
                    directory.mkdir(parents=True, exist_ok=True)
                    pred = directory / "predictions.npz"
                    np.savez_compressed(pred, paths=np.array(domain_ids), labels=labels, logits=current)
                    current_report = dataset_report(
                        "VZC" if domain == "VZC_TLD" else domain,
                        entries,
                        labels,
                        current,
                        meta,
                        atlas_entries,
                    )
                    current_report.update(
                        checkpoint_sha256=job["sha256"],
                        epoch=job["epoch"],
                        weights="EMA",
                        temperature=1.0,
                        checkpoint_selection=job["selection"],
                        predictions_sha256=sha256_file(pred),
                        verification=verification,
                        completed_at=now(),
                    )
                    write_json(directory / "report.json", to_jsonable(current_report))
                    results[domain][job["name"]] = current_report
                    status["completed_models"].append(domain + ":" + job["name"])
                    write_json(out / "status.json", status)
                    print(
                        f"Completed {domain}/{job['name']}: mAP={current_report['metrics']['mAP']:.4f}",
                        flush=True,
                    )
                del loader, dataset
            # Cached reference scores are recomputed after exact membership/label alignment.
            exemplar = out / domain / "A/predictions.npz"
            new = read_predictions(exemplar)
            if domain == "DTLD":
                meta = dict(
                    cities=[entry["city"] for entry in entries],
                    max_lamp_h=np.array([item["max_lamp_h"] for item in base_ds.items]),
                )
            for name, reference in references[domain].items():
                if domain == "DTLD":
                    reference = ([Path(path).stem for path in reference[0]], reference[1], reference[2])
                _, labels, _, logits = aligned(new, reference, False)
                current_report = dataset_report(
                    "VZC" if domain == "VZC_TLD" else domain, entries, labels, logits, meta, atlas_entries
                )
                current_report["prediction_source"] = (
                    "Existing frozen evaluation; raw logits aligned by exact image identity and labels"
                )
                results[domain][name] = current_report
                directory = out / domain / name
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "report.json", to_jsonable(current_report))
            export_summary(out, results, jobs) if len(results) == 3 else None
        for key in ("source_sha256", "input_sha256"):
            if any(sha256_file(REPO / path) != digest for path, digest in plan[key].items()):
                raise ValueError("Frozen sources or inputs changed during evaluation")
        if any(sha256_file(job["checkpoint"]) != job["sha256"] for job in jobs):
            raise ValueError("Frozen checkpoint changed")
        export_summary(out, results, jobs)
        status.update(
            state="complete", stage="complete", elapsed_seconds=time.monotonic() - started, completed_at=now()
        )
        write_json(out / "status.json", status)
        print(f"All fixed-model diagnostic evaluations complete: {out / 'README.md'}", flush=True)
    except BaseException as error:
        status.update(state="failed", error=str(error), updated_at=now())
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
