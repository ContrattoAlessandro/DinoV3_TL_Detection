"""Frozen VZC-TLD transfer evaluation of existing DTLD checkpoints.

No training, checkpoint selection, or calibration fitting uses VZC. All COCO
annotations are retained, including iscrowd=1, which marks almost all lamps.
Download the dataset separately and supply its verified manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.data import global_label_from_states
from dinov3_global.backbone import BACKBONE_ID, BACKBONE_REVISION
from dinov3_global.dtld import STATES
from dinov3_global.engine import load_model
from dinov3_global.config import architecture_name
from dinov3_global.metrics import report, softmax_np, to_jsonable
from dinov3_global.preprocessing import preprocess_for_config

CLASSES = ["RR", "RG", "NoR"]
CATEGORIES = [
    "not_relevant_unknown",
    "not_relevant_green",
    "not_relevant_yellow",
    "not_relevant_red",
    "relevant_unknown",
    "relevant_green",
    "relevant_yellow",
    "relevant_red",
]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(to_jsonable(data), indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.1 * 2**attempt, 1))


def source_hashes():
    files = [
        Path(__file__),
        *[
            REPO / "src/dinov3_global" / f"{name}.py"
            for name in ("backbone", "model", "head", "engine", "data", "dtld", "metrics", "config")
        ],
        REPO / "src/dinov3_global/inference.py",
        REPO / "src/dinov3_global/preprocessing.py",
        REPO / "src/dinov3_global/evidence_head.py",
    ]
    return {str(p.relative_to(REPO)): sha256(p) for p in files}


def build_labels(root, manifest):
    """Derive image labels using the existing DTLD rule, before model inference."""
    root = root.resolve()
    remote = {r["name"]: r for r in manifest["files"]}
    entries, audit, annotation_hashes = [], {}, {}
    seen_paths, seen_hashes, seen_image_ids = set(), set(), set()
    hash_groups = defaultdict(list)
    intersections = {}
    for split in ("test", "train"):
        label_key = f"labels/{split}_v1.json"
        path = root / label_key
        annotation_hashes[split] = sha256(path)
        if annotation_hashes[split] != remote[label_key]["sha256"]:
            raise ValueError(f"Annotations changed: {split}")
        data = json.loads(path.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in data["categories"]}
        if names != dict(enumerate(CATEGORIES)):
            raise ValueError("Unexpected VZC category schema")
        ids = [i["id"] for i in data["images"]]
        id_set = set(ids)
        if len(ids) != len(set(ids)) or seen_image_ids.intersection(ids):
            raise ValueError("Duplicate image IDs within/across published splits")
        seen_image_ids.update(ids)
        annotation_ids = [a["id"] for a in data["annotations"]]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("Duplicate COCO annotation IDs")
        lamps = defaultdict(list)
        for annotation in data["annotations"]:
            if annotation["image_id"] not in id_set or annotation["category_id"] not in names:
                raise ValueError("Invalid annotation reference")
            lamps[annotation["image_id"]].append(annotation)
        split_entries = []
        for image in sorted(data["images"], key=lambda i: i["file_name"]):
            key = image["file_name"]
            path = (root / key).resolve()
            if key in seen_paths or not path.is_relative_to(root) or key not in remote:
                raise ValueError(f"Duplicate/invalid image: {key}")
            image_hash = sha256(path)
            if image_hash != remote[key]["sha256"]:
                raise ValueError(f"Image contents changed: {key}")
            if (image["width"], image["height"]) != (remote[key]["width"], remote[key]["height"]):
                raise ValueError(f"Image geometry disagrees with COCO: {key}")
            seen_paths.add(key)
            seen_hashes.add(image_hash)
            annotations = lamps[image["id"]]
            category_ids = sorted(a["category_id"] for a in annotations)
            relevance = np.array([int(c >= 4) for c in category_ids], dtype=int)
            states = np.array([STATES.index(CATEGORIES[c].split("_")[-1]) for c in category_ids], dtype=int)
            label, debug = global_label_from_states(relevance, states)
            relevant_states = sorted({CATEGORIES[c].split("_")[-1] for c in category_ids if c >= 4})
            group = "+".join(relevant_states) if relevant_states else "no_relevant_lamp"
            entry = dict(
                file=key,
                split=split,
                image_id=image["id"],
                label=int(label),
                class_name=CLASSES[label],
                sha256=image_hash,
                relevant_states=relevant_states,
                relevant_group=group,
                pseudo_nor=label == 2 and debug["n_rel_off_unknown"] > 0,
                n_annotations=len(annotations),
                n_relevant=int(relevance.sum()),
                max_lamp_h=max((a["bbox"][3] for a in annotations), default=0),
            )
            split_entries.append(entry)
            hash_groups[image_hash].append(dict(file=key, split=split, label=int(label)))
        audit[split] = dict(
            n=len(split_entries),
            n_annotations=len(data["annotations"]),
            classes=dict(Counter(e["class_name"] for e in split_entries)),
            relevant_groups=dict(Counter(e["relevant_group"] for e in split_entries)),
            pseudo_nor=sum(e["pseudo_nor"] for e in split_entries),
            no_annotations=sum(e["n_annotations"] == 0 for e in split_entries),
            iscrowd=dict(Counter(a.get("iscrowd", 0) for a in data["annotations"])),
        )
        intersections[split] = {Path(e["file"]).stem.rsplit("_", 1)[0] for e in split_entries}
        entries.extend(split_entries)
    audit["train_test_shared_filename_location_prefixes"] = len(
        intersections["train"] & intersections["test"]
    )
    audit["identical_image_groups"] = [group for group in hash_groups.values() if len(group) > 1]
    if any(len({e["label"] for e in group}) > 1 for group in audit["identical_image_groups"]):
        raise ValueError("Identical images have conflicting published labels")
    audit["unreferenced_downloaded_images"] = sorted(
        k for k in remote if k.endswith(".png") and k not in seen_paths
    )
    return entries, audit, annotation_hashes


def metrics(entries, logits, temperature=1.0):
    y = np.array([e["label"] for e in entries])
    result = report(y, logits, T=temperature, max_lamp_h=np.array([e["max_lamp_h"] for e in entries]))
    probabilities = softmax_np(np.asarray(logits, dtype=float) / temperature)
    pred = probabilities.argmax(1)
    result["metrics"].update(
        accuracy=float((pred == y).mean()),
        nll=float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean()),
        brier=float(((probabilities - np.eye(3)[y]) ** 2).sum(1).mean()),
    )
    result["class_names"] = CLASSES
    result["class_counts"] = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    result["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    classification = classification_report(
        y, pred, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0
    )
    result["classification"] = classification
    for average, key in (("macro", "macro avg"), ("weighted", "weighted avg")):
        for name in ("precision", "recall", "f1-score"):
            result["metrics"][f"{average}_{name}"] = classification[key][name]
    if result["metrics"]["n_classes"] != 3:
        result["metrics"]["mAP_present_classes"] = result["metrics"]["mAP"]
        result["metrics"]["mAP"] = None
    for bucket in result.get("slices", {}).get("size", {}).values():
        if bucket["n_classes"] != 3:
            bucket["mAP_present_classes"] = bucket["mAP"]
            bucket["mAP"] = None
    return to_jsonable(result)


def model_report(entries, logits, temperature):
    result = {}
    for split in ("test", "train", "all"):
        mask = np.array([split == "all" or e["split"] == split for e in entries])
        rows = [e for e, include in zip(entries, mask) if include]
        current = logits[mask]
        result[split] = metrics(rows, current)
        strict = np.array([not e["pseudo_nor"] for e in rows])
        result[split]["exclude_unknown_relevant_sensitivity"] = metrics(
            [e for e, keep in zip(rows, strict) if keep], current[strict]
        )
        seen = set()
        unique = []
        for entry in rows:
            unique.append(entry["sha256"] not in seen)
            seen.add(entry["sha256"])
        unique = np.array(unique)
        result[split]["unique_images_sensitivity"] = metrics(
            [e for e, keep in zip(rows, unique) if keep], current[unique]
        )
        result[split]["relevant_state_groups"] = {}
        for group in sorted({e["relevant_group"] for e in rows}):
            group_mask = np.array([e["relevant_group"] == group for e in rows])
            group_y = np.array([e["label"] for e, keep in zip(rows, group_mask) if keep])
            group_pred = current[group_mask].argmax(1)
            result[split]["relevant_state_groups"][group] = dict(
                n=int(group_mask.sum()),
                accuracy=float((group_pred == group_y).mean()),
                predictions={c: int((group_pred == i).sum()) for i, c in enumerate(CLASSES)},
            )
        if temperature is not None:
            result[split]["saved_DTLD_temperature"] = metrics(rows, current, temperature)
    return result


def evaluation_jobs(checkpoints, names=None):
    if names is not None and len(names) != len(checkpoints):
        raise ValueError("One model name is required for each checkpoint")
    jobs = []
    for index, path in enumerate(checkpoints):
        cfg = torch.load(path, map_location="cpu", weights_only=True)["cfg"]
        architecture = architecture_name(cfg)
        name = names[index] if names is not None else f"{architecture}_fold{index}"
        if not name or Path(name).name != name or name in (".", "..") or "\\" in name or ":" in name:
            raise ValueError("Model names must be simple directory names")
        jobs.append(
            dict(
                name=name,
                checkpoint=str(path.resolve()),
                checkpoint_sha256=sha256(path),
                architecture=architecture,
                seed=cfg["optim"]["seed"],
            )
        )
    if len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate model names")
    return jobs


def export(out, entries, model_reports, model_logits):
    summary, classes, confusion = [], [], []
    for name, result in model_reports.items():
        for split in ("test", "train", "all"):
            for policy in (
                "primary",
                "exclude_unknown_relevant_sensitivity",
                "unique_images_sensitivity",
                "saved_DTLD_temperature",
            ):
                if policy != "primary" and policy not in result[split]:
                    continue
                report_row = result[split] if policy == "primary" else result[split][policy]
                summary.append(dict(model=name, split=split, policy=policy, **report_row["metrics"]))
                for cls in CLASSES:
                    classes.append(
                        dict(
                            model=name,
                            split=split,
                            policy=policy,
                            class_name=cls,
                            **report_row["classification"][cls],
                        )
                    )
                for i, true in enumerate(CLASSES):
                    for j, predicted in enumerate(CLASSES):
                        confusion.append(
                            dict(
                                model=name,
                                split=split,
                                policy=policy,
                                true=true,
                                predicted=predicted,
                                count=report_row["confusion_matrix"][i][j],
                            )
                        )
        with (out / name / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "file",
                    "split",
                    "label",
                    "pred",
                    "logit_RR",
                    "logit_RG",
                    "logit_NoR",
                    "P_RR_T1",
                    "P_RG_T1",
                    "P_NoR_T1",
                    "pseudo_nor",
                ]
            )
            logits = model_logits[name]
            for entry, row, probability in zip(entries, logits, softmax_np(logits.astype(float))):
                writer.writerow(
                    [
                        entry["file"],
                        entry["split"],
                        entry["class_name"],
                        CLASSES[int(row.argmax())],
                        *row.tolist(),
                        *probability.tolist(),
                        entry["pseudo_nor"],
                    ]
                )
    for filename, rows in (
        ("metrics.csv", summary),
        ("per_class_metrics.csv", classes),
        ("confusion_matrices.csv", confusion),
    ):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    # Mean individual-model metrics are separate from the logit ensemble.
    grouped = defaultdict(list)
    for row in summary:
        if "ensemble" not in row["model"]:
            grouped[(row["split"], row["policy"])].append(row)
    means = []
    for (split, policy), rows in grouped.items():
        combined = dict(model=f"mean_{len(rows)}_individual_checkpoints", split=split, policy=policy)
        for key in rows[0]:
            if key in ("model", "split", "policy"):
                continue
            values = [row[key] for row in rows if isinstance(row[key], (int, float))]
            combined[key] = float(np.mean(values)) if values else None
        means.append(combined)
    columns = list(dict.fromkeys(key for row in means for key in row))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(means)
    atomic_json(out / "summary.json", means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "datasets/VZC_TLD")
    parser.add_argument("--manifest", type=Path, default=REPO / "datasets/VZC_TLD/download_manifest.json")
    parser.add_argument("--out", type=Path, default=REPO / "runs/v5/vzc")
    parser.add_argument("--ckpt", nargs="+", type=Path)
    parser.add_argument("--model-names", nargs="+", help="Names matching --ckpt order")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("Batch size must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = args.ckpt or [REPO / f"runs/v5/city_cv/fold{f}/best.pt" for f in range(4)]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("image_integrity_verified"):
        raise ValueError("Download manifest must verify image integrity")
    entries, audit, annotation_hashes = build_labels(args.dataset, manifest)
    label_snapshot = dict(
        class_names=CLASSES,
        label_policy="DTLD map_to_nor; RR has priority over RG",
        revision=manifest["revision"],
        annotation_sha256=annotation_hashes,
        images=entries,
    )
    label_path = args.out / "labels_snapshot.json"
    if label_path.exists() and json.loads(label_path.read_text(encoding="utf-8")) != label_snapshot:
        raise ValueError("Existing frozen labels differ")
    atomic_json(label_path, label_snapshot)
    atomic_json(args.out / "label_audit.json", audit)
    jobs = evaluation_jobs(checkpoints, args.model_names)
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    backbone_files = {}
    checkpoint_configs = [torch.load(p, map_location="cpu", weights_only=True)["cfg"] for p in checkpoints]
    configurations = [cfg["backbone"] for cfg in checkpoint_configs]
    if any(backbone != configurations[0] for backbone in configurations[1:]):
        raise ValueError("All evaluation checkpoints must use the same frozen backbone settings")
    local = configurations[0].get("local_ckpt")
    snapshot = (REPO / local).resolve() if local else None
    for filename in ("config.json", "model.safetensors"):
        if snapshot is not None:
            path = str(snapshot / filename)
        else:
            path = try_to_load_from_cache(BACKBONE_ID, filename, revision=BACKBONE_REVISION)
            if not isinstance(path, str):
                path = hf_hub_download(BACKBONE_ID, filename, revision=BACKBONE_REVISION)
        backbone_files[filename] = dict(path=path, sha256=sha256(path))
    plan = dict(
        dataset=manifest["repo_id"],
        revision=manifest["revision"],
        jobs=jobs,
        backbone_files=backbone_files,
        dataset_manifest_sha256=sha256(args.manifest),
        labels_sha256=sha256(label_path),
        source_sha256=source_hashes(),
        n=len(entries),
        batch=args.batch,
        crop_sides=0,
        target_hw=checkpoint_configs[0]["data"]["target_hw"],
        metric_input="raw logits T=1; final softmax AP",
        weights="best DTLD-selected EMA checkpoints",
        ensemble="arithmetic mean of raw logits",
        policy="No VZC training, per-fold checkpoint selection, or temperature fitting; exploratory transfer",
    )
    if any(cfg["data"].get("preprocessing", "legacy") != "legacy" for cfg in checkpoint_configs):
        plan["preprocessing"] = [
            dict(mode=cfg["data"].get("preprocessing", "legacy"), target_hw=cfg["data"]["target_hw"])
            for cfg in checkpoint_configs
        ]
    plan_path = args.out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Evaluation plan/provenance changed")
    atomic_json(plan_path, plan)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dev = torch.device(args.device)
    status = dict(state="running", pid=os.getpid(), started_at=now(), total_jobs=len(jobs), completed_jobs=[])
    started = time.monotonic()
    atomic_json(args.out / "status.json", status)
    all_logits, all_reports = {}, {}
    try:
        for job in jobs:
            directory = args.out / job["name"]
            directory.mkdir(exist_ok=True)
            predictions_path = directory / "predictions.npz"
            result_path = directory / "report.json"
            status.update(current_job=job["name"], images_completed=0, updated_at=now())
            atomic_json(args.out / "status.json", status)
            if predictions_path.exists() and result_path.exists():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    existing["checkpoint_sha256"] != job["checkpoint_sha256"]
                    or existing["labels_sha256"] != plan["labels_sha256"]
                ):
                    raise ValueError("Cached checkpoint/label provenance changed")
                if existing["predictions_sha256"] != sha256(predictions_path):
                    raise ValueError("Cached raw predictions changed")
                with np.load(predictions_path, allow_pickle=False) as data:
                    if data["paths"].tolist() != [e["file"] for e in entries] or data["labels"].tolist() != [
                        e["label"] for e in entries
                    ]:
                        raise ValueError("Cached prediction membership changed")
                    logits = data["logits"]
                temperature = existing["temperature_from_DTLD"]
            else:
                model, cfg, state = load_model(job["checkpoint"], str(REPO), dev)
                temperature = float(state["metrics"]["temperature"])
                chunks = []
                print(f"{now()} Evaluating {job['name']} on {len(entries)} images (test first)", flush=True)
                with torch.inference_mode(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    for offset in range(0, len(entries), args.batch):
                        batch_entries = entries[offset : offset + args.batch]
                        prepared = [
                            preprocess_for_config(str(args.dataset / e["file"]), cfg) for e in batch_entries
                        ]
                        metadata = {
                            k: torch.stack([item[2][k] for item in prepared]).to(dev) for k in prepared[0][2]
                        }
                        batch = torch.stack([item[0] for item in prepared]).to(dev).float().div_(255)
                        chunks.append(model(batch, **metadata)["logits"].float().cpu().numpy())
                        completed = offset + len(batch_entries)
                        if completed % 200 == 0 or completed == len(entries):
                            status.update(
                                images_completed=completed,
                                updated_at=now(),
                                elapsed_seconds=time.monotonic() - started,
                            )
                            atomic_json(args.out / "status.json", status)
                            print(f"  {job['name']}: {completed}/{len(entries)}", flush=True)
                logits = np.concatenate(chunks)
                del model, state
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if sha256(job["checkpoint"]) != job["checkpoint_sha256"]:
                    raise ValueError("Checkpoint changed during evaluation")
                np.savez_compressed(
                    predictions_path,
                    logits=logits,
                    labels=np.array([e["label"] for e in entries]),
                    paths=np.array([e["file"] for e in entries]),
                )
            if logits.shape != (len(entries), 3) or not np.isfinite(logits).all():
                raise ValueError("Invalid model predictions")
            result = model_report(entries, logits, temperature)
            result.update(
                checkpoint_sha256=job["checkpoint_sha256"],
                labels_sha256=plan["labels_sha256"],
                predictions_sha256=sha256(predictions_path),
                temperature_from_DTLD=temperature,
                completed_at=now(),
            )
            atomic_json(result_path, result)
            all_logits[job["name"]], all_reports[job["name"]] = logits, result
            status["completed_jobs"].append(job["name"])
            atomic_json(args.out / "status.json", status)
            print(f"Completed {job['name']}: test mAP={result['test']['metrics']['mAP']:.4f}", flush=True)
        if len(jobs) > 1:
            architectures = {j["architecture"] for j in jobs}
            prefix = next(iter(architectures)) if len(architectures) == 1 else "mixed"
            name = f"{prefix}_ensemble{len(jobs)}"
            logits = np.mean(list(all_logits.values()), axis=0)
            (args.out / name).mkdir(exist_ok=True)
            result = model_report(entries, logits, None)
            result.update(
                labels_sha256=plan["labels_sha256"], aggregation="mean raw logits", completed_at=now()
            )
            atomic_json(args.out / name / "report.json", result)
            np.savez_compressed(
                args.out / name / "predictions.npz",
                logits=logits,
                labels=np.array([e["label"] for e in entries]),
                paths=np.array([e["file"] for e in entries]),
            )
            all_logits[name], all_reports[name] = logits, result
        export(args.out, entries, all_reports, all_logits)
        if (
            source_hashes() != plan["source_sha256"]
            or sha256(args.manifest) != plan["dataset_manifest_sha256"]
        ):
            raise ValueError("Sources/dataset manifest changed during evaluation")
        if any(sha256(j["checkpoint"]) != j["checkpoint_sha256"] for j in jobs):
            raise ValueError("Checkpoint changed during suite")
        if any(sha256(f["path"]) != f["sha256"] for f in backbone_files.values()):
            raise ValueError("Frozen backbone changed during evaluation")
        atomic_json(args.out / "comparison.json", dict(plan=plan, audit=audit, models=all_reports))
        status.update(
            state="complete", current_job=None, updated_at=now(), elapsed_seconds=time.monotonic() - started
        )
        atomic_json(args.out / "status.json", status)
        print(f"VZC evaluation complete in {status['elapsed_seconds']:.1f}s", flush=True)
    except BaseException as error:
        status.update(state="failed", error=str(error), updated_at=now())
        atomic_json(args.out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
