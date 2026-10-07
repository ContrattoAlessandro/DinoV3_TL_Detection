"""Evaluate fixed Experiment C checkpoints on the official DTLD test split.

Raw T=1 metrics are primary. A single checkpoint can additionally use its saved
DTLD-validation temperature. No calibration is fitted on test data.
"""

import argparse
import copy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from torch.utils.data import DataLoader

from dinov3_global.backbone import BACKBONE_ID, BACKBONE_REVISION
from dinov3_global.config import validate_config, architecture_name
from dinov3_global.data import collate_global
from dinov3_global.engine import build_test_loader, load_model
from dinov3_global.evaluation import audit_test_split, portable_path, predict_heads, sha256_file, write_json
from dinov3_global.metrics import report, to_jsonable
from dinov3_global.runtime import loader_kwargs, set_training_threads, validate_runtime


def save_predictions(directory, labels, logits, meta, pseudo_nor):
    np.savez_compressed(
        directory / "predictions.npz",
        labels=labels,
        logits=logits,
        paths=np.asarray(meta["paths"]),
        cities=np.asarray(meta["cities"]),
        max_lamp_h=meta["max_lamp_h"],
        pseudo_nor=pseudo_nor,
    )


def test_report(labels, logits, meta, pseudo_nor):
    result = report(labels, logits, T=1.0, cities=meta["cities"], max_lamp_h=meta["max_lamp_h"])
    # Preserve the full official membership; this fixed diagnostic only
    # measures sensitivity to relevant-off/unknown-only frames mapped to NoR.
    clean = ~pseudo_nor
    result["without_pseudo_nor"] = report(labels[clean], logits[clean], T=1.0)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, nargs="+")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out", default="runs/evaluations/dtld_test", help="directory for report and raw predictions"
    )
    ap.add_argument("--runtime-loader", type=Path, help="execution-only worker/thread settings")
    ap.add_argument(
        "--calibrated",
        action="store_true",
        help="also report the saved validation temperature (one checkpoint)",
    )
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.calibrated and len(args.ckpt) != 1:
        ap.error("Saved per-model temperatures do not define a calibrated logit ensemble")
    directory = Path(args.out)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Evaluation directory is not empty; choose a new --out")
    directory.mkdir(parents=True, exist_ok=True)
    runtime = (
        validate_runtime(json.loads(args.runtime_loader.read_text(encoding="utf-8")))
        if args.runtime_loader
        else None
    )
    set_training_threads(runtime)
    states = [torch.load(path, map_location="cpu", weights_only=True) for path in args.ckpt]
    cfg = states[0]["cfg"]
    for state in states:
        validate_config(state["cfg"])
        for section in ("backbone", "decoder", "data"):
            if state["cfg"][section] != cfg[section]:
                raise ValueError(f"Checkpoints differ in {section}; cannot share encoder/test membership")
        if "ema" not in state:
            raise ValueError("The official DTLD evaluation requires retained EMA weights")
    checkpoint_hashes = [sha256_file(path) for path in args.ckpt]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("Duplicate ensemble checkpoint")
    members = [
        dict(
            name=f"member{index}",
            checkpoint=portable_path(path, repo_root),
            checkpoint_sha256=digest,
            epoch=state["epoch"],
            seed=state["cfg"]["optim"]["seed"],
            weights="EMA",
            head_parameters=sum(t.numel() for t in state["ema"].values()),
        )
        for index, (path, digest, state) in enumerate(zip(args.ckpt, checkpoint_hashes, states))
    ]
    write_json(directory / "status.json", dict(status="auditing", n_models=len(members)))
    tel, dataset = build_test_loader(cfg, repo_root)
    if runtime is not None:
        tel = DataLoader(
            dataset,
            batch_size=cfg["optim"]["batch"],
            shuffle=False,
            collate_fn=collate_global,
            **loader_kwargs(cfg, runtime),
        )
    audit, membership = audit_test_split(cfg, dataset, repo_root)
    write_json(directory / "data_audit.json", audit)
    write_json(directory / "membership.json", membership)
    plan = dict(
        schema_version=1,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        architecture=architecture_name(cfg),
        split="DTLD v2.0 official test",
        checkpoints=members,
        predictor="mean raw logits, then softmax" if len(members) > 1 else "single EMA head",
        primary_temperature=1.0,
        checkpoint_selection=f"City-validation EMA {states[0].get('selection_metric', 'mAP')} at T=1; fixed before test inference",
        test_fitting=False,
        calibration="None for primary metrics; no test-fitted temperature or thresholds",
        backbone=dict(hf_id=BACKBONE_ID, revision=BACKBONE_REVISION),
        data=cfg["data"],
        runtime_loader=runtime or cfg.get("loader", {}),
        batch_size=tel.batch_size,
        precision=f"CUDA {torch.get_autocast_dtype('cuda')} autocast"
        if dev.type == "cuda"
        else "CPU float32",
        device=str(dev),
        gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        environment=dict(
            python=platform.python_version(),
            platform=platform.system(),
            packages={
                name: version(name)
                for name in (
                    "torch",
                    "transformers",
                    "huggingface-hub",
                    "numpy",
                    "Pillow",
                    "PyYAML",
                    "scikit-learn",
                    "tqdm",
                )
            },
            cuda_runtime=torch.version.cuda,
        ),
        source_sha256={
            path: sha256_file(Path(repo_root) / path)
            for path in (
                "scripts/evaluate.py",
                "src/dinov3_global/evaluation.py",
                "src/dinov3_global/model.py",
                "src/dinov3_global/head.py",
                "src/dinov3_global/backbone.py",
                "src/dinov3_global/data.py",
                "src/dinov3_global/dtld.py",
                "src/dinov3_global/metrics.py",
                "src/dinov3_global/runtime.py",
                "src/dinov3_global/config.py",
                "src/dinov3_global/engine.py",
                "src/dinov3_global/preprocessing.py",
                "src/dinov3_global/pooling.py",
            )
        },
        data_audit=audit,
    )
    write_json(directory / "plan.json", plan)
    print(f"Protocol frozen: {len(dataset)} test images, {len(members)} EMA heads, T=1", flush=True)
    model, _, _ = load_model(args.ckpt[0], repo_root, dev)
    heads = [model.head]
    for state in states[1:]:
        head = copy.deepcopy(model.head)
        head.load_state_dict(state["ema"], strict=True)
        heads.append(head.to(dev).eval())
    ys, member_logits, meta, verification = predict_heads(
        model, heads, tel, dev, repo_root, progress=lambda value: write_json(directory / "status.json", value)
    )
    expected_paths = [row["image"] for row in membership["test"]]
    expected_labels = np.array([("RR", "RG", "NoR").index(row["label"]) for row in membership["test"]])
    if meta["paths"] != expected_paths or not np.array_equal(ys, expected_labels):
        raise ValueError("Predictions differ from audited official test membership/order")
    pseudo_nor = np.array([row["pseudo_nor"] for row in membership["test"]], dtype=bool)
    for member, logits in zip(members, member_logits):
        out = directory / member["name"]
        out.mkdir()
        result = test_report(ys, logits, meta, pseudo_nor)
        result["checkpoint"] = member
        write_json(out / "report.json", to_jsonable(result))
        save_predictions(out, ys, logits, meta, pseudo_nor)
    lgs = np.mean(member_logits, axis=0)
    rep = test_report(ys, lgs, meta, pseudo_nor)
    if args.calibrated:
        rep["saved_validation_calibration"] = report(
            ys,
            lgs,
            T=float(states[0]["metrics"]["temperature"]),
            cities=meta["cities"],
            max_lamp_h=meta["max_lamp_h"],
        )
    rep.update(
        checkpoints=members,
        split="DTLD official test",
        metric_input="raw logits T=1; AP from final softmax",
        predictor=plan["predictor"],
        protocol_sha256=sha256_file(directory / "plan.json"),
        verification=verification,
    )
    met = rep["metrics"]
    print(json.dumps(to_jsonable(met), indent=2))
    for c, m in rep.get("slices", {}).get("city", {}).items():
        print(
            f"city {c:>12}: mAP={m['mAP']:.4f} acc_bal={m['acc_bal']:.4f} "
            f"n={m['n']} n_classes={m.get('n_classes', 3)}"
        )
    for k, m in rep.get("slices", {}).get("size", {}).items():
        print(
            f"size {k:>14}: mAP={m['mAP']:.4f} acc_bal={m['acc_bal']:.4f} "
            f"n={m['n']} n_classes={m.get('n_classes', 3)}"
        )
    out = directory / "report.json"
    write_json(out, to_jsonable(rep))
    print(f"report -> {out}")
    save_predictions(directory, ys, lgs, meta, pseudo_nor)
    write_json(
        directory / "status.json",
        dict(
            status="complete",
            images_done=len(ys),
            images_total=len(dataset),
            n_models=len(members),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            metrics=to_jsonable(met),
        ),
    )


if __name__ == "__main__":
    main()
