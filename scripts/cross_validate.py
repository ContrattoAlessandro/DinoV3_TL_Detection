"""Audited, resumable city-disjoint cross-validation of the v5 architecture."""

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from dinov3_global.config import load_config
from dinov3_global.data import DTLDGlobalDataset, class_frequencies, collate_global
from dinov3_global.engine import _build_model, _make_sampler, fit, predict_split, set_seed, _pseudo_flags
from dinov3_global.metrics import report
from dinov3_global.validation import make_folds, digest, atomic_json, fold_manifest
from dinov3_global.runtime import loader_kwargs, validate_runtime, set_training_threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="runs/v5/city_cv")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-items", type=int, default=0, help="smoke: cap items (0 = all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fold-indices", type=int, nargs="+", help="run selected folds only")
    ap.add_argument("--fold-manifest", type=Path, help="shared immutable membership audit")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume last epoch / skip completed folds")
    ap.add_argument("--warmup-epochs", type=int)
    ap.add_argument("--early-stop-patience", type=int)
    ap.add_argument("--runtime-loader", type=Path, help="audited execution-only loader settings")
    args = ap.parse_args()
    runtime = validate_runtime(json.loads(args.runtime_loader.read_text())) if args.runtime_loader else None
    initial_cpu_threads = torch.get_num_threads()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    cfg = load_config(cfg_path)
    cfg = copy.deepcopy(cfg)
    cfg["optim"]["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        cfg["optim"]["warmup_epochs"] = args.warmup_epochs
    if args.early_stop_patience is not None:
        cfg["optim"]["early_stop_patience"] = args.early_stop_patience
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_root = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(out_root, exist_ok=True)

    d = cfg["data"]
    ld, ir = os.path.join(repo_root, d["label_dir"]), os.path.join(repo_root, d["img_root"])
    th, tw = d.get("target_hw", [720, 1280])
    full = DTLDGlobalDataset(
        ld,
        ir,
        "train",
        train=False,
        crop_sides=d.get("crop_sides", 114),
        label_crop_sides=d.get("label_crop_sides", 114),
        target_hw=(th, tw),
        label_policy=d.get("label_policy", "map_to_nor"),
        policy_weight=float(d.get("policy_weight", 0.3)),
    )
    folds, by_city = make_folds(full.items, args.folds)
    if args.max_items and args.max_items < len(full.items):
        stride = len(full.items) / args.max_items  # stride sample: keep all cities
        full.items = [full.items[int(i * stride)] for i in range(args.max_items)]
    print("city sizes:", dict(by_city))
    print("folds:", folds)

    annotation_sha = hashlib.sha256(Path(ld, "DTLD_train.json").read_bytes()).hexdigest()
    manifest = fold_manifest(full.items, folds, d, annotation_sha)
    manifest_path = args.fold_manifest or Path(out_root, "fold_manifest.json")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("fold manifest differs from current data; use a new experiment")
    else:
        atomic_json(manifest_path, manifest)
    atomic_json(Path(out_root, "fold_manifest.json"), manifest)
    config_path = Path(out_root, "effective_config.json")
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("effective config differs from existing run; use a new --out")
    atomic_json(config_path, cfg)
    if args.audit_only:
        print(f"Audited {len(full)} frames, {len(folds)} folds -> {manifest_path}")
        return

    selected = set(range(args.folds) if args.fold_indices is None else args.fold_indices)
    if not selected.issubset(range(args.folds)):
        raise ValueError("invalid fold index")
    results = {}
    for fi in range(args.folds):
        completed = Path(out_root, f"fold{fi}", "fold_report.json")
        if completed.exists():
            results[f"fold{fi}"] = json.loads(completed.read_text())

    def write_summary():
        maps = [r["metrics"]["mAP"] for r in results.values()]
        abals = [r["metrics"]["acc_bal"] for r in results.values()]
        summary = {
            "folds": results,
            "completed_folds": len(results),
            "total_folds": args.folds,
            "complete": len(results) == args.folds,
            "manifest_sha256": digest(manifest),
            "config_sha256": digest(cfg),
            "mean_mAP": float(np.mean(maps)) if maps else None,
            "worst_fold_mAP": float(np.min(maps)) if maps else None,
            "mean_acc_bal": float(np.mean(abals)) if abals else None,
        }
        atomic_json(Path(out_root, "cv_city_report.json"), summary)
        return summary

    for fi, cities in enumerate(folds):
        if fi not in selected:
            continue
        if f"fold{fi}" in results:
            if not args.resume:
                raise ValueError(f"fold{fi} already complete; use --resume or a new --out")
            print(f"fold{fi}: already complete; skipped")
            continue
        fold_dir = os.path.join(out_root, f"fold{fi}")
        last_path = Path(fold_dir, "last.pt")
        has_artifacts = any(Path(fold_dir, n).exists() for n in ("best.pt", "last.pt", "history.jsonl"))
        if has_artifacts and not args.resume:
            raise ValueError(f"fold{fi} has training artifacts; use --resume or a new --out")
        if has_artifacts and not last_path.exists():
            raise ValueError(f"fold{fi} cannot resume without last.pt")
        tr_items = [it for it in full.items if it["city"] not in cities]
        va_items = [it for it in full.items if it["city"] in cities]
        hist = np.bincount([it["y"] for it in va_items], minlength=3).tolist()
        print(
            f"\n=== fold {fi}: val cities={cities} "
            f"({len(tr_items)} train / {len(va_items)} val, val_hist={hist}) ==="
        )
        if len(va_items) < 20 or len(tr_items) < 20 or min(hist) == 0:
            raise ValueError("empty fold, tiny fold, or missing class")
        tr_ds = DTLDGlobalDataset(
            ld,
            ir,
            "train",
            train=True,
            crop_sides=d.get("crop_sides", 114),
            label_crop_sides=d.get("label_crop_sides", 114),
            target_hw=(th, tw),
            label_policy=d.get("label_policy", "map_to_nor"),
            policy_weight=float(d.get("policy_weight", 0.3)),
            items=tr_items,
        )
        va_ds = DTLDGlobalDataset(
            ld,
            ir,
            "train",
            train=False,
            crop_sides=d.get("crop_sides", 114),
            label_crop_sides=d.get("label_crop_sides", 114),
            target_hw=(th, tw),
            label_policy=d.get("label_policy", "map_to_nor"),
            policy_weight=float(d.get("policy_weight", 0.3)),
            items=va_items,
        )
        bs = int(cfg["optim"].get("batch", 2))
        common = dict(collate_fn=collate_global, **loader_kwargs(cfg, runtime))
        sampler = _make_sampler(tr_ds, d.get("sampler", "none"))
        tl = DataLoader(tr_ds, shuffle=(sampler is None), sampler=sampler, batch_size=bs, **common)
        vl = DataLoader(va_ds, shuffle=False, batch_size=bs, **common)

        log_pi = torch.log(torch.tensor(class_frequencies(tr_ds), dtype=torch.float32) + 1e-8)
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["optim"]["seed"] = int(cfg["optim"].get("seed", 0)) + fi
        # Keep initialisation on the original CPU threading setting. Runtime
        # tuning applies only after the head has been created/restored.
        torch.set_num_threads(initial_cpu_threads)
        set_seed(fold_cfg["optim"]["seed"])
        model = _build_model(fold_cfg, dev, repo_root)
        # Different head sizes consume different initialisation draws. Reset
        # before fitting so sampler order starts from the same fold seed.
        set_seed(fold_cfg["optim"]["seed"])
        resume_kw = {}
        if last_path.exists():
            sd = torch.load(last_path, map_location="cpu", weights_only=True)
            if sd["cfg"] != fold_cfg:
                raise ValueError("checkpoint config differs from current fold")
            model.head.load_state_dict(sd["head"])
            resume_kw = dict(
                start_ep=sd["epoch"],
                opt_state=sd.get("opt"),
                scaler_state=sd.get("scaler"),
                ema_state=sd.get("ema"),
                best_init=sd["best_score"],
                stale_init=sd.get("stale_epochs", 0),
                rng_state=sd.get("rng_state"),
            )
            del sd
        set_training_threads(runtime)
        if runtime is not None:
            os.makedirs(fold_dir, exist_ok=True)
            with open(Path(fold_dir, "execution_history.jsonl"), "a") as f:
                f.write(
                    json.dumps(
                        {
                            "resume_epoch": resume_kw.get("start_ep", 0),
                            "runtime_loader": runtime,
                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        }
                    )
                    + "\n"
                )
            print(f"Execution-only loader override: {runtime}; scientific config unchanged")
        if not Path(fold_dir, "STOPPED").exists():
            fit(
                fold_cfg, model, tl, vl, fold_dir, dev, log_pi, pseudo_flags=_pseudo_flags(va_ds), **resume_kw
            )

        # final fold report from best.pt
        sd = torch.load(os.path.join(fold_dir, "best.pt"), map_location="cpu", weights_only=True)
        model.head.load_state_dict(sd["ema"])
        ys, lgs, meta = predict_split(model, vl, dev)
        rep = report(ys, lgs, cities=meta["cities"], max_lamp_h=meta["max_lamp_h"])
        results[f"fold{fi}"] = {
            "val_cities": cities,
            "metrics": rep["metrics"],
            "slices": rep.get("slices", {}),
            "best_epoch": sd["epoch"],
            "seed": fold_cfg["optim"]["seed"],
            "audit": manifest["folds"][fi],
        }
        np.savez_compressed(
            Path(fold_dir, "val_predictions.npz"),
            labels=ys,
            logits=lgs,
            paths=np.asarray(meta["paths"]),
            cities=np.asarray(meta["cities"]),
        )
        atomic_json(Path(fold_dir, "fold_report.json"), results[f"fold{fi}"])
        write_summary()
        print(f"fold{fi}: mAP={rep['metrics']['mAP']:.4f} acc_bal={rep['metrics']['acc_bal']:.4f}")
        del model, sd, tl, vl, tr_ds, va_ds, resume_kw
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    summary = write_summary()
    print(
        f"\nCompleted {len(results)}/{args.folds} folds: {summary['mean_mAP']=} "
        f"{summary['worst_fold_mAP']=} {summary['mean_acc_bal']=}"
    )
    print(f"report -> {os.path.join(out_root, 'cv_city_report.json')}")


if __name__ == "__main__":
    main()
