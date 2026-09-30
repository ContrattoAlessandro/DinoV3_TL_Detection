"""City-held-out cross-validation - the generalisation gate (M1/M3).

Session-disjoint val (default training) still shares cities with train. A model
that wins on random sessions but collapses on an unseen city will transfer
poorly to other datasets. This script folds DTLD *cities* (Berlin, Koeln, ...)
so every fold validates on geographically unseen drives:

  python dinov3_global/scripts/crossval_city.py --config dinov3_global/configs/base.yaml \
      --folds 4 --epochs 12 --out dinov3_global/runs/cv_city

Reports per-fold mAP/acc_bal + mean/worst city mAP in cv_city_report.json.
Short epochs by default: this is for CONFIG SELECTION, not the final model.
"""
import argparse
import copy
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dinov3_global.dinov3_global.data_global import (DTLDGlobalDataset,
                                                     class_frequencies,
                                                     collate_global)
from dinov3_global.dinov3_global.engine import _build_model, _make_sampler, fit
from dinov3_global.dinov3_global.eval_global import report


def make_folds(items, k):
    """Greedy balanced folds over city groups (city = first path component)."""
    by_city = defaultdict(int)
    for it in items:
        by_city[it["city"]] += 1
    order = sorted(by_city, key=lambda c: -by_city[c])
    folds = [[] for _ in range(k)]
    sizes = [0] * k
    for c in order:
        j = int(np.argmin(sizes))
        folds[j].append(c)
        sizes[j] += by_city[c]
    return folds, by_city


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov3_global/configs/base.yaml")
    ap.add_argument("--out", default="dinov3_global/runs/cv_city")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-items", type=int, default=0, help="smoke: cap items (0 = all)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["optim"]["epochs"] = args.epochs
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_root = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(out_root, exist_ok=True)

    d = cfg["data"]
    ld, ir = os.path.join(repo_root, d["label_dir"]), os.path.join(repo_root, d["img_root"])
    th, tw = d.get("target_hw", [720, 1280])
    full = DTLDGlobalDataset(ld, ir, "train", train=False,
                             crop_sides=d.get("crop_sides", 114),
                             label_crop_sides=d.get("label_crop_sides", 114),
                             target_hw=(th, tw),
                             label_policy=d.get("label_policy", "map_to_nor"),
                             policy_weight=float(d.get("policy_weight", 0.3)))
    folds, by_city = make_folds(full.items, args.folds)
    if args.max_items and args.max_items < len(full.items):
        stride = len(full.items) / args.max_items  # stride sample: keep all cities
        full.items = [full.items[int(i * stride)] for i in range(args.max_items)]
    print("city sizes:", dict(by_city))
    print("folds:", folds)

    results = {}
    for fi, cities in enumerate(folds):
        tr_items = [it for it in full.items if it["city"] not in cities]
        va_items = [it for it in full.items if it["city"] in cities]
        hist = np.bincount([it["y"] for it in va_items], minlength=3).tolist()
        print(f"\n=== fold {fi}: val cities={cities} "
              f"({len(tr_items)} train / {len(va_items)} val, val_hist={hist}) ===")
        if len(va_items) < 20 or len(tr_items) < 20 or min(hist) == 0:
            print("skip: empty fold, tiny fold, or missing class")
            continue
        tr_ds = DTLDGlobalDataset(ld, ir, "train", train=True,
                                  crop_sides=d.get("crop_sides", 114),
                                  label_crop_sides=d.get("label_crop_sides", 114),
                                  target_hw=(th, tw),
                                  label_policy=d.get("label_policy", "map_to_nor"),
                                  policy_weight=float(d.get("policy_weight", 0.3)),
                                  items=tr_items)
        va_ds = DTLDGlobalDataset(ld, ir, "train", train=False,
                                  crop_sides=d.get("crop_sides", 114),
                                  label_crop_sides=d.get("label_crop_sides", 114),
                                  target_hw=(th, tw),
                                  label_policy=d.get("label_policy", "map_to_nor"),
                                  policy_weight=float(d.get("policy_weight", 0.3)),
                                  items=va_items)
        nw = int(cfg.get("loader", {}).get("num_workers", 0))
        bs = int(cfg["optim"].get("batch", 2))
        common = dict(collate_fn=collate_global, num_workers=nw,
                      **({"prefetch_factor": 2, "persistent_workers": True} if nw > 0 else {}))
        sampler = _make_sampler(tr_ds, d.get("sampler", "none"))
        tl = DataLoader(tr_ds, shuffle=(sampler is None), sampler=sampler,
                        batch_size=bs, **common)
        vl = DataLoader(va_ds, shuffle=False, batch_size=bs, **common)

        log_pi = torch.log(torch.tensor(class_frequencies(tr_ds), dtype=torch.float32) + 1e-8)
        torch.manual_seed(int(cfg["optim"].get("seed", 0)) + fi)
        model = _build_model(cfg, dev, repo_root)
        fold_dir = os.path.join(out_root, f"fold{fi}")
        fit(cfg, model, tl, vl, fold_dir, dev, log_pi)

        # final fold report from best.pt
        sd = torch.load(os.path.join(fold_dir, "best.pt"), map_location=dev)
        model.decoder.load_state_dict(sd["ema"])
        from dinov3_global.dinov3_global.engine import predict_split
        ys, lgs, meta = predict_split(model, vl, dev)
        rep = report(ys, lgs, cities=meta["cities"], max_lamp_h=meta["max_lamp_h"])
        results[f"fold{fi}"] = {"val_cities": cities, "metrics": rep["metrics"],
                                "slices": rep.get("slices", {})}
        print(f"fold{fi}: mAP={rep['metrics']['mAP']:.4f} "
              f"acc_bal={rep['metrics']['acc_bal']:.4f}")

    maps = [r["metrics"]["mAP"] for r in results.values() if not np.isnan(r["metrics"]["mAP"])]
    abals = [r["metrics"]["acc_bal"] for r in results.values()]
    summary = {"folds": results,
               "mean_mAP": float(np.mean(maps)) if maps else float("nan"),
               "worst_fold_mAP": float(np.min(maps)) if maps else float("nan"),
               "mean_acc_bal": float(np.mean(abals)) if abals else float("nan")}
    with open(os.path.join(out_root, "cv_city_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nmean mAP={summary['mean_mAP']:.4f} worst fold mAP={summary['worst_fold_mAP']:.4f} "
          f"mean acc_bal={summary['mean_acc_bal']:.4f}")
    print(f"report -> {os.path.join(out_root, 'cv_city_report.json')}")


if __name__ == "__main__":
    main()
