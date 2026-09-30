"""Final test evaluation - run ONCE per model after all selection is done.

  python dinov3_global/scripts/evaluate.py --ckpt dinov3_global/runs/exp2/best.pt
  # multi-ckpt = logits ensemble (M3 soup/seed ensemble):
  python dinov3_global/scripts/evaluate.py --ckpt a/best.pt b/best.pt c/best.pt

Prints the paper-protocol metrics (AP_RR/AP_RG/AP_NoR/mAP/acc_bal) plus
generalisation slices (per-city, per-lamp-size) and calibration, and writes
runs/eval_test_<name>.json. Beat check: Faster-RCNN w/attr 81.5 mAP / 79.3
acc_bal; Trinci et al. end-to-end 83.1 / 80.8 (paper Table II).

Protocol notes (v4):
- The temperature is fit on the session-disjoint **val** split (per ckpt, or on
  the averaged logits for an ensemble) and then applied to test. The old code
  reused the T stored in the checkpoint - which was fit on val but applied to
  an ensemble's *last* member only.
- The report carries metrics at BOTH T = 1 (comparable with exp2/exp3 reports
  and with best.pt selection) and at the fitted T (calibrated probabilities).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dinov3_global.dinov3_global.engine import (build_loaders, build_test_loader,
                                                load_model, predict_split)
from dinov3_global.dinov3_global.eval_global import (fit_temperature, report,
                                                     to_jsonable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, nargs="+")
    ap.add_argument("--config", default="dinov3_global/configs/base.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="report JSON path")
    ap.add_argument("--no-calibration", action="store_true",
                    help="skip the val pass used to fit the temperature")
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    lgs_list, val_lgs_list, ys, meta, cfg = [], [], None, None, None
    for p in args.ckpt:
        model, cfg, sd = load_model(p, repo_root, dev)
        tel, _ = build_test_loader(cfg, repo_root)
        ys, lgs, meta = predict_split(model, tel, dev)
        lgs_list.append(lgs)
        if not args.no_calibration:
            _, vl, _, _, _ = build_loaders(cfg, repo_root)
            vys, vlgs, _ = predict_split(model, vl, dev)
            val_lgs_list.append((vys, vlgs))
        del model
    lgs = np.mean(lgs_list, axis=0)

    # temperature from val only (never test); ensemble fits T on averaged logits
    if val_lgs_list:
        vys = val_lgs_list[0][0]
        vlgs = np.mean([v for _, v in val_lgs_list], axis=0)
        T = fit_temperature(vlgs, vys)
        print(f"temperature fit on val (n={len(vys)}): T={T:.4f}")
    else:
        T = 1.0

    rep = report(ys, lgs, T=T, cities=meta["cities"], max_lamp_h=meta["max_lamp_h"])
    rep["metrics_T1"] = report(ys, lgs, T=1.0)["metrics"]
    met = rep["metrics"]
    print(json.dumps(to_jsonable(met), indent=2))
    print("at T=1: " + json.dumps(to_jsonable(rep["metrics_T1"])))
    for c, m in rep.get("slices", {}).get("city", {}).items():
        print(f"city {c:>12}: mAP={m['mAP']:.4f} acc_bal={m['acc_bal']:.4f} "
              f"n={m['n']} n_classes={m.get('n_classes', 3)}")
    for k, m in rep.get("slices", {}).get("size", {}).items():
        print(f"size {k:>14}: mAP={m['mAP']:.4f} acc_bal={m['acc_bal']:.4f} "
              f"n={m['n']} n_classes={m.get('n_classes', 3)}")
    print("beat check DTLD: Faster-RCNN w/attr 81.5 mAP / 79.3 acc_bal; "
          "Trinci end-to-end 83.1 / 80.8 (paper Table II)")

    out = args.out or os.path.join(repo_root, "dinov3_global", "runs",
                                   f"eval_test_{'_'.join(os.path.basename(os.path.dirname(p)) for p in args.ckpt)}.json")
    with open(out, "w") as f:
        json.dump(to_jsonable(rep), f, indent=2, allow_nan=False)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
