"""Diagnose evidence vs CLS scene shortcuts on unlabeled OOD images.

Counterfactuals change trained components at inference and are diagnostic only.
They are not replacement models and do not measure OOD accuracy without labels.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dinov3_global.dinov3_global.engine import load_model
from dinov3_global.scripts.infer_folder import EXTS, NAMES, preprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "dinov3_global/runs/ood_context_audit")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_model(args.ckpt, str(REPO), dev)
    head = model.decoder
    if head.head_kind != "mil" or not head.nor_sees_evidence:
        raise ValueError("This audit requires an MIL head with nor_sees_evidence")
    if head.nor_scene_bound is not None:
        raise ValueError("This audit targets legacy unbounded CLS heads, not bounded v5")
    captured = {}
    handles = [head.nor_head.register_forward_pre_hook(lambda m, a: captured.update(nor_input=a[0]))]
    if head.cls_modulates_evidence:
        handles.append(head.cls_mod.register_forward_hook(lambda m, a, out: captured.update(scene_prior=out)))
    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in EXTS)
    if args.max_images:
        files = files[:args.max_images]
    rows = []
    with torch.inference_mode(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        for p in files:
            x, _ = preprocess(str(p), target_hw=tuple(cfg["data"]["target_hw"]))
            out = model(x.unsqueeze(0).to(dev).float().div_(255))
            logits = out["logits"][0].float()
            prior = captured.get("scene_prior", torch.zeros(1, 2, device=dev))[0].float()
            if head.scene_prior_bound is not None:
                prior = head.scene_prior_bound * prior.tanh()
            evidence = captured["nor_input"][0, -2:].float().clone()
            neutral = captured["nor_input"].clone()
            neutral[:, :-2] = 0
            neutral_nor = head.nor_head(neutral)[0, 0].float()
            a = logits.clone(); a[:2] -= prior
            b = logits.clone(); b[2] = neutral_nor
            c = a.clone(); c[2] = neutral_nor
            maps = out["maps"]
            rows.append({"file": p.relative_to(args.images).as_posix(),
                         "pred": NAMES[int(logits.argmax())],
                         "without_scene_prior": NAMES[int(a.argmax())],
                         "without_nor_cls": NAMES[int(b.argmax())],
                         "without_both": NAMES[int(c.argmax())],
                         "evidence_RR": float(evidence[0]), "evidence_RG": float(evidence[1]),
                         "scene_prior_RR": float(prior[0]), "scene_prior_RG": float(prior[1]),
                         "nor_logit": float(logits[2]), "nor_neutral_cls": float(neutral_nor),
                         "lamp_peak": float(maps["lamp"].max()),
                         "rel_peak": float(maps["rel"].max())})
    for handle in handles:
        handle.remove()
    if not rows:
        raise ValueError("No images found")
    summary = {"n": len(rows), "checkpoint": args.ckpt,
               "prediction_counts": {mode: {c: sum(r[mode] == c for r in rows) for c in NAMES}
                                     for mode in ("pred", "without_scene_prior", "without_nor_cls", "without_both")},
               "means": {key: float(np.mean([r[key] for r in rows])) for key in rows[0]
                         if isinstance(rows[0][key], float)},
               "caution": "Counterfactual ablations, not validated models. No accuracy inferred from unlabeled images."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
