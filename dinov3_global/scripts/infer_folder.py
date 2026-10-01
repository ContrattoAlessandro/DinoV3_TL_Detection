"""Run the global RR/RG/NoR model on ANY folder of images (no labels needed).

This is the tool for manual cross-dataset inspection:

  python dinov3_global/scripts/infer_folder.py --ckpt runs/exp2/best.pt \
      --images /path/to/other_dataset [--out runs/infer_folder] [--overlays]

Outputs:
  - predictions.csv: file, P_RR, P_RG, P_NoR, pred
  - overlays (with --overlays): lamp/rel/rr/rg evidence heatmaps, useful to
    judge *why* the model says what it says on a foreign dataset.

Images are resized to 1280x720 (bicubic). Use --crop-sides 114 for 2048x1024
DTLD-style captures (2:1 -> 16:9 crop); other aspect ratios just get resized.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dinov3_global.dinov3_global.engine import load_model

NAMES = ["RR", "RG", "NoR"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def overlay(img_arr: np.ndarray, heat: np.ndarray, size=(1280, 720)) -> Image.Image:
    base = Image.fromarray(img_arr).convert("RGB").resize(size)
    a = np.asarray(heat, dtype=np.float32)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
    hm = Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return Image.blend(base, hm.convert("RGB"), 0.45)


def preprocess(path: str, crop_sides: int = 0, target_hw=(720, 1280)):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if crop_sides > 0 and w > 2 * crop_sides:
        img = img.crop((crop_sides, 0, w - crop_sides, h))
    img = img.resize((target_hw[1], target_hw[0]), Image.BICUBIC)
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1), np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True, help="folder (scanned recursively)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--crop-sides", type=int, default=0)
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = args.out or os.path.join(repo_root, "dinov3_global", "runs", "infer_folder")
    os.makedirs(out, exist_ok=True)

    model, cfg, sd = load_model(args.ckpt, repo_root, dev)
    T = float(sd.get("metrics", {}).get("temperature", 1.0))

    files = []
    for root, _, names in os.walk(args.images):
        for n in sorted(names):
            if n.lower().endswith(EXTS):
                files.append(os.path.join(root, n))
    files.sort()
    if args.max_images:
        files = files[:args.max_images]
    print(f"{len(files)} images from {args.images} (calibration T={T:.3f})")

    rows = []
    target_hw = tuple(cfg.get("data", {}).get("target_hw", (720, 1280)))
    with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
        for p in files:
            x, vis = preprocess(p, args.crop_sides, target_hw)
            out_m = model(x.unsqueeze(0).to(dev).float().div_(255.0))
            raw_lg = out_m["logits"].float().cpu().numpy()[0]
            lg = raw_lg / T
            e = np.exp(lg - lg.max())
            prob = e / e.sum()
            pred = int(prob.argmax())
            rows.append([os.path.relpath(p, args.images), *[f"{v:.4f}" for v in prob],
                         NAMES[pred], *[f"{v:.8f}" for v in raw_lg]])
            if args.overlays:
                maps = out_m.get("maps")
                stem = os.path.splitext(os.path.basename(p))[0]
                if maps is not None:
                    heats = {"lamp": maps["lamp_grid"][0].float().cpu().numpy(),
                             "rel": maps["rel_grid"][0].float().cpu().numpy(),
                             "rr": maps["rr"][0].float().cpu().numpy(),
                             "rg": maps["rg"][0].float().cpu().numpy()}
                else:
                    heats = {"attn": out_m["attn"][0].float().cpu().numpy()}
                for k, h in heats.items():
                    overlay(vis, h).save(os.path.join(out, f"{stem}_{NAMES[pred]}_{k}.jpg"))
            if len(rows) % 200 == 0:
                print(f"  {len(rows)}/{len(files)}")

    csv_path = os.path.join(out, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "P_RR", "P_RG", "P_NoR", "pred",
                    "logit_RR", "logit_RG", "logit_NoR"])
        w.writerows(rows)
    counts = {n: sum(1 for r in rows if r[-1] == n) for n in NAMES}
    print(f"predictions -> {csv_path}  counts={counts}")


if __name__ == "__main__":
    main()
