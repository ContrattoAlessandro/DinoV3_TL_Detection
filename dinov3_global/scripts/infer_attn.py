"""Token-map overlays for visual inspection (MIL head).

  python dinov3_global/scripts/infer_attn.py --ckpt runs/exp2/best.pt --max-frames 20

For each frame saves 4 overlays (1280x720):
  lamp  - learned lampness   (does the head find the lamps?)
  rel   - learned relevance  (does it look at ego-lane lamps?)
  rr    - RR evidence        (lamp*rel*(red+yellow+red_yellow))
  rg    - RG evidence        (lamp*rel*green)
Filename encodes true/pred class: 0001_t0p0_RR.jpg.
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dinov3_global.dinov3_global.data_global import DTLDGlobalDataset
from dinov3_global.dinov3_global.engine import load_model

NAMES = ["RR", "RG", "NoR"]


def overlay(img_arr: np.ndarray, heat: np.ndarray) -> Image.Image:
    base = Image.fromarray(img_arr).convert("RGB").resize((1280, 720))
    a = np.asarray(heat, dtype=np.float32)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
    hm = Image.fromarray((a * 255).astype(np.uint8)).resize((1280, 720), Image.BILINEAR)
    return Image.blend(base, hm.convert("RGB"), 0.45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="dinov3_global/configs/base.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, cfg, _ = load_model(args.ckpt, repo_root, dev)

    d = cfg["data"]
    ds = DTLDGlobalDataset(os.path.join(repo_root, d["label_dir"]),
                           os.path.join(repo_root, d["img_root"]), args.split,
                           train=False, crop_sides=d.get("crop_sides", 114),
                           label_crop_sides=d.get("label_crop_sides", 114))
    outdir = os.path.join(repo_root, "dinov3_global", "runs", "attn")
    os.makedirs(outdir, exist_ok=True)
    n = min(args.max_frames, len(ds) - args.start)
    with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
        for i in range(n):
            b = ds[args.start + i]
            img = b["image"].unsqueeze(0).to(dev).float().div_(255.0)
            out = model(img)
            pred = int(out["logits"].argmax(1)[0])
            gt = int(b["label"])
            base = b["image"].permute(1, 2, 0).numpy()
            maps = out.get("maps")
            if maps is not None:
                heats = {"lamp": maps["lamp_grid"][0].float().cpu().numpy(),
                         "rel": maps["rel_grid"][0].float().cpu().numpy(),
                         "rr": maps["rr"][0].float().cpu().numpy(),
                         "rg": maps["rg"][0].float().cpu().numpy()}
            else:  # legacy attn head
                heats = {"attn": out["attn"][0].float().cpu().numpy()}
            stem = f"{args.start+i:05d}_t{gt}p{pred}_{NAMES[pred]}"
            for k, h in heats.items():
                overlay(base, h).save(os.path.join(outdir, f"{stem}_{k}.jpg"))
            print("saved", stem)
    print(f"saved to {outdir}")


if __name__ == "__main__":
    main()
