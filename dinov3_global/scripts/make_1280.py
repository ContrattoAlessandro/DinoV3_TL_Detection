"""Offline dump: datasets/DTLD_jpg (2048x1024) -> datasets/DTLD_1280 (1280x720).

Bakes in the exact deterministic preprocessing from data_global.py
(crop 114px each side -> BICUBIC resize) so training dataloaders skip the
expensive per-epoch 2048px decode+resize. Label semantics unchanged.

Usage:
  python dinov3_global/scripts/make_1280.py --src datasets/DTLD_jpg --dst datasets/DTLD_1280

Afterwards set in configs/base.yaml: img_root: datasets/DTLD_1280, crop_sides: 0.
Windows-safe via ProcessPoolExecutor under __main__ guard.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image
from tqdm import tqdm

SRC_W, SRC_H = 2048, 1024
CROP_SIDES = 114
TARGET = (1280, 720)


def _convert_one(args) -> str:
    src, dst = args
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w == SRC_W and h == SRC_H:
                im = im.crop((CROP_SIDES, 0, w - CROP_SIDES, h))
            if im.size != TARGET:
                im = im.resize(TARGET, Image.BICUBIC)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im.save(dst, "JPEG", quality=95)
        return ""
    except Exception as e:  # noqa: BLE001 — report path, keep going
        return f"{src}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/DTLD_jpg")
    ap.add_argument("--dst", default="datasets/DTLD_1280")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src = os.path.join(repo_root, args.src)
    dst = os.path.join(repo_root, args.dst)
    jobs = []
    for split in ("train", "test"):
        sdir = os.path.join(src, split)
        for f in sorted(os.listdir(sdir)):
            if f.lower().endswith(".jpg"):
                jobs.append((os.path.join(sdir, f), os.path.join(dst, split, f)))
    print(f"{len(jobs)} images -> {dst} with {args.workers} workers")
    errs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in tqdm(ex.map(_convert_one, jobs), total=len(jobs)):
            if msg:
                errs += 1
                print(msg)
    print(f"done, errors={errs}")


if __name__ == "__main__":
    main()
