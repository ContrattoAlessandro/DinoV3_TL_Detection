"""Prepare the deterministic 1280x720 DTLD RGB dump.

Native 2048x1024 JPEGs are cropped by 114 pixels on each side and resized with
bicubic interpolation. Targets are mapped with the same geometry in data.py.
The prepared dump uses JPEG quality 95, matching the training data.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
