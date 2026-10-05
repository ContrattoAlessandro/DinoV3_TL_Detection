"""Convert annotated DTLD Bayer TIFFs to RGB JPEGs using the native parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from dtld_parsing.driveu_dataset import DriveuDatabase

REPO = Path(__file__).resolve().parents[2]


def convert_split(raw, labels, destination, split, overwrite=False):
    database = DriveuDatabase(str(labels / f"DTLD_{split}.json"))
    if not database.open(str(raw)):
        raise RuntimeError(f"Cannot open DTLD {split} annotations under {labels}")
    target = destination / split
    target.mkdir(parents=True, exist_ok=True)
    converted, skipped = 0, 0
    for frame in tqdm(database.images, desc=f"Convert {split}"):
        path = target / f"{Path(frame.file_path).stem}.jpg"
        if path.exists() and not overwrite:
            skipped += 1
            continue
        bgr = frame.get_labeled_image()
        if bgr is None:
            raise RuntimeError(f"Cannot decode {frame.file_path}")
        # Preserve the original parser's Bayer conversion and JPEG defaults.
        Image.fromarray(bgr[..., ::-1]).save(path)
        converted += 1
    print(f"{split}: {converted} converted; {skipped} existing files retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=REPO / "datasets/DTLD")
    parser.add_argument("--labels", type=Path, default=REPO / "datasets/DTLD/v2.0")
    parser.add_argument("--out", type=Path, default=REPO / "datasets/DTLD_jpg")
    parser.add_argument("--overwrite", action="store_true", help="replace existing JPEGs individually")
    args = parser.parse_args()
    if not args.raw.is_dir() or not args.labels.is_dir():
        parser.error("The raw data and annotation directories must exist")
    destination = args.out.resolve()
    if destination in (args.raw.resolve(), args.labels.resolve()):
        parser.error("The output must be separate from the raw data and annotations")
    for split in ("train", "test"):
        convert_split(args.raw, args.labels, destination, split, args.overwrite)


if __name__ == "__main__":
    main()
