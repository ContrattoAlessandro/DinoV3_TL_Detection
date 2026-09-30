"""Sequential DTLD TIF->JPG conversion (same logic as convert_tif.py).

Original convert_tif.py needs ALL cities extracted at once (~300GB).
This script extracts ONE city zip at a time, converts only that city's
images listed in DTLD_train/test.json, then deletes the extracted TIFs
(and optionally the zip) before the next city.

Conversion itself is IDENTICAL to the paper/repo:
  DriveuDatabase + img.get_labeled_image() (Bayer GB2BGR, >>4)
  + [..., ::-1] (BGR->RGB) + PIL save as .jpg
"""
from __future__ import print_function

import argparse
import logging
import sys
import os
import shutil
import glob
import zipfile
from tqdm import tqdm

from dtld_parsing.driveu_dataset import DriveuDatabase
from PIL import Image

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Small -> large so pipeline is validated early, big zips last
DEFAULT_ORDER = ["Bochum", "Bremen", "Fulda", "Kassel", "Berlin",
                 "Essen", "Frankfurt", "Duesseldorf", "Dortmund",
                 "Koeln", "Hannover"]


def parse_args():
    p = argparse.ArgumentParser(description="Sequential DTLD conversion, same logic as convert_tif.py")
    p.add_argument("--zip_dir", type=str, required=True,
                   help="Folder containing Berlin.zip, Bochum.zip, ...")
    p.add_argument("--label_path", type=str, required=True,
                   help="Folder containing DTLD_train.json / DTLD_test.json (extracted v2.0)")
    p.add_argument("--target_path", type=str, required=True,
                   help="Where to store jpg train/test folders (accumulated across cities)")
    p.add_argument("--work_dir", type=str, required=True,
                   help="Temp dir for one extracted city at a time")
    p.add_argument("--cities", nargs="*", default=None,
                   help="Subset to process, e.g. --cities Bochum Bremen. Default: all in size order.")
    p.add_argument("--delete-zip", action="store_true",
                   help="Delete the city .zip after successful conversion (frees ~150GB total)")
    p.add_argument("--yes", action="store_true",
                   help="Don't prompt; resume/append into existing target_path")
    return p.parse_args()


def convert_city(city, work_dir, label_path, target_path):
    """Convert only images of <city>. Returns (n_ok, n_missing, n_skipped_existing)."""
    n_ok = n_missing = n_skipped = 0
    for split in ["train", "test"]:
        db = DriveuDatabase(os.path.join(label_path, f"DTLD_{split}.json"))
        # data_base_dir remaps ./City/... -> work_dir/City/...
        db.open(work_dir)
        out_dir = os.path.join(target_path, split)
        os.makedirs(out_dir, exist_ok=True)
        for img in tqdm(db.images, desc=f"{city}/{split}", unit="img"):
            # Keep EXACT same filename logic as convert_tif.py:50-58
            if f"/{city}/" not in img.file_path.replace("\\", "/"):
                continue
            img_name = f"{img.file_path.split('/')[-1].split('.')[0]}"
            out_file = os.path.join(out_dir, f"{img_name}.jpg")
            if os.path.exists(out_file):
                n_skipped += 1
                continue
            # Same existence check as DriveuImage.get_image, but skip instead of sys.exit
            # (other cities are not extracted right now)
            resolved = img.file_path  # already remapped by db.open(work_dir)
            if not os.path.isfile(resolved):
                n_missing += 1
                continue
            img_color = img.get_labeled_image()  # Bayer->BGR, >>4 : identical to paper
            img_rgb = img_color[..., ::-1]
            Image.fromarray(img_rgb).save(out_file)
            n_ok += 1
    return n_ok, n_missing, n_skipped


def main(args):
    cities = args.cities or DEFAULT_ORDER
    # keep requested order but validate zips exist
    if args.cities:
        missing = [c for c in cities
                   if not os.path.isfile(os.path.join(args.zip_dir, f"{c}.zip"))]
        if missing:
            raise FileNotFoundError(f"Missing zips for: {missing} in {args.zip_dir}")

    if os.path.exists(args.target_path) and not args.yes:
        ans = input(f"Target {args.target_path} exists, append/resume? type yes\n")
        if ans != "yes":
            raise Exception("Aborting. Use --yes to resume without prompt.")
    os.makedirs(os.path.join(args.target_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.target_path, "test"), exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    total_ok = 0
    for city in cities:
        zip_path = os.path.join(args.zip_dir, f"{city}.zip")
        city_dir = os.path.join(args.work_dir, city)
        if not os.path.isfile(zip_path):
            logging.warning(f"{zip_path} not found, skipping {city}")
            continue
        if os.path.isdir(city_dir):
            logging.info(f"Resuming with already-extracted {city_dir}")
        else:
            logging.info(f"Extracting {zip_path} -> {args.work_dir} ...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(args.work_dir)

        ok, miss, skip = convert_city(city, args.work_dir, args.label_path, args.target_path)
        total_ok += ok
        logging.info(f"{city}: converted={ok} missing_tif={miss} already_done={skip}")

        logging.info(f"Deleting extracted {city_dir} to free space ...")
        shutil.rmtree(city_dir, ignore_errors=True)

        if args.delete_zip:
            logging.info(f"Deleting {zip_path} ...")
            os.remove(zip_path)

    n_train = len(glob.glob(os.path.join(args.target_path, "train", "*.jpg")))
    n_test = len(glob.glob(os.path.join(args.target_path, "test", "*.jpg")))
    logging.info(f"DONE. total newly converted={total_ok}. jpg on disk: train={n_train} test={n_test}")


if __name__ == "__main__":
    main(parse_args())
