"""Predict RR/RG/NoR from a folder of images using a saved DinoGlobal-MIL checkpoint."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.engine import load_model
from dinov3_global.inference import overlay
from dinov3_global.preprocessing import preprocess_for_config
from dinov3_global.metrics import softmax_np

CLASSES = ["RR", "RG", "NoR"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO / "runs/inference")
    parser.add_argument("--crop-sides", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--overlays", action="store_true")
    parser.add_argument("--calibrated", action="store_true", help="use the saved DTLD validation temperature")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.crop_sides < 0 or args.max_images < 0:
        parser.error("crop-sides and max-images must be nonnegative")
    image_root = args.images.resolve()
    files = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.max_images:
        files = files[: args.max_images]
    if not files:
        raise ValueError(f"No images found under {image_root}")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "predictions.csv").exists():
        raise ValueError("Predictions already exist; choose a new --out")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, checkpoint = load_model(args.ckpt, str(REPO), device)
    temperature = float(checkpoint["metrics"]["temperature"]) if args.calibrated else 1.0
    counts = Counter()
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "P_RR", "P_RG", "P_NoR", "pred", "logit_RR", "logit_RG", "logit_NoR"])
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            for index, path in enumerate(files):
                image, visible, metadata = preprocess_for_config(path, cfg, args.crop_sides)
                metadata = {k: v.unsqueeze(0).to(device) for k, v in metadata.items()}
                result = model(image.unsqueeze(0).to(device).float().div_(255), **metadata)
                logits = result["logits"][0].float().cpu().numpy()
                probabilities = softmax_np((logits / temperature)[None])[0]
                predicted = CLASSES[int(probabilities.argmax())]
                counts[predicted] += 1
                relative = path.relative_to(image_root)
                writer.writerow([relative.as_posix(), *probabilities.tolist(), predicted, *logits.tolist()])
                if args.overlays:
                    for name, key in (
                        ("lamp", "lamp_grid"),
                        ("relevance", "rel_grid"),
                        ("RR", "rr"),
                        ("RG", "rg"),
                    ):
                        output = args.out / "overlays" / relative.parent / f"{relative.stem}_{name}.jpg"
                        output.parent.mkdir(parents=True, exist_ok=True)
                        overlay(
                            visible,
                            result["maps"][key][0].float().cpu().numpy(),
                            size=(visible.shape[1], visible.shape[0]),
                        ).save(output)
                if (index + 1) % 200 == 0:
                    print(f"{index + 1}/{len(files)}")
    print(f"Predictions: {args.out / 'predictions.csv'}; counts={dict(counts)}; T={temperature:.4f}")


if __name__ == "__main__":
    main()
