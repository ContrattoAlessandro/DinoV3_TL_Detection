"""Deterministic full-image preprocessing and evidence overlays."""

import numpy as np
from PIL import Image
import torch


def preprocess(path, crop_sides=0, target_hw=(720, 1280)):
    with Image.open(path) as source:
        image = source.convert("RGB")
        w, _ = image.size
        if crop_sides > 0 and w > 2 * crop_sides:
            image = image.crop((crop_sides, 0, w - crop_sides, image.height))
        image = image.resize((target_hw[1], target_hw[0]), Image.Resampling.BICUBIC)
        array = np.array(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1), array


def overlay(image_array, heat, size=(1280, 720)):
    base = Image.fromarray(image_array).convert("RGB").resize(size)
    value = np.asarray(heat, dtype=np.float32)
    value = (value - value.min()) / max(value.max() - value.min(), 1e-6)
    grayscale = Image.fromarray((value * 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    return Image.blend(base, grayscale.convert("RGB"), 0.45)
